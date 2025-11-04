#Run the kSZ 4-point function
#
#
from os.path import join as opj, dirname
import os

#import sys
#sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ksz4.cross import four_split_K, split_phi_to_cl, mcrdn0_s4, mcn1_ABCD, mcrdn0
from ksz4.reconstruction import setup_ABCD_recon
from ksz4.utils import get_cl_smooth
from pixell import curvedsky, enmap, utils as putils
from scipy.signal import savgol_filter
from cmbsky import safe_mkdir, get_disable_mpi
from falafel import utils, qe
import healpy as hp
import yaml
import argparse
from orphics import maps, mpi
import numpy as np
import pickle
from string import Template
from pytempura import noise_spec


disable_mpi = get_disable_mpi()
if not disable_mpi:
    from mpi4py import MPI
    comm = mpi.MPI.COMM_WORLD
else:
    comm = None

NSPLIT=4
#MASK_PATH="/global/homes/m/maccrann/cmb/lensing/code/so-lenspipe/bin/planck/act_mask_20220316_GAL060_rms_70.00_d2.fits"
#DATA_DIR="/global/cscratch1/sd/maccrann/cmb/act_dr6/ilc_cldata_smooth-301-2_modelsub_v1"

#Read in kSZ alms
default_ksz_alm = utils.change_alm_lmax(hp.fitsfunc.read_alm("../tests/alms_4e3_2048_50_50_ksz.fits"),
                                6000)
#cl_rksz = get_cl_fg_smooth(ksz_alm)
DEFAULT_CL_RKSZ=get_cl_smooth(default_ksz_alm)

with open(opj(dirname(__file__),"../run_auto_defaults.yml"),'rb') as f:
    DEFAULTS=yaml.safe_load(f)

def get_config(defaults=DEFAULTS,
               description="Do 4pt measurement"):
    parser = argparse.ArgumentParser(description='Prepare Planck alms')
    #only required arg is output_dir
    parser.add_argument("output_dir", type=str)
    #can also add a config file
    parser.add_argument("-c", "--config_file", type=str, default=None)
    #and otherwise variables are set to defaults,
    #or overwritten on command line
    updated_defaults = {}
    for key,val in defaults.items():
        nargs=None
        if val=="iNone":
            t,val = int, None
        elif val=="fNone":
            t,val = float, None
        elif val=="sNone":
            t,val = str, None
        elif val=="liNone":
            t,val,nargs = int,None,'*'
        elif val=="lfNone":
            t,val,nargs = float,None,'*'
        elif val=="lsNone":
            t,val,nargs = str,None,'*'
        elif isinstance(val, list):
            t = type(val[0])
            nargs='*'
        elif val is True:
            raise ValueError("not set up for command line provided False. So for now doesn't make sense to have default True values here")
        else:
            t = type(val)
        updated_defaults[key] = val
        parser.add_argument("--%s"%key, type=t, nargs=nargs)
        
    #This parser will have optional arguments set to
    #None if not set. So if that's the case, replace
    #with the default value
    args_dict=vars(parser.parse_args())
    config_file = args_dict.pop("config_file")
    output_dir = args_dict.pop("output_dir")

    config = {}
    if config_file is None:
        config_from_file = {}
    else:
        with open(config_file,"rb") as f:
            config_from_file = yaml.safe_load(f)
        config.update(config_from_file)
        
    config['output_dir'] = output_dir

    for key,val in args_dict.items():
        if key in config:
            if val is not None:
                config[key] = val
        else:
            if val is not None:
                config[key] = val
            else:                
                config[key] = updated_defaults[key]

    #couple of special things
    config["est_maps"] = ((config["est_maps"]).strip()).split("_")    
    config["do_qe"] = (not config["skip_qe"])
    config["do_auto"] = (not config["skip_auto"])
                
    #I think most convenient to return
    #a namespace
    from argparse import Namespace
    config_namespace = Namespace(**config)

    #also return the config from the file,
    #and the defaults
    return config_namespace, config_from_file, dict(DEFAULTS)

def mean_alm_from_splits(split_alms):
    n_split = 0
    alm_sum = 0
    for split_alm in split_alms:
        alm_sum += split_alm
        n_split += 1
    return alm_sum / n_split
    

def get_alms_dr6(path_template, freqs=["90","150"], sim_seed=None, mlmax=None,
                 apply_extra_mask=None, verbose=False):
    alms = {}
    print(path_template)
    for freq in freqs:
        alms[freq] = []
        for split in range(NSPLIT):
            map_filename = Template(path_template).substitute(freq=freq, split=split)
            if verbose:
                print("reading data alm from %s"%map_filename)
            alm = hp.read_alm(map_filename)
            lmax=hp.Alm.getlmax(len(alm))
            if apply_extra_mask is not None:
                if verbose:
                    print("applying extra mask")
                z = enmap.zeros(shape=apply_extra_mask.shape,
                                wcs=apply_extra_mask.wcs)
                m = curvedsky.alm2map(alm, z, tweak=True)
                m *= apply_extra_mask
                alm = curvedsky.map2alm(m, lmax=lmax, tweak=True)
                
            if mlmax is not None:
                alm = utils.change_alm_lmax(alm, mlmax)
            alms[freq].append(alm)
            
    return alms

def get_sim_model_from_data(path_template, freqs=["90","150"], mlmax=None,
                            nsplit=NSPLIT, sg_window=51, sg_order=2,
                            w2=1.):
    alms = {}
    print(path_template)
    for freq in freqs:
        alms[freq] = []
        for split in range(nsplit):
            map_filename = Template(path_template).substitute(freq=freq, split=split)
            alm = hp.read_alm(map_filename)
            if mlmax is not None:
                alm = utils.change_alm_lmax(alm, mlmax)
            alms[freq].append(alm)

    #we want noise spectra for each frequency (get this from diffs)
    #and cross spectra for each frequency pair
    cl_signal={}
    cl_noise={}
    cl_split_auto = {}
    cl_split_cross = {}
    for i,freq_i in enumerate(freqs):
        for j,freq_j in enumerate(freqs):
            if (freq_j,freq_i) in cl_signal:
                continue

            #cross correlations
            cl_split_auto[(freq_i,freq_j)] = []
            cl_split_cross[(freq_i,freq_j)] = []
            for p in range(nsplit):
                for q in range(p,nsplit):
                    if (p==q):
                        cl_split_auto[(freq_i,freq_j)].append(
                            curvedsky.alm2cl(alms[freq_i][p],
                                             alms[freq_j][p])
                            )
                    else:
                        cl_split_cross[(freq_i,freq_j)].append(
                            curvedsky.alm2cl(alms[freq_i][p],
                                             alms[freq_j][q])
                            )
            cl_split_auto[(freq_i,freq_j)] = np.array(cl_split_auto[(freq_i,freq_j)]).mean(axis=0) / w2
            cl_split_cross[(freq_i,freq_j)] = np.array(cl_split_cross[(freq_i,freq_j)]).mean(axis=0) / w2
            if sg_window is not None:
                cl_split_auto[(freq_i,freq_i)] = savgol_filter(
                    cl_split_auto[(freq_i,freq_i)], sg_window, sg_order)
                cl_split_cross[(freq_i,freq_i)] = savgol_filter(
                    cl_split_cross[(freq_i,freq_i)], sg_window, sg_order)

            
            #first signal
            cl_ij = []
            for p in range(nsplit):
                for q in range(p,nsplit):
                    if (p==q):
                        continue
                    cl_ij.append(
                        curvedsky.alm2cl(alms[freq_i][p],
                                         alms[freq_j][q])
                        )
            cl_signal[freq_i,freq_j] = (np.array(cl_ij).mean(axis=0))/w2
            if sg_window is not None:
                cl_signal[freq_i,freq_j] = savgol_filter(
                    cl_signal[freq_i,freq_j], sg_window, sg_order)
            cl_signal[freq_j,freq_i] = cl_signal[freq_i,freq_j]
            #now noise
            #There are two cases
            #i=j - this is simple, just take the
            #mean of the diffs
            #i!=j - the noise term may be non-zero where we have
            #correlated "channels" e.g. if we're using different
            #ilc maps in each leg. I think in this case it should 
            #be only equal splits that have correlated noise (i.e. p=q).
            #Hmm...actually this is only strictly true when not using Planck
            #(we don't have Planck splits).
            if i==j:
                cl_diffs = []
                for p in range(nsplit-1):
                    for q in range(p+1,nsplit):
                        cl_diffs.append(
                            curvedsky.alm2cl(alms[freq_i][p]-alms[freq_i][q]
                        ))
                #noise cl is half of the diff cl
                cl_noise[(freq_i,freq_i)] = 0.5 * (np.array(cl_diffs)).mean(axis=0)/w2
                if sg_window is not None:
                    cl_noise[(freq_i,freq_i)] = savgol_filter(
                        cl_noise[(freq_i,freq_i)], sg_window, sg_order)
            else:
                cl_total_ps = []
                for p in range(nsplit):
                    cl_total_ps.append(
                        curvedsky.alm2cl(alms[freq_i][p], alms[freq_j][p]
                    ))
                cl_total_p = (np.array(cl_total_ps)).mean(axis=0)/w2
                if sg_window is not None:
                    cl_total_p = savgol_filter(cl_total_p, sg_window, sg_order)
                #noise cl is total - signal
                cl_noise[(freq_i,freq_j)] = cl_total_p - cl_signal[freq_i,freq_j]
                cl_noise[(freq_j,freq_i)] = cl_noise[(freq_i,freq_j)]

    return cl_signal, cl_noise, cl_split_auto, cl_split_cross


def get_sims_from_cls(cl_split_auto_dict, cl_split_cross_dict,
                      mask,
                      signal_seed, noise_seed,
                      nsplit=NSPLIT, w1=1.,
                      lmax_out=None):

    if lmax_out is None:
        lmax_out = mlmax
    
    channel_pairs = cl_split_auto_dict.keys()
    channels = []
    for p in channel_pairs:
        channels += list(p)
    channels = list(set(channels))
    print("getting sims for channels:",channels)

    #make covariance - should be NxNx(lmax+1) where N=len(channels)*nsplit
    print("getting cov")
    N = len(channels)*nsplit
    cov = np.zeros((N,N,mlmax+1))
    for i,c_i in enumerate(channels):
        for j,c_j in enumerate(channels):
            if (c_i, c_j) in cl_split_auto_dict:
                cl_auto_ij = cl_split_auto_dict[c_i,c_j]
                cl_cross_ij = cl_split_cross_dict[c_i,c_j]
            else:
                cl_auto_ij = cl_split_auto_dict[c_j,c_i]
                cl_cross_ij = cl_split_cross_dict[c_j,c_i]
                
            if mlmax is not None:
                cl_auto_ij = cl_auto_ij[:mlmax+1]
                cl_cross_ij = cl_cross_ij[:mlmax+1]
            for p in range(nsplit):
                for q in range(nsplit):
                    if p==q:
                        cov[i*nsplit+p, j*nsplit+q,:] = cl_auto_ij
                    else:
                        cov[i*nsplit+p, j*nsplit+q,:] = cl_cross_ij
    print("generating alms")
    alms = curvedsky.rand_alm(cov, seed=signal_seed)

    alms_masked = []
    for alm in alms:
        m = enmap.zeros(mask.shape,mask.wcs)
        curvedsky.alm2map(alm, m, tweak=True)
        m *= mask
        alms_masked.append(
            curvedsky.map2alm(m, lmax=lmax_out, tweak=True)
            )

    total_alms = {}
    for i,c in enumerate(channels):
        total_alms[c] = alms_masked[i*nsplit:(i+1)*nsplit]
    return total_alms, None, None

def get_alms_dr6_hilc(data_dir="/global/cscratch1/sd/maccrann/cmb/act_dr6/ilc_cldata_smooth-301-2_v1",
                      map_names=["hilc", "hilc-tszandcibd"], sim_seed=None, mlmax=None):

    #get splits
    alms = {}
    for map_name in map_names:
        alms[map_name] = []
        for split in range(NSPLIT):
            if sim_seed is None:
                map_filename = "%s_split%d.fits"%(map_name, split)
                map_path = opj(data_dir, map_filename)
            else:
                map_filename = opj("sim_planck%03d_act%02d_%05d"%(sim_seed),
                                   "%s_split%d.fits"%(map_name, split))
                map_path = opj(data_dir, map_filename)
            alm = hp.read_alm(map_path)
            if mlmax is not None:
                alm = utils.change_alm_lmax(alm, mlmax)
            alms[map_name].append(alm)
            
    return alms


def get_sim_alm_dr6_nilc(map_name, sim_seed, sim_template_path, mlmax=None,
                         apply_extra_mask=None, no_cross=True):
    if no_cross:
        alm_filenames = [Template(sim_template_path).substitute(
            actseed="%04d"%sim_seed,
            freq=map_name)]
    alms=[]
    for f in alm_filenames:
        alm=hp.read_alm(f)
        if mlmax is not None:
            print("apply mlmax cut *before* masking - this is probably the wrong thing to do, but avoids memory issues")
            alm = utils.change_alm_lmax(alm, mlmax)
        lmax = hp.Alm.getlmax(len(alm))
        if apply_extra_mask is not None:
            print("applying extra mask")
            z = enmap.zeros(shape=apply_extra_mask.shape,
                            wcs=apply_extra_mask.wcs)
            m = curvedsky.alm2map(alm, z, tweak=True)
            m *= apply_extra_mask
            alm = curvedsky.map2alm(m, lmax=lmax, tweak=True)

        
        if mlmax is not None:
            alm = utils.change_alm_lmax(alm, mlmax)
        alms.append(alm)
    return np.array(alms)

def get_sim_alm_dr6_hilc(map_name, sim_seed, sim_template_path, mlmax=None,
                         apply_extra_mask=None, no_cross=False):
    sim_seed = (200+sim_seed, 0, sim_seed+1)
    alm_filenames = [Template(sim_template_path).substitute(
        planckseed=sim_seed[0], actset="%02d"%sim_seed[1], actseed="%05d"%sim_seed[2],
        freq=map_name, split=split) for split in range(NSPLIT)]
    alms=[]
    for f in alm_filenames:
        alm=hp.read_alm(f)
        if mlmax is not None:
            print("apply mlmax cut *before* masking - this is probably the wrong thing to do, but avoids memory issues")
            alm = utils.change_alm_lmax(alm, mlmax)
        lmax = hp.Alm.getlmax(len(alm))
        if apply_extra_mask is not None:
            print("applying extra mask")
            z = enmap.zeros(shape=apply_extra_mask.shape,
                            wcs=apply_extra_mask.wcs)
            m = curvedsky.alm2map(alm, z, tweak=True)
            m *= apply_extra_mask
            alm = curvedsky.map2alm(m, lmax=lmax, tweak=True)

        
        if mlmax is not None:
            alm = utils.change_alm_lmax(alm, mlmax)
        alms.append(alm)
    return alms

def get_data_Cltot_dict(alm_dict,
                   sg_window=301, sg_order=2,
                   w2=None):
    #just do a straight average of splits
    #and use Savgol filter
    map_names = list(alm_dict.keys())
    cl_dict = {}
    for i,map_i in enumerate(map_names):
        mean_alm_i = np.average(np.array(alm_dict[map_i]), axis=0)
        assert mean_alm_i.shape == (alm_dict[map_i][0]).shape
        for j,map_j in enumerate(map_names[i:]):
            mean_alm_j = np.average(np.array(alm_dict[map_j]), axis=0)
            cl = savgol_filter(
                curvedsky.alm2cl(mean_alm_i, mean_alm_j),
                sg_window, sg_order)
            if w2 is not None:
                cl /= w2
            cl_dict[(map_i,map_j)] = cl
            cl_dict[(map_j,map_i)] = cl
    return cl_dict


def get_weight_map(args, verbose=False):
    #get w2 and w4
    mask = enmap.read_map(args.mask)
    extra_mask = None
    if args.apply_extra_mask is not None:
        if verbose:
            print("using extra weight map %s"%args.apply_extra_mask)
        if "," in args.apply_extra_mask:
            extra_mask_file, ind = args.apply_extra_mask.split(",")
            extra_mask = enmap.read_map(extra_mask_file)[int(ind)]
        else:
            extra_mask = enmap.read_map(args.apply_extra_mask)

        if args.smooth_extra_mask is not None:
            if verbose:
                print("smoothing weight map by Gaussian with sigma %.1f arcmin"%args.smooth_extra_mask)
            extra_mask = enmap.smooth_gauss(extra_mask,
                                            np.radians(args.smooth_extra_mask/60.)
                                            )
            
        if args.extra_mask_power is not None:
            if verbose:
                print("raising by power %.1f"%args.extra_mask_power)
            extra_mask = extra_mask ** args.extra_mask_power


        total_mask = mask*extra_mask

        #I think it's going to be convenient here to normalise the mask
        #such that it has maximum value = 1
        total_mask /= total_mask.max()
            
        w1 = maps.wfactor(1, total_mask)
        w2 = maps.wfactor(2, total_mask)
        w4 = maps.wfactor(4, total_mask)
    else:
        total_mask = mask

    return total_mask, extra_mask
    

def do_setup(args, verbose=False):
    if verbose:
        print("args:")
        print(args)
    
    #recon_config = vars(args)
    safe_mkdir(args.output_dir)

    #signal filter
    if args.rksz_cl is None:
        cl_rksz = DEFAULT_CL_RKSZ
    else:
        cl_rksz = np.load(args.rksz_cl)
    
    #get w2 and w4
    mask = enmap.read_map(args.mask)
    
    w1 = maps.wfactor(1, mask)
    w2 = maps.wfactor(2, mask)
    w4 = maps.wfactor(4, mask)
    #w2 = 0.2758258447605825
    #w4 = 0.2708454607428685
    if verbose:
        print("w1:",w1)
        print("w2:",w2)
        print("w4:",w4)

    
    #read in alms for data
    if verbose:
        print("reading data alms")
    #est_maps = (args.est_maps.strip()).split("_")

    total_mask, extra_mask = get_weight_map(args, verbose=verbose)
    w1 = maps.wfactor(1, total_mask)
    w2 = maps.wfactor(2, total_mask)
    w4 = maps.wfactor(4, total_mask)

    data_alms = get_alms_dr6(args.data_template_path,
                             freqs=list(set(args.est_maps)), mlmax=args.mlmax,
                             apply_extra_mask=extra_mask,
                             verbose=verbose)
    if verbose:
        print("data_alms.keys():")
        print(data_alms.keys())
    #data_alms = get_alms_dr6_hilc(mlmax=args.mlmax,
    #                              map_names=list(set(args.est_maps)))

    #get total Cls...hmm - maybe easiest to
    #estimate these from input maps to start with,
    #but also add an estimate from sims option
    if args.use_Cltot_dict_from_file is not None:
        print("using total Cl from file:",args.use_Cltot_dict_from_file)
        with open(args.use_Cltot_dict_from_file, "rb") as f:
            cltot_dict = pickle.load(f)
    else:
        if verbose:
            print("getting data Cls")
        #get from data
        cltot_dict = get_data_Cltot_dict(data_alms,
                   sg_window=301, sg_order=2,
                w2=w2)
        #save total Cls
        with open(opj(args.output_dir,
                      "total_Cl_from_data.pkl"), "wb") as f:
            pickle.dump(cltot_dict, f)
    """
    else:
        raise NotImplementedError("not implemented other options")
        #get from sims
    """

    #do setup
    #decide on lmin, lmax etc.
    #and what combination of maps to use
    #args.est_maps = recon_config["args.est_maps"]
    lmin, lmax = args.K_lmin, args.K_lmax

    n_alm = len( data_alms[ list(data_alms.keys())[0] ][0] )
    if args.mlmax is None:
        mlmax = hp.Alm.getlmax(n_alm)
        args.mlmax = mlmax
    else:
        mlmax = args.mlmax

    if verbose:
        print("getting data C_ls for sims")
        print("mlmax = %d"%args.mlmax)
    if args.mask_sim_lmax is None:
        args.mask_sim_lmax=mlmax

    if verbose:
        print("mask_sim_lmax = %d"%args.mask_sim_lmax)

    data_cl_signal_dict, data_cl_noise_dict, cl_split_auto_dict, cl_split_cross_dict = get_sim_model_from_data(
        args.data_template_path, freqs=list(set(args.est_maps)),
        mlmax=args.mask_sim_lmax, nsplit=NSPLIT, sg_window=args.sg_window, sg_order=args.sg_order,
        w2=w2)
    if verbose:
        print("data_cl_signal_dict.keys()")
        print(data_cl_signal_dict.keys())
    #save these dictionaries for debugging
    if verbose:
        with open(opj(args.output_dir, "data_cl_signal_dict.pkl"), "wb") as f:
            pickle.dump(data_cl_signal_dict, f)
        with open(opj(args.output_dir, "data_cl_noise_dict.pkl"), "wb") as f:
            pickle.dump(data_cl_noise_dict, f)
    
    if verbose:
        print("lmin,lmax:", lmin,lmax)
        print("mlmax:",mlmax)

    if verbose:
        print("Setting up estimator")
    if args.px_fullsky_0p5:
        shape,wcs = enmap.fullsky_geometry(res=0.5*putils.arcmin)
        px=qe.pixelization(shape=shape, wcs=wcs)
    else:
        px = qe.pixelization(shape=total_mask.shape,wcs=total_mask.wcs)
    
    cltot_A = cltot_dict[(args.est_maps[0], args.est_maps[0])][:mlmax+1]
    cltot_B = cltot_dict[(args.est_maps[1], args.est_maps[1])][:mlmax+1]
    cltot_C = cltot_dict[(args.est_maps[2], args.est_maps[2])][:mlmax+1]
    cltot_D = cltot_dict[(args.est_maps[3], args.est_maps[3])][:mlmax+1]
    cltot_AC = cltot_dict[(args.est_maps[0], args.est_maps[2])][:mlmax+1]
    cltot_BD = cltot_dict[(args.est_maps[1], args.est_maps[3])][:mlmax+1]
    cltot_AD = cltot_dict[(args.est_maps[0], args.est_maps[3])][:mlmax+1]
    cltot_BC = cltot_dict[(args.est_maps[1], args.est_maps[2])][:mlmax+1]
    
    setup = setup_ABCD_recon(
        px, lmin, lmax, mlmax,
        cl_rksz[:mlmax+1], 
        cltot_A, cltot_B, cltot_C, cltot_D,
        cltot_AC, cltot_BD, cltot_AD, cltot_BC,
        do_lh=True, do_psh=True)

    setup["cltot_A"] = cltot_A
    setup["cltot_B"] = cltot_B
    setup["cltot_C"] = cltot_C
    setup["cltot_D"] = cltot_D
    setup["cltot_AC"] = cltot_AC
    setup["cltot_BD"] = cltot_BD
    setup["cltot_AD"] = cltot_AD
    setup["cltot_BC"] = cltot_BC
    
    
    profile = setup["profile"]

    setup["w1"] = w1
    setup["w2"] = w2
    setup["w4"] = w4
    setup["cl_rksz"] = cl_rksz[:mlmax+1]
    setup["weight_map"] = total_mask
    setup["extra_mask"] = extra_mask
    
    if verbose:
        print("filter_A:",profile/cltot_A)
        print("N0_ABCD_K[:300]:", setup["N0_ABCD_K"][:300])
        print("N0_ABCD_K_lh[:300]:", setup["N0_ABCD_K_lh"][:300])
        print("N0_ABCD_K_psh[:300]:", setup["N0_ABCD_K_psh"][:300])

    #also get noiseless N0 - for split estimator, true N0 should
    #be somewhere in between? Use signal only Cl (and here we
    #should divide by w4 - didn't need to before to just generate
    #gaussian alms)
    wLK_A = profile[:lmax+1]/(cltot_dict[(args.est_maps[0], args.est_maps[0])][:lmax+1])
    wLK_C = profile[:lmax+1]/(cltot_dict[(args.est_maps[2], args.est_maps[2])][:lmax+1])
    wGK_D = profile[:lmax+1]/(cltot_dict[(args.est_maps[3], args.est_maps[3])][:lmax+1])/2
    wGK_B = profile[:lmax+1]/(cltot_dict[(args.est_maps[1], args.est_maps[1])][:lmax+1])/2

    N0_K_nonoise_nonorm = noise_spec.qtt_asym(
        'src', mlmax, lmin, lmax,
        wLK_A, wGK_B, wLK_C, wGK_D,
        data_cl_signal_dict[(args.est_maps[0], args.est_maps[2])][:lmax+1],
        data_cl_signal_dict[(args.est_maps[1], args.est_maps[3])][:lmax+1],
        data_cl_signal_dict[(args.est_maps[0], args.est_maps[3])][:lmax+1],
        data_cl_signal_dict[(args.est_maps[1], args.est_maps[2])][:lmax+1])[0]/profile[:lmax+1]**2
    N0_K_nonoise = N0_K_nonoise_nonorm * setup["norm_K_CD"] * setup["norm_K_AB"]
    setup["N0_K_nonoise"] = N0_K_nonoise

    #filter alms
    if verbose:
        print("filtering data alms")
    data_alms_Af = [setup["filter_A"](
        data_alms[args.est_maps[0]][split])
                    for split in range(NSPLIT)]
    data_alms_Bf = [setup["filter_B"](
        data_alms[args.est_maps[1]][split])
                    for split in range(NSPLIT)]
    if args.est_maps[2]!=args.est_maps[0]:
        data_alms_Cf = [setup["filter_C"](
        data_alms[args.est_maps[2]][split])
                    for split in range(NSPLIT)]
    else:
        data_alms_Cf = data_alms_Af
        
    if args.est_maps[3]!=args.est_maps[0]:
        data_alms_Df = [setup["filter_D"](
        data_alms[args.est_maps[3]][split])
                    for split in range(NSPLIT)]
    else:
        data_alms_Df = data_alms_Af   

    data_alms_f = (data_alms_Af, data_alms_Bf, data_alms_Cf, data_alms_Df)
    return args, setup, data_alms_f

def run_ClKK(args, setup, data_alms_f, comm=None, verbose=False,
             do_lh=False, do_psh=False, ksz_theory_alm=None, no_cross=False):

    if comm is not None:
        rank=comm.Get_rank()
    else:
        rank=0
    ret_dict = {}
    #optionally pass ksz_alms_f to run on theory kSZ-only alms
    if ksz_theory_alm is not None:
        cl_ksz_smooth = get_cl_smooth(ksz_theory_alm)
        print("Assuming full-sky, nside=4096 geometry for ksz theory alms")
        px_ksz = qe.pixelization(nside=4096)
        setup_ksz = setup_ABCD_recon(
            px_ksz, setup["lmin"], setup["lmax"], setup["mlmax"],
            setup["cl_rksz"],
            setup["cltot_A"], setup["cltot_B"], setup["cltot_C"], setup["cltot_D"],
            setup["cltot_AC"], setup["cltot_BD"], setup["cltot_AD"], setup["cltot_BC"],
            do_lh=do_lh, do_psh=do_psh)                

        ksz_theory_alms_f = [
            f(ksz_theory_alm) for f in [
                setup_ksz["filter_A"], setup_ksz["filter_B"], setup_ksz["filter_C"], setup_ksz["filter_D"]]
            ]

        K_ksz_AB = setup_ksz["qfunc_K_AB"](ksz_theory_alms_f[0],
                                       ksz_theory_alms_f[1])
        K_ksz_CD = setup_ksz["qfunc_K_CD"](ksz_theory_alms_f[2],
                                       ksz_theory_alms_f[3])
        Cl_KK_ksz_raw = curvedsky.alm2cl(K_ksz_AB, K_ksz_CD)
        ksz_N0 = setup_ksz["get_fg_trispectrum_N0_ABCD"](cl_ksz_smooth, cl_ksz_smooth, cl_ksz_smooth, cl_ksz_smooth)
        ret_dict["Cl_KK_ksz_theory_raw"] = Cl_KK_ksz_raw
        ret_dict["N0_ksz_theory"] = ksz_N0
        ret_dict["Cl_KK_ksz_theory"] = Cl_KK_ksz_raw-ksz_N0

        if args.do_lh:
            K_ksz_AB = setup_ksz["qfunc_K_AB_lh"](ksz_theory_alms_f[0],
                                           ksz_theory_alms_f[1])
            K_ksz_CD = setup_ksz["qfunc_K_CD_lh"](ksz_theory_alms_f[2],
                                           ksz_theory_alms_f[3])
            Cl_KK_ksz_lh_raw = curvedsky.alm2cl(K_ksz_AB, K_ksz_CD)
            ksz_N0_lh = setup_ksz["get_fg_trispectrum_N0_ABCD_lh"](cl_ksz_smooth, cl_ksz_smooth, cl_ksz_smooth, cl_ksz_smooth)
            ret_dict["Cl_KK_ksz_theory_raw_lh"] = Cl_KK_ksz_lh_raw
            ret_dict["N0_ksz_theory_lh"] = ksz_N0_lh
            ret_dict["Cl_KK_ksz_theory_lh"] = Cl_KK_ksz_lh_raw - ksz_N0_lh

        if args.do_psh:
            K_ksz_AB = setup_ksz["qfunc_K_AB_psh"](ksz_theory_alms_f[0],
                                           ksz_theory_alms_f[1])
            K_ksz_CD = setup_ksz["qfunc_K_CD_psh"](ksz_theory_alms_f[2],
                                           ksz_theory_alms_f[3])
            Cl_KK_ksz_psh_raw = curvedsky.alm2cl(K_ksz_AB, K_ksz_CD)
            ksz_N0_psh = setup_ksz["get_fg_trispectrum_N0_ABCD_psh"](cl_ksz_smooth, cl_ksz_smooth, cl_ksz_smooth, cl_ksz_smooth)
            ret_dict["Cl_KK_ksz_theory_raw_psh"] = Cl_KK_ksz_psh_raw
            ret_dict["N0_ksz_theory_psh"] = ksz_N0_psh
            ret_dict["Cl_KK_ksz_theory_psh"] = Cl_KK_ksz_psh_raw - ksz_N0_psh
    
    data_alms_Af, data_alms_Bf, data_alms_Cf, data_alms_Df = data_alms_f
    w2,w4 = setup["w2"], setup["w4"]
    #K from Q(ilc, ilc-tszandcibd)
    print("running K_AB")
    if no_cross:
        data_alm_A_mean = mean_alm_from_splits(data_alms_Af)
        data_alm_B_mean = mean_alm_from_splits(data_alms_Bf)
        Ks_ab = setup["qfunc_K_AB"](data_alm_A_mean,data_alm_B_mean)
        if rank==0:
            np.save(opj(args.output_dir, "K_ab.npy"), Ks_ab)
    else:

        if not args.skip_qe:

            Ks_ab = four_split_K(
                setup["qfunc_K_AB"],
                data_alms_Af[0], data_alms_Af[1],
                data_alms_Af[2], data_alms_Af[3],
                Xdatp_0=data_alms_Bf[0],
                Xdatp_1=data_alms_Bf[1],
                Xdatp_2=data_alms_Bf[2],
                Xdatp_3=data_alms_Bf[3],
                )
            #Actually save all the Ks - we'll need these for combining with mean-field
            if rank==0:
                np.save(opj(args.output_dir, "K_ab.npy"), Ks_ab)
    

    if args.do_lh:

        Ks_ab_lh = four_split_K(
            setup["qfunc_K_AB_lh"],
            data_alms_Af[0], data_alms_Af[1],
            data_alms_Af[2], data_alms_Af[3],
            Xdatp_0=data_alms_Bf[0],
            Xdatp_1=data_alms_Bf[1],
            Xdatp_2=data_alms_Bf[2],
            Xdatp_3=data_alms_Bf[3],
            )
        if rank==0:
            np.save(opj(args.output_dir, "K_ab_lh.npy"), Ks_ab_lh)

    if args.do_psh:
        Ks_ab_psh = four_split_K(
            setup["qfunc_K_AB_psh"],
            data_alms_Af[0], data_alms_Af[1],
            data_alms_Af[2], data_alms_Af[3],
            Xdatp_0=data_alms_Bf[0],
            Xdatp_1=data_alms_Bf[1],
            Xdatp_2=data_alms_Bf[2],
            Xdatp_3=data_alms_Bf[3],
            )
        if rank==0:
            np.save(opj(args.output_dir, "K_ab_psh.npy"), Ks_ab_psh)

    #K from Q(ilc,ilc)
    print("running K_CD")
    do_cd=True
    if (args.est_maps[0],args.est_maps[1]) == (args.est_maps[2],args.est_maps[3]):
        do_cd=False
        
    if not do_cd:
        if not args.skip_qe:
            Ks_cd = Ks_ab
        if args.do_lh:
            Ks_cd_lh = Ks_ab_lh.copy()
        if args.do_psh:
            Ks_cd_psh = Ks_ab_psh.copy()
    else:
        if no_cross:
            data_alm_C_mean = mean_alm_from_splits(data_alms_Cf)
            data_alm_D_mean = mean_alm_from_splits(data_alms_Df)
            Ks_cd = setup["qfunc_K_CD"](data_alm_C_mean,data_alm_D_mean)
        else:
            if not args.skip_qe:
                print("calling four_split_K for CD")
                Ks_cd = four_split_K(
                    setup["qfunc_K_CD"],
                    data_alms_Cf[0], data_alms_Cf[1],
                    data_alms_Cf[2], data_alms_Cf[3],
                    data_alms_Df[0], data_alms_Df[1],
                    data_alms_Df[2], data_alms_Df[3],
                    )
        #save K^hat^X alm 
        if args.do_lh:
            Ks_cd_lh = four_split_K(
                setup["qfunc_K_CD_lh"],
            data_alms_Cf[0], data_alms_Cf[1],
            data_alms_Cf[2], data_alms_Cf[3],
                data_alms_Df[0], data_alms_Df[1],
                data_alms_Df[2], data_alms_Df[3],
                )
        if args.do_psh:
            Ks_cd_psh = four_split_K(
                setup["qfunc_K_CD_psh"],
            data_alms_Cf[0], data_alms_Cf[1],
            data_alms_Cf[2], data_alms_Cf[3],
                data_alms_Df[0], data_alms_Df[1],
                data_alms_Df[2], data_alms_Df[3],
                )

    if rank==0:
        if not args.skip_qe:
            np.save(opj(args.output_dir,"K_cd.npy"), Ks_cd)
    if args.do_lh:
        if rank==0:
            np.save(opj(args.output_dir,"K_cd_lh.npy"), Ks_cd_lh)
    if args.do_psh:
        if rank==0:
            np.save(opj(args.output_dir,"K_cd_psh.npy"), Ks_cd_psh)

                
    print("getting CL_KK")
    if no_cross:
        cl_abcd = curvedsky.alm2cl(Ks_ab, Ks_cd)/w4
    else:
        if not args.skip_qe:
            cl_abcd = split_phi_to_cl(Ks_ab, Ks_cd)/w4
    """
    #check we get the same thing from reading in Ks
    Ks_ab_raw = np.load(opj(args.output_dir,"K_ab.npy"))
    Ks_cd_raw = np.load(opj(args.output_dir,"K_cd.npy"))
    #with open(opj(args.output_dir,"K_ab.pkl"),"rb") as f:
    #    Ks_ab_raw = pickle.load(f)
    #with open(opj(args.output_dir,"K_cd.pkl"),"rb") as f:
    #    Ks_cd_raw = pickle.load(f)
    cl_kk_from_file = split_phi_to_cl(Ks_ab_raw, Ks_cd_raw)/w4
    assert np.allclose(cl_abcd, cl_kk_from_file)
    """
    
    #return raw auto, as well as Ks arrays - we need to save these for mean field
    print("done")
    if not args.skip_qe:
        ret_dict["cl_KK_raw"] = cl_abcd
        ret_dict["Ks_ab"] = Ks_ab
        ret_dict["Ks_cd"] = Ks_cd
    
    if do_lh:
        cl_abcd_lh = split_phi_to_cl(Ks_ab_lh, Ks_cd_lh)/w4
        ret_dict["cl_KK_lh_raw"] = cl_abcd_lh
        ret_dict["Ks_ab_lh"] = Ks_ab_lh
        ret_dict["Ks_cd_lh"] = Ks_cd_lh
    if do_psh:
        cl_abcd_psh = split_phi_to_cl(Ks_ab_psh, Ks_cd_psh)/w4
        ret_dict["cl_KK_psh_raw"] = cl_abcd_psh
        ret_dict["Ks_ab_psh"] = Ks_ab_psh
        ret_dict["Ks_cd_psh"] = Ks_cd_psh

    return ret_dict


def main():

    #setup mpi if we want it
    if not disable_mpi:
        comm = MPI.COMM_WORLD
        rank,size = comm.Get_rank(), comm.Get_size()
    else:
        comm = None
        rank,size = 0,1
    
    #get args
    args,_,_ = get_config()
    #args.do_qe = (not args.skip_qe)
    if rank==0:
        print("args:")
        print(args)
    args, setup, data_alms_f = do_setup(args,
                           verbose=(rank==0)
                          )
    data_alms_Af, data_alms_Bf, data_alms_Cf, data_alms_Df = data_alms_f

    if args.do_auto:
        if rank==0:
            print("running auto")

        #Now run on data
        if args.ksz_theory_alm_file is not None:
            ksz_theory_alm = utils.change_alm_lmax(
                hp.read_alm(args.ksz_theory_alm_file), args.mlmax)
        else:
            ksz_theory_alm = None

        outputs = run_ClKK(args, setup, data_alms_f, comm=comm, verbose=(rank==0), do_lh=args.do_lh, do_psh=args.do_psh,
                                ksz_theory_alm=ksz_theory_alm, no_cross=args.no_cross)
        outputs["N0"] = setup["N0_ABCD_K"]
        if args.do_lh:
            outputs["N0_lh"] = setup["N0_ABCD_K_lh"]
        if args.do_psh:
            outputs["N0_psh"] = setup["N0_ABCD_K_psh"]
        outputs["N0_nonoise"] = setup["N0_K_nonoise"]
        outputs["norm_K_AB"] = setup["norm_K_AB"]
        outputs["norm_K_CD"] = setup["norm_K_CD"]
        outputs["cl_rksz"] = setup["cl_rksz"] #save what we used for filters
        outputs["cltot_A"] = setup["cltot_A"]
        outputs["cltot_B"] = setup["cltot_B"]
        outputs["cltot_C"] = setup["cltot_C"]
        outputs["cltot_D"] = setup["cltot_D"]
        outputs["cltot_AC"] = setup["cltot_AC"]
        outputs["cltot_BD"] = setup["cltot_BD"]
        outputs["cltot_AD"] = setup["cltot_AD"]
        outputs["cltot_BC"] = setup["cltot_BC"]
        outputs["w1"] = setup["w1"]
        outputs["w2"] = setup["w2"]
        outputs["w4"] = setup["w4"]

        for leg in ["A","B","C","D"]:
            outputs["total_filter_%s"%leg] = setup["profile"] / setup["cltot_%s"%leg]

        outputs["lmin"] = setup["lmin"]
        outputs["lmax"] = setup["lmax"]
        outputs["profile"] = setup["profile"]

        #save pkl
        if rank==0:
            with open(opj(args.output_dir, "auto_outputs.pkl"), 'wb') as f:
                pickle.dump(outputs, f)
            #also save weight map
            enmap.write_map(opj(args.output_dir, "weight_map.fits"), setup["weight_map"])

    def run_meanfield(setup, nsim, use_mpi=False, est=None,
                      combine_with_auto=False,
                      Ks_ab=None, Ks_cd=None, 
                      Ks_ab_lh=None, Ks_cd_lh=None,
                      fg_power_data=None, meanfield_tag=None,
                      no_cross=False, get_sim_func=None,
                      ):

        mlmax = setup["mlmax"]
        w1,w2,w4 = setup["w1"],setup["w2"],setup["w4"]
        
        #For the mean field we just loop through sims, saving
        #sim auto
        #get sim alm functions
        if meanfield_tag is not None:
            mean_field_outdir = opj(args.output_dir, "mean_field_nsim%d_%s"%(nsim, meanfield_tag))
        else:
            mean_field_outdir = opj(args.output_dir, "mean_field_nsim%d"%nsim)

        if est=="lh":
            mean_field_outdir = mean_field_outdir + "_lh"
        elif est=="psh":
            mean_field_outdir = mean_field_outdir + "_psh"
            
        mlmax=setup["mlmax"]
            
        safe_mkdir(mean_field_outdir)

        #function to get sims
        get_sim_func = eval(args.get_sim_func)
        def get_sim_alms_A(i):
            #alms = get_sim_alm_dr6_hilc(
            #    args.est_maps[0], (200+i,0,i+1),
            #    data_dir=DATA_DIR, mlmax=mlmax)
            print("getting sim from template path:", args.sim_template_path)
            alms = get_sim_func(
                args.est_maps[0], i,
                args.sim_template_path,
                mlmax=setup["mlmax"], apply_extra_mask=setup["extra_mask"],
                no_cross=no_cross)
            alms_f = [setup["filter_A"](alm) for alm in alms]
            return alms_f

        def get_sim_alms_B(i):
            #alms = get_sim_alm_dr6_hilc(
            #    args.est_maps[1], (200+i,0,i+1),
            #    data_dir=DATA_DIR, mlmax=mlmax)
            alms = get_sim_func(
                args.est_maps[1], i,
                args.sim_template_path,
                mlmax=setup["mlmax"], apply_extra_mask=setup["extra_mask"],
                no_cross=no_cross)
            alms_f = [setup["filter_B"](alm) for alm in alms]
            return alms_f

        Ks_ab_list = []
        Ks_cd_list = []

        if est=="lh":
            qfunc_AB = setup["qfunc_K_AB_lh"]
            qfunc_CD = setup["qfunc_K_CD_lh"]
        elif est=="psh":
            qfunc_AB = setup["qfunc_K_AB_psh"]
            qfunc_CD = setup["qfunc_K_CD_psh"]
        else:
            qfunc_AB = setup["qfunc_K_AB"]
            qfunc_CD = setup["qfunc_K_CD"]

        
        for isim in range(nsim):
            if isim % size != rank:
                continue
            print("rank %d doing sim %d"%(rank, isim))
            alms_Af = get_sim_alms_A(isim)
            alms_Bf = get_sim_alms_B(isim)
            print("lmax alms_Af:", hp.Alm.getlmax(len(alms_Af[0])))

            if fg_power_data is not None:
                #Also add foreground power
                cov_fg = np.zeros((4, 4, mlmax+1))
                for i, est_map_i in enumerate(args.est_maps):
                    for j, est_map_j in enumerate(args.est_maps):
                        try:
                            cov_fg[i,j] = fg_power_data["%s_%s"%(est_map_i,est_map_j)][:mlmax+1]
                        except ValueError:
                            cov_fg[i,j] = fg_power_data["%s_%s"%(est_map_j,est_map_i)][:mlmax+1]
                        cov_fg[j,i] = cov_fg[i,j]

                fg_alms = curvedsky.rand_alm(cov_fg, seed=123+isim)
                #need to convert to map and mask
                fg_alms_masked = []
                for fg_alm in fg_alms:
                    z = enmap.zeros(shape=mask.shape, wcs=mask.wcs)
                    print("!!!!!!!!!!!!!!!!!!!!!")
                    print("using hard-coded Gaussian 2' mean for foreground power")
                    print("!!!!!!!!!!!!!!!!!!!!!")
                    bl = maps.gauss_beam(np.arange(mlmax+1), 2.)
                    fg_alm_beamed = curvedsky.almxfl(fg_alm, bl)
                    fg_map = curvedsky.alm2map(fg_alm_beamed, z, tweak=True)
                    fg_map *= mask
                    fg_alm_beamed_masked = curvedsky.map2alm(
                        fg_map, lmax=mlmax, tweak=True)
                    fg_alms_masked.append(
                        curvedsky.almxfl(
                        fg_alm_beamed_masked, 1./bl)
                        )
                
                if rank==0:
                    print("adding foreground power")
                    print("fg_alms_masked:", fg_alms_masked)
                fg_alms_Af = setup["filter_A"](fg_alms_masked[0])
                fg_alms_Bf = setup["filter_B"](fg_alms_masked[1])

                alms_Af = [a + fg_alms_Af for a in alms_Af]
                alms_Bf = [a + fg_alms_Bf for a in alms_Bf]

            if no_cross:
                alm_Af_mean = mean_alm_from_splits(alms_Af)
                alm_Bf_mean = mean_alm_from_splits(alms_Bf)
                Ks_ab = qfunc_AB(alm_Af_mean,alm_Bf_mean)

            else:
                
                Ks_ab = four_split_K(
                    qfunc_AB,
                    alms_Af[0], alms_Af[1],
                    alms_Af[2], alms_Af[3],
                    Xdatp_0=alms_Bf[0],
                    Xdatp_1=alms_Bf[1],
                    Xdatp_2=alms_Bf[2],
                    Xdatp_3=alms_Bf[3],
                    )

            with open(opj(mean_field_outdir, "Ks_ab_sim%d.pkl"%isim), "wb") as f:
                pickle.dump(Ks_ab, f)

            #K from Q(ilc,ilc)
            print("running K_CD")
            if (args.est_maps[0],args.est_maps[1]) == (args.est_maps[2],args.est_maps[3]):
                Ks_cd = Ks_ab
                
            else:
                assert (args.est_maps[2]==args.est_maps[0] and args.est_maps[3]==args.est_maps[0])
                if no_cross:
                    alm_Cf_mean = alms_Af_mean
                    alm_Df_mean = alms_Af_mean
                    Ks_cd = qfunc_CD(alm_Cf_mean, alm_Df_mean)

                else:

                    Ks_cd = four_split_K(
                        qfunc_CD,
                        alms_Af[0], alms_Af[1],
                        alms_Af[2], alms_Af[3],
                        alms_Af[0], alms_Af[1],
                        alms_Af[2], alms_Af[3],
                        )
                """
                Ks_cd_lh = four_split_K(
                    setup["qfunc_K_CD_lh"],
                    alms_Af[0], alms_Af[1],
                    alms_Af[2], alms_Af[3],
                    )
                """

            with open(opj(mean_field_outdir, "Ks_cd_sim%d.pkl"%isim), "wb") as f:
                pickle.dump(Ks_cd, f)
                
                    
        comm.Barrier()
        
        if rank==0:
            print("rank 0 collecting sims and saving mean")
            #collect
            """
            #For each simulation we have a list of K_ab_xy and K_cd_xy
            #For each rank we have a list of these lists (one for each sim that rank did)
            #So first of all just concatenate all these lists together
            """
            #n_collected=1
            """
            while n_collected<size:
                r = comm.recv(source=MPI.ANY_SOURCE)
                if r==0:
                    n_collected+=1
            """
                    
            #Now get the mean K_ab list and K_cd list
            with open(opj(mean_field_outdir, "Ks_ab_sim0.pkl"), "rb") as f:
                Ks_ab_sim0 = pickle.load(f)
            with open(opj(mean_field_outdir, "Ks_cd_sim0.pkl"), "rb") as f:
                Ks_cd_sim0 = pickle.load(f)

            Ks_ab_sum = np.array(Ks_ab_sim0)
            Ks_cd_sum = np.array(Ks_cd_sim0)

            for isim in range(1,nsim):
                with open(opj(mean_field_outdir, "Ks_ab_sim%d.pkl"%isim), "rb") as f:
                    Ks_ab_simi = pickle.load(f)
                with open(opj(mean_field_outdir, "Ks_cd_sim%d.pkl"%isim), "rb") as f:
                    Ks_cd_simi = pickle.load(f)
                Ks_ab_sum += np.array(Ks_ab_simi)
                Ks_cd_sum += np.array(Ks_cd_simi)

            Ks_ab_mean = Ks_ab_sum/nsim
            Ks_cd_mean = Ks_cd_sum/nsim
            #for iK in range(len(Ks_ab_sim0)):
            #    Ks_ab_mean.append( (np.array([ Ks_ab[iK] for Ks_ab in Ks_ab_list])).mean(axis=0) )
            #    Ks_cd_mean.append( (np.array([ Ks_cd[iK] for Ks_cd in Ks_cd_list])).mean(axis=0) )
                
            #save these means
            with open(opj(mean_field_outdir, "Ks_ab_mean.pkl"), "wb") as f:
                pickle.dump(Ks_ab_mean, f)
            with open(opj(mean_field_outdir, "Ks_cd_mean.pkl"), "wb") as f:
                pickle.dump(Ks_cd_mean, f)

            return Ks_ab_mean, Ks_cd_mean
            #save pkl
            #with open(opj(outdir,"cls.pkl"), 'wb') as f:
            #    pickle.dump(cl_dict, f)

        else:
            return 0

            """
            print("getting CL_KK")
            cl_abcd = split_phi_to_cl(Ks_ab, Ks_cd)
            cl_abcd_lh = split_phi_to_cl(Ks_ab_lh, Ks_cd_lh)
            #apply w4
            cl_abcd /= w4
            cl_abcd_lh /= w4
            return cl_abcd, cl_abcd_lh
            print("done")
            """

    if args.do_meanfield:
        if args.fg_power_data is not None:
            fg_power_data = np.load(args.fg_power_data)
            print(fg_power_data.dtype.names)
            print(args.est_maps)
        else:
            fg_power_data = None

        if args.do_qe:
            print("running mean-field for qe case")
            run_meanfield(setup, args.nsim_meanfield,  
                          combine_with_auto=args.combine_with_auto,
                          Ks_ab=None, Ks_cd=None, est=None,
                          Ks_ab_lh=None, Ks_cd_lh=None,
                          fg_power_data = fg_power_data,
                          meanfield_tag = args.meanfield_tag,
                          no_cross=args.no_cross
                          )

        if args.do_lh:
            print("running mean-field for lensing-hardening case")
            run_meanfield(setup, args.nsim_meanfield,
                          combine_with_auto=args.combine_with_auto,
                          Ks_ab=None, Ks_cd=None,
                          Ks_ab_lh=None, Ks_cd_lh=None,
                          fg_power_data = fg_power_data,
                          meanfield_tag = args.meanfield_tag,
                          est="lh", no_cross=args.no_cross
                          )

        if args.do_psh:
            print("running mean-field for psh case")
            run_meanfield(setup, args.nsim_meanfield,
                          combine_with_auto=args.combine_with_auto,
                          Ks_ab=None, Ks_cd=None,
                          Ks_ab_lh=None, Ks_cd_lh=None,
                          fg_power_data = fg_power_data,
                          meanfield_tag = args.meanfield_tag,
                          est="psh", no_cross=args.no_cross
                          )

            
                  
    def run_rdn0(setup, nsim, use_mpi=False,
                 est=None, no_cross=False):

        mlmax=setup["mlmax"]
        w1,w2,w4 = setup["w1"],setup["w2"],setup["w4"]
        
        #get sim alm functions
        get_sim_func  = eval(args.get_sim_func)
        def get_sim_alms_A(i):
            #alms = get_sim_alm_dr6_hilc(
            #    args.est_maps[0], (200+i,0,i+1),
            #    data_dir=DATA_DIR, mlmax=mlmax)
            print("getting sim from template path:", args.sim_template_path)
            alms = get_sim_func(
                args.est_maps[0], i,
                args.sim_template_path,
                mlmax=mlmax, apply_extra_mask=setup["extra_mask"])
            alms_f = [setup["filter_A"](alm) for alm in alms]
            return alms_f

        def get_sim_alms_B(i):
            #alms = get_sim_alm_dr6_hilc(
            #    args.est_maps[1], (200+i,0,i+1),
            #    data_dir=DATA_DIR, mlmax=mlmax)
            alms = get_sim_func(
                args.est_maps[1], i,
                args.sim_template_path,
                mlmax=mlmax, apply_extra_mask=setup["extra_mask"])
            alms_f = [setup["filter_B"](alm) for alm in alms]
            return alms_f

        if args.est_maps[2]==args.est_maps[0]:
            get_sim_alms_C = None
        else:
            raise NotImplementedError("")
        if args.est_maps[3]==args.est_maps[0]:
            get_sim_alms_D = None
        else:
            raise NotImplementedError("")

        """
        Copy the function definition here to help get 
        the arguments right.
        mcrdn0_s4(nsims, power, qfunc_AB, split_K_func,
              get_sim_alms_A, data_split_alms_A,
              get_sim_alms_B=None, get_sim_alms_C=None, get_sim_alms_D=None,
              data_split_alms_B=None, data_split_alms_C=None,data_split_alms_D=None,
              qfunc_CD=None,
              use_mpi=True, verbose=True, skip_rd=False, power_mcn0=None):
        """
        if est=="lh":
            qfunc_AB = setup["qfunc_K_AB_lh"]
            qfunc_CD = setup["qfunc_K_CD_lh"]
        elif est=="psh":
            qfunc_AB = setup["qfunc_K_AB_psh"]
            qfunc_CD = setup["qfunc_K_CD_psh"]
        else:
            qfunc_AB = setup["qfunc_K_AB"]
            qfunc_CD = setup["qfunc_K_CD"]


        if no_cross:
            assert len(set(args.est_maps))==1
            def get_kmap(i):
                alms_f = get_sim_alms_A(i)
                return mean_alm_from_splits(alms_f)
            
            rdn0, mcn0 = mcrdn0(0, get_kmap, curvedsky.alm2cl, nsim,
                                qfunc_AB, Xdat=mean_alm_from_splits(data_alms_Af),
                                use_mpi=use_mpi)

        else:

            rdn0,mcn0 = mcrdn0_s4(nsim, split_phi_to_cl,
                                  qfunc_AB,
                                  four_split_K,
                                  get_sim_alms_A, data_alms_Af,
                                  get_sim_alms_B = get_sim_alms_B,
                                  data_split_alms_B = data_alms_Bf,
                                  qfunc_CD=qfunc_CD,
                                  use_mpi=use_mpi)

        rdn0 /= w4
        mcn0 /= w4

        return rdn0, mcn0

    if args.do_rdn0:
        if args.do_qe:
            if rank==0:
                print("running qe rdn0")
            rdn0, mcn0 = run_rdn0(setup, args.nsim_rdn0, use_mpi=args.use_mpi, no_cross=args.no_cross)
            outputs = {}
            outputs["rdn0"] = rdn0
            outputs["mcn0"] = mcn0
            outputs["theory_N0"] = setup["N0_ABCD_K"]
            if rank==0:
                print("done qe rdn0")
                
            rdn0_file = opj(args.output_dir, "rdn0_outputs_nsim%d.pkl"%args.nsim_rdn0)
            print("saving rdn0 outputs to %s"%rdn0_file)
            with open(rdn0_file, 'wb') as f:
                pickle.dump(outputs, f)

        if args.do_lh:
            if rank==0:
                print("running lh rdn0") 
            rdn0_lh, mcn0_lh = run_rdn0(setup, args.nsim_rdn0, use_mpi=args.use_mpi,
                                  est="lh")
            outputs = {}
            outputs["rdn0"] = rdn0_lh
            outputs["mcn0"] = mcn0_lh            
            outputs["theory_N0"] = setup["N0_ABCD_K_lh"]
            if rank==0:
                print("done lh rdn0")
                
            rdn0_file = opj(args.output_dir, "rdn0_outputs_lh_nsim%d.pkl"%args.nsim_rdn0)
            print("saving rdn0 outputs to %s"%rdn0_file)
            with open(rdn0_file, 'wb') as f:
                pickle.dump(outputs, f)
            if rank==0:
                print("done lh rdn0")

        if args.do_psh:
            if rank==0:
                print("running psh rdn0") 
            rdn0_psh, mcn0_psh = run_rdn0(setup, args.nsim_rdn0, use_mpi=args.use_mpi,
                                  est="psh")
            outputs = {}
            outputs["rdn0"] = rdn0_psh
            outputs["mcn0"] = mcn0_psh            
            outputs["theory_N0"] = setup["N0_ABCD_K_psh"]
            if rank==0:
                print("done psh rdn0")
                
            rdn0_file = opj(args.output_dir, "rdn0_outputs_psh_nsim%d.pkl"%args.nsim_rdn0)
            print("saving rdn0 outputs to %s"%rdn0_file)
            with open(rdn0_file, 'wb') as f:
                pickle.dump(outputs, f)
            if rank==0:
                print("done psh rdn0")

    def run_n1(setup, nsim, use_mpi=False, est=None):
    
        try:
            assert len(set(args.est_maps))==1
        except AssertionError as e:
            print("only implemented N1 for same map in each leg")
            raise(e)
    
        #get sim alm functions
        def get_sim_alms_A(i):
            #alms = get_sim_alm_dr6_hilc(
            #    args.est_maps[0], (200+i,0,i+1),
            #    data_dir=DATA_DIR, mlmax=mlmax)
            print("getting sim from template path:", args.sim_template_path)
            alms = get_sim_alm_dr6_hilc(
                args.est_maps[0], (200+i,0,i+1),
                args.sim_template_path,
                mlmax=mlmax, apply_extra_mask=extra_mask)
            alms_f = [setup["filter_A"](alm) for alm in alms]
            return alms_f

        def get_sim_alms_B(i):
            #alms = get_sim_alm_dr6_hilc(
            #    args.est_maps[1], (200+i,0,i+1),
            #    data_dir=DATA_DIR, mlmax=mlmax)
            alms = get_sim_alm_dr6_hilc(
                args.est_maps[1], (200+i,0,i+1),
                args.sim_template_path,
                mlmax=mlmax, apply_extra_mask=extra_mask)
            alms_f = [setup["filter_B"](alm) for alm in alms]
            return alms_f

        if args.est_maps[2]==args.est_maps[0]:
            get_sim_alms_C = None
        else:
            raise NotImplementedError("")
        if args.est_maps[3]==args.est_maps[0]:
            get_sim_alms_D = None
        else:
            raise NotImplementedError("")

        """
        Copy the function definition here to help get 
        the arguments right.
        mcrdn0_s4(nsims, power, qfunc_AB, split_K_func,
              get_sim_alms_A, data_split_alms_A,
              get_sim_alms_B=None, get_sim_alms_C=None, get_sim_alms_D=None,
              data_split_alms_B=None, data_split_alms_C=None,data_split_alms_D=None,
              qfunc_CD=None,
              use_mpi=True, verbose=True, skip_rd=False, power_mcn0=None):
        """
        if est=="lh":
            qfunc_AB = setup["qfunc_K_AB_lh"]
            qfunc_CD = setup["qfunc_K_CD_lh"]
        elif est=="psh":
            qfunc_AB = setup["qfunc_K_AB_psh"]
            qfunc_CD = setup["qfunc_K_CD_psh"]
        else:
            qfunc_AB = setup["qfunc_K_AB"]
            qfunc_CD = setup["qfunc_K_CD"]

        mcn1_data = mcn1_ABCD(0, get_sim_alms_A, curvedsky.alm2cl,
                             nsim, qfunc1=qfunc_AB,
                             qfunc2=qfunc_CD,
                             comm=comm)/w4
        mcn1_mean = mcn1_data.mean(axis=0)


        return mcn1_data, mcn1_mean

    if args.do_n1:
        if args.do_qe:
            if rank==0:
                print("not doing n1 for qe")
            pass

        if args.do_lh:
            if rank==0:
                print("running lh rdn0") 
                
            mcn1_data, mcn1_mean = run_n1(setup, args.nsim_n1, use_mpi=args.use_mpi,
                                          est="lh")                
            outputs = {}
            outputs["mcn1"] = mcn1_mean
            outputs["mcn1_allsims"] = mcn1_data
            if rank==0:
                print("done lh n1")
                
            n1_file = opj(args.output_dir, "n1_outputs_lh_nsim%d.pkl"%args.nsim_n1)
            print("saving n1 outputs to %s"%n1_file)
            with open(n1_file, 'wb') as f:
                pickle.dump(outputs, f)
            if rank==0:
                print("done lh n1")

        if args.do_psh:
            if rank==0:
                print("not doing n1 for qe")
            pass
    

if __name__=="__main__":
    main()
