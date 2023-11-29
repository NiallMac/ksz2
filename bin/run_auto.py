#Run the kSZ 4-point function
#
#
from os.path import join as opj, dirname
import os

#import sys
#sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ksz4.cross import four_split_K, split_phi_to_cl, mcrdn0_s4
from ksz4.reconstruction import setup_recon, setup_ABCD_recon, get_cl_fg_smooth
from pixell import curvedsky, enmap
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
#ksz_alm = utils.change_alm_lmax(hp.fitsfunc.read_alm("../tests/alms_4e3_2048_50_50_ksz.fits"),
#                                6000)
#cl_rksz = get_cl_fg_smooth(ksz_alm)
DEFAULT_CL_RKSZ=np.load(opj(dirname(__file__), "../tests/cl_4e3_2048_50_50_ksz.npy"))

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
            config_from_file = yaml.load(f)
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
    #I think most convenient to return
    #a namespace
    from argparse import Namespace
    config_namespace = Namespace(**config)

    #also return the config from the file,
    #and the defaults
    return config_namespace, config_from_file, dict(DEFAULTS)
    
def get_config_old(description="Do 4pt measurement"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("output_dir", type=str)

    defaults = {}
    bool_keys = []
    for key,val in DEFAULTS.items():
        nargs=None
        if val=="iNone":
            t,val = int, None
        elif val=="fNone":
            t,val = float, None
        elif val=="sNone":
            t,val = str, None
        elif val=="liNone":
            t,val,nargs = int,None,'*'
        elif val=="lsNone":
            t,val,nargs = str,None,'*'
        elif type(val) is bool:
            t = str
            bool_keys.append(key)
        else:
            t = type(val)
        defaults[key] = val
        parser.add_argument("--%s"%key, type=t,
                            nargs=nargs)

    #This parser will have optional arguments set to
    #None if not set. So if that's the case, replace
    #with the default value
    args_dict=vars(parser.parse_args())
    for key in bool_keys:
        if args_dict[key] is not None:
            if args_dict[key]=='True':
                args_dict[key]=True
            elif args_dict[key]=='False':
                args_dict[key]=False
            else:
                raise ValueError('%s must be True or False'%key)

    output_dir = args_dict.pop("output_dir")
    """
    config_file = opj(args_dict['output_dir'], 'config_from_file.yml')
    with open(config_file,"rb") as f:
        try:
            config = yaml.load(f)
        except KeyError:
            config = {}
    """
    config = {}
    config['output_dir'] = output_dir
    for key,val in args_dict.items():
        if key in config:
            if val is not None:
                config[key] = val
        else:
            if val is not None:
                config[key] = val
            else:                
                config[key] = defaults[key]
    #I think most convenient to return
    #a namespace
    from argparse import Namespace
    config_namespace = Namespace(**config)
    return config_namespace

def get_alms_dr6(path_template, freqs=["90","150"], sim_seed=None, mlmax=None,
                 apply_extra_mask=None):
    alms = {}
    print(path_template)
    for freq in freqs:
        alms[freq] = []
        for split in range(NSPLIT):
            map_filename = Template(path_template).substitute(freq=freq, split=split)
            alm = hp.read_alm(map_filename)
            lmax=hp.Alm.getlmax(len(alm))
            if apply_extra_mask is not None:
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

def get_sims_from_cls_old(cl_signal_dict, cl_noise_dict,
                      signal_seed, noise_seed, mlmax=None,
                      nsplit=NSPLIT, w1=1.):
    
    channel_pairs = cl_signal_dict.keys()
    channels = []
    for p in channel_pairs:
        channels += list(p)
    channels = list(set(channels))
    print("getting sims for channels:",channels)

    #make covariance - should be NxN where N=len(channels)*mlmax
    print("getting signal cov")
    N = len(channels)
    signal_cov = np.zeros((N,N,mlmax+1))
    for i,c_i in enumerate(channels):
        for j,c_j in enumerate(channels):
            if (c_i, c_j) in cl_signal_dict:
                cl_ij = cl_signal_dict[c_i,c_j]
            else:
                cl_ij = cl_signal_dict[c_j,c_i]
            if mlmax is not None:
                cl_ij = cl_ij[:mlmax+1]
            signal_cov[i,j,:] = cl_ij
            signal_cov[j,i,:] = cl_ij
    print("signal_cov.shape:", signal_cov.shape)

    print("generating signal alms")
    #Now can generate signal alms
    signal_alms = curvedsky.rand_alm(signal_cov, seed=signal_seed)
    signal_alms = [s*w1 for s in signal_alms]
    signal_alm_dict = {}
    for i,c in enumerate(channels):
        signal_alm_dict[c] = signal_alms[i]

    #now generate noise - should be independent
    print("Getting noise alms")
    noise_alm_dict = {}
    if len(channels)==1:
        for c in channels:
            noise_cov = np.zeros((nsplit, nsplit, mlmax+1))
            for i in range(nsplit):
                noise_cov[i,i,:] = cl_noise_dict[(c,c)][:mlmax+1]
            noise_alms = curvedsky.rand_alm(noise_cov, seed=noise_seed)
            #apply w1
            noise_alms = [n*w1 for n in noise_alms]
            noise_alm_dict[c] = list(noise_alms)

    else:
        #if we're e.g. cross-correlating different ilc versions,
        #then the noise is not independent between "channels".
        #but it should be independent between splits.
        noise_cov = np.zeros((len(channels),len(channels), mlmax+1))
        for i,c_i in enumerate(channels):
            for j,c_j in enumerate(channels):
                noise_cov[i,j,:] = cl_noise_dict[(c_i,c_j)][:mlmax+1]
                #noise_cov[j,i,:] = cl_noise_dict[(c_i,c_j)][:mlmax+1]
        noise_alm_dict = {}
        for c in channels:
            noise_alm_dict[c] = []
        for split in range(nsplit):
            noise_alms = curvedsky.rand_alm(noise_cov, seed=noise_seed+split)
            #apply w1w
            noise_alms = [n*w1 for n in noise_alms]
            for i,c in enumerate(channels):
                noise_alm_dict[c].append(noise_alms[i])

    total_alms = {}
    for c in channels:
        total_alms[c] = [n+signal_alm_dict[c] for n in noise_alm_dict[c]]

    return total_alms, signal_alm_dict, noise_alms

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


def get_sim_alm_dr6_hilc_old(map_name, sim_seed, data_dir="/global/cscratch1/sd/maccrann/cmb/act_dr6/ilc_cldata_smooth-301-2_v1",
                         mlmax=None):
    map_filenames = [opj("sim_planck%03d_act%02d_%05d"%(sim_seed),
                                   "%s_split%d.fits"%(map_name, split))
                     for split in range(NSPLIT)]
    map_paths = [opj(data_dir, f) for f in map_filenames]
    alms = []
    for p in map_paths:
        alm = hp.read_alm(p)
        if mlmax is not None:
            alm = utils.change_alm_lmax(alm, mlmax)
        alms.append(alm)
    return alms

def get_sim_alm_dr6_hilc(map_name, sim_seed, sim_template_path, mlmax=None,
                         apply_extra_mask=None):
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

def get_cl_model_from_data(alm_dict, sg_window=301, sg_order=2):
    map_names = list(alm_dict.keys())
    return 

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
    if rank==0:
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
    px = qe.pixelization(shape=mask.shape,wcs=mask.wcs)
    w1 = maps.wfactor(1, mask)
    w2 = maps.wfactor(2, mask)
    w4 = maps.wfactor(4, mask)
    #w2 = 0.2758258447605825
    #w4 = 0.2708454607428685
    if rank==0:
        print("w1:",w1)
        print("w2:",w2)
        print("w4:",w4)

    
    #read in alms for data
    if rank==0:
        print("reading data alms")
    est_maps = (args.est_maps.strip()).split("_")


    extra_mask = None
    if args.apply_extra_mask is not None:
        extra_mask = enmap.read_map(args.apply_extra_mask)
        total_mask = mask*extra_mask
        w1 = maps.wfactor(1, total_mask)
        w2 = maps.wfactor(2, total_mask)
        w4 = maps.wfactor(4, total_mask)
    else:
        total_mask = mask

    data_alms = get_alms_dr6(args.data_template_path,
                             freqs=list(set(est_maps)), mlmax=args.mlmax,
                             apply_extra_mask=extra_mask)
    print("data_alms.keys():")
    print(data_alms.keys())
    #data_alms = get_alms_dr6_hilc(mlmax=args.mlmax,
    #                              map_names=list(set(est_maps)))

    #get total Cls...hmm - maybe easiest to
    #estimate these from input maps to start with,
    #but also add an estimate from sims option
    if args.total_Cl_from_data:
        if rank==0:
            print("getting data Cls")
        #get from data
        cltot_dict = get_data_Cltot_dict(data_alms,
                   sg_window=301, sg_order=2,
                w2=w2)
        #save total Cls
        with open(opj(args.output_dir,
                      "total_Cl_from_data.pkl"), "wb") as f:
            pickle.dump(cltot_dict, f)
    else:
        pass
        #get from sims

    #do setup
    #decide on lmin, lmax etc.
    #and what combination of maps to use
    #est_maps = recon_config["est_maps"]
    lmin, lmax = args.K_lmin, args.K_lmax

    n_alm = len( data_alms[ list(data_alms.keys())[0] ][0] )
    if args.mlmax is None:
        mlmax = hp.Alm.getlmax(n_alm)
        args.mlmax = mlmax
    else:
        mlmax = args.mlmax

    print("getting data C_ls for sims")
    print("mlmax = %d"%args.mlmax)
    if args.mask_sim_lmax is None:
        args.mask_sim_lmax=mlmax

    print("mask_sim_lmax = %d"%args.mask_sim_lmax)

    data_cl_signal_dict, data_cl_noise_dict, cl_split_auto_dict, cl_split_cross_dict = get_sim_model_from_data(
        args.data_template_path, freqs=list(set(est_maps)),
        mlmax=args.mask_sim_lmax, nsplit=NSPLIT, sg_window=args.sg_window, sg_order=args.sg_order,
        w2=w2)
    
    print("data_cl_signal_dict.keys()")
    print(data_cl_signal_dict.keys())
    #save these dictionaries for debugging
    if rank==0:
        with open(opj(args.output_dir, "data_cl_signal_dict.pkl"), "wb") as f:
            pickle.dump(data_cl_signal_dict, f)
        with open(opj(args.output_dir, "data_cl_noise_dict.pkl"), "wb") as f:
            pickle.dump(data_cl_noise_dict, f)
    
    if rank==0:
        print("lmin,lmax:", lmin,lmax)
        print("mlmax:",mlmax)

    """
    cltot_A = cltot_dict[(est_maps[0], est_maps[0])][:mlmax+1]
    cltot_B = cltot_dict[(est_maps[1], est_maps[1])][:mlmax+1]
    cltot_C = cltot_dict[(est_maps[2], est_maps[2])][:mlmax+1]
    cltot_D = cltot_dict[(est_maps[3], est_maps[3])][:mlmax+1]
    
    cltot_Y = cltot_dict[("hilc-tszandcibd", "hilc-tszandcibd")][:mlmax+1]
    cltot_XY = cltot_dict[("hilc", "hilc-tszandcibd")][:mlmax+1]
    """
    if rank==0:
        print("Setting up estimator")
    px = qe.pixelization(shape=mask.shape,wcs=mask.wcs)
    
    cltot_A = cltot_dict[(est_maps[0], est_maps[0])][:mlmax+1]
    cltot_B = cltot_dict[(est_maps[1], est_maps[1])][:mlmax+1]
    cltot_C = cltot_dict[(est_maps[2], est_maps[2])][:mlmax+1]
    cltot_D = cltot_dict[(est_maps[3], est_maps[3])][:mlmax+1]
    cltot_AC = cltot_dict[(est_maps[0], est_maps[2])][:mlmax+1]
    cltot_BD = cltot_dict[(est_maps[1], est_maps[3])][:mlmax+1]
    cltot_AD = cltot_dict[(est_maps[0], est_maps[3])][:mlmax+1]
    cltot_BC = cltot_dict[(est_maps[1], est_maps[2])][:mlmax+1]
    
    setup = setup_ABCD_recon(
        px, lmin, lmax, mlmax,
        cl_rksz[:mlmax+1], 
        cltot_A, cltot_B, cltot_C, cltot_D,
        cltot_AC, cltot_BD, cltot_AD, cltot_BC,
        do_lh=True, do_psh=True)
    profile = setup["profile"]
    print("filter_A:",profile/cltot_A)

    #also get noiseless N0 - for split estimator, true N0 should
    #be somewhere in between? Use signal only Cl (and here we
    #should divide by w4 - didn't need to before to just generate
    #gaussian alms)
    wLK_A = profile[:lmax+1]/(cltot_dict[(est_maps[0], est_maps[0])][:lmax+1])
    wLK_C = profile[:lmax+1]/(cltot_dict[(est_maps[2], est_maps[2])][:lmax+1])
    wGK_D = profile[:lmax+1]/(cltot_dict[(est_maps[3], est_maps[3])][:lmax+1])/2
    wGK_B = profile[:lmax+1]/(cltot_dict[(est_maps[1], est_maps[1])][:lmax+1])/2

    N0_K_nonoise_nonorm = noise_spec.qtt_asym(
        'src', mlmax, lmin, lmax,
        wLK_A, wGK_B, wLK_C, wGK_D,
        data_cl_signal_dict[(est_maps[0], est_maps[2])][:lmax+1],
        data_cl_signal_dict[(est_maps[1], est_maps[3])][:lmax+1],
        data_cl_signal_dict[(est_maps[0], est_maps[3])][:lmax+1],
        data_cl_signal_dict[(est_maps[1], est_maps[2])][:lmax+1])[0]/profile[:lmax+1]**2
    N0_K_nonoise = N0_K_nonoise_nonorm * setup["norm_K_CD"] * setup["norm_K_AB"]
    
    #run some measurements
    #filter
    if rank==0:
        print("filtering data alms")
    data_alms_Af = [setup["filter_A"](
        data_alms[est_maps[0]][split])
                    for split in range(NSPLIT)]
    data_alms_Bf = [setup["filter_B"](
        data_alms[est_maps[1]][split])
                    for split in range(NSPLIT)]
    """
    alm_hilc_f = [iici_setup["filter_A"](data_alms["hilc"][split])
                  for split in range(NSPLIT)]
    alm_tszandcibd_f = [iici_setup["filter_B"](data_alms["hilc-tszandcibd"][split])
                        for split in range(NSPLIT)]
    data_alms_Af = alm_hilc_f
    data_alms_Bf = alm_tszandcibd_f
    """
    if est_maps[2]!=est_maps[0]:
        data_alms_Cf = [setup["filter_C"](
        data_alms[est_maps[2]][split])
                    for split in range(NSPLIT)]
    else:
        data_alms_Cf = data_alms_Af
        
    if est_maps[3]!=est_maps[0]:
        data_alms_Cf = [setup["filter_D"](
        data_alms[est_maps[3]][split])
                    for split in range(NSPLIT)]
    else:
        data_alms_Df = data_alms_Af   
        
    #assert est_maps[2]==est_maps[0]
    #assert est_maps[3]==est_maps[0]
    #data_alms_Cf = data_alms_Af
    #data_alms_Df = data_alms_Af


    def run_ClKK(comm=None):
        #K from Q(ilc, ilc-tszandcibd)
        print("running K_AB")
        Ks_ab = four_split_K(
            setup["qfunc_K_AB"],
            data_alms_Af[0], data_alms_Af[1],
            data_alms_Af[2], data_alms_Af[3],
            Xdatp_0=data_alms_Bf[0],
            Xdatp_1=data_alms_Bf[1],
            Xdatp_2=data_alms_Bf[2],
            Xdatp_3=data_alms_Bf[3],
            )
        Ks_ab_lh = four_split_K(
            setup["qfunc_K_AB_lh"],
            data_alms_Af[0], data_alms_Af[1],
            data_alms_Af[2], data_alms_Af[3],
            Xdatp_0=data_alms_Bf[0],
            Xdatp_1=data_alms_Bf[1],
            Xdatp_2=data_alms_Bf[2],
            Xdatp_3=data_alms_Bf[3],
            )
        #save K^hat^X alm 
        hp.write_alm(opj(args.output_dir, "K_ab_hatX.fits"), Ks_ab[0], overwrite=True)
        hp.write_alm(opj(args.output_dir, "K_ab_lh_hatX.fits"), Ks_ab_lh[0], overwrite=True)

        #Actually save all the Ks - we'll need these for combining with mean-field
        if rank==0:
            with open(opj(args.output_dir, "K_ab.pkl"), "wb") as f:
                pickle.dump(Ks_ab, f)
            with open(opj(args.output_dir, "K_ab_lh.pkl"), "wb") as f:
                pickle.dump(Ks_ab_lh, f)


        #K from Q(ilc,ilc)
        print("running K_CD")
        if (est_maps[0],est_maps[1]) == (est_maps[2],est_maps[3]):
            Ks_cd = Ks_ab
            Ks_cd_lh = Ks_ab_lh
        else:
            Ks_cd = four_split_K(
                setup["qfunc_K_CD"],
                data_alms_Af[0], data_alms_Af[1],
                data_alms_Af[2], data_alms_Af[3],
                )
            Ks_cd_lh = four_split_K(
                setup["qfunc_K_CD_lh"],
                data_alms_Af[0], data_alms_Af[1],
                data_alms_Af[2], data_alms_Af[3],
                )
            #save K^hat^X alm 
            hp.write_alm(opj(args.output_dir, "K_cd_hatX.fits"), Ks_cd[0], overwrite=True)
            hp.write_alm(opj(args.output_dir, "K_cd_lh_hatX.fits"), Ks_cd_lh[0], overwrite=True)

        #Actually save all the Ks - we'll need these for combining with mean-field
        if rank==0:
            with open(opj(args.output_dir, "K_cd.pkl"), "wb") as f:
                pickle.dump(Ks_cd, f)
            with open(opj(args.output_dir, "K_cd_lh.pkl"), "wb") as f:
                pickle.dump(Ks_cd_lh, f)

        print("getting CL_KK")
        cl_abcd = split_phi_to_cl(Ks_ab, Ks_cd)
        cl_abcd_lh = split_phi_to_cl(Ks_ab_lh, Ks_cd_lh)
        #apply w4
        cl_abcd /= w4
        cl_abcd_lh /= w4
        #return raw auto, as well as Ks arrays - we need to save these for mean field
        return cl_abcd, cl_abcd_lh, Ks_ab, Ks_ab_lh, Ks_cd, Ks_cd_lh
        print("done")

    if args.do_auto:
        if rank==0:
            print("running auto")
            outputs={}
            #We probably also want to run on the kSZ theory here as the exact
            #filters may affect slightly what we get
            if args.ksz_theory_alm_file is not None:
                ksz_theory_alm = utils.change_alm_lmax(
                    hp.read_alm(args.ksz_theory_alm_file), args.mlmax)
                cl_ksz = savgol_filter(
                    curvedsky.alm2cl(ksz_theory_alm), 301, 2)
                ksz_theory_alms_f = [
                    f(ksz_theory_alm) for f in [
                        setup["filter_A"], setup["filter_B"], setup["filter_C"], setup["filter_D"]]
                    ]
                if args.do_qe:
                    K_ksz_AB = setup["qfunc_K_AB"](ksz_theory_alms_f[0],
                                                   ksz_theory_alms_f[1])
                    K_ksz_CD = setup["qfunc_K_CD"](ksz_theory_alms_f[2],
                                                   ksz_theory_alms_f[3])
                    Cl_KK_ksz_raw = curvedsky.alm2cl(K_ksz_AB, K_ksz_CD)
                    ksz_N0 = setup["get_fg_trispectrum_N0_ABCD"](cl_ksz, cl_ksz, cl_ksz, cl_ksz)
                    outputs["Cl_KK_ksz_theory_raw"] = Cl_KK_ksz_raw
                    outputs["N0_ksz_theory"] = ksz_N0
                    outputs["Cl_KK_ksz_theory"] = Cl_KK_ksz_raw-ksz_N0

                if args.do_lh:
                    K_ksz_AB = setup["qfunc_K_AB_lh"](ksz_theory_alms_f[0],
                                                   ksz_theory_alms_f[1])
                    K_ksz_CD = setup["qfunc_K_CD_lh"](ksz_theory_alms_f[2],
                                                   ksz_theory_alms_f[3])
                    Cl_KK_ksz_raw = curvedsky.alm2cl(K_ksz_AB, K_ksz_CD)
                    #ksz_N0 = setup["get_fg_trispectrum_N0_ABCD_lh"](cl_ksz, cl_ksz, cl_ksz, cl_ksz)
                    outputs["Cl_KK_ksz_theory_raw_lh"] = Cl_KK_ksz_raw
                    #outputs["N0_ksz_theory_lh"] = ksz_N0
                    #outputs["Cl_KK_ksz_theory_lh"] = Cl_KK_ksz_raw-ksz_N0

                if args.do_psh:
                    K_ksz_AB = setup["qfunc_K_AB_psh"](ksz_theory_alms_f[0],
                                                   ksz_theory_alms_f[1])
                    K_ksz_CD = setup["qfunc_K_CD_psh"](ksz_theory_alms_f[2],
                                                   ksz_theory_alms_f[3])
                    Cl_KK_ksz_raw = curvedsky.alm2cl(K_ksz_AB, K_ksz_CD)
                    #ksz_N0 = setup["get_fg_trispectrum_N0_ABCD_psh"](cl_ksz, cl_ksz, cl_ksz, cl_ksz)
                    outputs["Cl_KK_ksz_theory_raw_psh"] = Cl_KK_ksz_raw
                    #outputs["N0_ksz_theory_psh"] = ksz_N0
                    #outputs["Cl_KK_ksz_theory_psh"] = Cl_KK_ksz_raw-ksz_N0


            #Now run on data
            cl_KK_raw, cl_KK_lh_raw, Ks_ab, Ks_ab_lh, Ks_cd, Ks_cd_lh = run_ClKK(comm=comm)
            outputs["cl_KK_raw"] = cl_KK_raw
            outputs["cl_KK_lh_raw"] = cl_KK_lh_raw
            outputs["N0"] = setup["N0_ABCD_K"]
            outputs["N0_lh"] = setup["N0_ABCD_K_lh"]
            outputs["N0_nonoise"] = N0_K_nonoise
            outputs["Ks_ab"] = Ks_ab
            outputs["Ks_ab_lh"] = Ks_ab_lh
            outputs["Ks_cd"] = Ks_cd
            outputs["Ks_cd_lh"] = Ks_cd_lh
            outputs["cl_rksz"] = cl_rksz #save what we used for filters
            outputs["cltot_A"] = cltot_A
            outputs["cltot_B"] = cltot_B
            outputs["cltot_C"] = cltot_C
            outputs["cltot_D"] = cltot_D
            outputs["cltot_AC"] = cltot_AC
            outputs["cltot_BD"] = cltot_BD
            outputs["cltot_AD"] = cltot_AD
            outputs["cltot_BC"] = cltot_BC

            for leg,cltot in zip(["A","B","C","D"], [cltot_A,cltot_B,cltot_C,cltot_D]):
                outputs["total_filter_%s"%leg] = profile / cltot

            outputs["lmin"] = lmin
            outputs["lmax"] = lmax
            outputs["profile"] = setup["profile"]

            #save pkl
            with open(opj(args.output_dir, "auto_outputs.pkl"), 'wb') as f:
                pickle.dump(outputs, f)

    def run_meanfield(setup, nsim, use_mpi=False, est=None,
                      combine_with_auto=False,
                      Ks_ab=None, Ks_cd=None, 
                      Ks_ab_lh=None, Ks_cd_lh=None,
                      fg_power_data=None, meanfield_tag=None
                      ):
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
            
            
        safe_mkdir(mean_field_outdir)
        
        if args.get_sims_from_data_cls:
            
            def get_sim_alms_A(i):
                print("getting sims A %d"%i)
                signal_seed = (args.seed*(i+1)*nsim)
                noise_seed = signal_seed+1
                print("signal seed:", signal_seed)
                print("noise seed:", noise_seed)
                #all_alms,_,_ = get_sims_from_cls(
                #    data_cl_signal_dict, data_cl_noise_dict,
                #    signal_seed, noise_seed, mlmax=mlmax,
                #    w1=w1)
                all_alms,_,_ = get_sims_from_cls(
                    cl_split_auto_dict, cl_split_cross_dict, total_mask,
                    signal_seed, noise_seed, mlmax=args.mask_sim_lmax,
                    w1=w1, lmax_out=mlmax)
                
                alms=all_alms[est_maps[0]]
                print("len(alms)",len(alms))
                alms_f = [setup["filter_A"](alm) for alm in alms]
                return alms_f

            def get_sim_alms_B(i):
                signal_seed = (args.seed*(i+1)*nsim)
                noise_seed = signal_seed+1
                all_alms,_,_ = get_sims_from_cls(
                    cl_split_auto_dict, cl_split_cross_dict, total_mask,
                    signal_seed, noise_seed, mlmax=args.mask_sim_lmax,
                    w1=w1, lmax_out=mlmax)
                alms=all_alms[est_maps[1]]
                alms_f = [setup["filter_B"](alm) for alm in alms]
                return alms_f
            
        else:
        
            def get_sim_alms_A(i):
                #alms = get_sim_alm_dr6_hilc(
                #    est_maps[0], (200+i,0,i+1),
                #    data_dir=DATA_DIR, mlmax=mlmax)
                print("getting sim from template path:", args.sim_template_path)
                alms = get_sim_alm_dr6_hilc(
                    est_maps[0], (200+i,0,i+1),
                    args.sim_template_path,
                    mlmax=mlmax, apply_extra_mask=extra_mask)
                alms_f = [setup["filter_A"](alm) for alm in alms]
                return alms_f

            def get_sim_alms_B(i):
                #alms = get_sim_alm_dr6_hilc(
                #    est_maps[1], (200+i,0,i+1),
                #    data_dir=DATA_DIR, mlmax=mlmax)
                alms = get_sim_alm_dr6_hilc(
                    est_maps[1], (200+i,0,i+1),
                    args.sim_template_path,
                    mlmax=mlmax, apply_extra_mask=extra_mask)
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
                for i, est_map_i in enumerate(est_maps):
                    for j, est_map_j in enumerate(est_maps):
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
            if (est_maps[0],est_maps[1]) == (est_maps[2],est_maps[3]):
                Ks_cd = Ks_ab
                Ks_cd_lh = Ks_ab_lh
                
            else:
                assert (est_maps[2]==est_maps[0] and est_maps[3]==est_maps[0])
                Ks_cd = four_split_K(
                    qfunc_CD,
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
            print(est_maps)
        else:
            fg_power_data = None

        if args.do_qe:
            print("running mean-field for qe case")
            run_meanfield(setup, args.nsim_meanfield,  
                          combine_with_auto=args.combine_with_auto,
                          Ks_ab=None, Ks_cd=None, est=None,
                          Ks_ab_lh=None, Ks_cd_lh=None,
                          fg_power_data = fg_power_data,
                          meanfield_tag = args.meanfield_tag
                          )

        if args.do_lh:
            print("running mean-field for lensing-hardening case")
            run_meanfield(setup, args.nsim_meanfield,
                          combine_with_auto=args.combine_with_auto,
                          Ks_ab=None, Ks_cd=None,
                          Ks_ab_lh=None, Ks_cd_lh=None,
                          fg_power_data = fg_power_data,
                          meanfield_tag = args.meanfield_tag,
                          est="lh"
                          )

        if args.do_psh:
            print("running mean-field for psh case")
            run_meanfield(setup, args.nsim_meanfield,
                          combine_with_auto=args.combine_with_auto,
                          Ks_ab=None, Ks_cd=None,
                          Ks_ab_lh=None, Ks_cd_lh=None,
                          fg_power_data = fg_power_data,
                          meanfield_tag = args.meanfield_tag,
                          est="psh"
                          )

            
                  
    def run_rdn0(setup, nsim, use_mpi=False,
                 est=None):
        if args.get_sims_from_data_cls:
            
            def get_sim_alms_A(i):
                print("getting sims A %d"%i)
                signal_seed = (args.seed*(i+1)*nsim)
                noise_seed = signal_seed+1
                print("signal seed:", signal_seed)
                print("noise seed:", noise_seed)
                #all_alms,_,_ = get_sims_from_cls(
                #    data_cl_signal_dict, data_cl_noise_dict,
                #    signal_seed, noise_seed, mlmax=mlmax,
                #    w1=w1)
                all_alms,_,_ = get_sims_from_cls(
                    cl_split_auto_dict, cl_split_cross_dict, mask,
                    signal_seed, noise_seed, mlmax=args.mask_sim_lmax,
                    w1=w1, lmax_out=mlmax)
                
                alms=all_alms[est_maps[0]]
                print("len(alms)",len(alms))
                alms_f = [setup["filter_A"](alm) for alm in alms]
                return alms_f

            def get_sim_alms_B(i):
                signal_seed = (args.seed*(i+1)*nsim)
                noise_seed = signal_seed+1
                all_alms,_,_ = get_sims_from_cls(
                    cl_split_auto_dict, cl_split_cross_dict, mask,
                    signal_seed, noise_seed, mlmax=args.mask_sim_lmax,
                    w1=w1, lmax_out=mlmax)

                alms=all_alms[est_maps[1]]
                alms_f = [setup["filter_B"](alm) for alm in alms]
                return alms_f

        else:
            #get sim alm functions
            def get_sim_alms_A(i):
                #alms = get_sim_alm_dr6_hilc(
                #    est_maps[0], (200+i,0,i+1),
                #    data_dir=DATA_DIR, mlmax=mlmax)
                print("getting sim from template path:", args.sim_template_path)
                alms = get_sim_alm_dr6_hilc(
                    est_maps[0], (200+i,0,i+1),
                    args.sim_template_path,
                    mlmax=mlmax, apply_extra_mask=extra_mask)
                alms_f = [setup["filter_A"](alm) for alm in alms]
                return alms_f

            def get_sim_alms_B(i):
                #alms = get_sim_alm_dr6_hilc(
                #    est_maps[1], (200+i,0,i+1),
                #    data_dir=DATA_DIR, mlmax=mlmax)
                alms = get_sim_alm_dr6_hilc(
                    est_maps[1], (200+i,0,i+1),
                    args.sim_template_path,
                    mlmax=mlmax, apply_extra_mask=extra_mask)
                alms_f = [setup["filter_B"](alm) for alm in alms]
                return alms_f

        if est_maps[2]==est_maps[0]:
            get_sim_alms_C = None
        else:
            raise NotImplementedError("")
        if est_maps[3]==est_maps[0]:
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
            rdn0, mcn0 = run_rdn0(setup, args.nsim_rdn0, use_mpi=args.use_mpi)
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


            
        """
        outputs["theory_N0"] = setup["N0_ABCD_K"]
        outputs["theory_N0_lh"] = setup["N0_ABCD_K_lh"]
        #save pkl

        output_file = "rdn0_outputs_nsim%d.pkl"%args.nsim_rdn0
        rdn0_file = opj(args.output_dir, "rdn0_outputs.pkl")
        print("saving rdn0 outputs to %s"%rdn0_file)
        with open(rdn0_file, 'wb') as f:
            pickle.dump(outputs, f)
    
    #run rdn0

    rdn0, mcn0 = run_rdn0(setup, nsim_rdn0, use_mpi=use_mpi)
    outputs = {}
    outputs["rdn0"] = rdn0
    outputs["mcn0"] = mcn0
    outputs["theory_N0"] = iici_setup["N0_ABCD_K"]
    outputs["theory_N0_lh"] = iici_setup["N0_ABCD_K_lh"]
    return outputs
            
            
    def run_rdn0(nsim):
        #get sim alm functions
        def get_sim_alms_A(i):
            alms = get_sim_alm_dr6_hilc(
                est_maps[0], (200+i,0,i+1),
                data_dir=DATA_DIR, mlmax=mlmax)
            alms_f = [iici_setup["filter_A"](alm) for alm in alms]
            return alms_f
        
        def get_sim_alms_B(i):
            alms = get_sim_alm_dr6_hilc(
                est_maps[1], (200+i,0,i+1),
                data_dir=DATA_DIR, mlmax=mlmax)
            alms_f = [iici_setup["filter_B"](alm) for alm in alms]
            return alms_f

        if est_maps[2]==est_maps[0]:
            get_sim_alms_C = None
        else:
            raise NotImplementedError("")
        if est_maps[3]==est_maps[0]:
            get_sim_alms_D = None
        else:
            raise NotImplementedError("")


        rdn0,mcn0 = mcrdn0_s4(args.nsim_rdn0, split_phi_to_cl,
                              iici_setup["qfunc_K_AB"],
                              four_split_K,
                              get_sim_alms_A, data_alms_Af,
                              get_sim_alms_B = get_sim_alms_B,
                              data_split_alms_B = data_alms_Bf,
                              qfunc_CD=iici_setup["qfunc_K_CD"],
                              use_mpi=args.use_mpi)

        rdn0 /= w4
        mcn0 /= w4

        return rdn0, mcn0

    if args.do_rdn0:
        rdn0, mcn0 = run_rdn0(args.nsim_rdn0)
        outputs = {}
        outputs["rdn0"] = rdn0
        outputs["mcn0"] = mcn0
        outputs["theory_N0"] = iici_setup["N0_ABCD_K"]
        outputs["theory_N0_lh"] = iici_setup["N0_ABCD_K_lh"]
        #save pkl
        rdn0_file = opj(args.output_dir, "rdn0_outputs.pkl")
        print("saving rdn0 outputs to %s"%rdn0_file)
        with open(rdn0_file, 'wb') as f:
            pickle.dump(outputs, f)
    """


    

if __name__=="__main__":
    main()
