#Having generated alms, do the reconstuction
#and cross-correlate with true kappa
import pickle
import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
from falafel import utils, qe
import pytempura
import solenspipe
from pixell import lensing, curvedsky, enmap
from pixell import utils as putils
from os.path import join as opj
import argparse
import yaml
from collections import OrderedDict
from cmbsky import safe_mkdir, get_disable_mpi, get_cmb_alm_unlensed, get_cmb_seeds
from orphics import maps
from websky_model import WebSky
from copy import deepcopy
import sys
from scipy.signal import savgol_filter

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=CB_color_cycle)

disable_mpi = get_disable_mpi()
if not disable_mpi:
    from mpi4py import MPI

with open("defaults.yaml",'rb') as f:
    DEFAULTS=yaml.load(f)

def get_cl_smooth(alms):
    cl = curvedsky.alm2cl(alms)
    l = np.arange(len(cl))
    d = l*(l+1)*cl
    #smooth with savgol
    d_smooth = savgol_filter(d, 5, 2)
    #if there's still negative values, set them
    #to zero
    d_smooth[d_smooth<0.] = 0.
    return np.where(
        l>0,d_smooth/l/(l+1),0.)
    
def get_config(section='reconstruction',
               description="Do reconstruction"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("output_dir", type=str)
    #parser.add_argument("--cmb_seeds", type=int,
    #                     nargs='*', default=None)
    #and otherwise variables are set to defaults,
    #or overwritten on command line
    defaults = {}
    bool_keys = []
    for key,val in DEFAULTS[section].items():
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
    
    config_file = opj(args_dict['output_dir'], 'config_from_file.yml')
    output_dir = args_dict.pop("output_dir")
    with open(config_file,"rb") as f:
        try:
            config = yaml.load(f)[section]
        except KeyError:
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

def dummy_teb(alms):
    return [alms, np.zeros_like(alms), np.zeros_like(alms)]

def white_noise(shape,wcs,noise_muK_arcmin,seed):
    """
    Generate a non-band-limited white noise map.
    """
    div = maps.ivar(shape,wcs,noise_muK_arcmin)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape) / np.sqrt(div)

def setup_recon(prep_map_config, recon_config,
                sim_start_ind=0):
    
    mlmax = recon_config['mlmax']
    lmin,lmax = recon_config['K_lmin'], recon_config['K_lmax']
    filter_noise_sigma = recon_config["noise_sigma"]

    res = prep_map_config["res"]*putils.arcmin
    filter_beam_fwhm = recon_config["beam_fwhm"]

    ells = np.arange(mlmax+1)
    bfact = maps.gauss_beam(ells, filter_beam_fwhm)**2.
    shape,wcs = enmap.fullsky_geometry(res=res,
                                     proj='car')    
    Nl_tt = (filter_noise_sigma*np.pi/180./60.)**2./bfact
    nells = {"TT":Nl_tt, "EE":2*Nl_tt, "BB":2*Nl_tt}

    #At present, if we haven't added noise, we never
    #actually make a map, so the pixelization is still
    #that of the input signal i.e. healpix nisde 4096
    if prep_map_config['disable_noise']:
        px = qe.pixelization(nside=recon_config['nside'])
    else:
        px = qe.pixelization(shape=shape,wcs=wcs)

    #CMB theory for filters
    ucls,tcls = utils.get_theory_dicts(grad=True,
                                       nells=nells, lmax=mlmax)

    
    cl_total_cmb = (tcls['TT']).copy()
    if prep_map_config['disable_noise']:
        cl_total_cmb -= nells['TT']

    #It may be that we want to use a filter from
    #file
    if "Cl_tot" in recon_config:
        if recon_config["Cl_tot"] is not None:
            print("using Cl_tot from file")
            tcls["TT"] = recon_config["Cl_tot"]
        
    #And ksz2 filter
    #Read alms and get (smooth) Cl for filter
    #and rdn0
    alm_file = recon_config['ksz_reion_alms']
    alms=hp.fitsfunc.read_alm(alm_file)
    alm_lmax=hp.Alm.getlmax(len(alms))
    if alm_lmax>mlmax:
        alms = utils.change_alm_lmax(
            alms, mlmax)
    elif alm_lmax<mlmax:
        raise ValueError("alm_lmax (=%d) < mlmax (=%d)"%(
            alm_lmax, mlmax)
                         )
    cl = curvedsky.alm2cl(alms)
    d = ells*(ells+1)*cl
    d_smooth = savgol_filter(d, 101, 3)
    cl_smooth = d_smooth/ells/(ells+1)
    cl_smooth[0]=cl_smooth[1]
    
    #tcls['TT'][:len(cl_smooth)] += cl_smooth
    cl_total = tcls['TT']
    
    #Get qfunc and normalization
    #profile is Cl**0.5
    profile = cl_smooth**0.5
    norm_s = pytempura.get_norms(
        ['src'], ucls, tcls,
        lmin, lmax, k_ellmax=mlmax,
        profile=profile)['src']
    norm_s[0]=0.
    #The normalization of the Smith and
    #Ferraro estimator:
    norm_K = norm_s / (profile)**2
    #For the normalized estimator this
    #is also the N0.
    N0_K_normed = norm_K
    #otherwise, it is
    N0_K_nonorm = 1./norm_K

    print('getting point-source norm')
    print(tcls['TT'])
    print(np.any(tcls['TT']==0.))
    print(np.any(~np.isfinite(tcls['TT'])))
    norm_ps = pytempura.get_norms(
        ['src'], ucls, tcls,
        lmin, lmax, k_ellmax=mlmax)['src']

    #Also get lensing norms and cross-response for
    #lensing hardening
    print('getting lensing norm')
    norm_lens = pytempura.norm_lens.qtt(
        mlmax,lmin,lmax,ucls['TT'],
        tcls['TT'],gtype='')[0]
    
    print('getting source-lens response')
    print(tcls['TT'])
    print(profile)
    R_s_tt = pytempura.get_cross(
        'SRC','TT', ucls, tcls,
        lmin, lmax, k_ellmax=mlmax,
        profile=profile)

    print('getting source-point-source response')
    R_s_ps = (1./pytempura.get_norms(
        ['src'], ucls, tcls,
        lmin, lmax, k_ellmax=mlmax,
        profile = profile**0.5)['src'])
    R_s_ps[0] = 0.
    
    z = np.zeros_like(alms)
    def filter_alms(alms):
        if len(alms)!=3:
            alms = dummy_teb(alms)
        return utils.isotropic_filter(
            alms, tcls, lmin, lmax,
            ignore_te=True)[0]

    #unnormalized source estimator
    def qfunc_s(X, Y):
        return qe.qe_source(px,mlmax,profile=profile,
                         fTalm=Y,xfTalm=X)

    #Smith/Ferraro estimator (x 2!)
    def qfunc_sf(X,Y):
        s_nonorm = qfunc_s(X, Y)
        K_nonorm = curvedsky.almxfl(s_nonorm, profile)
        return K_nonorm

    #Normalized version of Smith/Ferraro estimator x 2
    #This works out as s_normed / (profile),
    #or norm_K * K_nonorm!
    def qfunc_sf_normed(X,Y):
        K_nonorm = qfunc_sf(X,Y)
        return curvedsky.almxfl(
            K_nonorm, norm_K
            )
    
    #qfunc for lensing
    qfunc_lens = lambda X,Y: qe.qe_all(
        px,ucls,mlmax,fTalm=Y,fEalm=None,fBalm=None,
        estimators=['TT'], xfTalm=X, xfEalm=None,
        xfBalm=None)['TT']
    
    def qfunc_lens_norm(X,Y):
        phi = qfunc_lens(X,Y)
        return curvedsky.almxfl(phi, norm_lens)

    def qfunc_lh_normed(X,Y):
        #use e.g. eqn. 26 of arxiv:1209.0091
        #s and phi are the biased source and phi
        #estimates respectively
        s = curvedsky.almxfl(qfunc_s(X,Y), norm_s)
        phi = curvedsky.almxfl(
            qfunc_lens(X,Y)[0], norm_lens) #0th element is gradient
        num = s - curvedsky.almxfl(
            phi, norm_s * R_s_tt)
        denom = 1 - norm_s * norm_lens * R_s_tt**2
        s_normed_lh = curvedsky.almxfl(num, 1./denom)
        K_normed_lh = curvedsky.almxfl(s_normed_lh, 1./profile)
        return K_normed_lh
    
    #Also compute N0 for this case
    N0_s_lh = norm_s / (1 - norm_s * norm_lens * R_s_tt**2)
    N0_K_lh_normed = N0_s_lh / (profile)**2
    N0_K_lh_nonorm = N0_K_lh_normed / norm_K**2
    
    #qfunc for lensing-hardened K estimator
    def qfunc_lh(X,Y):
        """
        #use e.g. eqn. 26 of arxiv:1209.0091
        #s and phi are the biased source and phi
        #estimates respectively
        s = curvedsky.almxfl(qfunc_s(X,Y), norm_s)
        phi = curvedsky.almxfl(
            qfunc_lens(X,Y)[0], norm_lens) #0th element is gradient
        num = s - curvedsky.almxfl(
            phi, norm_s * R_s_tt)
        denom = 1 - norm_s * norm_lens * R_s_tt**2
        s_lh = curvedsky.almxfl(num, 1./denom)
        #This is the normalized source estimator
        #we now need to apply following factors
        K_normed = curvedsky.almxfl(s_lh, 2*profile/norm_s)
        """
        K_normed_lh = qfunc_lh_normed(X,Y)
        return curvedsky.almxfl(K_normed_lh, 1./norm_K)
        
    def qfunc_ps(X,Y):
        return qe.qe_source(px, mlmax,
                            fTalm=Y,xfTalm=X)
    def qfunc_psh_normed(X,Y):
        s = curvedsky.almxfl(qfunc_s(X,Y), norm_s)
        ps = curvedsky.almxfl(
            qfunc_ps(X,Y), norm_ps)
        num = s - curvedsky.almxfl(
            ps, norm_s * R_s_ps)
        denom = 1 - norm_s * norm_ps * R_s_ps**2
        s_psh_normed = curvedsky.almxfl(num, 1./denom)
        K_normed_psh = curvedsky.almxfl(s_psh_normed,
                                        1./profile)
        return K_normed_psh
    #Also compute N0 for this case
    N0_s_psh = norm_s / (1 - norm_s * norm_ps * R_s_ps**2)
    N0_K_psh_normed = N0_s_psh / (profile)**2
    N0_K_psh_nonorm = N0_K_psh_normed / norm_K**2

    def qfunc_psh(X,Y):
        """
        #use e.g. eqn. 26 of arxiv:1209.0091
        #s and phi are the biased profile source
        #and point source
        #estimates respectively
        s = curvedsky.almxfl(qfunc_s(X,Y), norm_s)
        ps = curvedsky.almxfl(
            qfunc_ps(X,Y), norm_ps)
        num = s - curvedsky.almxfl(
            ps, norm_s * R_s_ps)
        denom = 1 - norm_s * norm_ps * R_s_ps**2
        s_psh = curvedsky.almxfl(num, 1./denom)
        #This is the normalized source estimator
        #we now need to apply following factors
        return curvedsky.almxfl(s_psh, 2*profile/norm_s)
        """
        K_normed_psh = qfunc_psh_normed(X,Y)
        return curvedsky.almxfl(K_normed_psh, 1./norm_K)


    
    
    assert prep_map_config['disable_noise']
    def get_sim_alm(seed, add_kszr=True,
                    add_cmb=True,
                    cl_fg_func=None,
                    lensed_cmb=True):
        print("calling get_sim_alm with")
        print("add_kszr:",add_kszr)
        print("add_cmb:",add_cmb)
        print("lensed_cmb:",lensed_cmb)
        if cl_fg_func is not None:
            print("adding Gaussian fgs")
        #generate alms
        #seed has form (icov, iset, inoise)
        #we just need one number - the following
        #should ensure we get unique realizations
        #for rdn0
        ksz2_seed = seed[1]*9999 + seed[2]
        if add_kszr:
            sim_alms = curvedsky.rand_alm(cl_smooth,
                                          seed=ksz2_seed)
        else:
            sim_alms = np.zeros(hp.Alm.getsize(mlmax),
                                dtype=np.complex128)

        s_i,s_set,noise_seed = solenspipe.convert_seeds(seed)
        if add_cmb:
            if lensed_cmb:
                cmb_alms = solenspipe.get_cmb_alm(s_i, s_set)[0]
            else:
                cmb_alms = get_cmb_alm_unlensed(s_i, s_set)[0]
            cmb_alms = utils.change_alm_lmax(cmb_alms.astype(np.complex128), mlmax)
            sim_alms += cmb_alms

        if cl_fg_func is not None:
            print("adding Gaussian foreground power:")
            cl_fg = cl_fg_func(np.arange(mlmax+1))
            print(cl_fg)
            fg_alms = curvedsky.rand_alm(
                cl_fg, seed=ksz2_seed)
            sim_alms += fg_alms
            
        #filter and return
        return filter_alms(sim_alms)

    recon_stuff = {"shape" : shape,
                   "wcs" : wcs,
                   "norm_s" : norm_s,
                   "norm_K" : norm_K,
                   "ucls" : ucls,
                   "tcls" : tcls,
                   "qfunc_sf" : qfunc_sf,
                   "qfunc_normed" : qfunc_sf_normed,
                   "qfunc_lh_sf" : qfunc_lh,
                   "qfunc_lh_normed" : qfunc_lh_normed,
                   "qfunc_psh_sf" : qfunc_psh,
                   "qfunc_psh_normed" : qfunc_psh_normed,
                   "qfunc_lens" : qfunc_lens_norm,
                   "filter_alms" : filter_alms,
                   "get_sim_alm" : get_sim_alm,
                   "cl_total":cl_total,
                   "cl_kszr": cl_smooth,
                   "cl_total_cmb": cl_total_cmb,
                   "profile": profile,
                   "norm_ps": norm_ps,
                   "norm_lens" : norm_lens,
                   "R_s_ps" : R_s_ps,
                   "R_s_tt" : R_s_tt,
                   "N0_K_normed" : N0_K_normed,
                   "N0_K_nonorm" : N0_K_nonorm,
                   "N0_K_lh_normed" : N0_K_lh_normed,
                   "N0_K_lh_nonorm" : N0_K_lh_nonorm,
                   "N0_K_psh_normed" : N0_K_psh_normed,
                   "N0_K_psh_nonorm" : N0_K_psh_nonorm,
        }

    return recon_stuff

def get_alm_files(map_dir, freq,
                  get_masked=False, get_unlensed=False):
    alm_files = OrderedDict([])
    alm_files["sky"] = opj(map_dir,"sky_alms_%s.fits"%freq)
    alm_files["cmb"] = opj(map_dir,"cmb_alms_%s.fits"%freq)
    alm_files["kszr"] = opj(map_dir,"kszr_alms_%s.fits"%freq)
    #if get_masked:
    #    alm_files["sky_masked"] = opj(
    #        map_dir,"sky_masked_alms_%s.fits"%freq)
    if get_unlensed:
        alm_files["sky_unlensed"] = opj(
            map_dir,"sky_unlensed_alms_%s.fits"%freq)
    return alm_files

from reconstruction import setup_recon

def main():

    args = get_config()
    print(args)
    recon_config = vars(args)

    if not disable_mpi:
        comm = MPI.COMM_WORLD
        rank,size = comm.Get_rank(), comm.Get_size()
    else:
        rank,size = 0,1
    
    #get config from prep map stage
    with open(opj(args.output_dir, 'prep_map', 'prepare_map_config.yml'),'r') as f:
        prep_map_config = yaml.load(f)
    map_dir = opj(args.output_dir, "prep_map")

    #pixelisation - for websky and sehgal assume
    #this is healpix
    px = qe.pixelization(nside=recon_config['nside'])

    map_dir = opj(args.output_dir, "prep_map")
    
    #make output subdirectory for reconstruction stuff
    recon_dir = opj(args.output_dir, "recon_%s"%args.tag)
    safe_mkdir(recon_dir)
    if recon_config['mlmax'] is None:
        recon_config['mlmax'] = prep_map_config["mlmax"]
    if recon_config['freqs'] is None:
        recon_config['freqs'] = prep_map_config['freqs']
    print(recon_config)
    #save args to output dir
    arg_file = opj(recon_dir,'recon_config.yml')
    if rank==0:
        with open(arg_file,'w') as f:
            yaml.dump(recon_config, f)
    
    #Get rksz power
    cl_rksz = np.load(recon_config["rksz_cl_file"])

    cmb_seeds = get_cmb_seeds(args)
    if cmb_seeds is None:
        cmb_seeds = prep_map_config['cmb_seeds']
    print("cmb_seeds:",cmb_seeds)
    if rank==0:
        print("doing cmb seeds:",cmb_seeds)

    if args.freqs is None:
        args.freqs = prep_map_config['freqs']
    if args.skip_tags is None:
        args.skip_tags = []

    print("normalizing estimator:", args.normalized)
        
    #Loop through cmb seeds
    for icmb,cmb_seed in enumerate(cmb_seeds):
        if icmb%size != rank:
            continue
        cmb_seed = cmb_seeds[icmb]
        if cmb_seed is None:
            map_dir_this_seed = opj(map_dir, "cmb_orig")
            recon_dir_this_seed = opj(recon_dir, "cmb_orig")
            print("doing reconstruction for cmb orig case")
        else:
            map_dir_this_seed = opj(map_dir, "cmb_%d"%cmb_seed)
            recon_dir_this_seed = opj(recon_dir, "cmb_%d"%cmb_seed)
            print("rank %d doing reconstruction for cmb_seed %d"%(rank, cmb_seed))

        safe_mkdir(recon_dir_this_seed)

        #Load total cl data
        f = opj(map_dir_this_seed,
                "Cltot_data.npy")
        cltot_data = np.load(f)
        for freq in args.freqs:

            #This may change but I'm allowing args.freqs
            #to be e.g. freqcoadd or tsz-sym as well as just
            #numbers. So check if freq is a number
            freq_is_number = True
            try:
                float(freq)
            except ValueError:
                freq_is_number = False

            is_freq_diff=False
            if '-' in str(freq):
                try:
                    float(freq.split('-')[0])
                    float(freq.split('-')[1])
                    is_freq_diff = True
                except ValueError:
                    pass

            #These are the cases where we're using the same map in
            #each leg
            
            if (freq_is_number or
                (freq in ['hilc','hilc-tszd','hilc-cibd',
                          'hilc-tszandcibd','freqcoadd'])
                or is_freq_diff):
                sky_alm_file = opj(map_dir_this_seed,
                               "sky_alms_%s.fits"%freq)
                sky_alms = hp.fitsfunc.read_alm(alm_file)
                sky_alms = utils.change_alm_lmax(
                    alms, recon_config['mlmax'])
                cmb_alm_file = opj(map_dir_this_seed,
                                   "cmb_alms.fits")
                cmb_alms = utils.change_alm_lmax(
                    hp.fitsfunc.read_alm(cmb_alm_file))
                
            #The asymmetric (and symmetrized cases). Not here I've
            #hardcoded that we're using ilc in one of the legs, we
            #may want to generalize that. Read in a tuple of foreground
            #alms 
            elif freq in ["tszd-sym", "cibd-sym", "tszandcibd-sym"]:
                raise NotImplementedError("not yet implemented assymetric estimator")
            else:
                raise ValueError("frequency %s not recognized"%freq)

            if isinstance(fg_alms, tuple):
                raise NotImplementedError("not yet implemented assymetric estimator")
            else:
                cl_sky_minus_cmb = get_cl_fg_smooth(sky_alms-cmb_alms)
            
            #Get noise for filters
            Nl_tt = cltot_data["Nltt_%s"%freq][:recon_config["mlmax"]+1]
            nells = {"TT":Nl_tt, "EE":2*Nl_tt, "BB":2*Nl_tt}
            _,tcls = utils.get_theory_dicts(grad=True, nells=nells,
                                            lmax=recon_config["mlmax"])
            tcls['TT'] = cltot_data["Cltt_total_%s"%freq][:recon_config["mlmax"]+1]
            if recon_config["add_rkszcl_to_filter"]:
                tcls["TT"] += cl_rksz

            lmin,lmax=recon_config["K_lmin"], recon_config["K_lmax"]
            recon_stuff = setup_recon(
                px, lmin, lmax, recon_config["mlmax"],
                cl_rksz, tcls, do_lh=args.do_lh,
                do_psh=args.do_psh)

            """
            if cltot_data is not None:
                print(cltot_data.dtype.names)
                if freq in cltot_data.dtype.names:
                    recon_config["Cl_tot"] = cltot_data[freq]
                else:
                    print("no Cl_tot for freq %s, will use other options"%freq)
                    recon_config["Cl_tot"] = None
            """

            filter_alms = recon_stuff['filter_alms_X']
            profile = recon_stuff['profile']
            qfuncs = [("qe", recon_stuff["qfunc_K"])]
            if args.do_psh:
                qfuncs.append(
                    ("psh", recon_stuff["qfunc_K_psh"])
                    )
            if args.do_lh:
                qfuncs.append(
                    ("lh", recon_stuff["qfunc_K_lh"])
                    )

                
            cl_rr_out = {}
            #cl_tag = get_cl_smooth(alms) #the smoothed cl for this tag
            filtered_alms = filter_alms(alms)
            K_lmin, K_lmax, mlmax = (recon_config['K_lmin'],
                                     recon_config['K_lmax'],
                                     recon_config['mlmax'])
            profile = recon_setup['profile']
            norm_s = recon_setup["norm_s"]
            norm_ps = recon_setup["norm_ps"]
            norm_K = recon_setup["norm_K"]
            R_s_ps = recon_setup["R_s_ps"]
            norm_lens = recon_setup["norm_lens"]
            R_K_lens = recon_setup["R_s_tt"]

            if args.do_qe:
                if args.normalized:
                    qfunc = recon_setup["qfunc_normed"]
                else:
                    qfunc = recon_setup["qfunc_sf"]

                #No bias hardening
                K_recon_alms = qfunc(filtered_alms,
                                       filtered_alms)
                hp.fitsfunc.write_alm(
                    opj(recon_dir_this_seed,
                        os.path.basename(alm_file).replace(".fits","_K-recon.fits"),
                    ),
                    K_recon_alms, overwrite=True
                        )
                cl_rr = curvedsky.alm2cl(
                    K_recon_alms)
                cl_rr_out[tag] = cl_rr

                #Also get analytic N_0
                if tag == "kszr":
                    cl_signal = recon_setup["cl_kszr"]
                    Ctot = recon_setup['cl_total']**2 / cl_signal
                    norm_ksz = pytempura.get_norms(
                        ['src'], recon_setup['ucls'], {'TT':Ctot},
                        K_lmin, K_lmax,
                        k_ellmax=mlmax,
                        profile=profile)['src']
                    N0 = norm_s**2 / norm_ksz

                elif tag == "cmb":
                    cl_signal = recon_setup["cl_total_cmb"]
                    Ctot = recon_setup['cl_total']**2 / cl_signal
                    norm_cmb = pytempura.get_norms(
                        ['src'], recon_setup['ucls'], {'TT':Ctot},
                        K_lmin, K_lmax,
                        k_ellmax=mlmax, profile=profile)['src']
                    N0 = norm_s**2 / norm_cmb

                elif tag == "sky":
                    if prep_map_config["disable_noise"]:
                        cl_signal = recon_setup["cl_kszr"] + recon_setup["cl_total_cmb"]
                        Ctot = recon_setup['cl_total']**2 / cl_signal
                        norm_sky = pytempura.get_norms(
                            ['src'], recon_setup['ucls'], {'TT':Ctot},
                            K_lmin, K_lmax,
                            k_ellmax=mlmax, profile=profile)['src']
                        N0 = norm_s**2 / norm_sky
                    else:
                        N0 = norm_s 

                else:
                    N0 = np.nan * np.ones_like(norm_s)

                #Convert from N0 for s to K
                N0 /= (profile)**2
                N0_nonorm = N0 / norm_K**2
                cl_rr_out[tag+"_N0_normed"] = N0
                cl_rr_out[tag+"_N0_nonorm"] = N0_nonorm

                """
                #Also save total N0 (including noise)
                if args.normalized:
                    total_N0 = norm_s * (2*profile)**2
                else:
                    total_N0 = (2 * profile)**2 / norm_s
                cl_rr_out["total_N0"] = total_N0
                """

            #With lensing hardening
            if args.do_lh:
                print("doing lensing-hardened reconstruction")
                if args.normalized:
                    qfunc_lh = recon_setup["qfunc_lh_normed"]
                else:
                    qfunc_lh = recon_setup["qfunc_lh_sf"]
                K_recon_alms_lh = qfunc_lh(
                    filtered_alms, filtered_alms)
                hp.fitsfunc.write_alm(
                    opj(recon_dir_this_seed,
                        os.path.basename(alm_file).replace(".fits","_K-recon-lh.fits"),
                    ),
                    K_recon_alms_lh, overwrite=True
                        )
                cl_rr_out[tag+"_lh"] = curvedsky.alm2cl(
                    K_recon_alms_lh)

                #It'll be interesting to save the
                #result of the lensing estimator on the
                #kSZ map
                if tag == "kszr":
                    kszr_phi_alms = qfunc_lh(
                        filtered_alms, filtered_alms)
                    hp.fitsfunc.write_alm(
                        opj(recon_dir_this_seed,
                            os.path.basename(alm_file).replace(".fits","_kszr_phi_recon.fits"),
                        ),
                        kszr_phi_alms, overwrite=True
                    )
                    cl_signal = recon_setup["cl_kszr"]
                    Ctot = recon_setup['cl_total']**2 / cl_signal
                    norm_lens = recon_setup["norm_lens"]
                    R_s_lens = recon_setup["R_s_tt"]
                    norm_lens_fg = pytempura.norm_lens.qtt(
                        mlmax, K_lmin, K_lmax, recon_setup['ucls']['TT'],
                        Ctot, gtype='')[0]
                    R_s_lens_fg = pytempura.get_cross(
                        'SRC','TT', recon_setup['ucls'], {"TT" : Ctot},
                        K_lmin, K_lmax, k_ellmax = mlmax,
                        profile = profile)
                    N0 = norm_s**2/norm_ksz
                    N0_lens = norm_lens**2 / norm_lens_fg
                    N0_bh = (N0 + R_s_lens**2 * norm_s**2 * N0_lens
                              - 2 * R_s_lens * norm_s**2 * norm_lens * R_s_lens_fg)
                    N0_bh /= (1 - norm_s*norm_lens*R_s_lens**2)**2

                    #Convert from N0 for s to K
                    N0_bh /= (profile)**2
                    N0_bh_nonorm = N0 / norm_K**2
                    cl_rr_out[tag+"_N0_lh_normed"] = N0_bh
                    cl_rr_out[tag+"_N0_lh_nonorm"] = N0_bh_nonorm

                #Also save the total N0 for this case
                """
                total_N0 = norm_s / (1 - norm_s * norm_lens * R_K_lens**2)
                total_N0 *=  (2 * profile)**2
                if not args.normalized:
                    total_N0 /= norm_s
                cl_rr_out["total_N0_lh"] = total_N0
                """

            if args.do_psh:
                print("doing point source-hardened reconstruction")
                if args.normalized:
                    qfunc_psh = recon_setup["qfunc_psh_normed"]
                else:
                    qfunc_psh = recon_setup["qfunc_psh_sf"]
                K_recon_alms_psh = qfunc_psh(
                    filtered_alms, filtered_alms)
                hp.fitsfunc.write_alm(
                    opj(recon_dir_this_seed,
                        os.path.basename(alm_file).replace(".fits","_K-recon-psh.fits"),
                    ),
                    K_recon_alms_psh, overwrite=True
                        )
                cl_rr_out[tag+"_psh"] = curvedsky.alm2cl(
                    K_recon_alms_psh)

                """
                #Also save the total N0 for this case
                total_N0 = norm_s / (1 - norm_s * norm_ps * R_s_ps**2)
                total_N0 *= (2 * profile)**2
                if not args.normalized:
                    total_N0 /= norm_s
                cl_rr_out["total_N0_psh"] = total_N0                
                """

            for key in ["N0_K_normed","N0_K_nonorm",
                        "N0_K_lh_normed","N0_K_lh_nonorm",
                        "N0_K_psh_normed","N0_K_psh_nonorm"]:
                cl_rr_out[key] = recon_setup[key]

            output_data = np.zeros(recon_config['mlmax']+1,
                                   dtype=[(k,float) for k in cl_rr_out.keys()])            
            for k,v in cl_rr_out.items():
                output_data[k] = v

            np.save(opj(recon_dir_this_seed, 'cl_%s_rr.npy'%freq), output_data)
            #also save normalizations and responses

            from solenspipe import weighted_bin1D
            bin_edges = np.linspace(10,200,15).astype(int)
            binner = weighted_bin1D(bin_edges)
            def bin_cl(cl):
                ix,weights = np.zeros_like(cl), np.ones_like(cl)
                return binner.bin(ix, cl, weights)
            ell_mids, cl_KK = bin_cl(cl_rr_out['kszr'])

if __name__=="__main__":
    main()
        
    
