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
from prepare_maps import safe_mkdir, get_disable_mpi, DEFAULTS, get_cmb_alm_unlensed, get_cmb_seeds
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
    
    mlmax = prep_map_config['mlmax']
    lmin,lmax = recon_config['K_lmin'], recon_config['K_lmax']
    noise_sigma = prep_map_config["noise_sigma"]
    res = prep_map_config["res"]*putils.arcmin
    beam_fwhm = prep_map_config["beam_fwhm"]
    beam_fn = lambda x: maps.gauss_beam(x, beam_fwhm)
    unbeam_fn = lambda x: 1./maps.gauss_beam(x, beam_fwhm)

    ells = np.arange(mlmax+1)
    bfact = maps.gauss_beam(beam_fwhm,ells)**2.
    shape,wcs = enmap.fullsky_geometry(res=res,
                                     proj='car')    
    Nl_tt = (noise_sigma*np.pi/180./60.)**2./bfact
    nells = {"TT":Nl_tt, "EE":2*Nl_tt, "BB":2*Nl_tt}

    #At present, if we haven't added noise, we never
    #actually make a map, so the pixelization is still
    #that of the input signal i.e. healpix nisde 4096
    if prep_map_config['disable_noise']:
        px = qe.pixelization(nside=4096)
    else:
        px = qe.pixelization(shape=shape,wcs=wcs)

    #CMB theory for filters
    ucls,tcls = utils.get_theory_dicts(grad=True,
                                       nells=nells, lmax=mlmax)

    #And ksz2 filter
    #Read alms and get (smooth) Cl for filter
    #and rdn0
    alm_file = prep_map_config['ksz_reion_alms']
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
    cl_smooth[0]=0.

    tcls['TT'][:len(cl_smooth)] += cl_smooth

    #Get qfunc and normalization
    #profile is Cl**0.5
    profile = cl_smooth**0.5
    norm_s = pytempura.get_norms(
        ['src'], ucls, tcls,
        lmin, lmax, k_ellmax=mlmax,
        profile=profile)['src']
    norm_s[0]=0.

    norm_ps = pytempura.get_norms(
        ['src'], ucls, tcls,
        lmin, lmax, k_ellmax=mlmax)['src']

    #Also get lensing norms and cross-response for
    #lensing hardening
    norm_lens = pytempura.norm_lens.qtt(
        mlmax,lmin,lmax,ucls['TT'],tcls['TT'],gtype='')[0]
    print("norm_s.shape:",norm_s.shape)
    print("norm_lens.shape:",norm_lens.shape)
    #assert norm_lens.shape == (mlmax+1,)
    R_src_tt = pytempura.get_cross(
        'SRC','TT', ucls, tcls,
        lmin, lmax, k_ellmax=mlmax,
        profile=profile)

    R_src_ps = (1./pytempura.get_norms(
        ['src'], ucls, tcls,
        lmin, lmax, k_ellmax=mlmax,
        profile = profile**0.5)['src'])
    R_src_ps[0] = 0.
    
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

    #qfunc for lensing
    qfunc_lens = lambda X,Y: qe.qe_all(
        px,ucls,mlmax,fTalm=Y,fEalm=None,fBalm=None,
        estimators=['TT'], xfTalm=X, xfEalm=None,
        xfBalm=None)['TT']
    #qfunc for lensing-hardened K estimator
    def qfunc_lh(X,Y):
        #use e.g. eqn. 26 of arxiv:1209.0091
        #s and phi are the biased source and phi
        #estimates respectively
        s = curvedsky.almxfl(qfunc_s(X,Y), norm_s)
        phi = curvedsky.almxfl(
            qfunc_lens(X,Y)[0], norm_lens) #0th element is gradient
        num = s - curvedsky.almxfl(
            phi, norm_s * R_src_tt)
        denom = 1 - norm_s * norm_lens * R_src_tt**2
        s_lh = curvedsky.almxfl(num, 1./denom)
        #This is the normalized source estimator
        #we now need to apply following factors
        return curvedsky.almxfl(s_lh, 2*profile/norm_s)


    def qfunc(X,Y):
        s = qfunc_s(X, Y)
        K = 2 * curvedsky.almxfl(s, profile)
        return K

    def qfunc_ps(X,Y):
        return qe.qe_source(px, mlmax,
                            fTalm=Y,xfTalm=X)

    def qfunc_psh(X,Y):
        #use e.g. eqn. 26 of arxiv:1209.0091
        #s and phi are the biased source and phi
        #estimates respectively
        print("norm_s:",norm_s)
        print("norm_ps:",norm_ps)
        print("R_src_ps:", R_src_ps)
        s = curvedsky.almxfl(qfunc_s(X,Y), norm_s)
        ps = curvedsky.almxfl(
            qfunc_ps(X,Y), norm_ps)
        num = s - curvedsky.almxfl(
            ps, norm_s * R_src_ps)
        denom = 1 - norm_s * norm_ps * R_src_ps**2
        s_psh = curvedsky.almxfl(num, 1./denom)
        #This is the normalized source estimator
        #we now need to apply following factors
        return curvedsky.almxfl(s_psh, 2*profile/norm_s)
    """
    def qfunc_lh2(X,Y):
        #There is another version of lensing
        #hardeneing where we ignore the phi
        #contamination
        s = qfunc_s(X, Y)
        K_hat = 2 * curvedsky.almxfl(s, profile)
        R = 2*profile*R_src_tt
        phi = curvedsky.almxfl(
            qfunc_lens(X,Y)[0], norm_lens) #0th element is gradient
        return K_hat - curvedsky.almxfl(phi, R)
    """

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
                   "ucls" : ucls,
                   "tcls" : tcls,
                   "qfunc" : qfunc,
                   "qfunc_lh" : qfunc_lh,
                   "qfunc_psh" : qfunc_psh,
                   "filter_alms" : filter_alms,
                   "get_sim_alm" : get_sim_alm,
        }

    return recon_stuff

def get_alm_files(config, map_dir, freq,
                  get_masked=False, get_unlensed=False):
    alm_files = OrderedDict([])
    alm_files["sky"] = opj(map_dir,"sky_alms_%s.fits"%freq)
    alm_files["cmb"] = opj(map_dir,"cmb_alms_%s.fits"%freq)
    alm_files["kszr"] = opj(map_dir,"kszr_alms_%s.fits"%freq)
    if get_masked:
        alm_files["sky_masked"] = opj(
            map_dir,"sky_masked_alms_%s.fits"%freq)
    if get_unlensed:
        alm_files["sky_unlensed"] = opj(
            map_dir,"sky_unlensed_alms_%s.fits"%freq)
    return alm_files

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
    #has_masked=False
    has_masked = ((prep_map_config["halo_mask_fgs"])
                  or (prep_map_config["cib_flux_cut"] is not None))
    map_dir = opj(args.output_dir, "prep_map")

    #make output subdirectory for reconstruction stuff
    recon_dir = opj(args.output_dir, "recon_%s"%args.tag)
    safe_mkdir(recon_dir)
    if recon_config['mlmax'] is None:
        recon_config['mlmax'] = prep_map_config["mlmax"]
    if recon_config['freqs'] is None:
        recon_config['freqs'] = prep_map_config['freqs']
    #save args to output dir
    arg_file = opj(recon_dir,'recon_config.yml')
    if rank==0:
        with open(arg_file,'w') as f:
            yaml.dump(recon_config, f)
    
    print("doing setup")
    recon_setup = setup_recon(
        prep_map_config, recon_config)
    filter_alms = recon_setup['filter_alms']
    qfunc = recon_setup['qfunc']
    qfunc_lh = recon_setup['qfunc_lh']
    qfunc_psh = recon_setup['qfunc_psh']

    print(prep_map_config)
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
        
    #Loop through cmb seeds
    for icmb,cmb_seed in enumerate(cmb_seeds):
        if icmb%size != rank:
            continue
        cmb_seed = cmb_seeds[icmb]
        if cmb_seed is None:
            map_dir_this_seed = opj(map_dir, "no_cmb")
            recon_dir_this_seed = opj(recon_dir, "no_cmb")
            print("doing reconstruction for no cmb case")
        else:
            map_dir_this_seed = opj(map_dir, "cmb_%d"%cmb_seed)
            recon_dir_this_seed = opj(recon_dir, "cmb_%d"%cmb_seed)
            print("rank %d doing reconstruction for cmb_seed %d"%(rank, cmb_seed))

        safe_mkdir(recon_dir_this_seed)

        for freq in args.freqs:
            #Read alms for reconstruction
            alm_files = get_alm_files(prep_map_config, map_dir_this_seed,
                                      freq, get_masked=has_masked,
                                      get_unlensed=False)
            print(alm_files)
            dtype = [(tag,float) for tag in alm_files]
            if args.do_lh:
                dtype += [(tag+"_lh",float) for tag in alm_files]
            if args.do_psh:
                dtype += [(tag+"_psh",float) for tag in alm_files]
            cl_rr_out = np.zeros(recon_config['mlmax']+1,
                                 dtype = dtype)
            for tag, alm_file in alm_files.items():
                if tag in args.skip_tags:
                    print("skipping %s reconstruction"%tag)
                    continue
                print("doing %s reconstruction"%tag)
                alms = hp.fitsfunc.read_alm(
                    alm_file)
                alms = utils.change_alm_lmax(alms, recon_config['mlmax'])
                filtered_alms = filter_alms(alms)
                if args.do_qe:
                    #No bias hardening
                    K_recon_alms = qfunc(filtered_alms,
                                           filtered_alms)
                    hp.fitsfunc.write_alm(
                        opj(recon_dir_this_seed,
                            os.path.basename(alm_file).replace(".fits","_K-recon.fits"),
                        ),
                        K_recon_alms, overwrite=True
                            )
                    cl_rr_out[tag] = curvedsky.alm2cl(
                        K_recon_alms)

                #With lensing hardening
                if args.do_lh:
                    print("doing lensing-hardened reconstruction")
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

                if args.do_psh:
                    print("doing point source-hardened reconstruction")
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
                
            np.save(opj(recon_dir_this_seed, 'cl_%s_rr.npy'%freq), cl_rr_out)

            from solenspipe import weighted_bin1D
            bin_edges = np.linspace(10,200,15).astype(int)
            binner = weighted_bin1D(bin_edges)
            def bin_cl(cl):
                ix,weights = np.zeros_like(cl), np.ones_like(cl)
                return binner.bin(ix, cl, weights)
            ell_mids, cl_KK = bin_cl(cl_rr_out['kszr'])


if __name__=="__main__":
    main()
        
    
