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
from cmbsky import safe_mkdir, get_disable_mpi
from orphics import maps
from copy import deepcopy
import sys
from scipy.signal import savgol_filter

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=CB_color_cycle)

WEBSKY_DIR="/global/cscratch1/sd/maccrann/cmb/websky/"

disable_mpi = get_disable_mpi()
if not disable_mpi:
    from mpi4py import MPI

def get_cl_fg_smooth(alms):
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

    
with open("fg_term_defaults.yaml",'rb') as f:
    DEFAULTS=yaml.load(f)
        
def get_config(description="Get fg terms"):
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
        print(key, t, nargs)
        parser.add_argument("--%s"%key, type=t,
                            nargs=nargs)
    #This parser will have optional arguments set to
    #None if not set. So if that's the case, replace
    #with the default value
    args_dict=vars(parser.parse_args())
    print("args_dict:",args_dict)
    for key in bool_keys:
        if args_dict[key] is not None:
            if args_dict[key]=='True':
                args_dict[key]=True
            elif args_dict[key]=='False':
                args_dict[key]=False
            else:
                raise ValueError('%s must be True or False'%key)
    print("args_dict:",args_dict)            
    
    config_file = opj(args_dict['output_dir'], 'config_from_file.yml')
    output_dir = args_dict.pop("output_dir")
    with open(config_file,"rb") as f:
        try:
            config = yaml.load(f)
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


from do_reconstruction import setup_recon


def trispectrum_N0_qe(cl_total, cl_fg, norm_s,
                      ucls, K_lmin, K_lmax, mlmax,
                      profile):
    
    Ctot = cl_total**2/cl_fg
    norm_fg = pytempura.get_norms(
        ['src'], ucls, {'TT' : Ctot},
        K_lmin, K_lmax, k_ellmax=mlmax,
        profile=profile)['src']
    
    L = np.arange(mlmax+1)
    N0_tri = norm_s**2 / norm_fg / profile**2
    return N0_tri

def trispectrum_N0_lh(cl_total, cl_fg, norm_s,
                      ucls, K_lmin, K_lmax, mlmax,
                      profile, norm_lens, R_s_lens):
    
    Ctot = cl_total**2/cl_fg
    norm_lens_fg = pytempura.norm_lens.qtt(
        mlmax, K_lmin, K_lmax, cl_total,
        Ctot, gtype='')[0]
    R_s_lens_fg = pytempura.get_cross(
        'SRC','TT', ucls, {"TT" : Ctot},
        K_lmin, K_lmax, k_ellmax = mlmax,
        profile = profile)

    N0 = norm_s**2/norm_fg
    N0_lens = norm_lens**2 / norm_lens_fg
    N0_tri = (N0 + R_s_lens**2 * norm_s**2 * N0_lens
              - 2 * R_s_lens * norm_s**2 * norm_lens * R_s_lens_fg)
    N0_tri /= (1 - norm_s*norm_lens*R_s_lens**2)**2
    N0_tri /= profile**2
    return N0_tri

def trispectrum_N0_psh(cl_total, cl_fg, norm_s,
                      ucls, K_lmin, K_lmax, mlmax,
                       profile, norm_ps, R_s_ps):
    norm_ps = recon_stuff["norm_ps"]
    R_s_ps = recon_stuff["R_s_ps"]
    norm_ps_fg = pytempura.get_norms(
        ['src'], recon_stuff['ucls'], {'TT':Ctot},
        K_lmin, K_lmax,
        k_ellmax=mlmax)['src']
    R_s_ps_fg = (1./pytempura.get_norms(
        ['src'], recon_stuff['ucls'], {'TT':Ctot},
        K_lmin, K_lmax, k_ellmax=mlmax,
        profile = profile**0.5)['src'])
    R_s_ps_fg[0] = 0.

    N0 = norm_s**2/norm_fg
    N0_ps = norm_ps**2 / norm_ps_fg
    N0_tri = (N0 + R_s_ps**2 * norm_s**2 * N0_ps
              - 2 * R_s_ps * norm_s**2 * norm_ps * R_s_ps_fg)
    N0_tri /= (1 - norm_s*norm_ps*R_s_ps**2)**2
    N0_tri /= profile**2
    return N0_tri
    

    
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

    if recon_config['mlmax'] is None:
        recon_config['mlmax'] = prep_map_config["lmax_out"]
    if recon_config['freqs'] is None:
        recon_config['freqs'] = prep_map_config['freqs']

    """
    has_masked = ((prep_map_config["halo_mask_fgs"])
                  or (prep_map_config["cib_flux_cut"] is not None))
    has_modelsub = (prep_map_config["fg_model_alms"] is not None)
    """
    has_masked, has_modelsub = False,False
        
    if rank==0:
        print("has_masked:",has_masked)
        print("has_modelsub:",has_modelsub)
    map_dir = opj(args.output_dir, "prep_map")

    #make output subdirectory for reconstruction stuff
    out_dir = opj(args.output_dir, "ksz2_fg_terms_%s"%args.tag)
    safe_mkdir(out_dir)
    if recon_config['mlmax'] is None:
        recon_config['mlmax'] = prep_map_config["lmax_out"]
    if recon_config['freqs'] is None:
        recon_config['freqs'] = prep_map_config['freqs']
    #save args to output dir
    arg_file = opj(out_dir,'recon_config.yml')
    if rank==0:
        with open(arg_file,'w') as f:
            yaml.dump(recon_config, f)
    
    #print("doing setup")
    #shape, wcs, norm_stuff, ucls, tcls, filter_alms, qfunc, qfunc_bh, _ = setup_recon(
    #   prep_map_config, recon_config)
    #recon_stuff = setup_recon(
    #    prep_map_config, recon_config)
    #filter_alms = recon_stuff['filter_alms']
    #profile = recon_stuff['profile']

    #Get kappa alms
    print("reading kappa alms")
    if recon_config['sim_name']=="websky":
        kap_alm_file = opj(recon_config['websky_dir'],
                           'kappa_alm_lmax6000.fits'
                           )
    elif recon_config['sim_name']=='sehgal':
        kap_alm_file = opj(recon_config['sehgal_dir'],
                           'healpix_4096_KappaeffLSStoCMBfullsky_almlmax6000.fits'
                           )
    else:
        raise ValueError("sim_name should be websky or sehgal")
    kappa_alms = hp.fitsfunc.read_alm(kap_alm_file)

    kappa_alms = utils.change_alm_lmax(kappa_alms, recon_config['mlmax'])
    #convert to phi
    Ls = np.arange(recon_config['mlmax']+1)
    phi_alms = curvedsky.almxfl(kappa_alms, 1/(Ls*(Ls+1)/2))
    
    """
    #cl_kk = curvedsky.alm2cl(kappa_alms)
    #print("cl_kk.shape:",cl_kk.shape)
    """


            
    for freq in args.freqs:
        print("freq: %s"%freq)
        #Read alms for reconstruction
        if recon_config["sim_name"]=="websky":
            if recon_config["cmb_name"] is None:
                recon_config['cmb_name']="cmb_orig"
        else:
            if recon_config["cmb_name"] is None:
                recon_config['cmb_name']="cmb_orig"
        if args.use_cltot_file:
            f = opj(map_dir, recon_config["cmb_name"],
                    "Cltot_data.npy")
            print("using Cl^tot from %s"%f)
            cltot_data = np.load(f)
        else:
            cltot_data = None

        if cltot_data is not None:
            print(cltot_data.dtype.names)
            if freq in cltot_data.dtype.names:
                recon_config["Cl_tot"] = cltot_data[freq]
            else:
                print("no Cl_tot for freq %s, will use other options"%freq)
                recon_config["Cl_tot"] = None

        recon_stuff = setup_recon(
            prep_map_config, recon_config)
        filter_alms = recon_stuff['filter_alms']
        profile = recon_stuff['profile']

        if args.freqs is None:
            args.freqs = prep_map_config['freqs']

        if args.normalized:
            qfuncs = [("qe",recon_stuff['qfunc_normed']),
                      ("psh",recon_stuff['qfunc_psh_normed'])]
            if args.do_lh:
                qfuncs.append(("lh", recon_stuff["qfunc_lh_normed"]))
        else:
            qfuncs = [("qe",recon_stuff['qfunc_sf']),
                      ("psh",recon_stuff['qfunc_psh_sf'])]
            if args.do_lh:
                qfuncs.append(("lh", recon_stuff["qfunc_lh_sf"]))
        
        print(recon_stuff["norm_s"])
        deltas_phi = curvedsky.almxfl(
                phi_alms, recon_stuff["norm_s"] * recon_stuff["R_s_tt"]
            )
        deltaK_phi = curvedsky.almxfl(
            deltas_phi, 1./profile)

        if not args.normalized:
            deltaK_phi = curvedsky.almxfl(
                deltaK_phi, 1./recon_stuff["norm_K"]
                )
        cl_phiphi = curvedsky.alm2cl(deltaK_phi)

        
        cmb_name = recon_config['cmb_name']
        outputs = OrderedDict()
        outputs['cl_KphiKphi'] = cl_phiphi
        #outputs['cl_kk'] = cl_kk

        for key in ["N0_K_normed", "N0_K_nonorm",
                    "N0_K_lh_normed", "N0_K_lh_normed",
                    "N0_K_psh_normed", "N0_K_psh_nonorm"]:
            outputs[key] = recon_stuff[key]
        
        fg_alms = hp.fitsfunc.read_alm(opj(map_dir, cmb_name,
                                       "fg_nonoise_alms_%s.fits"%freq))
        fg_alms = utils.change_alm_lmax(fg_alms, recon_config['mlmax'])
        cl_fg = get_cl_fg_smooth(fg_alms)
        fg_alms_filtered = recon_stuff['filter_alms'](fg_alms)

        K_lmin,K_lmax,mlmax = (recon_config['K_lmin'],
                                       recon_config['K_lmax'],
                                       recon_config['mlmax'])

        jobs = []
        for (estimator_name, qfunc) in qfuncs:
                jobs.append((fg_alms_filtered, cl_fg, estimator_name, qfunc))
        njobs = len(jobs)
        print("%d jobs"%len(jobs))

        for i,job in enumerate(jobs):
            if rank>=njobs:
                continue
            if i%size != rank:
                continue
            fg_alm_filt_use, cl_fg_use, estimator_name, qfunc = job
                
            label = estimator_name
            print("estimator: %s"%(estimator_name))
            s_fg_fg = qfunc(fg_alm_filt_use, fg_alm_filt_use)

            #Do trispectrum
            cl_raw = curvedsky.alm2cl(s_fg_fg, s_fg_fg)
            #N0 is (A^s)^2 / A_fg (see eqn. 9 of 1310.7023
            #for the normal estimator, and a bit more complex for
            #the bias hardened case
            Ctot = recon_stuff['cl_total']**2 / cl_fg_use
            norm_fg = pytempura.get_norms(
                ['src'], recon_stuff['ucls'], {'TT' : Ctot},
                K_lmin, K_lmax, k_ellmax=mlmax,
                profile=profile)['src']

            L = np.arange(mlmax+1)
            norm_s = recon_stuff["norm_s"]
            if estimator_name == "qe":
                N0_tri = norm_s**2 / norm_fg

            elif estimator_name == "lh":
                norm_lens = recon_stuff["norm_lens"]
                R_s_lens = recon_stuff["R_s_tt"]
                norm_lens_fg = pytempura.norm_lens.qtt(
                    mlmax, K_lmin, K_lmax, recon_stuff['ucls']['TT'],
                    Ctot, gtype='')[0]
                R_s_lens_fg = pytempura.get_cross(
                    'SRC','TT', recon_stuff['ucls'], {"TT" : Ctot},
                    K_lmin, K_lmax, k_ellmax = mlmax,
                    profile = profile)

                N0 = norm_s**2/norm_fg
                N0_lens = norm_lens**2 / norm_lens_fg
                N0_tri = (N0 + R_s_lens**2 * norm_s**2 * N0_lens
                          - 2 * R_s_lens * norm_s**2 * norm_lens * R_s_lens_fg)
                N0_tri /= (1 - norm_s*norm_lens*R_s_lens**2)**2

            elif estimator_name == "psh":
                norm_ps = recon_stuff["norm_ps"]
                R_s_ps = recon_stuff["R_s_ps"]
                norm_ps_fg = pytempura.get_norms(
                    ['src'], recon_stuff['ucls'], {'TT':Ctot},
                    K_lmin, K_lmax,
                    k_ellmax=mlmax)['src']
                R_s_ps_fg = (1./pytempura.get_norms(
	            ['src'], recon_stuff['ucls'], {'TT':Ctot},
                    K_lmin, K_lmax, k_ellmax=mlmax,
                    profile = profile**0.5)['src'])
                R_s_ps_fg[0] = 0.

                N0 = norm_s**2/norm_fg
                N0_ps = norm_ps**2 / norm_ps_fg
                N0_tri = (N0 + R_s_ps**2 * norm_s**2 * N0_ps
                          - 2 * R_s_ps * norm_s**2 * norm_ps * R_s_ps_fg)
                N0_tri /= (1 - norm_s*norm_ps*R_s_ps**2)**2

            #convert from s to K
            N0_tri /= profile**2
            if not args.normalized:
                N0_tri /= norm_K**2
                
            outputs['trispectrum_'+label] = cl_raw - N0_tri
            outputs['trispectrum_N0_'+label] = N0_tri

        if rank==0:
            outputs_all = outputs
            for i in range(1,size):
                o = comm.recv(source=i)
                outputs_all.update(o)
            """
            try:
                assert len(outputs_all) == njobs+1 #(+1 because we also add cl_kk)
            except AssertionError:
                print("len(outputs_all)=",len(outputs_all))
                print("njobs=",njobs)
                raise(e)
            """
            output_data = np.zeros((mlmax+1),
                                   dtype=[(k,float) for k in outputs_all.keys()])
            for k,v in outputs_all.items():
                output_data[k] = v
            output_file = opj(out_dir, 'fg_terms_%s.npy'%freq)
            np.save(output_file, output_data)

        else:
            comm.send(outputs, dest=0)
            
            
        """

        output_data = np.zeros((mlmax+1),
                               dtype=[(k,float) for k in outputs.keys()])
        for k,v in outputs.items():
            output_data[k] = v
        output_file = opj(out_dir, 'fg_terms_%s.npy'%freq)
        np.save(output_file, output_data)
        """

if __name__=="__main__":
    main()
        
    
