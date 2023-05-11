#Get meanfield and N^0
import pickle
import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
from falafel import utils, qe
import pytempura
import solenspipe
from solenspipe.bias import mcrdn0, mcn1
from pixell import lensing, curvedsky, enmap
from pixell import utils as putils
from os.path import join as opj
import argparse
import yaml
from websky_model import WebSky
from collections import OrderedDict
from prepare_maps import safe_mkdir, get_disable_mpi
from orphics import maps
from do_reconstruction import setup_recon, get_alm_files, dummy_teb, get_config, get_cmb_seeds
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

disable_mpi = get_disable_mpi()
if not disable_mpi:
    from mpi4py import MPI


def get_cl_fg_func(alms):
    cl = curvedsky.alm2cl(alms)
    l = np.arange(len(cl))
    d = l*(l+1)*cl
    #smooth with savgol
    d_smooth = savgol_filter(d, 5, 2)
    #if there's still negative values, set them
    #to zero
    d_smooth[d_smooth<0.] = 0.
    d_smooth_interp = interp1d(
        l, d_smooth, bounds_error=False,
    )
    def cl_fg_func(l):
        return np.where(
            l>0,d_smooth_interp(l)/l/(l+1),0.)
    return cl_fg_func
    
def main():

    args = get_config(section="bias",
                      description="estimate biases")
    print("args:",args)
    bias_config = vars(args)

    #get config from prep map stage
    with open(opj(args.output_dir, 'prep_map', 'prepare_map_config.yml'),'r') as f:
        prep_map_config = yaml.load(f)
    has_masked = ((prep_map_config["halo_mask_fgs"])
                  or (prep_map_config["cib_flux_cut"] is not None))
    map_dir = opj(args.output_dir, "prep_map")
    recon_dir = opj(args.output_dir, "recon_%s"%args.recon_tag)
    with open(opj(recon_dir, "recon_config.yml")) as f:
        recon_config  = yaml.load(f)

    if not disable_mpi:
        comm = MPI.COMM_WORLD
        rank,size = comm.Get_rank(), comm.Get_size()
    else:
        rank,size = 0,1
    
    #make subdirectory for bias output
    output_dir = opj(args.output_dir, "bias_%s"%args.tag)
    safe_mkdir(output_dir)

    if args.kappa_lmax is not None:
        recon_config['kappa_lmax']=args.kappa_lmax
    
    #get reconstruction setup stuff
    #_, _, norm_stuff, ucls, tcls, filter_alms, qfunc, qfunc_bh, get_sim_alm = setup_recon(
    #        prep_map_config, recon_config)
    recon_setup = setup_recon(
        prep_map_config, recon_config, sim_start_ind=args.sim_start)
    filter_alms = recon_setup['filter_alms']
    qfunc = recon_setup['qfunc_normed']

    sim_range_tag = "%s-%s"%(str(args.sim_start).zfill(3),
                             str(args.sim_start+args.nsim_n0).zfill(3)
                             )
    
    ells = np.arange(recon_config['mlmax']+1)

    powfunc = lambda x,y: curvedsky.alm2cl(x,y)

    cmb_seeds = get_cmb_seeds(args)
    if cmb_seeds is None:
        cmb_seeds = prep_map_config['cmb_seeds']
    if rank==0:
        print("cmb_seeds:",cmb_seeds)
        
    if cmb_seeds is not None:
        num_cmbs = len(cmb_seeds)
        use_websky_cmb=False
    else:
        use_websky_cmb=True
        num_cmbs = 1

    if args.freqs is None:
        args.freqs = prep_map_config['freqs']

    if args.skip_tags is None:
        args.skip_tags = []
    if not args.do_unlensed:
        args.skip_tags.append("sky_unlensed")

    #Loop through cmb seeds
    for icmb,cmb_seed in enumerate(cmb_seeds):
        bias_outputs={}
        cmb_seed = cmb_seeds[icmb]
        if cmb_seed is None:
            map_dir_this_seed = opj(map_dir, "no_cmb")
            recon_dir_this_seed = opj(map_dir, "no_cmb")
            bias_dir_this_seed = opj(output_dir, "no_cmb")
            
        else:
            map_dir_this_seed = opj(map_dir, "cmb_%d"%cmb_seed)
            recon_dir_this_seed = opj(recon_dir, "cmb_%d"%cmb_seed)
            bias_dir_this_seed = opj(output_dir, "cmb_%d"%cmb_seed)
            if rank==0:
                print("doing cmb_seed %d"%cmb_seed)

        safe_mkdir(bias_dir_this_seed)
        
        if args.do_n0:
            for freq in args.freqs:
                alm_files = get_alm_files(
                    prep_map_config, map_dir_this_seed, freq,
                    get_masked=has_masked, get_unlensed=False)
                bias_outputs[freq] = OrderedDict()
                for tag,alm_file in alm_files.items():
                    if args.skip_tags is not None:
                        if tag in args.skip_tags:
                            if rank==0:
                                print("skipping tag %s"%tag)
                            continue
                    if rank==0:
                        print("doing tag:",tag)
                    if tag=="sky":
                        add_cmb=prep_map_config['do_cmb']
                        lensed_cmb=True
                        add_kszr=prep_map_config["do_reion_ksz"]
                    elif tag=="sky_masked":
                        add_cmb=prep_map_config['do_cmb']
                        lensed_cmb=True
                        add_kszr=prep_map_config["do_reion_ksz"]
                    elif tag=="sky_unlensed":
                        add_cmb=prep_map_config['do_cmb']
                        lensed_cmb=False
                        add_kszr=prep_map_config["do_reion_ksz"]
                    elif tag=="cmb":
                        add_cmb=True
                        add_kszr=False
                        lensed_cmb=True
                    elif tag=="kszr":
                        add_cmb=False
                        lensed_cmb=True
                        add_kszr=True
                    else:
                        raise ValueError("Need to sepcify add_cmb option for tag %s"%tag)

                    #set up the get_sim_alm
                    #function for rdn0. There are a
                    #couple of options we may want to
                    #set here - what sim realization to
                    #start from, and whether to include
                    #Gaussian foreground power
                    if args.add_gaussian_fg:
                        if tag=="sky":
                            fg_alms_file = opj(
                                map_dir_this_seed,
                                "fg_raw_alms_%s.fits"%freq
                            )
                            print('reading gaussian fg from:')
                            print(fg_alms_file)
                            fg_alms = hp.fitsfunc.read_alm(
                                fg_alms_file)
                            #get C(l)
                            cl_fg_func = get_cl_fg_func(fg_alms)
                            np.save(
                                opj(bias_dir_this_seed, 'cl_fg_%s.npy'%freq),
                                cl_fg_func(ells)
                                )
                        elif tag=="sky_masked":
                            fg_alms_file = opj(
                                map_dir_this_seed,
                                "fg_masked_raw_alms_%s.fits"%freq
                                )
                            fg_alms = hp.fitsfunc.read_alm(
                                fg_alms_file)
                            #get C(l)
                            cl_fg_func = get_cl_fg_func(fg_alms)
                            np.save(
                                opj(bias_dir_this_seed, 'cl_fg_masked_%s.npy'%freq),
                                cl_fg_func(ells)
                                )
                        else:
                            cl_fg_func=None
                    else:
                        cl_fg_func=None

                    def get_sim_alm(seed):
                        return recon_setup['get_sim_alm'](
                            seed, cl_fg_func=cl_fg_func,
                            add_cmb=add_cmb,
                            lensed_cmb=lensed_cmb,
                            add_kszr=add_kszr)

                    data_alm = hp.fitsfunc.read_alm(alm_file)
                    #filter
                    data_alm_filtered = filter_alms(data_alm)

                    bias_outputs[freq][tag] = {}

                    if args.do_qe:
                        #get rdn0 - normal qe
                        rdn0,mcn0 = mcrdn0(
                                0, get_sim_alm, powfunc, args.nsim_n0, 
                                qfunc, qfunc2=None, Xdat=data_alm_filtered, use_mpi=True)
                        f = opj(bias_dir_this_seed,
                                "rdn0_%s_%s_%s.npz"%(
                                    freq, tag, sim_range_tag
                                    )
                                )
                        np.savez(f, rdn0=rdn0, mcn0=mcn0)

                    #lensing hardened
                    if args.do_lh:
                        qfunc_lh = recon_setup['qfunc_lh_normed']
                        rdn0_lh,mcn0_lh = mcrdn0(
                                0, get_sim_alm, powfunc, args.nsim_n0, 
                                qfunc_lh, qfunc2=None, Xdat=data_alm_filtered, use_mpi=True)
                        f = opj(bias_dir_this_seed,
                                "rdn0_%s_%s_%s.npz"%(
                                    freq, tag+"_lh", sim_range_tag
                                    )
                                )
                        np.savez(f, rdn0=rdn0_lh, mcn0=mcn0_lh)

                    if args.do_psh:
                        qfunc_psh = recon_setup['qfunc_psh_normed']
                        rdn0_psh,mcn0_psh = mcrdn0(
                                0, get_sim_alm, powfunc, args.nsim_n0, 
                                qfunc_psh, qfunc2=None, Xdat=data_alm_filtered, use_mpi=True)
                        f = opj(bias_dir_this_seed,
                                "rdn0_%s_%s_%s.npz"%(
                                    freq, tag+"_psh", sim_range_tag
                                    )
                                )
                        np.savez(f, rdn0=rdn0_psh, mcn0=mcn0_psh)
                        
                
        if args.do_n1:
            n1_sim_range_tag = "%s-%s"%(str(0).zfill(3),
                             str(args.nsims_n1).zfill(3)
                             )
            def get_sim_alm(seed):
                return recon_setup['get_sim_alm'](
                    seed, cl_fg_func=None,
	            add_cmb=True,
                    lensed_cmb=True,
                    add_kszr=False)
            if args.do_qe:
                n1 = mcn1(0, get_sim_alm, powfunc, args.nsims_n1, 
                          qfunc, qfunc2=None)
                f = opj(bias_dir_this_seed,
                        "n1_%s.npz"%(
                            n1_sim_range_tag))
                np.save(f, n1)

            if args.do_lh:
                n1 = mcn1(0, get_sim_alm, powfunc, args.nsims_n1,
                          recon_setup['qfunc_lh_normed'], qfunc2=None)
                f = opj(bias_dir_this_seed,
                        "n1_lh_%s.npz"%(
                            n1_sim_range_tag))
                np.save(f, n1)

            if args.do_lh:
                n1 = mcn1(0, get_sim_alm, powfunc, args.nsims_n1,
                          recon_setup['qfunc_psh_normed'], qfunc2=None)
                f = opj(bias_dir_this_seed,
                        "n1_psh_%s.npz"%(
                            n1_sim_range_tag))
                np.save(f, n1)

                
                
        bias_stuff_file = opj(bias_dir_this_seed, "bias_stuff.pkl")
        with open(bias_stuff_file,'wb') as f:
            pickle.dump(bias_outputs, f)

if __name__=="__main__":
    main()
