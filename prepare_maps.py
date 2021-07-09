#Prepare websky alms:
#- get lensed cmb
#- select and add included foregrounds
#- inpaint clusters based on halo positions
#- save beam-deconvolved sky map and ivar map.
from __future__ import print_function
import os,sys
from os.path import join as opj
from orphics import maps,io,cosmology,stats,pixcov
from pixell import enmap,curvedsky,utils,enplot
import numpy as np
import healpy as hp
import argparse
import falafel.utils as futils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from orphics import maps
from websky_model import WebSky
import argparse
import errno
import yaml
import shutil
import time
from datetime import timedelta
from collections import namedtuple

def get_disable_mpi():
    try:
        disable_mpi_env = os.environ['DISABLE_MPI']
        disable_mpi = True if disable_mpi_env.lower().strip() == "true" else False
    except:
        disable_mpi = False
    return disable_mpi

disable_mpi = get_disable_mpi()
if not disable_mpi:
    from mpi4py import MPI

def safe_mkdir(d):
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise(e)

defaults_file = opj(os.path.dirname(__file__),
                    "defaults.yaml")
with open(defaults_file,'rb') as f:
    DEFAULTS=yaml.load(f)
        
def get_config(section='prepare_map'):
    parser = argparse.ArgumentParser(description='Prepare websky map')
    #only required arg is output_dir
    parser.add_argument("output_dir", type=str)
    #can also add a config file
    parser.add_argument("-c", "--config_file", type=str, default=None)
    #and otherwise variables are set to defaults,
    #or overwritten on command line
    defaults = {}
    for key,val in DEFAULTS[section].items():
        nargs=None
        if val=="iNone":
            t,val = int, None
        elif val=="fNone":
            t,val = float, None
        elif val=="liNone":
            t,val,nargs = int,None,'*'
        elif isinstance(val, list):
            t = type(val[0])
            nargs='*'
        else:
            t = type(val)
        defaults[key] = val
        print(key, val, t)
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
        config.update(config_from_file[section])
        
    config['output_dir'] = output_dir

    for key,val in args_dict.items():
        print(key,val)
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

    #also return the config from the file,
    #and the defaults
    return config_namespace, config_from_file, dict(DEFAULTS)

from solenspipe import weighted_bin1D
bin_edges = np.linspace(0,3000,100).astype(int)
binner = weighted_bin1D(bin_edges)
def bin_cl(cl):
    ix,weights = np.zeros_like(cl), np.ones_like(cl)
    return binner.bin(ix, cl, weights)


config = futils.config
def get_cmb_alm_unlensed(i,iset,path=config['signal_path']):
    sstr = str(iset).zfill(2)
    istr = str(i).zfill(5)
    fname = path + "fullskyUnlensedCMB_alm_set%s_%s.fits" % (sstr,istr)
    return hp.read_alm(fname,hdu=(1,2,3))

def get_cmb_seeds(args):
    #sort out cmb seeds
    if args.cmb_seed_start is not None:
        assert args.cmb_seed_end is not None
        args.cmb_seeds = list(range(
            args.cmb_seed_start, args.cmb_seed_end+1
            ))
    return args.cmb_seeds

def main():
    
    config, config_from_file, defaults = get_config()


    
    if not disable_mpi:
        comm = MPI.COMM_WORLD
        rank,size = comm.Get_rank(), comm.Get_size()
    else:
        rank,size = 0,1

    if config.do_cmb:
        config.cmb_seeds = get_cmb_seeds(config)
        if rank==0:
            print("cmb seeds:",config.cmb_seeds)
    else:
        config.cmb_seeds = [None]
        
    theory = cosmology.default_theory()

    safe_mkdir(config.output_dir)
    output_dir = opj(config.output_dir, "prep_map")
    safe_mkdir(output_dir)
    if rank==0:
        #save config to output dir
        #save the prepare_map_section
        config_file = opj(output_dir,'prepare_map_config.yml')
        with open(config_file,'w') as f:
            yaml.dump(vars(config), f)
        #and the full config
        with open(opj(config.output_dir, "config_from_file.yml"),'w') as f:
            yaml.dump(config_from_file, f)
        with open(opj(config.output_dir, "defaults.yml"),'w') as f:
            yaml.dump(defaults, f)

    do_fgs=False
    if (config.do_tsz or config.do_cib or config.do_ksz):
        do_fgs=True
        print('initializing websky')
        websky = WebSky(
            directory_path = config.websky_directory,
            websky_version = config.websky_version)
    if config.halo_mask_fgs:
        halo_mask_fgs_config = {k:vars(config)[k] for k in
                            ['log10_m_min','zmax',
                             'mask_radius','num_halo']}
    else:
        halo_mask_fgs_config = None

    if config.do_reion_ksz:        
        #Read in reionization ksz alms
        kszr_alms = hp.fitsfunc.read_alm(
            config.ksz_reion_alms)
        kszr_alms = futils.change_alm_lmax(
            kszr_alms, config.mlmax)
    else:
        kszr_alms = None
        
    #Loop through cmb seeds
    for icmb,cmb_seed in enumerate(config.cmb_seeds):
        if icmb%size != rank:
            continue
        cmb_seed = config.cmb_seeds[icmb]
        if cmb_seed is not None:
            cmb_unlensed_alms = get_cmb_alm_unlensed(
                config.cmb_seeds[icmb], 0)[0]
            cmb_unlensed_alms = futils.change_alm_lmax(
                cmb_unlensed_alms, config.mlmax)
            print("rank %d doing cmb seed %d"%(rank, cmb_seed))
            cmb_seed_output_dir = opj(output_dir, "cmb_%d"%cmb_seed)
        else:
            cmb_unlensed_alms = None
            cmb_seed_output_dir = opj(output_dir, "no_cmb")
        safe_mkdir(cmb_seed_output_dir)

        cmb_alms=None
        if not do_fgs:
            cmb_alms = futils.get_cmb_alm(
                config.cmb_seeds[icmb], 0)[0]
            cmb_alms = futils.change_alm_lmax(
                cmb_alms, config.mlmax)
            print("cmb_alms.shape",cmb_alms.shape)

        for ifreq,freq in enumerate(config.freqs):
            #Get sky alms
            print('freq = %s'%freq)
            print('getting alms')

            sky_alms = np.zeros(hp.Alm.getsize(config.mlmax), dtype=np.complex128)
            sky_unlensed_alms = sky_alms.copy()
            if do_fgs:
                fg_sky_dict = websky.get_sky(
                    cmb = config.do_cmb, freq = freq, cib = config.do_cib,
                    tsz = config.do_tsz, ksz = config.do_ksz,
                    cmb_unlensed_alms = cmb_unlensed_alms,
                    cmb_alms = cmb_alms, cib_flux_cut = config.cib_flux_cut,
                    halo_mask_fgs_config = halo_mask_fgs_config,
                    lmax=config.mlmax,
                    udgrade_fill_factor=config.udgrade_fill_factor)
                if config.do_cmb:
                    cmb_alms = sky_dict['cmb_alms']
                else:
                    cmb_alms = None
                fg_alms = fg_sky_dict['fg_alms']
                if 'fg_masked_alms' in fg_sky_dict:
                    fg_masked_alms = fg_sky_dict['fg_masked_alms']
                else:
                    fg_masked_alms = None

            if cmb_alms is not None:
                sky_alms += cmb_alms
            if cmb_alms is not None:
                sky_unlensed_alms += cmb_unlensed_alms
            sky_masked_alms = None
            if do_fgs:
                sky_alms += fg_sky_dict['fg_alms']
                sky_unlensed_alms += fg_sky_dict['fg_alms']
                if fg_masked_alms is not None:
                    sky_masked_alms  = np.zeros(hp.Alm.getsize(config.mlmax), dtype=np.complex128)
                    if cmb_alms is not None:
                        sky_masked_alms += cmb_alms
                    sky_masked_alms += fg_masked_alms
            else:
                fg_alms, fg_masked_alms = None, None

            print('got alms')

            #Probably an idea to make some C(l) plots to make sure
            #nothing is too crazy
            ell_plot, cl_sky_raw = bin_cl(curvedsky.alm2cl(sky_alms))
            fig,ax=plt.subplots()
            
            ax.plot(ell_plot, ell_plot*(ell_plot+1)*cl_sky_raw,
                    label='sky input')
            if config.do_cmb:
                _,cl_cmb = bin_cl(curvedsky.alm2cl(cmb_alms))
                ax.plot(ell_plot, ell_plot*(ell_plot+1)*cl_cmb,
                        label='cmb input')
            if config.do_reion_ksz:
                _,cl_kszr = bin_cl(curvedsky.alm2cl(kszr_alms))
                ax.plot(ell_plot, ell_plot*(ell_plot+1)*cl_kszr,
                        label='high-z ksz')
            if do_fgs:
                _,cl_fg = bin_cl(curvedsky.alm2cl(fg_alms))
                ax.plot(ell_plot, ell_plot*(ell_plot+1)*cl_fg,
                        label='fgs input')

            #save some stuff
            if config.save_raw_alms:
                hp.fitsfunc.write_alm(opj(cmb_seed_output_dir,
                                          "sky_raw_alms_%s.fits"%freq),
                                      sky_alms, overwrite=True
                                      )
                if config.do_cmb:
                    hp.fitsfunc.write_alm(opj(cmb_seed_output_dir,
                                              "cmb_raw_alms_%s.fits"%freq),
                                          cmb_alms, overwrite=True
                    )
                if config.do_reion_ksz:
                    hp.fitsfunc.write_alm(opj(cmb_seed_output_dir,
                                              "kszr_raw_alms_%s.fits"%freq),
                                          kszr_alms, overwrite=True)
                if do_fgs:
                    hp.fitsfunc.write_alm(opj(cmb_seed_output_dir,
                                              "fg_raw_alms_%s.fits"%freq),
                                          fg_alms, overwrite=True
                                          )
                if fg_masked_alms is not None:
                    #save in case we want to use for adding gaussian
                    #power downstream
                    hp.fitsfunc.write_alm(
                        opj(cmb_seed_output_dir,
                            "fg_masked_raw_alms_%s.fits"%freq),
                        fg_masked_alms, overwrite=True
                                  )

            if not config.disable_noise:
                #Noise and beam
                beam_fn = lambda x: maps.gauss_beam(x, config.beam_fwhm)
                unbeam_fn = lambda x: 1./maps.gauss_beam(x, config.beam_fwhm)
                cmb_alms_beamed = curvedsky.almxfl(cmb_alms, beam_fn)
                sky_alms_beamed = curvedsky.almxfl(sky_alms, beam_fn)
                kszr_alms_beamed = curvedsky.almxfl(kszr_alms, beam_fn)
                
                #make map
                res = config.res*utils.arcmin
                shape,wcs = enmap.fullsky_geometry(res=res,
                                             proj='car')
                sky_map = enmap.zeros(shape=shape, wcs=wcs)
                curvedsky.alm2map(sky_alms_beamed, sky_map)
                cmb_map = enmap.zeros(shape=shape, wcs=wcs)
                curvedsky.alm2map(cmb_alms_beamed, cmb_map)
                kszr_map = enmap.zeros(shape=shape, wcs=wcs)
                curvedsky.alm2map(kszr_alms_beamed, kszr_map)

                if cmb_seed is not None:
                    if config.noise_seed is None:
                        print("setting noise seed equal to cmb_seed")
                        print("if this is not what you want, explicitly")
                        print("set noise_seed in the config file.")
                        config.noise_seed = cmb_seed

                noise_map = maps.white_noise(
                    shape, wcs, config.noise_sigma, seed=config.noise_seed)
                sky_map += noise_map
                cmb_map += noise_map
                kszr_map += noise_map
                #deconcolve and save alms
                sky_alms_beamed = curvedsky.map2alm(
                    sky_map, lmax=config.mlmax)
                sky_alms = curvedsky.almxfl(sky_alms_beamed, unbeam_fn)
                cmb_alms_beamed = curvedsky.map2alm(
                            cmb_map, lmax=config.mlmax)
                cmb_alms = curvedsky.almxfl(cmb_alms_beamed, unbeam_fn)
                kszr_alms_beamed = curvedsky.map2alm(
                            kszr_map, lmax=config.mlmax)
                kszr_alms = curvedsky.almxfl(kszr_alms_beamed, unbeam_fn)

                if fg_masked_alms is not None:
                    sky_masked_map = enmap.zeros(shape=shape, wcs=wcs)
                    sky_masked_alms_beamed = curvedsky.almxfl(sky_masked_alms, beam_fn)
                    curvedsky.alm2map(sky_masked_alms_beamed, fg_masked_map)
                    fg_masked_map += noise_map
                    #back to alms and save
                    sky_masked_alms_beamed = curvedsky.map2alm(
                        sky_map_masked, lmax=config.mlmax)
                    #deconvolve beam
                    sky_masked_alms = curvedsky.almxfl(sky_masked_alms_beamed,
                                                       unbeam_fn)
            else:
                print("adding noise disabled")
                    
            hp.fitsfunc.write_alm(
                opj(cmb_seed_output_dir,
                    "sky_alms_%s.fits"%freq),
                sky_alms, overwrite=True
            )
            if config.do_cmb:
                hp.fitsfunc.write_alm(
                    opj(cmb_seed_output_dir,
                        "cmb_alms_%s.fits"%freq),
                    cmb_alms, overwrite=True
                )
            hp.fitsfunc.write_alm(
                opj(cmb_seed_output_dir,
                    "sky_unlensed_alms_%s.fits"%freq),
                sky_unlensed_alms, overwrite=True
            )
            if config.do_reion_ksz:
                hp.fitsfunc.write_alm(
                    opj(cmb_seed_output_dir,
                        "kszr_alms_%s.fits"%freq),
                    kszr_alms, overwrite=True)

            if fg_alms is not None:
                hp.fitsfunc.write_alm(
                    opj(cmb_seed_output_dir,
                        "fg_alms_%s.fits"%freq),
                    fg_alms, overwrite=True
                )
            
            if fg_masked_alms is not None:
                #save in case we want to use for adding gaussian
                #power downstream
                hp.fitsfunc.write_alm(
                    opj(cmb_seed_output_dir,
                        "sky_masked_alms_%s.fits"%freq),
                    sky_masked_alms, overwrite=True
                )
                hp.fitsfunc.write_alm(
                    opj(cmb_seed_output_dir,
                        "fg_masked_alms_%s.fits"%freq),
                    sky_masked_alms, overwrite=True
                )
                _,cl_fg_mask = bin_cl(
                    curvedsky.alm2cl(sky_masked_alms))
                ax.plot(ell_plot,
                        ell_plot*(ell_plot+1)*cl_fg_mask,
                        label='sky fg masked')

            ax.legend()
            ax.set_yscale('log')
            ax.set_xlabel(r"$l$")
            ax.set_ylabel(r"$l(l+1)C_l$")
            fig.tight_layout()
            fig.savefig(opj(cmb_seed_output_dir,
                            "cl_tt_%s.png"%freq))
            
if __name__=="__main__":
    main()
