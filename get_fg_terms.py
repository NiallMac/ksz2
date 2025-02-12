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
from pixell import lensing, curvedsky, enmap, reproject
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
from ksz4.reconstruction import setup_recon, setup_asym_recon, setup_ABCD_recon, get_cl_smooth

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=CB_color_cycle)

WEBSKY_DIR="/global/cscratch1/sd/maccrann/cmb/websky/"

disable_mpi = get_disable_mpi()
if not disable_mpi:
    from mpi4py import MPI

    
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
    
def read_alm(alm_file, extra_weight_map=None, lmax=None):
    alm = hp.read_alm(alm_file)
    lmax_orig = hp.Alm.getlmax(len(alm))
    if extra_weight_map is not None:
        nside=hp.npix2nside(len(extra_weight_map))
        m = hp.alm2map(alm, nside=nside)*extra_weight_map
        alm = hp.map2alm(m, lmax=lmax_orig)
    if lmax is not None:
        alm = utils.change_alm_lmax(alm, lmax)
    return alm
    
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

    #check for a survey_mask_hpix
    #we may also want to apply an extra weight here
    #if not, the weight_map just becomes survey_mask_hpix
    if prep_map_config["survey_mask_hpix"] is not None:
        print("reading survey_mask_pix from %s"%prep_map_config["survey_mask_hpix"])
        weight_map = hp.read_map(prep_map_config["survey_mask_hpix"])
    else:
        weight_map = None
        
    if args.apply_extra_weight is not None:
        print("using weight map from %s"%args.apply_extra_weight)
        if "," in args.apply_extra_weight:
            extra_weight_filename, ext = (args.apply_extra_weight).split(",")
            extra_weight = enmap.read_map(extra_weight_filename)[0]
        else:
            extra_weight = enmap.read_map(args.apply_extra_weight)

        if args.extra_weight_power is not None:
            extra_weight = extra_weight**args.extra_weight_power
            
        extra_weight_hpix = reproject.map2healpix(extra_weight, 
                                                  nside=hp.npix2nside(len(weight_map)), 
                                                  method="spline")

        
        if weight_map is None:
            weight_map = extra_weight_hpix
        else:
            weight_map *= extra_weight_hpix
    else:
        extra_weight_hpix = None
        
    if weight_map is None:
        w2=1.
        w4=1.
    else:
        w2 = maps.wfactor(2, weight_map, equal_area=True)
        w4 = maps.wfactor(4, weight_map, equal_area=True)
        
    print("w2:",w2)
    print("w4:",w4)

    if recon_config['mlmax'] is None:
        recon_config['mlmax'] = prep_map_config["lmax_out"]
    if recon_config['freqs'] is None:
        recon_config['freqs'] = prep_map_config['freqs']

    #pixelisation - for websky and sehgal assume
    #this is healpix
    px = qe.pixelization(nside=recon_config['nside'])

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

    #Get rksz power
    rksz_alms=hp.fitsfunc.read_alm(
        recon_config['ksz_reion_alms'])
    rksz_alms = utils.change_alm_lmax(
            rksz_alms, recon_config["mlmax"])
    cl_rksz = get_cl_smooth(rksz_alms)
            
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

    nfreq = len(args.freqs)
    for ifreq,freq in enumerate(args.freqs):
        if args.parallelise_freqs:
            if rank>=nfreq:
                return
            if ifreq%size != rank:
                continue

        print("rank %d doing freq: %s"%(rank,freq))
        #Read alms for reconstruction
        cmb_name = recon_config['cmb_name']
        outputs = OrderedDict()

        #This may change but I'm allowing args.freqs
        #to be e.g. freqcoadd or tsz-sym as well as just
        #numbers. So check if freq is a number
        freq_is_number = True
        try:
            float(freq)
        except ValueError:
            freq_is_number = False
        if "_" in freq:
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
            (freq in ['ilc', 'hilc','hilc-tszd','hilc-cibd',
                      'hilc-tszandcibd','freqcoadd'])
            or (("_" not in freq) and freq.startswith("deproj"))
            or is_freq_diff):
            raise ValueError("not supporting this right now")
            fg_alm_file = opj(map_dir, cmb_name,
                              "fg_nonoise_alms_%s.fits"%freq)
            print(fg_alm_file)
            fg_alms = hp.fitsfunc.read_alm(opj(map_dir, cmb_name,
                                           "fg_nonoise_alms_%s.fits"%freq))
            fg_alms = utils.change_alm_lmax(fg_alms, recon_config['mlmax'])

        #The asymmetric (and symmetrized cases). Not here I've
        #hardcoded that we're using ilc in one of the legs, we
        #may want to generalize that. Read in a tuple of foreground
        #alms 
        elif freq.startswith("XY_"):
            raise ValueError("not supporting this right now")
            freqX = freq[3:].split("_")[0]
            freqY = freq[3:].split("_")[1]
            fg_alms_X = hp.fitsfunc.read_alm(
                opj(map_dir, cmb_name, "fg_nonoise_alms_%s.fits"%freqX)
                )
            fg_alms_X = utils.change_alm_lmax(fg_alms_X, recon_config['mlmax'])
            fg_alms_Y = hp.fitsfunc.read_alm(
                opj(map_dir, cmb_name, "fg_nonoise_alms_%s.fits"%(freqY))
                )
            fg_alms_Y = utils.change_alm_lmax(fg_alms_Y, recon_config['mlmax'])
            fg_alms = (fg_alms_X, fg_alms_Y)
            

        #elif freq == "hilc_hilc-tzd_hilc_hilc-cibd":
        elif len(freq.split("_"))==4:
            #this is the general ABCD case
            freqA = freq.split("_")[0]
            freqB = freq.split("_")[1]
            freqC = freq.split("_")[2]
            freqD = freq.split("_")[3]
            #freqA = "hilc"
            #freqB = "hilc-tszd"
            #freqC = "hilc"
            #freqD = "hilc-cibd"

            
            fg_alms_A = read_alm(opj(map_dir, cmb_name, "fg_nonoise_alms_%s.fits"%freqA),
                                 extra_weight_map=extra_weight_hpix,
                                 lmax=recon_config["mlmax"])
            fg_alms_B = read_alm(opj(map_dir, cmb_name, "fg_nonoise_alms_%s.fits"%freqB),
                                 extra_weight_map=extra_weight_hpix,
                                 lmax=recon_config["mlmax"])
            fg_alms_C = read_alm(opj(map_dir, cmb_name, "fg_nonoise_alms_%s.fits"%freqC),
                                 extra_weight_map=extra_weight_hpix,
                                 lmax=recon_config["mlmax"])
            fg_alms_D = read_alm(opj(map_dir, cmb_name, "fg_nonoise_alms_%s.fits"%freqD),
                                 extra_weight_map=extra_weight_hpix,
                                 lmax=recon_config["mlmax"])                        
            
            """
            fg_alms_A = utils.change_alm_lmax(
                hp.fitsfunc.read_alm(
                opj(map_dir, cmb_name, "fg_nonoise_alms_%s.fits"%freqA)
                ), recon_config["mlmax"]
            )
            fg_alms_B = utils.change_alm_lmax(
                hp.fitsfunc.read_alm(
                opj(map_dir, cmb_name, "fg_nonoise_alms_%s.fits"%freqB)
                ), recon_config["mlmax"]
            )
            fg_alms_C = utils.change_alm_lmax(
                hp.fitsfunc.read_alm(
                opj(map_dir, cmb_name, "fg_nonoise_alms_%s.fits"%freqC)
                ), recon_config["mlmax"]
            )
            fg_alms_D = utils.change_alm_lmax(
                hp.fitsfunc.read_alm(
                opj(map_dir, cmb_name, "fg_nonoise_alms_%s.fits"%freqD)
                ), recon_config["mlmax"]
            )
            """
            fg_alms = (fg_alms_A, fg_alms_B, fg_alms_C, fg_alms_D)

        else:
            raise ValueError("frequency %s not recognized"%freq)

        if isinstance(fg_alms, tuple):
            if len(fg_alms)==2:
                cl_fg_XX = get_cl_smooth(fg_alms[0])/w2
                cl_fg_YY = get_cl_smooth(fg_alms[1])/w2
                cl_fg_XY = get_cl_smooth(fg_alms[0], fg_alms[1])/w2
            elif len(fg_alms)==4:
                cl_fg_AA = get_cl_smooth(fg_alms[0])/w2
                cl_fg_BB = get_cl_smooth(fg_alms[1])/w2
                cl_fg_CC = get_cl_smooth(fg_alms[2])/w2
                cl_fg_DD = get_cl_smooth(fg_alms[3])/w2
                cl_fg_AB = get_cl_smooth(fg_alms[0], fg_alms[1])/w2
                cl_fg_AC = get_cl_smooth(fg_alms[0], fg_alms[2])/w2
                cl_fg_AD = get_cl_smooth(fg_alms[0], fg_alms[3])/w2
                cl_fg_BC = get_cl_smooth(fg_alms[1], fg_alms[2])/w2
                cl_fg_BD = get_cl_smooth(fg_alms[1], fg_alms[3])/w2
                cl_fg_CD = get_cl_smooth(fg_alms[2], fg_alms[3])/w2
        else:
            cl_fg = get_cl_smooth(fg_alms)/w2

        
        #Load total cl data
        f = opj(map_dir, recon_config["cmb_name"],
                "Cltot_data.npy")
        cltot_data = np.load(f)

        print("doing setup")
        if freq.startswith("XY_"): # in ["tszd-sym", "cibd-sym", "tszandcibd-sym"]:
            raise ValueError("not using this option now")
            assert isinstance(fg_alms, tuple)
            #The code for these asymmetric estimators is quite
            #different to what we had before. So for now just do
            #everything separately. But try and tidy things up
            #to make more uniform.
            f = opj(map_dir, recon_config["cmb_name"],
                    "Cltot_data.npy")
            cltot_data = np.load(f)

            cltot_X = cltot_data["Cltt_total_%s"%freqX]
            cltot_Y = cltot_data["Cltt_total_%s"%(freqY)]
            cltot_XY = cltot_data["Cltt_total_%s_%s"%(freqX, freqY)]
            if recon_config["add_rkszcl_to_filter"]:
                cltot_X = (cltot_X[:recon_config["mlmax"]+1] +
                           cl_rksz[:recon_config["mlmax"]+1])
                cltot_Y = (cltot_Y[:recon_config["mlmax"]+1] +
                           cl_rksz[:recon_config["mlmax"]+1])
                cltot_XY = (cltot_XY[:recon_config["mlmax"]+1] +
                            cl_rksz[:recon_config["mlmax"]+1])

            if not recon_config["include_fg_power_in_filter"]:
                cltot_X -= cltot_data["Clfg_%s"%freqX]
                cltot_Y -= cltot_data["Clfg_%s"%freqY]
                cltot_XY -= cltot_data["Clfg_%s_%s"%(freqX, freqY)]
                
            Nl_tt_X = cltot_data["Nltt_%s"%freqX][:recon_config["mlmax"]+1]
            if "Nlee_%s"%freqX in cltot_data.dtype.names:
                Nl_ee_X = cltot_data["Nlee_%s"%cltot_pol_label_X][:recon_config["mlmax"]+1]
                Nl_bb_X = cltot_data["Nlbb_%s"%cltot_pol_label_X][:recon_config["mlmax"]+1]
            else:
                print("no Nlee or Nlbb found in Cltot data file for freq: %s"%freq)
                print("using 2*Nltt")
                Nl_ee_X = 2*Nl_tt_X
                Nl_bb_X = 2*Nl_tt_X

            _,tcls_X = utils.get_theory_dicts(
                grad=True, nells={"TT":Nl_tt_X, "EE":Nl_ee_X, "BB":Nl_bb_X},
		lmax=recon_config["mlmax"])
            tcls_X["TT"] = cltot_X

            print("setting up XY estimator")
            lmin,lmax=recon_config["K_lmin"], recon_config["K_lmax"]
            recon_stuff = setup_asym_recon(
                px, lmin, lmax,
                recon_config["mlmax"],
                cl_rksz, cltot_X, cltot_Y[:recon_config["mlmax"]+1],
                cltot_XY[:recon_config["mlmax"]+1],
                do_lh=False, do_psh=False)

            profile = recon_stuff['profile']
            qfuncs = [("qe", recon_stuff["qfunc_K_XY"])]
            if args.do_psh:
                raise NotImplementedError(
                    "not implemented assymetric with bias hardening")
                qfuncs.append(
                    ("psh", recon_stuff["qfunc_K_psh"])
                    )
            if args.do_lh:
                raise NotImplementedError(
                    "not implemented assymetric with lensing hardening"
                    )
                qfuncs.append(
                    ("lh", recon_stuff["qfunc_K_lh"])
                    )

                """
                deltaK_phi = curvedsky.almxfl(
                        phi_alms, recon_stuff["norm_K_XY"] * recon_stuff["R_K_phi"]
                    )
                cl_phiphi = curvedsky.alm2cl(deltaK_phi)
                """

            cmb_name = recon_config['cmb_name']
            outputs = OrderedDict()
            outputs["profile"] = profile.copy()

            fg_alms_filtered_X = recon_stuff["filter_X"](fg_alms[0])
            fg_alms_filtered_Y = recon_stuff["filter_Y"](fg_alms[1])
            K_lmin,K_lmax,mlmax = (recon_config['K_lmin'],
                                   recon_config['K_lmax'],
                                   recon_config['mlmax'])
            
            outputs["N0_K"] = (recon_stuff["N0_XYXY_K"]).copy()
            outputs["norm_K_AB"] = recon_stuff["norm_K_AB"]
            outputs["norm_K_CD"] = recon_stuff["norm_K_CD"]
            
            jobs = [("qe", recon_stuff["qfunc_K_XY"],
                     recon_stuff["get_fg_trispectrum_N0_XYXY"])]
            if args.do_lh:
                outputs["N0_K_lh"] = (recon_stuff["N0_XYXY_K_lh"]).copy()
                jobs += [("psh", recon_stuff["qfunc_K_XY_lh"],
                          recon_stuff["get_fg_trispectrum_N0_XYXY_lh"])
                         ]
            if args.do_psh:
                outputs["N0_K_psh"] = (recon_stuff["N0_XYXY_K_psh"]).copy()
                jobs += [("lh", recon_stuff["qfunc_K_XY_psh"],
                          recon_stuff["get_fg_trispectrum_N0_XYXY_psh"])
                         ]

            #We'll also want to do the ksz-only case
            Xf_ksz, Yf_ksz = recon_stuff["filter_X"](rksz_alms.copy()), recon_stuff["filter_Y"](rksz_alms.copy())
            for i,job in enumerate(jobs):

                est_name, qfunc, get_tri_N0 = job
                print("rank %d doing %s"%(rank, est_name))
                K_fg_fg = qfunc(
                    fg_alms_filtered_X, fg_alms_filtered_Y)

                #Do trispectrum
                cl_tri_raw = curvedsky.alm2cl(K_fg_fg, K_fg_fg) / w4
                N0_tri = get_tri_N0(cl_fg_XX, cl_fg_YY, cl_fg_XY) 
                outputs['trispectrum_'+est_name] = cl_tri_raw - N0_tri
                outputs['trispectrum_N0_'+est_name] = N0_tri.copy()

                print("doing ksz only")
                K_ksz = qfunc(Xf_ksz, Yf_ksz)
                cl_K_ksz_raw = curvedsky.alm2cl(K_ksz)
                ksz_N0 = get_tri_N0(cl_rksz, cl_rksz, cl_rksz)
                cl_K_ksz = cl_K_ksz_raw - ksz_N0
                outputs["cl_K_ksz"] = cl_K_ksz.copy()

            output_data = np.zeros((recon_config["mlmax"]+1),
                                   dtype=[(k,float) for k in outputs.keys()])
            for k,v in outputs.items():
                try:
                    output_data[k] = v
                except ValueError as e:
                    print("failed to write column %s to output array"%k)
                    raise(e)
            output_file = opj(out_dir, 'fg_terms_%s.npy'%freq)
            print("saving to %s"%output_file)
            np.save(output_file, output_data)
            
        #elif freq == ("hilc_hilc-tzd_hilc_hilc-cibd"):
        elif len(freq.split("_"))==4:
            #this is the ABCD case
            assert isinstance(fg_alms, tuple)
            assert len(fg_alms)==4
                             
            #The code for these asymmetric estimators is quite
            #different to what we had before. So for now just do
            #everything separately. But try and tidy things up
            #to make more uniform.
            add_rkz = 0.
            if recon_config["add_rkszcl_to_filter"]:
                add_rksz = cl_rksz[:recon_config["mlmax"]+1]
            cltot_A = cltot_data["Cltt_total_%s"%freqA][:recon_config["mlmax"]+1] + add_rksz
            cltot_B = cltot_data["Cltt_total_%s"%freqB][:recon_config["mlmax"]+1] + add_rksz
            cltot_C = cltot_data["Cltt_total_%s"%freqC][:recon_config["mlmax"]+1] + add_rksz
            cltot_D = cltot_data["Cltt_total_%s"%freqD][:recon_config["mlmax"]+1] + add_rksz

            if not recon_config["include_fg_power_in_filter"]:
                cltot_A -= cltot_data["Cltt_total_%s"%freqA][:recon_config["mlmax"]+1]
                cltot_B -= cltot_data["Cltt_total_%s"%freqB][:recon_config["mlmax"]+1]
                cltot_C -= cltot_data["Cltt_total_%s"%freqC][:recon_config["mlmax"]+1]
                cltot_D -= cltot_data["Cltt_total_%s"%freqD][:recon_config["mlmax"]+1]

            
            def get_cl_tot_12(freq1, freq2):
                key = "Cltt_total_%s_%s"%(freq1, freq2) if (freq1!=freq2) else "Cltt_total_%s"%freq1
                if key not in cltot_data.dtype.names:
                    key = "Cltt_total_%s_%s"%(freq2, freq1) if (freq1!=freq2) else "Cltt_total_%s"%freq1
                cl = cltot_data[key][:recon_config["mlmax"]+1]
                if recon_config["add_rkszcl_to_filter"]:
                    cl += add_rksz
                if not recon_config["include_fg_power_in_filter"]:
                    cl -= cltot_data[key.replace("tt_total","fg")][:recon_config["mlmax"]+1]
                return cl

            cltot_AC = get_cl_tot_12(freqA, freqC)
            cltot_BD = get_cl_tot_12(freqB, freqD)
            cltot_AD = get_cl_tot_12(freqA, freqD)
            cltot_BC = get_cl_tot_12(freqB, freqC)
            
            #save cls for debugging N0
            cls_out = np.zeros(len(cltot_A), dtype=[
                (x,float) for x in ["A","B","C","D","AC","BD","AD","BC"]
            ])
            cls_out["A"] = cltot_A
            cls_out["B"] = cltot_B
            cls_out["C"] = cltot_C
            cls_out["D"] = cltot_D
            cls_out["AC"] = cltot_AC
            cls_out["BD"] = cltot_BD
            cls_out["AD"] = cltot_AD
            cls_out["BC"] = cltot_BC
            np.save("cltot_debug.npy", cls_out)
            
 
            print("setting up ABCD estimator")
            lmin,lmax=recon_config["K_lmin"], recon_config["K_lmax"]
            #def qtt_asym(est,lmax,rlmin,rlmax,wx0,wxy0,wx1,wxy1,a0a1,b0b1,a0b1,a1b0,gtype='')
            #AC, BD, AD, BC
            recon_stuff = setup_ABCD_recon(
                px, lmin, lmax,
                recon_config["mlmax"],
                cl_rksz, cltot_A, cltot_B,
                cltot_C, cltot_D,
                cltot_AC, cltot_BD,
                cltot_AD, cltot_BC,
                do_lh=args.do_lh, do_psh=args.do_psh)

            profile = recon_stuff['profile']
            qfuncs = [("qe", (recon_stuff["qfunc_K_AB"], recon_stuff["qfunc_K_CD"]))]
            if args.do_psh:
                #raise NotImplementedError(
                #    "not implemented assymetric with bias hardening")
                qfuncs.append(
                    ("psh", (recon_stuff["qfunc_K_AB_psh"], recon_stuff["qfunc_K_CD_psh"]))
                    )
            if args.do_lh:
                #raise NotImplementedError(
                #    "not implemented assymetric with lensing hardening")
                qfuncs.append(
                    ("lh", (recon_stuff["qfunc_K_AB_lh"], recon_stuff["qfunc_K_CD_lh"]))
                    )
                """
                deltaK_phi = curvedsky.almxfl(
                        phi_alms, recon_stuff["norm_K_XY"] * recon_stuff["R_K_phi"]
                    )
                cl_phiphi = curvedsky.alm2cl(deltaK_phi)
                """

            cmb_name = recon_config['cmb_name']
            outputs = OrderedDict()
            outputs["profile"] = profile.copy()

            fg_alms_filtered_A = recon_stuff["filter_A"](fg_alms[0])
            fg_alms_filtered_B = recon_stuff["filter_B"](fg_alms[1])
            fg_alms_filtered_C = recon_stuff["filter_C"](fg_alms[2])
            fg_alms_filtered_D = recon_stuff["filter_D"](fg_alms[3])
            
            K_lmin,K_lmax,mlmax = (recon_config['K_lmin'],
                                   recon_config['K_lmax'],
                                   recon_config['mlmax'])
            
            outputs["N0_K"] = (recon_stuff["N0_ABCD_K"]).copy()
            jobs = [("qe", (recon_stuff["qfunc_K_AB"], recon_stuff["qfunc_K_CD"]),
                     recon_stuff["get_fg_trispectrum_N0_ABCD"])]

            
            if args.do_lh:
                outputs["N0_K_lh"] = (recon_stuff["N0_ABCD_K_lh"]).copy()
                jobs += [("lh", (recon_stuff["qfunc_K_AB_lh"], recon_stuff["qfunc_K_CD_lh"]),
                          recon_stuff["get_fg_trispectrum_N0_ABCD_lh"])
                         ]
            if args.do_psh:
                outputs["N0_K_psh"] = (recon_stuff["N0_ABCD_K_psh"]).copy()
                jobs += [("psh", (recon_stuff["qfunc_K_AB_psh"], recon_stuff["qfunc_K_CD_psh"]),
                          recon_stuff["get_fg_trispectrum_N0_ABCD_psh"])
                         ]

            #We'll also want to do the ksz-only case
            Af_ksz, Bf_ksz = recon_stuff["filter_A"](rksz_alms.copy()), recon_stuff["filter_B"](rksz_alms.copy())
            Cf_ksz, Df_ksz = recon_stuff["filter_C"](rksz_alms.copy()), recon_stuff["filter_D"](rksz_alms.copy())
            for i,job in enumerate(jobs):

                est_name, qfuncs, get_tri_N0 = job
                print("rank %d doing %s"%(rank, est_name))
                KAB_fg_fg = qfuncs[0](
                    fg_alms_filtered_A, fg_alms_filtered_B)
                KCD_fg_fg = qfuncs[1](
                    fg_alms_filtered_C, fg_alms_filtered_D)

                #Do trispectrum
                cl_tri_raw = curvedsky.alm2cl(KAB_fg_fg, KCD_fg_fg) / w4
                N0_tri = get_tri_N0(cl_fg_AC, cl_fg_BD, cl_fg_AD, cl_fg_BC) 
                outputs['trispectrum_'+est_name] = cl_tri_raw - N0_tri
                outputs['trispectrum_raw_'+est_name] = cl_tri_raw.copy()
                outputs['trispectrum_N0_'+est_name] = N0_tri.copy()

                print("doing ksz only")
                KAB_ksz = qfuncs[0](Af_ksz, Bf_ksz)
                KCD_ksz = qfuncs[1](Cf_ksz, Df_ksz)
                
                cl_K_ksz_raw = curvedsky.alm2cl(KAB_ksz, KCD_ksz)
                ksz_N0 = get_tri_N0(cl_rksz, cl_rksz, cl_rksz, cl_rksz)
                cl_K_ksz = cl_K_ksz_raw - ksz_N0
                outputs["trispectrum_N0_ksz_"+est_name] = ksz_N0.copy()
                outputs["cl_K_ksz_"+est_name] = cl_K_ksz.copy()
                
                
                #mean-field?
                if args.do_meanfield:
                    if weight_map is not None:
                        nside = hp.get_nside(weight_map)
                    cov_fg_power = np.zeros((4,4,mlmax+1))
                    cov_fg_power[0,0,:] = cl_fg_AA 
                    cov_fg_power[0,1,:] = cl_fg_AB 
                    cov_fg_power[1,0,:] = cl_fg_AB 
                    cov_fg_power[0,2,:] = cl_fg_AC 
                    cov_fg_power[2,0,:] = cl_fg_AC 
                    cov_fg_power[0,3,:] = cl_fg_AD 
                    cov_fg_power[3,0,:] = cl_fg_AD 
                    cov_fg_power[1,1,:] = cl_fg_BB 
                    cov_fg_power[1,2,:] = cl_fg_BC 
                    cov_fg_power[2,1,:] = cl_fg_BC
                    cov_fg_power[1,3,:] = cl_fg_BD
                    cov_fg_power[3,1,:] = cl_fg_BD
                    cov_fg_power[2,2,:] = cl_fg_CC 
                    cov_fg_power[2,3,:] = cl_fg_CD 
                    cov_fg_power[3,2,:] = cl_fg_CD 
                    cov_fg_power[3,3,:] = cl_fg_DD 
                    K_ABs = []
                    K_CDs = []
                    CL_KK_SSs = []
                    MCN0s = []
                    for isim in range(args.nsim_meanfield):
                        #generate alms
                        fg_alms_g = curvedsky.rand_alm(cov_fg_power, seed=123+isim)
                        #apply mask
                        if weight_map is not None:
                            print("weighting mean-field sims")
                            print("[alm.shape for alm in fg_alms]", [alm.shape for alm in fg_alms_g] )
                            print("survey_mask.shape", weight_map.shape)
                            print("hp.get_nside(survey_mask)", hp.get_nside(weight_map))
                            fg_alms_masked = []
                            for alm in fg_alms_g:
                                print("alm.shape:",alm.shape)
                                m = hp.alm2map(alm, nside)*weight_map
                                print("m.shape:",m.shape)
                                fg_alms_masked.append( hp.map2alm(m, lmax=mlmax))
                        else:
                            fg_alms_masked = fg_alms_g
                            #fg_alms = [ hp.map2alm(hp.alm2map(alm, nside)*survey_mask, lmax=len(cl_fg_AA)+1)
                            #            for alm in fg_alms ]
                        fg_alms_filtered = []
                        for iX,X in enumerate(["A","B","C","D"]):
                            fg_alms_filtered.append(
                            recon_stuff["filter_%s"%X](fg_alms_masked[iX])
                            )
                        K_AB = qfuncs[0](fg_alms_filtered[0],fg_alms_filtered[1])
                        K_CD = qfuncs[1](fg_alms_filtered[2],fg_alms_filtered[3])
                        K_ABs.append(K_AB)
                        K_CDs.append(K_CD)
                        CL_KK_SSs.append(curvedsky.alm2cl(K_AB, K_CD))

                        if isim > 0:
                            K_AB_s_sp = qfuncs[0](fg_alms_filtered[0], fg_alms_filtered_prev[1])
                            K_CD_s_sp = qfuncs[1](fg_alms_filtered[2], fg_alms_filtered_prev[3])
                            K_CD_sp_s = qfuncs[1](fg_alms_filtered_prev[2], fg_alms_filtered[3])

                            MCN0s.append(
                                curvedsky.alm2cl(K_AB_s_sp, K_CD_s_sp)
                                + curvedsky.alm2cl(K_AB_s_sp, K_CD_sp_s)
                                )
                            
                        fg_alms_filtered_prev = list(fg_alms_filtered)

                    #if rank==0:
                    #    n_collected=1
                    #    while n_collected<size:
                    #        more_K_ABs, more_K_CDs, more_MCN0s = comm.recv(source=MPI.ANY_SOURCE)
                    #        K_ABs += more_K_ABs
                    #        K_CDs += more_K_CDs
                    #        MCN0s += more_MCN0s
                    #        n_collected+=1

                    K_AB_mean = np.mean(np.array(K_ABs), axis=0)
                    K_CD_mean = np.mean(np.array(K_CDs), axis=0)

                    CL_KK_meanfield_auto = curvedsky.alm2cl(K_AB_mean, K_CD_mean)/w4
                    CL_KK_meanfield_corrected = curvedsky.alm2cl(KAB_fg_fg-K_AB_mean,
                                                                 KCD_fg_fg-K_CD_mean)/w4
                    outputs["trispectrum_meanfield_auto_"+est_name] = CL_KK_meanfield_auto.copy()
                    outputs["trispectrum_raw_meanfield_corrected_"+est_name] = CL_KK_meanfield_corrected.copy()
                    

                    MCN0 = np.mean(np.array(MCN0s), axis=0)
                    outputs["trispectrum_MCN0_"+est_name] = (MCN0/w4).copy()
                    outputs["trispectrum_MCN0_SS_"+est_name] = (np.mean(np.array(CL_KK_SSs), axis=0)/w4).copy()
                        
                    
            output_data = np.zeros((recon_config["mlmax"]+1),
                                   dtype=[(k,float) for k in outputs.keys()])
            for k,v in outputs.items():
                try:
                    output_data[k] = v
                except ValueError as e:
                    print("failed to write column %s to output array"%k)
                    raise(e)
            output_file = opj(out_dir, 'fg_terms_%s.npy'%freq)
            print("saving to %s"%output_file)
            np.save(output_file, output_data)
            
        
        else:
            raise ValueError("Not using this option right now")
            #Get noise for filters
            Nl_tt = cltot_data["Nltt_%s"%freq][:recon_config["mlmax"]+1]
            nells = {"TT":Nl_tt, "EE":2*Nl_tt, "BB":2*Nl_tt}
            _,tcls = utils.get_theory_dicts(grad=True, nells=nells,
                                            lmax=recon_config["mlmax"])
            tcls['TT'] = cltot_data["Cltt_total_%s"%freq][:recon_config["mlmax"]+1]
            if recon_config["add_rkszcl_to_filter"]:
                tcls["TT"] += cl_rksz
            if not recon_config["include_fg_power_in_filter"]:
                print("removing fg power from filters")
                tcls["TT"] -= cltot_data["Clfg_%s"%freq][:recon_config["mlmax"]+1]

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

            filter_alms = recon_stuff['filter_X']
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

            deltaK_phi = curvedsky.almxfl(
                    phi_alms, recon_stuff["norm_K"] * recon_stuff["R_K_phi"]
                )
            cl_phiphi = curvedsky.alm2cl(deltaK_phi)

            cmb_name = recon_config['cmb_name']
            outputs = OrderedDict()
            outputs["profile"] = profile
            outputs['cl_KphiKphi'] = cl_phiphi


            fg_alms = utils.change_alm_lmax(fg_alms, recon_config['mlmax'])
            cl_fg = get_cl_smooth(fg_alms)
            fg_alms_filtered = filter_alms(fg_alms)

            K_lmin,K_lmax,mlmax = (recon_config['K_lmin'],
                                           recon_config['K_lmax'],
                                           recon_config['mlmax'])

            outputs["N0_K"] = recon_stuff["N0_K"]
            jobs = [("qe", recon_stuff["qfunc_K"],
                     recon_stuff["get_fg_trispectrum_K_N0"])]
            if args.do_lh:
                outputs["N0_K_lh"] = recon_stuff["N0_K_lh"]
                jobs += [("psh", recon_stuff["qfunc_K_psh"],
                          recon_stuff["get_fg_trispectrum_K_N0_lh"])
                         ]
            if args.do_psh:
                outputs["N0_K_psh"] = recon_stuff["N0_K_psh"]
                jobs += [("lh", recon_stuff["qfunc_K_lh"],
                          recon_stuff["get_fg_trispectrum_K_N0_lh"])
                         ]

            #We'll also want to do the ksz-only case
            Xf_ksz = filter_alms(rksz_alms.copy())
            print(Xf_ksz.shape)
            print(fg_alms.shape)
            for i,job in enumerate(jobs):

                est_name, qfunc, get_tri_N0 = job
                print("rank %d doing %s"%(rank, est_name))
                K_fg_fg = qfunc(
                    fg_alms_filtered, fg_alms_filtered)

                #Do trispectrum
                cl_tri_raw = curvedsky.alm2cl(K_fg_fg, K_fg_fg) / w4
                N0_tri = get_tri_N0(cl_fg) 
                outputs['trispectrum_'+est_name] = cl_tri_raw - N0_tri
                outputs['trispectrum_N0_'+est_name] = N0_tri

                print("doing ksz only")
                K_ksz = qfunc(Xf_ksz, Xf_ksz)
                cl_K_ksz_raw = curvedsky.alm2cl(K_ksz)
                ksz_N0 = get_tri_N0(cl_rksz)
                cl_K_ksz = cl_K_ksz_raw - ksz_N0
                outputs["cl_K_ksz"] = cl_K_ksz 

            output_data = np.zeros((recon_config["mlmax"]+1),
                                   dtype=[(k,float) for k in outputs.keys()])
            for k,v in outputs.items():
                try:
                    output_data[k] = v
                except ValueError as e:
                    print("failed to write column %s to output array"%k)
                    raise(e)
            output_file = opj(out_dir, 'fg_terms_%s.npy'%freq)
            print("saving to %s"%output_file)
            np.save(output_file, output_data)

if __name__=="__main__":
    main()    
