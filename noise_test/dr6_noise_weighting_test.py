#Test how much noise-weighting can help.
#I think we just need to calculate this numerically
#so generate ACT dr6-like noise, add CMB, apply 
#noise weighting, and run estimator
import pickle
import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
import os
os.environ["DISABLE_MPI"]="true"
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
from cmbsky import safe_mkdir, get_disable_mpi, get_cmb_alm_unlensed, get_cmb_seeds, ClBinner
from orphics import maps
from websky_model import WebSky
from copy import deepcopy
import sys
from scipy.signal import savgol_filter
sys.path.append("../")
from reconstruction import setup_recon, setup_asym_recon

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="run tests")
    parser.add_argument("-m", "--mpi", action='store_true',
                        help="use_mpi")
    parser.add_argument("--from_pkl", action="store_true",
                        help="read cls from pkl")
    parser.add_argument("-n", "--nsim", type=int, default=10)
    parser.add_argument("--smooth_ivar", default=None, type=float,
                        help="smooth ivar with Gaussian of this fwhm")
    parser.add_argument("--dg_ivar", default=8, type=int,
                        help="downgrade the ivar, then upgrade to original resolution")
    parser.add_argument("--clip_ivar", default=10, type=int,
                        help="for forming the weight map, clip the ivar at this factor times the unmasked median")
    args = parser.parse_args()
    return args

def get_cl_fg_smooth(alms, alm2=None):
    cl = curvedsky.alm2cl(alms, alm2=alm2)
    l = np.arange(len(cl))
    d = l*(l+1)*cl
    #smooth with savgol
    d_smooth = savgol_filter(d, 5, 2)
    #if there's still negative values, set them
    #to zero
    d_smooth[d_smooth<0.] = 0.
    return np.where(
        l>0,d_smooth/l/(l+1),0.)

def main():

    args = parse_args()
    
    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank,size = comm.Get_rank(), comm.Get_size()
    else:
        comm = None
        rank,size = 0,1
    if rank==0:
        outdir="dr6_weighting_noise_test_output"
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

    ivar_map_file = "/global/cscratch1/sd/maccrann/cmb/act_dr6/act_daynight_f090_ivar.fits"
    ivar_map = enmap.read_map(ivar_map_file)
    if args.dg_ivar:
        ivar_map_dg = ivar_map.downgrade(args.dg_ivar)
        ivar_map = ivar_map_dg.upgrade(args.dg_ivar)

    mask = enmap.read_map("/global/project/projectdirs/act/data/maccrann/dr6/dr6v2_default_union_mask.fits")
    print(ivar_map.shape, ivar_map.wcs)
    print(mask.shape, mask.wcs)
    
    #form the weight map
    weight_map = enmap.zeros(ivar_map.shape, ivar_map.wcs)
    unmasked = mask>1.e-9
    med = np.median(ivar_map[unmasked])
    weight_map[unmasked] = ivar_map[unmasked]
    print("clipping ivar at %d times median"%args.clip_ivar)
    weight_map[ivar_map > args.clip_ivar*med] = args.clip_ivar*med
    weight_map /= weight_map.max()
    w2, w4 = maps.wfactor(2, mask), maps.wfactor(4, mask)
    w2_w,w4_w = maps.wfactor(2, weight_map), maps.wfactor(4, weight_map)
    print("w2, w4 = ",w2, w4)

    assert(np.all(np.isfinite(weight_map)))

    #smooth this?
    smooth=60*putils.arcmin
    if smooth is not None:
        ivar_map = enmap.smooth_gauss(ivar_map, smooth)

    px=qe.pixelization(nside=4096)
    mlmax=6000
    lmin=3000
    lmax=6000
    binner=ClBinner(lmin=10, lmax=200, nbin=15)

    noise_sigma = 16.

    beam_fwhm=2.2
    ells = np.arange(mlmax+1)
    beam = maps.gauss_beam(ells, beam_fwhm)
    Nl_tt_X = (noise_sigma*np.pi/180./60.)**2./beam**2
    nells_X = {"TT":Nl_tt_X, "EE":2*Nl_tt_X, "BB":2*Nl_tt_X}

    _,tcls = utils.get_theory_dicts(grad=True, nells=nells_X, lmax=mlmax)
    _,tcls_nonoise = utils.get_theory_dicts(grad=True, lmax=mlmax)

    #Read in alms
    ksz_alm = utils.change_alm_lmax(hp.fitsfunc.read_alm("../tests/alms_4e3_2048_50_50_ksz.fits"),
                                    mlmax)
    ksz_map = enmap.zeros(shape=ivar_map.shape, wcs=ivar_map.wcs)
    ksz_map = curvedsky.alm2map(ksz_alm, ksz_map)
    ksz_alm_masked = curvedsky.map2alm(ksz_map*mask, lmax=mlmax) #masked alm
    ksz_alm_weighted = curvedsky.map2alm(ksz_map*mask*weight_map, lmax=lmax)
    
    cl_rksz = get_cl_fg_smooth(ksz_alm)[:mlmax+1]
    tcls['TT'] += cl_rksz

    cltot = tcls['TT'][:mlmax+1]

    #setup reconstruction
    normal_setup = setup_recon(
        px, lmin, lmax,
        mlmax,  cl_rksz, {"TT":cltot}
    )
    
    #do ksz only recon
    ksz_alm_f = normal_setup["filter_X"](ksz_alm_masked)
    ksz_alm_weighted_f = normal_setup["filter_X"](ksz_alm_weighted)
    K_kszonly = normal_setup["qfunc_K"](
        ksz_alm_f, ksz_alm_f
    )
    K_kszonly_weighted = normal_setup["qfunc_K"](
        ksz_alm_weighted_f, ksz_alm_weighted_f
    )
    
    cl_KK_kszonly_raw = curvedsky.alm2cl(K_kszonly)/w4
    N0_kszonly = normal_setup["get_fg_trispectrum_K_N0"](cl_rksz)
    cl_KK_kszonly =  cl_KK_kszonly_raw - N0_kszonly
    cl_KK_kszonly_weighted_raw = curvedsky.alm2cl(K_kszonly_weighted)/w4_w
    
    cl_dict = {"KKxi" : [],
               "KKxi_w" : [],
               "KK_raw" : [],
               "KK_w_raw" : [],
               "KKxi_noksz" : [],
               "KKxi_noksz_w" : [],
               "KK_noksz" : [],
               "KK_w_noksz" : []
              }

    nsim = 1
    for isim in range(nsim):
        if size>1:
            if isim%size != rank:
                continue
        print("rank %d running sim %d"%(rank, isim))
        cmb_alm = curvedsky.rand_alm(tcls_nonoise["TT"],
                                     lmax=mlmax, seed=isim*(10*nsim))
        cmb_alm = utils.change_alm_lmax(cmb_alm, mlmax)
        sky_alm = cmb_alm + ksz_alm
        
        sky_map = enmap.zeros(shape=ivar_map.shape, wcs=ivar_map.wcs)
        sky_map = curvedsky.alm2map(sky_alm, sky_map)

        #generate noise
        print("generating noise")
        noise_alm = curvedsky.rand_alm(Nl_tt_X, seed=isim*(10*nsim)+1)
        rng = np.random.default_rng(
            seed=isim*(10*nsim)+2)
        noise = rng.standard_normal(
            ivar_map.shape)
        ivar_nonzero = ivar_map>0.
        ivar_median = np.median(ivar_map[ivar_nonzero])
        valid_ivar = ivar_nonzero * (ivar_map>ivar_median/1.e6)
        noise[valid_ivar] /= np.sqrt(ivar_map[valid_ivar])
        noise *= valid_ivar.astype(float)
        noise_map = enmap.enmap(noise,
                                wcs=ivar_map.wcs,
                                copy=False)
        
        sky_map_wnoise = (sky_map + noise_map)*mask
        sky_map_wnoise_noksz = sky_map_wnoise.copy()
        sky_map_wnoise += ksz_map*mask

        sky_map_wnoise_noksz_weighted = sky_map_wnoise_noksz * weight_map
        sky_map_wnoise_weighted = sky_map_wnoise * weight_map
        
        #convert to alm, filter and run through the estimator
        sky_alm_wnoise = curvedsky.map2alm(sky_map_wnoise, lmax=mlmax)
        sky_alm_wnoise_noksz = curvedsky.map2alm(sky_map_wnoise_noksz, lmax=lmax)
        sky_alm_wnoise_weighted = curvedsky.map2alm(sky_map_wnoise_weighted, 
                                                    lmax=lmax)
        sky_alm_wnoise_noksz_weighted = curvedsky.map2alm(sky_map_wnoise_noksz_weighted, 
                                                    lmax=lmax)
        
        
        Xf = normal_setup["filter_X"](sky_alm_wnoise)
        Xwf = normal_setup["filter_X"](sky_alm_wnoise_weighted)
        Xf_noksz = normal_setup["filter_X"](sky_alm_wnoise_noksz)
        Xwf_noksz = normal_setup["filter_X"](sky_alm_wnoise_noksz_weighted)
        
        K = normal_setup["qfunc_K"](Xf,Xf)
        Kw = normal_setup["qfunc_K"](Xwf, Xwf)
        K_noksz = normal_setup["qfunc_K"](Xf_noksz,Xf_noksz)
        Kw_noksz = normal_setup["qfunc_K"](Xwf_noksz, Xwf_noksz)
        
        cl_dict["KKxi"].append(
            curvedsky.alm2cl(K, K_kszonly)/w4
        )
        cl_dict["KKxi_w"].append(
            curvedsky.alm2cl(Kw, K_kszonly)/w4_w
        )
        cl_dict["KKxi_noksz"].append(
            curvedsky.alm2cl(K_noksz, K_kszonly)/w4
        )
        cl_dict["KKxi_noksz_w"].append(
            curvedsky.alm2cl(Kw_noksz, K_kszonly)/w4_w
        )
        cl_dict["KK_raw"].append(
            curvedsky.alm2cl(K)/w4
        )
        cl_dict["KK_w_raw"].append(
            curvedsky.alm2cl(Kw)/w4_w
        )
        cl_dict["KK_noksz"].append(
            curvedsky.alm2cl(K_noksz)/w4
        )
        cl_dict["KK_w_noksz"].append(
            curvedsky.alm2cl(Kw_noksz)/w4_w
        )
        
        if rank==0:
            #collect and plot
            n_collected=1
            while n_collected<size:
                cls_to_add = comm.recv(source=MPI.ANY_SOURCE)
                for key in cl_dict.keys():
                    cl_dict[key] += cls_to_add[key]
                n_collected+=1
            #convert to arrays
            for key in cl_dict:
                cl_dict[key] = np.array(cl_dict[key])
            
            cl_dict["KK_kszonly_raw"] = cl_KK_kszonly_raw
            cl_dict["KK_kszonly"] = cl_KK_kszonly
            cl_dict["KK_N0_kszonly"] = normal_setup["get_fg_trispectrum_K_N0"](cl_rksz)
            cl_dict["N0_K"] = normal_setup["N0_K"]
            
            #save pkl
            with open(opj(outdir,"cls.pkl"), 'wb') as f:
                pickle.dump(cl_dict, f)

        else:
            comm.send(cl_dict, dest=0)
            return 0
        
if __name__=="__main__":
    main()
