#!/usr/bin/env python
# coding: utf-8
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
from cmbsky import safe_mkdir, get_disable_mpi, get_cmb_alm_unlensed, get_cmb_seeds, ClBinner
from orphics import maps
from websky_model import WebSky
from copy import deepcopy
import sys
from scipy.signal import savgol_filter
sys.path.append("../")
from reconstruction import setup_recon

OUTDIR="test_output"
safe_mkdir(OUTDIR)


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

def test_rksz_only():

    #setup
    noise_sigma=10.
    beam_fwhm=1.5
    px=qe.pixelization(nside=4096)
    mlmax=3000
    lmin=2000
    lmax=3000
    binner=ClBinner(lmin=5, lmax=200, nbin=10)

    ells = np.arange(mlmax+1)
    beam = maps.gauss_beam(ells, beam_fwhm)
    Nl = (noise_sigma*np.pi/180./60.)**2./beam**2
    nells = {"TT":Nl, "EE":2*Nl, "BB":2*Nl}

    ucls,tcls = utils.get_theory_dicts(grad=True, nells=nells, lmax=mlmax)

    #Read in alms
    ksz_alm = utils.change_alm_lmax(hp.fitsfunc.read_alm("alms_4e3_2048_50_50_ksz.fits"),
                                    mlmax)
    cl_rksz = get_cl_fg_smooth(ksz_alm)[:mlmax+1]
    tcls['TT'] += cl_rksz

    #setup reconstruction
    recon_stuff = setup_recon(px, lmin, lmax, mlmax,
                    cl_rksz, tcls, tcls_Y=None,
                    do_lh=False,
                    do_psh=False)

    rksz_alm_filtered = recon_stuff["filter_alms_X"](ksz_alm)[0]
    K_psh_recon_alms = recon_stuff["qfunc_K_psh"](rksz_alm_filtered, rksz_alm_filtered)
    K_recon_alms = recon_stuff["qfunc_K"](rksz_alm_filtered, rksz_alm_filtered)
    K_lh_recon_alms = recon_stuff["qfunc_K_lh"](rksz_alm_filtered, rksz_alm_filtered)

    cl_KK = curvedsky.alm2cl(K_recon_alms)
    cl_KK_lh = curvedsky.alm2cl(K_lh_recon_alms)
    cl_KK_psh = curvedsky.alm2cl(K_psh_recon_alms)

    N0 = recon_stuff["get_fg_trispectrum_K_N0"](cl_rksz)
    N0_lh = recon_stuff["get_fg_trispectrum_K_N0_lh"](cl_rksz)
    N0_psh = recon_stuff["get_fg_trispectrum_K_N0_psh"](cl_rksz)
    
    fig,ax=plt.subplots()

    ell_mids = binner.bin_mids

    ax.plot(ell_mids, binner(cl_KK-N0), label="qe")
    ax.plot(ell_mids, binner(cl_KK_lh-N0_lh), label='lh')
    ax.plot(ell_mids, binner(cl_KK_psh-N0_psh), label='psh')
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$C_L^{KK}$")
    fig.savefig(opj(OUTDIR, "test_signal1.png"))

    cl_dict = {"cl_KK_raw" : cl_KK,
               "cl_KK_lh_raw" : cl_KK_lh,
               "cl_KK_psh_raw" : cl_KK_psh,
               "cl_KK" : cl_KK-N0,
               "cl_KK_lh" : cl_KK_lh-N0_lh,
               "cl_KK_psh" : cl_KK_psh-N0_psh}
    dtype = [(c,float) for c in cl_dict.keys()]
    outdata = np.zeros(len(cl_dict["cl_KK"]), dtype=dtype)
    for c in cl_dict.keys():
        outdata[c] = cl_dict[c]
    np.save(opj(OUTDIR, "test_rksz_only.npy"), outdata)

def test_wcmb(nsim):

    #setup
    noise_sigma=10.
    beam_fwhm=1.5
    px=qe.pixelization(nside=4096)
    mlmax=3000
    lmin=2000
    lmax=3000
    binner=ClBinner(lmin=5, lmax=200, nbin=10)

    ells = np.arange(mlmax+1)
    beam = maps.gauss_beam(ells, beam_fwhm)
    Nl = (noise_sigma*np.pi/180./60.)**2./beam**2
    nells = {"TT":Nl, "EE":2*Nl, "BB":2*Nl}

    ucls,tcls = utils.get_theory_dicts(grad=True, nells=nells, lmax=mlmax)

    #Read in ksz alms
    ksz_alm = utils.change_alm_lmax(hp.fitsfunc.read_alm("alms_4e3_2048_50_50_ksz.fits"),
                                    mlmax)
    cl_rksz = get_cl_fg_smooth(ksz_alm)[:mlmax+1]
    tcls['TT'] += cl_rksz

    #setup reconstruction
    recon_stuff = setup_recon(px, lmin, lmax, mlmax,
                    cl_rksz, tcls, tcls_Y=None,
                    do_lh=False,
                    do_psh=False)


    pkl_filename = opj(OUTDIR, "test_data_wcmb.pkl")
    from_pkl=False
    if not from_pkl:
        cl_dict = {"KK" : [],
                   "KK_unlensed" : [],
                   "KK_lh" : [],
                   "KK_unlensed_lh" : [],
                   "KK_psh" : [],
                   "KK_unlensed_psh" : []
        }

        cl_dict["N0_K"] = recon_stuff["N0_K"]
        cl_dict["N0_K_lh"] = recon_stuff["N0_K_lh"]
        cl_dict["N0_K_psh"] = recon_stuff["N0_K_psh"]
        
        for isim in range(nsim):
            if isim%size != rank:
                continue
            print("rank %d doing sim %d"%(rank,isim))
            print("reading cmb and kappa alm")
            cmb_alm = futils.get_cmb_alm(isim,0)[0]
            cmb_alm = futils.change_alm_lmax(cmb_alm, mlmax)
            cmb_unlensed_alm = get_cmb_alm_unlensed(isim,0)[0]
            cmb_unlensed_alm = futils.change_alm_lmax(
                cmb_unlensed_alm, mlmax)
            ells = np.arange(mlmax+1)
            cl_dict["ells"] = ells
            cl_kk_binned = binner(curvedsky.alm2cl(kappa_alm))
            cl_dict['ii'].append(cl_kk_binned)

            print("generating noise")
            noise_alm = curvedsky.rand_alm(Nl, seed=isim*(10*nsim))
            cmb_alm_wnoise = cmb_alm+noise_alm
            cmb_unlensed_alm_wnoise = cmb_unlensed_alm+noise_alm

            sky_alm_wnoise = cmb_alm_wnoise+ksz_alm
            sky_unlensed_alm_wnoise = cmb_unlensed_alm_wnoise+ksz_alm

            X = recon_stuff["filter_alms_X"](sky_alm_wnoise)[0]
            X_unlensed = recon_stuff["filter_alms_X"](sky_unlensed_alm_wnoise)[0]
            
            print("running K estimators")
            ests = ["qe", "lh", "psh"]
            qfuncs = [recon_stuff["qfunc_K"], recon_stuff["qfunc_K_lh"],
                      recon_stuff["qfunc_K_psh"]]

            for est,qfunc in zip(ests,qfuncs):
                K = qfunc(X,X)
                K_unlensed = qfunc(X_unlensed, X_unlensed)
                cl_KK = curvedsky.alm2cl(K)
                cl_KK_unlensed = curvedsky.alm2cl(K_unlensed)
                cl_dict["KK_%s"%est].append(binner(cl_KK))
                cl_dict["KK_unlensed_%s"%est].append(binner(cl_KK_unlensed))

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
            #also save binner info
            cl_dict["ell_mids"] = binner.bin_mids
            cl_dict["lmin"] = binner.lmin
            cl_dict["lmax"] = binner.lmax
            cl_dict["nbin"] = binner.nbin
                
        else:
            comm.send(cl_dict, dest=0)
            return 0

        with open(pkl_filename, "wb") as f:
            pickle.dump(cl_dict, f)

    else:
        if rank==0:
            with open(pkl_filename, "rb") as f:
                cl_dict = pickle.load(f)

    if rank==0:
        fig,ax=plt.subplots()
        binner = ClBinner(lmin=cl_dict["lmin"], lmax=cl_dict["lmax"],
                          nbin=cl_dict["nbin"])
        ell_mids = binner.bin_mids
        ax.plot(ell_mids, binner(cl_dict["KK_qe"].mean(axis=0)-cl_dict["N0"]),
                 label="qe, lensed", color='C0')
        ax.plot(ell_mids, binner(cl_dict["KK_unlensed_qe"].mean(axis=0)-cl_dict["N0"]),
                 label="qe, unlensed", color='C0', linestyle='--')
        ax.plot(ell_mids, binner(cl_dict["KK_lh"].mean(axis=0)-cl_dict["N0_lh"]),
                 label="lh, lensed", color='C1')
        ax.plot(ell_mids, binner(cl_dict["KK_unlensed_lh"].mean(axis=0)-cl_dict["N0_lh"]),
                 label="lh, unlensed", color='C1', linestyle='--')
        ax.plot(ell_mids, binner(cl_dict["KK_psh"].mean(axis=0)-cl_dict["N0_psh"]),
                 label="psh, lensed", color='C2')
        ax.plot(ell_mids, binner(cl_dict["KK_unlensed_psh"].mean(axis=0)-cl_dict["N0_psh"]),
                 label="psh, unlensed", color='C2', linestyle='--')
        ax.legend()
        fig.savefig(opj(OUTDIR, "test_wcmb1.png"))    
    

def main():
    test_rksz_only()
    #test_wcmd(nsim=10)

if __name__=="__main__":
    main()
