#!/usr/bin/env python
# coding: utf-8

# In[5]:


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
from cmbsky import safe_mkdir, get_disable_mpi, get_cmb_alm_unlensed, get_cmb_seeds, ClBinner
from orphics import maps
from websky_model import WebSky
from copy import deepcopy
import sys
from scipy.signal import savgol_filter
sys.path.append("../")
from reconstruction import setup_recon, setup_asym_recon, get_cl_fg_smooth, setup_ABCD_recon
from orphics import mpi


# In[6]:



def test_sym(nsim, use_mpi=False, from_pkl=False, do_lh=True, no_cmb=False):
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank,size = comm.Get_rank(), comm.Get_size()
    else:
        comm = None
        rank,size = 0,1
    if rank==0:
        outdir="ABCD_estimator_test_output"
        if not os.path.isdir(outdir):
            os.makedirs(outdir)   

    if not from_pkl:

        px=qe.pixelization(nside=2048)
        mlmax=3000
        lmin=2500
        lmax=3000
        binner=ClBinner(lmin=10, lmax=200, nbin=15)

        noise_sigma_X = 10.
        beam_fwhm=1.
        ells = np.arange(mlmax+1)
        beam = maps.gauss_beam(ells, beam_fwhm)
        Nl_tt_X = (noise_sigma_X*np.pi/180./60.)**2./beam**2
        nells_X = {"TT":Nl_tt_X, "EE":2*Nl_tt_X, "BB":2*Nl_tt_X}

        _,tcls_X = utils.get_theory_dicts(grad=True, nells=nells_X, lmax=mlmax)
        _,tcls_nonoise = utils.get_theory_dicts(grad=True, lmax=mlmax)
        #Read in alms
        ksz_alm = utils.change_alm_lmax(hp.fitsfunc.read_alm("alms_4e3_2048_50_50_ksz.fits"),
                                        mlmax)
        cl_rksz = get_cl_fg_smooth(ksz_alm)[:mlmax+1]
        tcls_X['TT'] += cl_rksz

        cltot_X = tcls_X['TT'][:mlmax+1]
        if no_cmb:
            cltot_X -= tcls_nonoise['TT']

        # In[10]:
        recon_setup = setup_recon(
            px, lmin, lmax,
            mlmax,  cl_rksz, {"TT":cltot_X}
        )
        from cmbsky.utils import get_cmb_alm_unlensed

        cl_dict = {"KK_qe":[],
                   "KK_qe_gaussian":[],
                   "KK_qe_lensed":[],
                   "KK_lh":[],
                   "KK_lh_gaussian":[],
                   "KK_lh_lensed":[],
                  }

        do_qe=True

        #do ksz only
        X_ksz = recon_setup["filter_X"](ksz_alm.copy())
        print("doing ksz only")
        K_kszonly = recon_setup["qfunc_K"](X_ksz,X_ksz)

        for isim in range(nsim):
            if size>1:
                if isim%size != rank:
                    continue
            print("rank %d doing sim %d"%(rank,isim))
            print("generate unlensed Gaussian cmb")
            #for unlensed CMB, use Gaussian with lensed power
            #otherwise we'll get wrong N0.
            cmb_alm = curvedsky.rand_alm(tcls_nonoise["TT"],
                                         lmax=mlmax, seed=isim*(10*nsim))
            #cmb_alm = utils.change_alm_lmax(
            #    get_cmb_alm_unlensed(isim,0)[0], mlmax)
            cmb_alm_lensed = utils.change_alm_lmax(
                utils.get_cmb_alm(isim,0)[0], mlmax)
            
            #cmb_alm_lensed = get_cmb_alm(
            if no_cmb:
                cmb_alm*=0.
                cmb_alm_lensed*=0.
            #cmb_alm = utils.change_alm_lmax(cmb_alm, mlmax)

            print("generating noise")
            noise_alm_X = curvedsky.rand_alm(Nl_tt_X, seed=isim*(10*nsim)+1)

            A = cmb_alm+noise_alm_X+ksz_alm
            Af = recon_setup["filter_X"](A)
            Af_lensed = recon_setup["filter_X"](cmb_alm_lensed+noise_alm_X+ksz_alm)
            
            #A_noksz = cmb_alm+noise_alm_X
            #B_noksz = cmb_alm+noise_alm_Y
            #Af_noksz, Bf_noksz = ABCD_setup["filter_A"](A_noksz), ABCD_setup["filter_B"](B_noksz)
            #Cf_noksz, Df_noksz = Af_noksz.copy(), Af_noksz.copy()
            #Af_noksz_lensed = ABCD_setup["filter_A"](cmb_alm_lensed+noise_alm_X)
            #Bf_noksz_lensed = ABCD_setup["filter_B"](cmb_alm_lensed+noise_alm_Y)
            #Cf_noksz_lensed, Df_noksz_lensed = Af_noksz_lensed.copy(), Af_noksz_lensed.copy()
            ksz_gaussian = curvedsky.rand_alm(cl_rksz, 
                                              lmax=mlmax, seed=isim*(10*nsim)+3)
            A_gaussian = cmb_alm+noise_alm_X+ksz_gaussian
            Af_gaussian = recon_setup["filter_X"](A_gaussian)
         
            #X_fg_gaussian = curvedsky.rand_alm(cl_fg_X, seed=isim*(10*nsim)+3)
            #Y_fg_gaussian = 0.1*X_fg_gaussian

            print("running K estimators")
            ests = ["qe","lh"]
            qfuncs = [recon_setup["qfunc_K"],
                      recon_setup["qfunc_K_lh"]]
            for est,qfunc in zip(ests, qfuncs):
                print("qe case:")
                
                #unlensed CMB
                K = qfunc(Af, Af)
                print("qe K.shape:",K.shape)
                
                #lensed CMB
                K_lensed = qfunc(Af_lensed, Af_lensed)
                
                print("running Gaussian case")
                #unlensed CMB
                K_gaussian = qfunc(Af_gaussian, Af_gaussian)

                #auto
                cl_dict["KK_%s"%est].append(binner(
                    curvedsky.alm2cl(K, K)
                ))
                cl_dict["KK_%s_gaussian"%est].append(binner(
                    curvedsky.alm2cl(K_gaussian, K_gaussian)
                ))
                cl_dict["KK_%s_lensed"%est].append(binner(
                    curvedsky.alm2cl(K_lensed, K_lensed)
                ))
                
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

            cl_dict["KK_kszonly"] = binner(curvedsky.alm2cl(
                K_kszonly
            ))
            
            cl_dict["binner_lmin"] = binner.lmin
            cl_dict["binner_lmax"] = binner.lmax
            cl_dict["binner_nbin"] = binner.nbin
            
            cl_dict["KK_N0_kszonly"] = recon_setup["get_fg_trispectrum_K_N0"](cl_rksz)
            cl_dict["N0_K"] = recon_setup["N0_K"]
            cl_dict["N0_K_lh"] = recon_setup["N0_K_lh"]
            
            #save pkl
            with open(opj(outdir,"cls.pkl"), 'wb') as f:
                pickle.dump(cl_dict, f)

        else:
            comm.send(cl_dict, dest=0)
            return 0

    else:
        if rank==0:
            print("rank 0 reading pkl")
            with open(opj(outdir, "cls.pkl"),"rb") as f:
                cl_dict = pickle.load(f)
        else:
            return 0
        
    binner = ClBinner(lmin=cl_dict["binner_lmin"], 
                      lmax=cl_dict["binner_lmax"],
                      nbin=cl_dict["binner_nbin"])

    cl_KK_kszonly_binned_XX = cl_dict["KK_kszonly"] - binner(cl_dict["KK_N0_kszonly"])

    #plot N0
    #fig,ax=plt.subplots()
    #ax.plot(binner.bin_mids, 

    #plot signal
    fig,ax1=plt.subplots()
    cl_KK = cl_dict["KK_qe"].mean(axis=0) - binner(cl_dict["N0_K"])
    cl_KK_err = np.std(cl_dict["KK_qe"], axis=0) / np.sqrt(nsim)
    cl_KK_gaussian = cl_dict["KK_qe_gaussian"].mean(axis=0) - binner(cl_dict["N0_K"])
    cl_KK_gaussian_err = np.std(cl_dict["KK_qe_gaussian"], axis=0) / np.sqrt(nsim)
    cl_KK_lensed = cl_dict["KK_qe_lensed"].mean(axis=0)-binner(cl_dict["N0_K"])
    cl_KK_lensed_err = np.std(cl_dict["KK_qe_lensed"], axis=0) / np.sqrt(nsim)
    
    cl_KK_lensed_lh = cl_dict["KK_lh_lensed"].mean(axis=0) - binner(cl_dict["N0_K_lh"])
    cl_KK_lensed_lh_err = np.std(cl_dict["KK_lh_lensed"],axis=0) / np.sqrt(nsim)

    #cl_KK_XX = cl_dict["KK_XX"].mean(axis=0) - cl_dict["N0_K"]
    #cl_KK_XX_err = np.std(cl_dict["KK_XX"], axis=0) / np.sqrt(nsim)

    Lfac = 2
    
    offsets = np.linspace(-5,5,6)
    ax1.plot(binner.bin_mids+offsets[0], binner.bin_mids**Lfac*cl_KK_kszonly_binned_XX, 
             label="cl_KK ksz-only", marker='o')
    ax1.errorbar(binner.bin_mids+offsets[2], binner.bin_mids**Lfac*cl_KK, 
                 yerr=binner.bin_mids**Lfac*cl_KK_err, 
                 label="cl_KK no lensing", fmt='o')
    
    ax1.errorbar(binner.bin_mids+offsets[3], binner.bin_mids**Lfac*cl_KK_lensed, 
                 yerr=binner.bin_mids**Lfac*cl_KK_lensed_err, 
                 label="cl_KK with lensing", fmt='o')
    
    ax1.errorbar(binner.bin_mids+offsets[4], binner.bin_mids**Lfac*cl_KK_lensed_lh, 
                 yerr=binner.bin_mids**Lfac*cl_KK_lensed_lh_err, 
                 label="cl_LL lensed + lh", fmt='o')
    #ax.errorbar(binner.bin_mids+offsets[3], binner.bin_mids**Lfac*cl_KK_XX, yerr=binner.bin_mids**Lfac*cl_KK_XX_err, label="XX", fmt='o')
    #ax.errorbar(binner.bin_mids, cl_KK_XYxi, yerr=cl_KK_XYxi_err, label="XYxi")
    ax1.legend()
    fig.savefig(opj(outdir,"signal_test_nsim%d.png"%nsim))
    
    fig,ax=plt.subplots()
    ax.plot(binner.bin_mids, binner(cl_dict["N0_XX_K"]), color='k', label="N0 AA theory")
    ax.plot(binner.bin_mids, (cl_dict["KK_AA_gaussian"]).mean(axis=0), '--',color='k', label="N0 AA sim")
    ax.plot(binner.bin_mids, binner(cl_dict["N0_ABCD_K"]), color='C0', label="N0 ABAA theory")
    ax.plot(binner.bin_mids, (cl_dict["KK_ABCD_gaussian"]).mean(axis=0), '--',color='C0', label="N0 ABAA sim")
    #ax.plot(binner.bin_mids, np.std(cl_dict["KK_ABCD"], axis=0), ':',color='C0')
    ax.plot(binner.bin_mids, binner(cl_dict["N0_ABCD_K_lh"]), color='C1', label="N0 ABAA lh theory")
    ax.plot(binner.bin_mids, (cl_dict["KK_ABCD_gaussian_lh"]).mean(axis=0), '--',color='C1', label="N0 ABAA lh sim")
    #ax.plot(binner.bin_mids, np.std(cl_dict["KK_ABCD_lh"], axis=0), ':',color='C1')

    #N0 vs. noise test
    #In normal case (e.g. where all Ts are the same), 
    #variance goes as N0^2...we think this won't be the
    #case for more general ABCD case
    def get_var_clKK_binned(clKK, N0, binner):
        Ls = np.arange(len(N0))
        var_L = 2*(clKK + N0)**2/(2*Ls+1)
        #binning
        var_binned = np.zeros(binner.nbin)
        w = 2*Ls+1
        for i in range(binner.nbin):
            use = (Ls>=binner.bin_lims[i])*(Ls<binner.bin_lims[i+1])
            var_binned[i] = np.sum(w[use]**2 * var_L[use]) / ((w[use]).sum())**2
        return var_binned
    
    var_AA_binned = get_var_clKK_binned(0., cl_dict["N0_XX_K"], binner)
    print(var_AA_binned)
    fig,ax=plt.subplots()
    ax.plot(binner.bin_mids, np.var(cl_dict["KK_AA_gaussian"], axis=0), color='C0', label="AA var from sims")
    ax.plot(binner.bin_mids, var_AA_binned, linestyle='--', color='C0', label="AA var from N0")
    ax.legend()
    ax.set_yscale('log')
    fig.savefig(opj(outdir,"var_ABCD_test_nsim%d.png"%nsim))
        
        
    ax.set_yscale('log')
    ax.set_xlabel(r"$L$")
    ax.legend()
    fig.savefig(opj(outdir,"N0_ABCD_test_nsim%d.png"%nsim))
           
    fig,ax=plt.subplots()
    ax.plot(binner.bin_mids, (cl_dict["KK_AA_gaussian"]).mean(axis=0)/binner(cl_dict["N0_XX_K"])-1)
    fig.savefig(opj(outdir,"N0_ratio_nsim%d.png"%nsim))
    
    fig,ax=plt.subplots()
    ax.errorbar(binner.bin_mids, cl_KK_ABCD/cl_KK_kszonly_binned_XX-1, yerr=cl_KK_ABCD_err/cl_KK_kszonly_binned_XX, label="ABCD")
    ax.errorbar(binner.bin_mids, cl_KK_ABCD_lensed/cl_KK_kszonly_binned_XX-1, yerr=cl_KK_ABCD_lensed_err/cl_KK_kszonly_binned_XX, label="ABCD lensed CMB")
    ax.errorbar(binner.bin_mids, cl_KK_ABCD_lensed_lh/cl_KK_kszonly_binned_XX-1, yerr=cl_KK_ABCD_lensed_lh_err/cl_KK_kszonly_binned_XX, label="ABCD lensed CMB + lh")
    #ax.errorbar(binner.bin_mids, cl_KK_XX/cl_KK_binned_XX-1, yerr=cl_KK_XX_err/cl_KK_binned_XX, label="XX")
    ax.legend()
    fig.savefig(opj(outdir,"signal_fracbias_ABCD_test_nsim%d.png"%nsim))       

# In[7]:

def mcn1_ABCD(icov, get_kmap, power, nsims, qfunc1, 
         qfunc2=None, comm=None, verbose=True, shear=False):
    """
    MCN1 for alpha=XY cross beta=AB
    qfunc(x,y) returns QE reconstruction minus mean-field in fourier space


    Parameters
    ----------
    icov: int
        The index of the realization passed to get_kmap if performing 
    a covariance calculation - otherwise, set to zero.
    get_kmap: function
        Function for getting the filtered a_lms of  data and simulation
    maps. See notes at top of module.
    power: function
        Returns C(l) from two maps x,y, as power(x,y). 
    nsims: int
        Number of sims
    qfunc1: function
        Function for reconstructing lensing from maps x and y,
    called as e.g. qfunc(x, y). See e.g. SoLensPipe.qfunc.
    The x and y arguments accept a [T_alm,E_alm,B_alm] tuple. 
    The function should return an (N,...) array where N is typically
    two components for the gradient and curl. 
    qfunc2: function, optional
        Same as above, for the third and fourth legs of the 4-point
    MCN1.
    comm: object, optional
        MPI communicator
    verbose: bool, optional
        Whether to show progress

    Returns
    -------
    mcn1: (N*(N+1)/2,...) array
        Estimate of the MCN1 bias. If N=2 for gradient and curl,
    the three components correspond to the gradient MCN1, the
    curl MCN1 and the gradient x curl MCN1.
    
        Estimate of the MCN1 bias

    """
    comm,rank,my_tasks = mpi.distribute(nsims)
    n1evals = []
    for i in my_tasks:
        i=i+1
        if rank==0 and verbose: print("MCN1: Rank %d doing task %d" % (comm.rank,i))
        S    = get_kmap((icov,0,i)) # S
        Sp   = get_kmap((icov,1,i)) # S'
        Sphi   = get_kmap((icov,2,i)) # Sphi
        Sphip  = get_kmap((icov,3,i)) # Sphi'

        q1_Sphi_Sphip = qfunc1(Sphi[0], Sphip[1])
        q2_Sphi_Sphip = qfunc2(Sphi[2], Sphip[3])
        q2_Sphip_Sphi = qfunc2(Sphip[2], Sphi[3])
        q1_S_Sp = qfunc1(S[0], Sp[1])
        q2_S_Sp = qfunc2(S[2], Sp[2])
        q2_Sp_S = qfunc2(Sp[2], S[0])
        
        term = (power(q1_Sphi_Sphip, q2_Sphi_Sphip) 
                + power(q1_Sphi_Sphip, q2_Sphip_Sphi)
                - power(q1_S_Sp, q2_S_Sp) 
                - power(q1_S_Sp, q2_Sp_S)
               )
        n1evals.append(term.copy())
    n1s = putils.allgatherv(n1evals,comm)
    return n1s

def get_N1(nsim, do_lh=True, use_mpi=True):

    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank,size = comm.Get_rank(), comm.Get_size()
    else:
        comm = None
        rank,size = 0,1

    if rank==0:
        print("getting N1")

    px=qe.pixelization(nside=2048)
    mlmax=6000
    lmin=4000
    lmax=5000
    binner=ClBinner(lmin=10, lmax=200, nbin=15)

    noise_sigma_X = 2.
    noise_sigma_Y = 10.
    #cross-correlation coefficient 
    r = 0.5

    beam_fwhm=1.
    ells = np.arange(mlmax+1)
    beam = maps.gauss_beam(ells, beam_fwhm)
    Nl_tt_X = (noise_sigma_X*np.pi/180./60.)**2./beam**2
    Nl_tt_Y = (noise_sigma_Y*np.pi/180./60.)**2./beam**2
    nells_X = {"TT":Nl_tt_X, "EE":2*Nl_tt_X, "BB":2*Nl_tt_X}
    nells_Y = {"TT":Nl_tt_Y, "EE":2*Nl_tt_Y, "BB":2*Nl_tt_Y}

    _,tcls_X = utils.get_theory_dicts(grad=True, nells=nells_X, lmax=mlmax)
    _,tcls_Y = utils.get_theory_dicts(grad=True, nells=nells_Y, lmax=mlmax)
    _,tcls_nonoise = utils.get_theory_dicts(grad=True, lmax=mlmax)
    #Read in alms
    ksz_alm = utils.change_alm_lmax(hp.fitsfunc.read_alm("alms_4e3_2048_50_50_ksz.fits"),
                                    mlmax)
    cl_rksz = get_cl_fg_smooth(ksz_alm)[:mlmax+1]
    tcls_X['TT'] += cl_rksz
    tcls_Y['TT'] += cl_rksz
    #tcls_nonoise['TT'] += cl_rksz

    cltot_X = tcls_X['TT'][:mlmax+1]
    cltot_Y = tcls_Y['TT'][:mlmax+1]
    cltot_XY = (tcls_nonoise['TT'] + cl_rksz + r*np.sqrt(Nl_tt_X*Nl_tt_Y))[:mlmax+1]

    #We need N_l for generating uncorrelated component of Y,
    #call this map Z
    #We have noise_Y = a*noise_X + noise_Z
    #then N_l^Y = a^2 N_l^X + N_l^Z
    #and <XY> = a * N_l^X = r_l * sqrt(N_l^X N_l^Y) (from defintion of r_l)
    #then we have a = r_l*sqrt(N_l^X N_l^Y) / N_l^X
    #and N_l^Z = N_l^Y - a^2 N_l^X 
    a = r * np.sqrt(Nl_tt_X*Nl_tt_Y)/Nl_tt_X
    Nl_tt_Z = Nl_tt_Y - a**2 * Nl_tt_X


    # In[10]:
    normal_setup = setup_recon(
        px, lmin, lmax,
        mlmax,  cl_rksz, {"TT":cltot_X}
    )

    #XYXX case
    ABCD_setup = setup_ABCD_recon(px, lmin, lmax, mlmax,
                          cl_rksz, cltot_X, cltot_Y,
                          cltot_X, cltot_X,
                          cltot_X, cltot_XY,
                          cltot_X, cltot_XY, do_lh=True,
                          do_psh=False)
    icov=0

    def get_kmap(seed):
        #seed is of the form icov,iset,i
        #get_cmb_alm wants i,iset
        s_i, s_set, _ = solenspipe.convert_seeds(seed)
        alm = utils.get_cmb_alm(s_i,s_set)[0] #[0] picks out T only
        #then filter
        alms_f = [utils.change_alm_lmax(
            ABCD_setup["filter_%s"%x](alm), mlmax)
                  for x in ["A","B","C","D"]
                 ]
        return alms_f
    
    mcn1_data = mcn1_ABCD(icov, get_kmap, curvedsky.alm2cl,
                         nsim, qfunc1=ABCD_setup["qfunc_K_AB"],
                         qfunc2=ABCD_setup["qfunc_K_CD"],
                         comm=comm)
    mcn1_mean = mcn1_data.mean(axis=0)
    mcn1_mean_binned = binner(mcn1_mean)
    
    mcn1_data_lh = mcn1_ABCD(icov, get_kmap, curvedsky.alm2cl,
                         nsim, qfunc1=ABCD_setup["qfunc_K_AB_lh"],
                         qfunc2=ABCD_setup["qfunc_K_CD_lh"],
                         comm=comm)
    mcn1_lh_mean = mcn1_data_lh.mean(axis=0)
    mcn1_lh_mean_binned = binner(mcn1_lh_mean)

    if rank==0:
        outdir="ABCD_estimator_test_output_nsim%d"%nsim
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        n1_outputs = {"mcn1_allsims" : mcn1_data,
                      "mcn1" : mcn1_mean,
                      "mcn1_mean_binned" : mcn1_mean_binned,
                      "mcn1_lh_allsims" : mcn1_data_lh,
                      "mcn1_lh" : mcn1_lh_mean,
                      "mcn1_lh_mean_binned" : mcn1_lh_mean_binned,
                      "binner_lmin" : binner.lmin,
                      "binner_lmax" : binner.lmax,
                      "binner_nbin" : binner.nbin}
        filename = opj(outdir, "n1_data.pkl")
        print("saving N1 outputs to %s"%filename)
        with open(filename, "wb") as f:
            pickle.dump(n1_outputs, f)


def test_ABCD(nsim, use_mpi=False, from_pkl=False, no_cmb=False, do_lh=False,
              do_n1=False, seed=None, isim_start=0):
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank,size = comm.Get_rank(), comm.Get_size()
    else:
        comm = None
        rank,size = 0,1
    if rank==0:
        outdir="ABCD_estimator_test_output_nsim%d_start%d"%(nsim, isim_start)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)   

    if not from_pkl:

        px=qe.pixelization(nside=2048)
        mlmax=6000
        lmin=4000
        lmax=5000
        binner=ClBinner(lmin=1, lmax=200, nbin=15)

        noise_sigma_X = 2.
        noise_sigma_Y = 10.
        #cross-correlation coefficient 
        r = 0.5

        beam_fwhm=1.
        ells = np.arange(mlmax+1)
        beam = maps.gauss_beam(ells, beam_fwhm)
        Nl_tt_X = (noise_sigma_X*np.pi/180./60.)**2./beam**2
        Nl_tt_Y = (noise_sigma_Y*np.pi/180./60.)**2./beam**2
        nells_X = {"TT":Nl_tt_X, "EE":2*Nl_tt_X, "BB":2*Nl_tt_X}
        nells_Y = {"TT":Nl_tt_Y, "EE":2*Nl_tt_Y, "BB":2*Nl_tt_Y}

        _,tcls_X = utils.get_theory_dicts(grad=True, nells=nells_X, lmax=mlmax)
        _,tcls_Y = utils.get_theory_dicts(grad=True, nells=nells_Y, lmax=mlmax)
        _,tcls_nonoise = utils.get_theory_dicts(grad=True, lmax=mlmax)
        #Read in alms
        ksz_alm = utils.change_alm_lmax(hp.fitsfunc.read_alm("alms_4e3_2048_50_50_ksz.fits"),
                                        mlmax)
        cl_rksz = get_cl_fg_smooth(ksz_alm)[:mlmax+1]
        tcls_X['TT'] += cl_rksz
        tcls_Y['TT'] += cl_rksz
        #tcls_nonoise['TT'] += cl_rksz


        cltot_X = tcls_X['TT'][:mlmax+1]
        cltot_Y = tcls_Y['TT'][:mlmax+1]
        cltot_XY = (tcls_nonoise['TT'] + cl_rksz + r*np.sqrt(Nl_tt_X*Nl_tt_Y))[:mlmax+1]
        if no_cmb:
            cltot_X -= tcls_nonoise['TT']
            cltot_Y -= tcls_nonoise['TT']
            cltot_XY -= tcls_nonoise["TT"]
        #We need N_l for generating uncorrelated component of Y,
        #call this map Z
        #We have noise_Y = a*noise_X + noise_Z
        #then N_l^Y = a^2 N_l^X + N_l^Z
        #and <XY> = a * N_l^X = r_l * sqrt(N_l^X N_l^Y) (from defintion of r_l)
        #then we have a = r_l*sqrt(N_l^X N_l^Y) / N_l^X
        #and N_l^Z = N_l^Y - a^2 N_l^X 
        a = r * np.sqrt(Nl_tt_X*Nl_tt_Y)/Nl_tt_X
        Nl_tt_Z = Nl_tt_Y - a**2 * Nl_tt_X


        # In[10]:
        normal_setup = setup_recon(
            px, lmin, lmax,
            mlmax,  cl_rksz, {"TT":cltot_X}
        )
        
        #XYXX case
        ABCD_setup = setup_ABCD_recon(px, lmin, lmax, mlmax,
                              cl_rksz, cltot_X, cltot_Y,
                              cltot_X, cltot_X,
                              cltot_X, cltot_XY,
                              cltot_X, cltot_XY, do_lh=True,
                              do_psh=False)
        
        from cmbsky.utils import get_cmb_alm_unlensed

        cl_dict = {"KK_ABCD" : [],
                   "KK_ABCD_lh" : [],
                   #"KK_ABCD_noksz" : [],
                   "KK_ABCD_gaussian" : [],
                   "KK_ABCD_lensed" : [],
                   #"KK_ABCD_noksz_lensed" : [],
                   "KK_ABCD_lensed_lh" : [],
                   "KK_ABCD_gaussian_lh" : [],
                   "KK_AA_gaussian" : [],
                   "KK_AA_gaussian_lh" : []
                  }

        do_qe=True

        #do ksz only
        Af_ksz, Bf_ksz = ABCD_setup["filter_A"](ksz_alm.copy()), ABCD_setup["filter_B"](ksz_alm.copy())
        #C and D same as A here
        Cf_ksz = Af_ksz.copy()
        Df_ksz = Af_ksz.copy()
        
        print("doing ksz only")
        K_AB_kszonly = ABCD_setup["qfunc_K_AB"](Af_ksz, Bf_ksz)
        K_CD_kszonly = ABCD_setup["qfunc_K_CD"](Cf_ksz, Df_ksz)
        K_XX_kszonly = normal_setup["qfunc_K"](Af_ksz,Af_ksz)
        
        #cl_dict["KK_XX_kszonly"]

        for isim in range(isim_start, nsim+isim_start):
            if size>1:
                if (isim-isim_start)%size != rank:
                    continue
            print("rank %d doing sim %d"%(rank,isim))
            print("generate unlensed Gaussian cmb")
            #for unlensed CMB, use Gaussian with lensed power
            #otherwise we'll get wrong N0.
            cmb_alm = curvedsky.rand_alm(tcls_nonoise["TT"],
                                         lmax=mlmax, seed=seed+isim*(10*nsim))
            #cmb_alm = utils.change_alm_lmax(
            #    get_cmb_alm_unlensed(isim,0)[0], mlmax)
            cmb_alm_lensed = utils.change_alm_lmax(
                utils.get_cmb_alm(isim,0)[0], mlmax)
            
            #cmb_alm_lensed = get_cmb_alm(
            if no_cmb:
                cmb_alm*=0.
                cmb_alm_lensed*=0.
            #cmb_alm = utils.change_alm_lmax(cmb_alm, mlmax)

            print("generating noise")
            noise_alm_X = curvedsky.rand_alm(Nl_tt_X, seed=seed+isim*(10*nsim)+1)
            noise_alm_Z = curvedsky.rand_alm(Nl_tt_Z, seed=seed+isim*(10*nsim)+2)
            noise_alm_Y = curvedsky.almxfl(noise_alm_X,a) + noise_alm_Z

            A = cmb_alm+noise_alm_X+ksz_alm
            B = cmb_alm+noise_alm_Y+ksz_alm
            Af, Bf = ABCD_setup["filter_A"](A), ABCD_setup["filter_B"](B)
            #Xf_ksz, Yf_ksz = asym_setup["filter_X"](ksz_alm.copy()), asym_setup["filter_Y"](ksz_alm.copy())
            Cf = Af.copy()
            Df = Af.copy()
            
            Af_lensed = ABCD_setup["filter_A"](cmb_alm_lensed+noise_alm_X+ksz_alm)
            Bf_lensed = ABCD_setup["filter_B"](cmb_alm_lensed+noise_alm_Y+ksz_alm)
            Cf_lensed,Df_lensed = Af_lensed.copy(), Af_lensed.copy()
            
            #A_noksz = cmb_alm+noise_alm_X
            #B_noksz = cmb_alm+noise_alm_Y
            #Af_noksz, Bf_noksz = ABCD_setup["filter_A"](A_noksz), ABCD_setup["filter_B"](B_noksz)
            #Cf_noksz, Df_noksz = Af_noksz.copy(), Af_noksz.copy()
            #Af_noksz_lensed = ABCD_setup["filter_A"](cmb_alm_lensed+noise_alm_X)
            #Bf_noksz_lensed = ABCD_setup["filter_B"](cmb_alm_lensed+noise_alm_Y)
            #Cf_noksz_lensed, Df_noksz_lensed = Af_noksz_lensed.copy(), Af_noksz_lensed.copy()
            ksz_gaussian = curvedsky.rand_alm(cl_rksz, 
                                              lmax=mlmax, seed=seed+isim*(10*nsim)+3)
            A_gaussian = cmb_alm+noise_alm_X+ksz_gaussian
            B_gaussian = cmb_alm+noise_alm_Y+ksz_gaussian
            Af_gaussian, Bf_gaussian = ABCD_setup["filter_A"](A_gaussian), ABCD_setup["filter_B"](B_gaussian)
            Cf_gaussian, Df_gaussian = Af_gaussian.copy(), Af_gaussian.copy()
         
            #X_fg_gaussian = curvedsky.rand_alm(cl_fg_X, seed=isim*(10*nsim)+3)
            #Y_fg_gaussian = 0.1*X_fg_gaussian

            print("running K estimators")
            if do_qe:
                print("qe case:")
                
                #unlensed CMB
                K_AB = ABCD_setup["qfunc_K_AB"](Af, Bf)
                K_CD = ABCD_setup["qfunc_K_CD"](Cf, Df)
                print("qe K_AB.shape:",K_AB.shape)
                print("qe K_CD.shape:",K_CD.shape)
                
                #lensed CMB
                K_AB_lensed = ABCD_setup["qfunc_K_AB"](Af_lensed, Bf_lensed)
                K_CD_lensed = ABCD_setup["qfunc_K_CD"](Cf_lensed, Df_lensed)
                
                print("running Gaussian case")
                #unlensed CMB
                K_AB_gaussian = ABCD_setup["qfunc_K_AB"](Af_gaussian, Bf_gaussian)
                K_CD_gaussian = ABCD_setup["qfunc_K_CD"](Cf_gaussian, Df_gaussian)
                
                KK_AA_gaussian = normal_setup["qfunc_K"](Af_gaussian, Af_gaussian)

                #auto
                cl_dict["KK_ABCD"].append(binner(
                    curvedsky.alm2cl(K_AB, K_CD)
                ))
                cl_dict["KK_ABCD_gaussian"].append(binner(
                    curvedsky.alm2cl(K_AB_gaussian, K_CD_gaussian)
                ))
                cl_dict["KK_ABCD_lensed"].append(binner(
                    curvedsky.alm2cl(K_AB_lensed, K_CD_lensed)
                ))
                cl_dict["KK_AA_gaussian"].append(binner(
                    curvedsky.alm2cl(KK_AA_gaussian)
                ))
                #cl_dict["KK_ABCD_noksz_lensed"].append(binner(
                #    curvedsky.alm2cl(K_AB_noksz_lensed, K_CD_noksz_lensed)
                #))
                
            if do_lh:
                print("lh case")
                #unlensed CMB
                K_AB = ABCD_setup["qfunc_K_AB_lh"](Af, Bf)
                K_CD = ABCD_setup["qfunc_K_CD_lh"](Cf, Df)
                print("lh K_AB.shape:",K_AB.shape)
                print("lh K_CD.shape:",K_CD.shape)
                
                #lensed CMB
                K_AB_lensed = ABCD_setup["qfunc_K_AB_lh"](Af_lensed, Bf_lensed)
                K_CD_lensed = ABCD_setup["qfunc_K_CD_lh"](Cf_lensed, Df_lensed)
                
                print("running Gaussian case")
                #unlensed CMB
                K_AB_gaussian = ABCD_setup["qfunc_K_AB_lh"](Af_gaussian, Bf_gaussian)
                K_CD_gaussian = ABCD_setup["qfunc_K_CD_lh"](Cf_gaussian, Df_gaussian)
                
                KK_AA_gaussian = normal_setup["qfunc_K_lh"](Af_gaussian, Af_gaussian)

                #auto
                cl_dict["KK_ABCD_lh"].append(binner(
                    curvedsky.alm2cl(K_AB, K_CD)
                ))
                cl_dict["KK_ABCD_gaussian_lh"].append(binner(
                    curvedsky.alm2cl(K_AB_gaussian, K_CD_gaussian)
                ))
                cl_dict["KK_ABCD_lensed_lh"].append(binner(
                    curvedsky.alm2cl(K_AB_lensed, K_CD_lensed)
                ))
                cl_dict["KK_AA_gaussian_lh"].append(binner(
                    curvedsky.alm2cl(KK_AA_gaussian)
                ))
                
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

            cl_dict["KK_XX_kszonly"] = binner(curvedsky.alm2cl(
                K_XX_kszonly
            ))
            cl_dict["KK_ABCD_kszonly"] = binner(curvedsky.alm2cl(
                K_AB_kszonly, K_CD_kszonly
            ))
            cl_dict["KK_AA_kszonly_nobin"] = curvedsky.alm2cl(
                K_XX_kszonly
            )
            
            cl_dict["binner_lmin"] = binner.lmin
            cl_dict["binner_lmax"] = binner.lmax
            cl_dict["binner_nbin"] = binner.nbin
            
            cl_dict["KK_XX_N0_kszonly"] = normal_setup["get_fg_trispectrum_K_N0"](cl_rksz)
            cl_dict["KK_ABCD_N0_kszonly"] = ABCD_setup["get_fg_trispectrum_N0_ABCD"](
                cl_rksz, cl_rksz, cl_rksz, cl_rksz)

            cl_dict["N0_ABCD_K"] = ABCD_setup["N0_ABCD_K"]
            cl_dict["N0_XX_K"] = normal_setup["N0_K"]
            if do_lh:
                cl_dict["N0_ABCD_K_lh"] = ABCD_setup["N0_ABCD_K_lh"]
            
            #save pkl
            with open(opj(outdir,"cls.pkl"), 'wb') as f:
                pickle.dump(cl_dict, f)

        else:
            comm.send(cl_dict, dest=0)
            return 0


        
    else:
        if rank==0:
            pkl_file = opj(outdir, "cls.pkl")
            print("rank 0 reading pkl file %s"%pkl_file)
            with open(pkl_file, "rb") as f:
                cl_dict = pickle.load(f)
        else:
            return 0
        
    if rank==0:
        binner = ClBinner(lmin=cl_dict["binner_lmin"], 
                          lmax=cl_dict["binner_lmax"],
                          nbin=cl_dict["binner_nbin"])

        
        cl_KK_kszonly_binned_XX = cl_dict["KK_XX_kszonly"] - binner(cl_dict["KK_XX_N0_kszonly"])
        cl_KK_kszonly_binned_ABCD = cl_dict["KK_ABCD_kszonly"] - binner(cl_dict["KK_ABCD_N0_kszonly"])

        #read in N1 
        if args.n1_file is None:
            n1_file = opj(outdir, "n1_data.pkl")
        else:
            n1_file = args.n1_file
        sub_n1 = False
        if os.path.isfile(n1_file):
            with open(n1_file, "rb") as f:
                n1_data = pickle.load(f)
            sub_n1=True
        else:
            print("found no n1 file %s"%n1_file)
                
        
        #plot N0
        #fig,ax=plt.subplots()
        #ax.plot(binner.bin_mids, 

        #plot signal
        fig,ax1=plt.subplots()
        cl_KK_ABCD = cl_dict["KK_ABCD"].mean(axis=0) - binner(cl_dict["N0_ABCD_K"])
        cl_KK_ABCD_err = np.std(cl_dict["KK_ABCD"], axis=0) / np.sqrt(nsim)
        cl_KK_ABCD_gaussian = cl_dict["KK_ABCD_gaussian"].mean(axis=0) - binner(cl_dict["N0_ABCD_K"])
        cl_KK_ABCD_gaussian_err = np.std(cl_dict["KK_ABCD_gaussian"], axis=0) / np.sqrt(nsim)
        cl_KK_ABCD_lensed = cl_dict["KK_ABCD_lensed"].mean(axis=0)-binner(cl_dict["N0_ABCD_K"])
        cl_KK_ABCD_lensed_err = np.std(cl_dict["KK_ABCD_lensed"], axis=0) / np.sqrt(nsim)

        cl_KK_ABCD_lensed_lh = cl_dict["KK_ABCD_lensed_lh"].mean(axis=0) - binner(cl_dict["N0_ABCD_K_lh"])
        cl_KK_ABCD_lensed_lh_err = np.std(cl_dict["KK_ABCD_lensed_lh"],axis=0) / np.sqrt(nsim)
        
        if sub_n1:
             cl_KK_ABCD_lensed_lh_wn1 = cl_KK_ABCD_lensed_lh - n1_data["mcn1_lh_mean_binned"]
             cl_KK_ABCD_lensed_lh_err_wn1 = np.std(cl_dict["KK_ABCD_lensed_lh"],axis=0) / np.sqrt(nsim)
                

        #cl_KK_XX = cl_dict["KK_XX"].mean(axis=0) - cl_dict["N0_K"]
        #cl_KK_XX_err = np.std(cl_dict["KK_XX"], axis=0) / np.sqrt(nsim)

        Lfac = 2

        offsets = np.linspace(-5,5,6)
        ax1.plot(binner.bin_mids+offsets[0], binner.bin_mids**Lfac*cl_KK_kszonly_binned_XX, label="cl_KK AA ksz-only", marker='o')
        ax1.plot(binner.bin_mids+offsets[1], binner.bin_mids**Lfac*cl_KK_kszonly_binned_ABCD, label="cl_KK ABCD ksz-only", marker='o')
        ax1.errorbar(binner.bin_mids+offsets[2], binner.bin_mids**Lfac*cl_KK_ABCD, yerr=binner.bin_mids**Lfac*cl_KK_ABCD_err, label="ABCD", fmt='o')
        ax1.errorbar(binner.bin_mids+offsets[3], binner.bin_mids**Lfac*cl_KK_ABCD_lensed, yerr=binner.bin_mids**Lfac*cl_KK_ABCD_lensed_err, label="ABCD lensed", fmt='o')
        ax1.errorbar(binner.bin_mids+offsets[4], binner.bin_mids**Lfac*cl_KK_ABCD_lensed_lh, yerr=binner.bin_mids**Lfac*cl_KK_ABCD_lensed_lh_err, label="ABCD lensed + lh", fmt='o')
        if sub_n1:
            ax1.errorbar(binner.bin_mids+offsets[4], binner.bin_mids**Lfac*cl_KK_ABCD_lensed_lh_wn1, yerr=binner.bin_mids**Lfac*cl_KK_ABCD_lensed_lh_err_wn1, label="ABCD lensed + lh, N1 subtracted", fmt='o')
        #ax.errorbar(binner.bin_mids+offsets[3], binner.bin_mids**Lfac*cl_KK_XX, yerr=binner.bin_mids**Lfac*cl_KK_XX_err, label="XX", fmt='o')
        #ax.errorbar(binner.bin_mids, cl_KK_XYxi, yerr=cl_KK_XYxi_err, label="XYxi")
        ax1.legend()
        ax1.set_yscale('log')
        fig.savefig(opj(outdir,"signal_ABCD_test_nsim%d.png"%nsim))

        fig,ax=plt.subplots()
        ax.plot(binner.bin_mids, binner(cl_dict["N0_XX_K"]), color='k', label="N0 AA theory")
        ax.plot(binner.bin_mids, (cl_dict["KK_AA_gaussian"]).mean(axis=0), '--',color='k', label="N0 AA sim")
        ax.plot(binner.bin_mids, binner(cl_dict["N0_ABCD_K"]), color='C0', label="N0 ABAA theory")
        ax.plot(binner.bin_mids, (cl_dict["KK_ABCD_gaussian"]).mean(axis=0), '--',color='C0', label="N0 ABAA sim")
        #ax.plot(binner.bin_mids, np.std(cl_dict["KK_ABCD"], axis=0), ':',color='C0')
        ax.plot(binner.bin_mids, binner(cl_dict["N0_ABCD_K_lh"]), color='C1', label="N0 ABAA lh theory")
        ax.plot(binner.bin_mids, (cl_dict["KK_ABCD_gaussian_lh"]).mean(axis=0), '--',color='C1', label="N0 ABAA lh sim")
        #ax.plot(binner.bin_mids, np.std(cl_dict["KK_ABCD_lh"], axis=0), ':',color='C1')

        #N0 vs. noise test
        #In normal case (e.g. where all Ts are the same), 
        #variance goes as N0^2...we think this won't be the
        #case for more general ABCD case
        def get_var_clKK_binned(clKK, N0, binner):
            Ls = np.arange(len(N0))
            var_L = 2*(clKK + N0)**2/(2*Ls+1)
            #binning
            var_binned = np.zeros(binner.nbin)
            w = 2*Ls+1
            for i in range(binner.nbin):
                use = (Ls>=binner.bin_lims[i])*(Ls<binner.bin_lims[i+1])
                var_binned[i] = np.sum(w[use]**2 * var_L[use]) / ((w[use]).sum())**2
            return var_binned

        var_AA_binned = get_var_clKK_binned(0., cl_dict["N0_XX_K"], binner)
        print(var_AA_binned)
        fig,ax=plt.subplots()
        ax.plot(binner.bin_mids, np.var(cl_dict["KK_AA_gaussian"], axis=0), color='C0', label="AA var from sims")
        ax.plot(binner.bin_mids, var_AA_binned, linestyle='--', color='C0', label="AA var from N0")
        ax.legend()
        ax.set_yscale('log')
        fig.savefig(opj(outdir,"var_ABCD_test_nsim%d.png"%nsim))


        ax.set_yscale('log')
        ax.set_xlabel(r"$L$")
        ax.legend()
        fig.savefig(opj(outdir,"N0_ABCD_test_nsim%d.png"%nsim))

        fig,ax=plt.subplots()
        ax.plot(binner.bin_mids, (cl_dict["KK_AA_gaussian"]).mean(axis=0)/binner(cl_dict["N0_XX_K"])-1)
        fig.savefig(opj(outdir,"N0_ratio_nsim%d.png"%nsim))

        #Fractional bias
        fig,ax=plt.subplots(figsize=(5,4))
        ax.errorbar(binner.bin_mids, cl_KK_ABCD/cl_KK_kszonly_binned_XX-1, yerr=cl_KK_ABCD_err/cl_KK_kszonly_binned_XX, fmt='o', label="ABCD")
        ax.errorbar(binner.bin_mids, cl_KK_ABCD_lensed/cl_KK_kszonly_binned_XX-1, yerr=cl_KK_ABCD_lensed_err/cl_KK_kszonly_binned_XX, fmt='o', label="ABCD lensed CMB")
        ax.errorbar(binner.bin_mids, cl_KK_ABCD_lensed_lh/cl_KK_kszonly_binned_XX-1, yerr=cl_KK_ABCD_lensed_lh_err/cl_KK_kszonly_binned_XX, fmt='o', label="ABCD lensed CMB + lh")
        if sub_n1:
            ax.errorbar(binner.bin_mids, cl_KK_ABCD_lensed_lh_wn1/cl_KK_kszonly_binned_XX-1, yerr=cl_KK_ABCD_lensed_lh_err_wn1/cl_KK_kszonly_binned_XX, fmt='o', label="ABCD lensed CMB + lh \n N1-subtracted")
        #ax.errorbar(binner.bin_mids, cl_KK_XX/cl_KK_binned_XX-1, yerr=cl_KK_XX_err/cl_KK_binned_XX, label="XX")
        ax.legend()
        if args.frac_bias_ylim is not None:
            ax.set_ylim(args.frac_bias_ylim[0], args.frac_bias_ylim[1])
        ax.set_ylabel("fractional bias on $C_L^{KK}$")
        ax.set_xlabel("$L$")
        fig.tight_layout()
        fig.savefig(opj(outdir,"signal_fracbias_ABCD_test_nsim%d.png"%nsim))       
        
        
def test_signal(nsim, use_mpi=False, from_pkl=False, no_cmb=False):

    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank,size = comm.Get_rank(), comm.Get_size()
    else:
        comm = None
        rank,size = 0,1
    if rank==0:
        outdir="asym_estimator_test_output"
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
            
    if not from_pkl:

        px=qe.pixelization(nside=4096)
        mlmax=6000
        lmin=3000
        lmax=6000
        binner=ClBinner(lmin=10, lmax=200, nbin=15)

        noise_sigma_X = 2.
        noise_sigma_Y = 10.
        #cross-correlation coefficient 
        r = 0.5

        beam_fwhm=1.
        ells = np.arange(mlmax+1)
        beam = maps.gauss_beam(ells, beam_fwhm)
        Nl_tt_X = (noise_sigma_X*np.pi/180./60.)**2./beam**2
        Nl_tt_Y = (noise_sigma_Y*np.pi/180./60.)**2./beam**2
        nells_X = {"TT":Nl_tt_X, "EE":2*Nl_tt_X, "BB":2*Nl_tt_X}
        nells_Y = {"TT":Nl_tt_Y, "EE":2*Nl_tt_Y, "BB":2*Nl_tt_Y}

        _,tcls_X = utils.get_theory_dicts(grad=True, nells=nells_X, lmax=mlmax)
        _,tcls_Y = utils.get_theory_dicts(grad=True, nells=nells_Y, lmax=mlmax)
        _,tcls_nonoise = utils.get_theory_dicts(grad=True, lmax=mlmax)


        # In[8]:


        #Read in alms
        ksz_alm = utils.change_alm_lmax(hp.fitsfunc.read_alm("alms_4e3_2048_50_50_ksz.fits"),
                                        mlmax)
        cl_rksz = get_cl_fg_smooth(ksz_alm)[:mlmax+1]
        tcls_X['TT'] += cl_rksz
        tcls_Y['TT'] += cl_rksz
        #tcls_nonoise['TT'] += cl_rksz


        cltot_X = tcls_X['TT'][:mlmax+1]
        cltot_Y = tcls_Y['TT'][:mlmax+1]
        cltot_XY = (tcls_nonoise['TT'] + cl_rksz + r*np.sqrt(Nl_tt_X*Nl_tt_Y))[:mlmax+1]
        if no_cmb:
            cltot_X -= tcls_nonoise['TT']
            cltot_Y -= tcls_nonoise['TT']
            cltot_XTY -= tcls_nonoise["TT"]
        #We need N_l for generating uncorrelated component of Y,
        #call this map Z
        #We have noise_Y = a*noise_X + noise_Z
        #then N_l^Y = a^2 N_l^X + N_l^Z
        #and <XY> = a * N_l^X = r_l * sqrt(N_l^X N_l^Y) (from defintion of r_l)
        #then we have a = r_l*sqrt(N_l^X N_l^Y) / N_l^X
        #and N_l^Z = N_l^Y - a^2 N_l^X 
        a = r * np.sqrt(Nl_tt_X*Nl_tt_Y)/Nl_tt_X
        Nl_tt_Z = Nl_tt_Y - a**2 * Nl_tt_X


        # In[10]:


        px = qe.pixelization(nside=4096)
        normal_setup = setup_recon(
            px, lmin, lmax,
            mlmax,  cl_rksz, {"TT":cltot_X}
        )
        asym_setup = setup_asym_recon(
            px, lmin, lmax,
            mlmax,  cl_rksz, cltot_X, cltot_Y,
            cltot_XY,
            do_lh=False, do_psh=False)
        asym_setup_XX = setup_asym_recon(
            px, lmin, lmax,
            mlmax,  cl_rksz, cltot_X, cltot_X,
            cltot_X,
            do_lh=False, do_psh=False)


        # In[11]:


        print(asym_setup["norm_K_XY"])
        print(len(asym_setup["N0_XYXY_K"]))


        # In[12]:


        from cmbsky.utils import get_cmb_alm_unlensed

        cl_dict = {"KK_XY" : [],
                   "KK_XX_asym" : [], #using sym_setup_XX
                   "KK_XX" : [], #using recon_setup
                   "KK_XYxi" : [],
                   "KK_XXxi" : [],
                   #"KK_XY_kszonly" : [],
                   #"KK_XX_kszonly" : [],
                   "KK_XY_noksz" : [],
                   "KK_XX_noksz" : [], 
                   "KK_XX_asym_noksz" : [],
                  }

        do_qe=True

        #do ksz only
        Xf_ksz, Yf_ksz = asym_setup["filter_X"](ksz_alm.copy()), asym_setup["filter_Y"](ksz_alm.copy())
        print("doing ksz only")
        K_XY_kszonly = asym_setup["qfunc_K_XY"](Xf_ksz, Yf_ksz)
        K_XX_kszonly = normal_setup["qfunc_K"](Xf_ksz,Xf_ksz)

        if rank==0:
            ksz_output_stuff = np.zeros((mlmax+1),
                             dtype=[("cl_KK_raw",float), ("cl_KK",float),
                              ("N0",float)])
            ksz_output_stuff["cl_KK_raw"] = curvedsky.alm2cl(K_XX_kszonly)
            ksz_output_stuff["N0"] = normal_setup["get_fg_trispectrum_K_N0"](cl_rksz)
            ksz_output_stuff["cl_KK"] = ksz_output_stuff["cl_KK_raw"] - normal_setup["N0_K"]
            np.save(opj(outdir,"cl_KK_lmin%d_lmax%d.npy"%(lmin, lmax)), 
                    ksz_output_stuff)
        
        
        # In[13]:
        for isim in range(nsim):
            if size>1:
                if isim%size != rank:
                    continue
            print("rank %d doing sim %d"%(rank,isim))
            print("generate unlensed Gaussian cmb")
            cmb_alm = curvedsky.rand_alm(tcls_nonoise["TT"],
                                         lmax=mlmax, seed=isim*(10*nsim))

            
                
            #cmb_alm = get_cmb_alm_unlensed(isim,0)[0]
            if no_cmb:
                cmb_alm*=0.
            cmb_alm = utils.change_alm_lmax(cmb_alm, mlmax)

            print("generating noise")
            noise_alm_X = curvedsky.rand_alm(Nl_tt_X, seed=isim*(10*nsim)+1)
            noise_alm_Z = curvedsky.rand_alm(Nl_tt_Z, seed=isim*(10*nsim)+2)
            noise_alm_Y = curvedsky.almxfl(noise_alm_X,a) + noise_alm_Z

            X_ksz = ksz_alm.copy()
            X = cmb_alm+noise_alm_X+ksz_alm
            Y = cmb_alm+noise_alm_Y+ksz_alm
            Xf, Yf = asym_setup["filter_X"](X), asym_setup["filter_Y"](Y)
            Xf_ksz, Yf_ksz = asym_setup["filter_X"](ksz_alm.copy()), asym_setup["filter_Y"](ksz_alm.copy())

            X_noksz = cmb_alm+noise_alm_X
            Y_noksz = cmb_alm+noise_alm_Y
            Xf_noksz, Yf_noksz = asym_setup["filter_X"](X_noksz), asym_setup["filter_Y"](Y_noksz)
            
            #X_fg_gaussian = curvedsky.rand_alm(cl_fg_X, seed=isim*(10*nsim)+3)
            #Y_fg_gaussian = 0.1*X_fg_gaussian

            print("running K estimators")
            if do_qe:
                print("qe case:")
                K_XY = asym_setup["qfunc_K_XY"](Xf, Yf)
                K_XX_asym = asym_setup_XX["qfunc_K_XY"](Xf,Xf)
                K_XX = normal_setup["qfunc_K"](Xf,Xf)

                print("running no ksz case")
                K_XY_noksz = asym_setup["qfunc_K_XY"](Xf_noksz, Yf_noksz)
                K_XX_noksz = normal_setup["qfunc_K"](Xf_noksz, Xf_noksz)
                K_XX_asym_noksz = asym_setup_XX["qfunc_K_XY"](Xf_noksz, Xf_noksz)

                #auto
                cl_dict["KK_XY"].append(binner(
                    curvedsky.alm2cl(K_XY)
                ))
                cl_dict["KK_XX_asym"].append(binner(
                    curvedsky.alm2cl(K_XX_asym)
                ))              
                cl_dict["KK_XX"].append(binner(
                    curvedsky.alm2cl(K_XX)
                ))
                cl_dict["KK_XY_noksz"].append(binner(
                    curvedsky.alm2cl(K_XY_noksz)
                ))
                cl_dict["KK_XX_noksz"].append(binner(
                    curvedsky.alm2cl(K_XX_noksz)
                ))
                cl_dict["KK_XX_asym_noksz"].append(binner(
                    curvedsky.alm2cl(K_XX_asym_noksz)
                ))
                cl_dict["KK_XYxi"].append(binner(
                    curvedsky.alm2cl(K_XY, K_XY_kszonly)
                ))
                cl_dict["KK_XXxi"].append(binner(
                    curvedsky.alm2cl(K_XX, K_XX_kszonly)
                ))


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

            cl_dict["KK_XY_kszonly"] = binner(curvedsky.alm2cl(
                K_XY_kszonly
            ))
            cl_dict["KK_XX_kszonly"] = binner(curvedsky.alm2cl(
                K_XX_kszonly
            ))
            
            cl_dict["binner_lmin"] = binner.lmin
            cl_dict["binner_lmax"] = binner.lmax
            cl_dict["binner_nbin"] = binner.nbin
            
            cl_dict["KK_XX_N0_kszonly"] = binner(normal_setup["get_fg_trispectrum_K_N0"](cl_rksz))
            cl_dict["KK_XY_N0_kszonly"] = binner(asym_setup["get_fg_trispectrum_N0_XYXY"](cl_rksz, cl_rksz, cl_rksz))

            cl_dict["N0_XYXY_K"] = binner(asym_setup["N0_XYXY_K"])
            cl_dict["N0_K"] = binner(normal_setup["N0_K"])
            
            #save pkl
            with open(opj(outdir,"cls.pkl"), 'wb') as f:
                pickle.dump(cl_dict, f)

        else:
            comm.send(cl_dict, dest=0)
            return 0
            
    else:
        if rank==0:
            print("rank 0 reading pkl")
            with open(opj(outdir, "cls.pkl"),"rb") as f:
                cl_dict = pickle.load(f)
        else:
            return 0
        
    binner = ClBinner(lmin=cl_dict["binner_lmin"], 
                      lmax=cl_dict["binner_lmax"],
                      nbin=cl_dict["binner_nbin"])

    cl_KK_binned_XX = cl_dict["KK_XX_kszonly"] - cl_dict["KK_XX_N0_kszonly"]
    cl_KK_binned_XY = cl_dict["KK_XY_kszonly"] - cl_dict["KK_XY_N0_kszonly"]
    
    #plot N0
    #fig,ax=plt.subplots()
    #ax.plot(binner.bin_mids, 

    #plot signal
    fig,ax=plt.subplots()
    cl_KK_XY = cl_dict["KK_XY"].mean(axis=0) - cl_dict["N0_XYXY_K"]
    cl_KK_XY_err = np.std(cl_dict["KK_XY"], axis=0) / np.sqrt(nsim)

    cl_KK_XX = cl_dict["KK_XX"].mean(axis=0) - cl_dict["N0_K"]
    cl_KK_XX_err = np.std(cl_dict["KK_XX"], axis=0) / np.sqrt(nsim)

    cl_KK_XYxi = cl_dict["KK_XYxi"].mean(axis=0) - cl_dict["KK_XY_N0_kszonly"]
    cl_KK_XYxi_err = np.std(cl_dict["KK_XYxi"], axis=0) / np.sqrt(nsim)

    Lfac = 2
    
    offsets = np.linspace(-5,5,6)
    ax.plot(binner.bin_mids+offsets[0], binner.bin_mids**Lfac*cl_KK_binned_XX, label="cl_KK XX ksz-only", marker='o')
    ax.plot(binner.bin_mids+offsets[1], binner.bin_mids**Lfac*cl_KK_binned_XY, label="cl_KK XY ksz-only", marker='o')
    ax.errorbar(binner.bin_mids+offsets[2], binner.bin_mids**Lfac*cl_KK_XY, yerr=binner.bin_mids**Lfac*cl_KK_XY_err, label="XY", fmt='o')
    ax.errorbar(binner.bin_mids+offsets[3], binner.bin_mids**Lfac*cl_KK_XX, yerr=binner.bin_mids**Lfac*cl_KK_XX_err, label="XX", fmt='o')
    ax.errorbar(binner.bin_mids, cl_KK_XYxi, yerr=cl_KK_XYxi_err, label="XYxi")
    
    ax.legend()
    fig.savefig(opj(outdir,"signal_XY_test_nsim%d.png"%nsim))       

    fig,ax=plt.subplots()
    ax.errorbar(binner.bin_mids, cl_KK_XY/cl_KK_binned_XY-1, yerr=cl_KK_XY_err/cl_KK_binned_XY, label="XY")
    ax.errorbar(binner.bin_mids, cl_KK_XX/cl_KK_binned_XX-1, yerr=cl_KK_XX_err/cl_KK_binned_XX, label="XX")
    ax.errorbar(binner.bin_mids, cl_KK_XYxi/cl_KK_binned_XX-1, yerr=cl_KK_XYxi_err/cl_KK_binned_XX, label="XYxi")
    ax.legend()
    fig.savefig(opj(outdir,"signal_fracbias_XY_test_nsim%d.png"%nsim))       
        

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="run tests")
    parser.add_argument("-m", "--mpi", action='store_true',
                        help="use_mpi")
    parser.add_argument("--from_pkl", action="store_true",
                        help="read cls from pkl")
    parser.add_argument("-n", "--nsim", type=int, default=10)
    parser.add_argument("--n1_file", type=str, default=None)
    parser.add_argument("--frac_bias_ylim", type=float, nargs=2, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    test_ABCD(args.nsim, use_mpi=args.mpi, from_pkl=args.from_pkl, no_cmb=False, do_lh=True,
              seed=args.seed)
    #get_N1(args.nsim, use_mpi=args.mpi)


