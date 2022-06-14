import numpy as np
import healpy as hp
import pytempura
from pytempura import norm_general, noise_spec
import os
from falafel import utils as futils, qe
from orphics import maps
from pixell import lensing, curvedsky
import matplotlib.pyplot as plt
from os.path import join as opj
import pickle
import solenspipe

try:
    WEBSKY_DIR=os.environ["WEBSKY_DIR"]
except KeyError:
    WEBSKY_DIR="/global/project/projectdirs/act/data/maccrann/websky"
try:
    SEHGAL_DIR=os.environ["SEHGAL_DIR"]
except KeyError:
    SEHGAL_DIR="/global/project/projectdirs/act/data/maccrann/sehgal"


def filter_T(T_alm, cltot, lmin, lmax):
    """                                                                                                                                                                                                    
    filter by 1/cltot within lmin<=l<=lmax                                                                                                                                                                 
    return zero otherwise                                                                                                                                                                                  
    """
    mlmax=qe.get_mlmax(T_alm)
    filt = np.zeros_like(cltot)
    ls = np.arange(filt.size)
    assert lmax<=ls.max()
    assert lmin<lmax
    filt[2:] = 1./cltot[2:]
    filt[ls<lmin] = 0.
    filt[ls>lmax] = 0.
    return curvedsky.almxfl(T_alm.copy(), filt)

def dummy_teb(alms):
    return [alms, np.zeros_like(alms), np.zeros_like(alms)]

def setup_assym_recon(px, lmin, lmax, mlmax,
                      cl_rksz, tcls_X, tcls_Y,
                      tcls_XY, do_lh=False,
                      do_psh=False):

    outputs = {}
    #CMB theory for filters
    ucls,_ = futils.get_theory_dicts(grad=True,
                                    lmax=mlmax)
    outputs["ucls"] = ucls
    #Get qfunc and normalization
    #profile is Cl**0.5
    profile = cl_rksz**0.5
    outputs["profile"] = profile

    def filter_X(X):
        return filter_T(X, cltot_X, lmin, lmax)
    output["filter_X"] = filter_X
    def filter_Y(Y):
        return filter_T(Y, cltot_Y, lmin, lmax)
    output["filter_Y"] = filter_Y
    return
    

def setup_recon(px, lmin, lmax, mlmax,
                cl_rksz, tcls_X, tcls_Y=None,
                tcls_XY=None, do_lh=False,
                do_psh=False):

    outputs = {}
    
    #CMB theory for filters
    ucls,_ = futils.get_theory_dicts(grad=True,
                                    lmax=mlmax)
    outputs["ucls"] = ucls
    #Get qfunc and normalization
    #profile is Cl**0.5
    profile = cl_rksz**0.5
    outputs["profile"] = profile

    def filter_alms_X(alms):
        if len(alms)!=3:
            alms = dummy_teb(alms)
        alms_filtered = futils.isotropic_filter(alms,
                tcls_X, lmin, lmax, ignore_te=True)
        return alms_filtered
    outputs["filter_alms_X"] = filter_alms_X
    
    if tcls_Y is None:
        norm_K = pytempura.get_norms(
            ['src'], ucls, tcls_X,
            lmin, lmax, k_ellmax=mlmax,
            profile=profile)['src']
        norm_K[0]=0.
        outputs["norm_K"] = norm_K

        #The Smith and Ferraro estimator Ksf differs
        #slightly from our profile estimator,K
        #Ksf = 2*profile*K (both unnormalised)
        #So normalizations should be related via:
        norm_Ksf = norm_K / 2 / profile
        outputs["norm_Ksf"] = norm_Ksf

        #For the normalized estimator this
        #is also the N0.
        N0_K = norm_K
        outputs["N0_K"] = N0_K
        
        #otherwise, it is
        N0_K_nonorm = 1./norm_K

        print('getting point-source norm')
        print(np.any(tcls_X['TT']==0.))
        print(np.any(~np.isfinite(tcls_X['TT'])))
        norm_ps = pytempura.get_norms(
            ['src'], ucls, tcls_X,
            lmin, lmax, k_ellmax=mlmax)['src']
        outputs["norm_ps"] = norm_ps

        #Also get lensing norms and cross-response for
        #lensing hardening
        print('getting lensing norm')
        norm_phi = pytempura.norm_lens.qtt(
            mlmax, lmin, lmax, ucls['TT'],
            tcls_X['TT'], gtype='')[0]
        outputs["norm_phi"] = norm_phi

        print('getting source-lens response')
        print(tcls_X['TT'])
        print(profile)
        R_K_phi = pytempura.get_cross(
            'SRC','TT', ucls, tcls_X,
            lmin, lmax, k_ellmax=mlmax,
            profile=profile)
        outputs["R_K_phi"] = R_K_phi

        print('getting source-point-source response')
        R_K_ps = (1./pytempura.get_norms(
            ['src'], ucls, tcls_X,
            lmin, lmax, k_ellmax=mlmax,
            profile = profile**0.5)['src'])
        R_K_ps[0] = 0.
        outputs["R_K_ps"] = R_K_ps

        #unnormalized source estimator
        def qfunc_prof(X, Y):
            return qe.qe_source(px,mlmax,profile=profile,
                             fTalm=Y,xfTalm=X)

        #Smith/Ferraro estimator (x 2!)
        def qfunc_Ksf_nonorm(X,Y):
            s_nonorm = qfunc_prof(X, Y)
            K_nonorm = curvedsky.almxfl(s_nonorm, norm_Ksf)
            return K_nonorm

        def qfunc_K(X,Y):
            K_nonorm = qfunc_prof(X,Y)
            return curvedsky.almxfl(
                K_nonorm, norm_K
                )
        outputs["qfunc_K"] = qfunc_K
        qfunc_K_incfilter = lambda X,Y: qfunc_K(filter_alms_X(X),
                                                filter_alms_X(Y))
        outputs["qfunc_K_incfilter"] = qfunc_K_incfilter

        def get_fg_trispectrum_K_N0(cl_fg):
            #N0 is (A^K)^2 / A_fg (see eqn. 9 of 1310.7023
            #for the normal estimator, and a bit more complex for
            #the bias hardened case
            Ctot = tcls_X['TT']**2 / cl_fg
            norm_fg = pytempura.get_norms(
                ['src'], ucls, {"TT" : Ctot},
                lmin, lmax, k_ellmax=mlmax,
                profile=profile)['src']
            
            return N0_K**2 / norm_fg
        outputs["get_fg_trispectrum_K_N0"] = get_fg_trispectrum_K_N0

        #qfunc for lensing
        qfunc_phi_nonorm = lambda X,Y: qe.qe_all(
            px,ucls,mlmax,fTalm=Y,fEalm=None,fBalm=None,
            estimators=['TT'], xfTalm=X, xfEalm=None,
            xfBalm=None)['TT']

        def qfunc_phi(X,Y):
            phi = qfunc_phi_nonorm(X,Y)
            return curvedsky.almxfl(phi[0], norm_phi)
        outputs["qfunc_phi"] = qfunc_phi
        
        def qfunc_lh(X,Y):
            #(K_est  )  = ( 1       A^K R^K\phi )(K   )
            #(\phi_est)    ( A^\phi R^K\phi    1 )(\phi)
            # So bias-hardened K_BH given by
            # K_BH = (K_est - A^K R^K\phi \phi_est) / (1 - A^K*A^\phi*(R^K\phi)^2)
            K = qfunc_K(X,Y)
            #s = curvedsky.almxfl(qfunc_prof(X,Y), norm_prof)
            phi = qfunc_phi(X,Y)
            #phi = curvedsky.almxfl(
            #    qfunc_lens_nonorm(X,Y)[0], norm_lens) #0th element is gradient
            num = K - curvedsky.almxfl(
                phi, norm_K * R_K_phi)
            denom = 1 - norm_K * norm_phi * R_K_phi**2
            s_normed_lh = curvedsky.almxfl(num, 1./denom)
            K_normed_lh = curvedsky.almxfl(s_normed_lh, 1./profile)
            return K_normed_lh
        outputs["qfunc_K_lh"] = qfunc_lh
        qfunc_lh_incfilter = lambda X,Y: qfunc_lh(filter_alms_X(X),
                                                  filter_alms_X(Y))
        outputs["qfunc_lh_incfilter"] = qfunc_lh_incfilter

        
        N0_K_lh = N0_K / (1 - norm_K * norm_phi * R_K_phi**2)
        outputs["N0_K_lh"] = N0_K_lh
        def get_fg_trispectrum_K_N0_lh(cl_fg):
            Ctot = tcls_X['TT']**2 / cl_fg
            norm_fg = pytempura.get_norms(
                ['src'], ucls, {"TT" : Ctot},
                lmin, lmax, k_ellmax=mlmax,
                profile=profile)['src']
            norm_phi_fg = pytempura.get_norms(
                ['TT'], ucls, {'TT':Ctot},
                lmin, lmax,
                k_ellmax=mlmax)['TT'][0]
            R_K_phi_fg = pytempura.get_cross(
                'SRC','TT', ucls, {'TT':Ctot},
                lmin, lmax,
                k_ellmax=mlmax, profile=profile)
            N0 = N0_K**2/norm_fg
            N0_phi = norm_phi**2/norm_phi_fg
            N0_tri = (N0 + R_K_phi**2 * norm_K**2 * N0_phi
                      - 2 * R_K_phi * norm_K**2 * norm_phi * R_K_phi_fg)
            N0_tri /= (1 - norm_K*norm_phi*R_K_phi**2)**2
            return N0_tri
        outputs["get_fg_trispectrum_K_N0_lh"] = get_fg_trispectrum_K_N0_lh
        
        def qfunc_ps_nonorm(X,Y):
            return qe.qe_source(px, mlmax,
                                fTalm=Y,xfTalm=X)
        def qfunc_ps(X,Y):
            ps = qfunc_ps_nonorm(X, Y)
            return qfunc_ps(ps, norm_ps)
        outputs["qfunc_ps"] = qfunc_ps

        def qfunc_psh(X,Y):
            K = qfunc_K(X,Y)
            ps = qfunc_ps(X,Y)
            num = K - curvedsky.almxfl(ps, norm_K*R_K_phi)
            denom = 1 - norm_K * norm_phi * R_K_phi
            K_psh = curvedsky.almxfl(num, 1./denom)
            return K_psh
        outputs["qfunc_K_psh"] = qfunc_psh
        qfunc_psh_incfilter = lambda X,Y: qfunc_psh(filter_alms_X(X),
                                                  filter_alms_X(Y))
        outputs["qfunc_psh_incfilter"] = qfunc_psh_incfilter

        N0_K_psh = N0_K / (1 - norm_K * norm_ps * R_K_ps**2)
        outputs["N0_K_psh"] = N0_K_psh


        def get_fg_trispectrum_K_N0_psh(cl_fg):
            Ctot = tcls_X['TT']**2 / cl_fg
            norm_fg = pytempura.get_norms(
                ['src'], ucls, {"TT" : Ctot},
                lmin, lmax, k_ellmax=mlmax,
                profile=profile)['src']
            norm_ps_fg = pytempura.get_norms(
                ['src'], ucls, {'TT':Ctot},
                lmin, lmax,
                k_ellmax=mlmax)['src'][0]
            R_K_ps_fg = (1./pytempura.get_norms(
                ['src'], ucls, {'TT':Ctot},
                lmin, lmax, k_ellmax=mlmax,
                profile = profile**0.5)['src'])
            R_K_ps_fg[0] = 0.
            N0 = N0_K**2/norm_fg
            N0_ps = norm_ps**2/norm_ps_fg
            N0_tri = (N0 + R_K_ps**2 * norm_K**2 * N0_ps
                      - 2 * R_K_ps * norm_K**2 * norm_ps * R_K_ps_fg)
            N0_tri /= (1 - norm_K*norm_ps*R_K_ps**2)**2
            return N0_tri
        outputs["get_fg_trispectrum_K_N0_psh"] = get_fg_trispectrum_K_N0_psh

        return outputs

    
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
                cmb_alms = futils.change_alm_lmax(cmb_alms.astype(np.complex128), mlmax)
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

        outputs["get_sim_alm"] = get_sim_alm
        
        recon_stuff = {"shape" : shape,
                       "wcs" : wcs,
                       "norm_prof" : norm_prof,
                       "norm_K" : norm_K,
                       "ucls" : ucls,
                       "tcls" : tcls,
                       "qfunc_Ksf_nonorm" : qfunc_Ksf_nonorm,
                       "qfunc_K" : qfunc_K,
                       "qfunc_lh_sf" : qfunc_lh,
                       "qfunc_lh_normed" : qfunc_lh_normed,
                       "qfunc_psh_sf" : qfunc_psh,
                       "qfunc_psh_normed" : qfunc_psh_normed,
                       "qfunc_lens" : qfunc_lens,
                       "filter_alms" : filter_alms,
                       "get_sim_alm" : get_sim_alm,
                       "cl_total":cl_total,
                       "cl_kszr": cl_smooth,
                       "cl_total_cmb": cl_total_cmb,
                       "profile": profile,
                       "norm_ps": norm_ps,
                       "norm_lens" : norm_lens,
                       "R_prof_ps" : R_prof_ps,
                       "R_prof_tt" : R_prof_tt,
                       "N0_K_normed" : N0_K_normed,
                       "N0_K_nonorm" : N0_K_nonorm,
                       "N0_K_lh_normed" : N0_K_lh_normed,
                       "N0_K_lh_nonorm" : N0_K_lh_nonorm,
                       "N0_K_psh_normed" : N0_K_psh_normed,
                       "N0_K_psh_nonorm" : N0_K_psh_nonorm,
            }

        return recon_stuff
