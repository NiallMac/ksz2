#Run the kSZ 4-point function
#
#
from os.path import join as opj, dirname
import os
os.environ["DISABLE_MPI"]="true"
from ksz4.cross import four_split_K, split_phi_to_cl
from ksz4.reconstruction import setup_recon, setup_ABCD_recon, get_cl_smooth
from pixell import curvedsky, enmap
from scipy.signal import savgol_filter
from cmbsky import safe_mkdir, get_disable_mpi, ClBinner
from falafel import utils, qe
import healpy as hp
import yaml
import argparse
from orphics import maps
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy


def fit_linear_model(x,y,ycov,funcs,dofs=None,deproject=False,Cinv=None,Cy=None):
    """
    Given measurements with known uncertainties, this function fits those to a linear model:
    y = a0*funcs[0](x) + a1*funcs[1](x) + ...
    and returns the best fit coefficients a0,a1,... and their uncertainties as a covariance matrix
    """
    s = solve if deproject else np.linalg.solve
    C = ycov
    y = y[:,None] 
    A = np.zeros((y.size,len(funcs)))
    print(A.shape)
    for i,func in enumerate(funcs):
        A[:,i] = func
    CA = s(C,A) if Cinv is None else np.dot(Cinv,A)
    cov = np.linalg.inv(np.dot(A.T,CA))
    if Cy is None: Cy = s(C,y) if Cinv is None else np.dot(Cinv,y)
    b = np.dot(A.T,Cy)
    X = np.dot(cov,b)
    YAX = y - np.dot(A,X)
    CYAX = s(C,YAX) if Cinv is None else np.dot(Cinv,YAX)
    chisquare = np.dot(YAX.T,CYAX)
    dofs = len(x)-len(funcs)-1 if dofs is None else dofs
    pte = 1 - chi2.cdf(chisquare, dofs)    
    return X,cov,chisquare/dofs,pte

def get_fg_trispectrum(tag, fg_tag, freq, est, divide_by_uL=False, lpower=0, prefactor=None):
    fg_data = np.load(opj("/pscratch/sd/m/maccrann/cmb/fg_outputs/", tag, "ksz2_fg_terms_%s"%fg_tag, "fg_terms_%s.npy"%freq))
    fg_bias = (fg_data["trispectrum_raw_meanfield_corrected_%s"%est]-fg_data["trispectrum_MCN0_%s"%est])
    if divide_by_uL:
        fg_bias /= fg_data["profile"]**2
        assert prefactor is None
    elif prefactor is not None:
        print("len(prefactor)", len(prefactor))
        print("len(fg_bias)", len(fg_bias))
        if len(prefactor)<len(fg_bias):
            fg_bias = fg_bias[:len(prefactor)]*prefactor
        else:
            fg_bias = fg_bias[:len(fg_bias)]*prefactor[:len(fg_bias)]
    Ls = np.arange(len(fg_bias))
    print("lpower:",lpower)
    fg_bias *= Ls**lpower
    return fg_bias, fg_data["profile"]


def get_CLKK_stuff(data_dir, rdn0_file, binner, w1=None, w4=None, nsim_meanfield=None, meanfield_dir=None, use_mcn0=False,
            est="qe", sim_CLKKs=None, no_cross=False, divide_by_uL=True, lpower=0, check_auto=True, transfer_func=1., 
                   use_alt_norm=False, use_r24_norm=False, correct_mf_auto=True):
    """
    Parameters
    ----------
    sim_CLKKs: string (optional)
        If not None, read array of CL_KK realizations from file,
        compute covariance, and use for errorabrs etc.
    """

    print("est:",est)
    
    outputs = {}
    with open(opj(data_dir, "auto_outputs.pkl"), "rb") as f:
        auto_outputs = pickle.load(f)
    u_L = auto_outputs["profile"]
    outputs["profile"] = u_L
    
    if w1 is None:
        w1 = auto_outputs["w1"]
    if w4 is None:
        w4 = auto_outputs["w4"]

    #Get alternative norm
    # \int d^2l / (2pi)^2 w_l w_L-l
    # \approx \int d^2l / (2pi)^2 w_l**2
    # \approx \int_lmin^lmax l dl / (2pi) w_l**2
    lmin, lmax = auto_outputs["lmin"], auto_outputs["lmax"]
    from scipy.integrate import quad
    ls = np.arange(len(u_L))
    w_l_A = u_L / auto_outputs["cltot_A"]
    w_l_B = u_L / auto_outputs["cltot_B"]
    w_l_C = u_L / auto_outputs["cltot_C"]
    w_l_D = u_L / auto_outputs["cltot_D"]
    
    if use_alt_norm:
        alt_norm_integrand_AB = ls * w_l_A * w_l_B #/ 2 / np.pi
        alt_norm_integrand_CD = ls * w_l_C * w_l_D #/ 2 / np.pi
        alt_norm_AB = 1./np.sum(alt_norm_integrand_AB[lmin:lmax])
        print("alt_norm_AB:",alt_norm_AB)
        print("norm_K_AB:",auto_outputs["norm_K_AB"])
        print("u_L:",u_L)
        alt_norm_CD = 1./np.sum(alt_norm_integrand_CD[lmin:lmax])

        outputs["alt_norm_over_orig_norm"] = ((2*u_L)**2 * alt_norm_AB * alt_norm_CD /
                                              (auto_outputs["norm_K_AB"] * auto_outputs["norm_K_CD"]))
        
    elif use_r24_norm:
        assert (not use_alt_norm)
        assert (not divide_by_uL)
        alt_norm_integrand_AB = 1./ls * w_l_A * w_l_B #/ 2 / np.pi
        alt_norm_integrand_CD = 1./ls * w_l_C * w_l_D #/ 2 / np.pi
        alt_norm_AB = 1./np.sum(alt_norm_integrand_AB[lmin:lmax])
        print("alt_norm_AB:",alt_norm_AB)
        print("norm_K_AB:",auto_outputs["norm_K_AB"])
        print("u_L:",u_L)
        alt_norm_CD = 1./np.sum(alt_norm_integrand_CD[lmin:lmax])

        outputs["alt_norm_over_orig_norm"] = ((2*u_L)**2 * alt_norm_AB * alt_norm_CD/
                                              (auto_outputs["norm_K_AB"] * auto_outputs["norm_K_CD"]))
    
    prefactor=1.
    if divide_by_uL:
        prefactor = (1./u_L)**2
    elif use_alt_norm:
        assert (divide_by_uL is False)
        print("only use altnorm when divide_by_uL is False")
        prefactor = copy.copy(outputs["alt_norm_over_orig_norm"])
    elif use_r24_norm:
        prefactor = copy.copy(outputs["alt_norm_over_orig_norm"])
        
    if lpower != 0:
        Ls = np.arange(len(u_L))
        prefactor *= Ls**lpower
    outputs["prefactor"] = prefactor

    if est=="lh":
        outputs["CL_KK_raw"] = prefactor * auto_outputs["cl_KK_lh_raw"]
        K_ab_file=opj(data_dir,"K_ab_lh.npy")
        K_cd_file=opj(data_dir,"K_cd_lh.npy")

    elif est=="psh":
        outputs["CL_KK_raw"] = prefactor * auto_outputs["cl_KK_psh_raw"]
        K_ab_file=opj(data_dir,"K_ab_psh.npy")
        K_cd_file=opj(data_dir,"K_cd_psh.npy")
    elif est=="qe":
        outputs["CL_KK_raw"] = prefactor * auto_outputs["cl_KK_raw"]
        K_ab_file=opj(data_dir,"K_ab.npy")
        K_cd_file=opj(data_dir,"K_cd.npy")
    else:
        raise ValueError("invalid est:", est)
    
    if os.path.isfile(K_ab_file):
        Ks_ab_raw = np.load(K_ab_file)
    else:
        print("didn't find %s, will try .pkl instead"%K_ab_file)
        K_ab_file_pkl = K_ab_file.replace(".npy",".pkl")
        with open(K_ab_file_pkl, "rb") as f:
            Ks_ab_raw = pickle.load(f)
    if os.path.isfile(K_cd_file):
        Ks_cd_raw = np.load(K_cd_file)
    else:
        print("didn't find %s, will try .pkl instead"%K_cd_file)
        K_cd_file_pkl = K_cd_file.replace(".npy",".pkl")
        with open(K_cd_file_pkl, "rb") as f:
            Ks_cd_raw = pickle.load(f)              
    
    if np.any(np.isnan(outputs["CL_KK_raw"])):
        print("nans in CL_KK_raw")
        
    outputs["CL_KK_raw_binned"] = binner(outputs["CL_KK_raw"])
    theory_N0_wnoise = prefactor * auto_outputs["N0"+("_%s"%est if (est!="qe") else "")]
    if no_cross:
        CL_KK_raw_from_Ks = prefactor * curvedsky.alm2cl(Ks_ab_raw, Ks_cd_raw)/w4
    else:
        CL_KK_raw_from_Ks = prefactor * split_phi_to_cl(Ks_ab_raw, Ks_cd_raw)/w4

    print(np.abs(CL_KK_raw_from_Ks[binner.lmin:binner.lmax] / outputs["CL_KK_raw"][binner.lmin:binner.lmax]-1) > 1.e-2)

    if check_auto:
        try:
            assert np.allclose(CL_KK_raw_from_Ks[binner.lmin:binner.lmax], outputs["CL_KK_raw"][binner.lmin:binner.lmax], rtol=1.e-3)
        except AssertionError as e:
            print("mismatch in CL_KK_raw:")
            print("from auto_outputs.pkl:", (outputs["CL_KK_raw"]).shape, outputs["CL_KK_raw"])
            print("from K files:",CL_KK_raw_from_Ks.shape, CL_KK_raw_from_Ks)
            print(CL_KK_raw_from_Ks/outputs["CL_KK_raw"])
            frac_diff = (CL_KK_raw_from_Ks[:binner.lmax]-outputs["CL_KK_raw"][:binner.lmax])/outputs["CL_KK_raw"][:binner.lmax]
            print((np.abs(frac_diff)).max())
            print(frac_diff)
            print(opj(data_dir, "auto_outputs.pkl"))
            print(K_ab_file, K_cd_file)
            print(opj(data_dir, "auto_outputs.pkl"))
            raise(e)

        
    #Do meanfield 
    if meanfield_dir is None:
        meanfield_dir = "mean_field_nsim%d"%nsim_meanfield
        
    if not os.path.isabs(meanfield_dir):
        meanfield_dir = opj(data_dir, meanfield_dir)
    print("reading meanfield from %s"%meanfield_dir)
    
    with open(opj(meanfield_dir,"Ks_ab_mean.pkl"), "rb") as f:
        Ks_ab_mf=pickle.load(f)
    with open(opj(meanfield_dir,"Ks_cd_mean.pkl"), "rb") as f:
        Ks_cd_mf=pickle.load(f)     
        
    if no_cross:
        Ks_ab = Ks_ab_raw - Ks_ab_mf
        Ks_cd = Ks_cd_raw - Ks_cd_mf
    else:
        Ks_ab, Ks_cd = [], []
        for i in range(len(Ks_ab_raw)):
            Ks_ab.append( np.array(Ks_ab_raw[i]) - np.array(Ks_ab_mf[i]))
            Ks_cd.append( np.array(Ks_cd_raw[i]) - np.array(Ks_cd_mf[i]))

    #meanfield bias
    if no_cross:
        outputs["CL_KK_mfcorrected"] = prefactor * curvedsky.alm2cl(Ks_ab, Ks_cd) / w4 / transfer_func
        outputs["meanfield_auto"] = prefactor * curvedsky.alm2cl(Ks_ab_mf.copy(), Ks_cd_mf.copy()) / w4 / transfer_func
    else:
        outputs["CL_KK_mfcorrected"] = prefactor * split_phi_to_cl(Ks_ab, Ks_cd) / w4 / transfer_func
        outputs["meanfield_auto"] = prefactor * split_phi_to_cl(Ks_ab_mf, Ks_cd_mf) / w4 / transfer_func
    outputs["meanfield_correction"] = -(outputs["CL_KK_mfcorrected"]-outputs["CL_KK_raw"])


    if np.any(np.isnan(outputs["meanfield_auto"])):
        print("nans in meanfield")
    
    #Get rdn0
    if not os.path.isabs(rdn0_file):
        rdn0_file = opj(data_dir, rdn0_file)
    print("reading rdn0 from %s"%rdn0_file)
    with open(rdn0_file, 'rb') as f:
        rdn0_outputs = pickle.load(f)

    outputs["rdn0"] = prefactor * rdn0_outputs["rdn0"].mean(axis=0) / transfer_func
    outputs["mcn0"] = prefactor * rdn0_outputs["mcn0"].mean(axis=0) / transfer_func
    outputs["rdn0_binned"] = binner(outputs["rdn0"])
    outputs["mcn0_binned"] = binner(outputs["mcn0"])
    
    if np.any(np.isnan(outputs["rdn0"])):
        print("nans in rdn0")
    
    #get binned errors on rdn0/mcn0
    nsim=rdn0_outputs["rdn0"].shape[0]
    print(nsim)
    binned_rdn0s = np.zeros((rdn0_outputs["rdn0"].shape[0], binner.nbin))
    binned_mcn0s = np.zeros_like(binned_rdn0s)
    for i in range(rdn0_outputs["rdn0"].shape[0]):
        binned_rdn0s[i] = binner(prefactor * rdn0_outputs["rdn0"][i] / transfer_func)
        binned_mcn0s[i] = binner(prefactor * rdn0_outputs["mcn0"][i] / transfer_func)
    rdn0_err = np.std(binned_rdn0s, axis=0)/np.sqrt(nsim-1)
    mcn0_err = np.std(binned_mcn0s, axis=0)/np.sqrt(nsim-1)
    outputs["rdn0_bined_err"] = rdn0_err
    outputs["mcn0_err"] = mcn0_err
    
    
    if use_mcn0:
        CL_KK = outputs["CL_KK_mfcorrected"]-outputs["mcn0"]
    else:
        CL_KK = outputs["CL_KK_mfcorrected"]-outputs["rdn0"]

        
    outputs["CL_KK"] = CL_KK
    if correct_mf_auto:
        mf_auto_correction = outputs["mcn0"] / nsim_meanfield
    outputs["CL_KK"] -= mf_auto_correction
    outputs["meanfield_auto"] -= mf_auto_correction
    
    
    
    CL_KK_binned = binner(CL_KK)
    outputs["CL_KK_binned"] = CL_KK_binned
    outputs["CL_KK_mfcorrected_binned"] = binner(outputs["CL_KK_mfcorrected"])
    outputs["meanfield_auto_binned"] = binner(outputs["meanfield_auto"])
    outputs["meanfield_correction_binned"] = binner(outputs["meanfield_correction"])
        
    #also get errorbars
    CL_KK_cov_use = None
    CK_KK_cov_binned_use = None
    if sim_CLKKs is not None:
        sim_CLKKs_div_uL2 = np.zeros_like(sim_CLKKs)
        sim_CLKKs_binned = np.zeros((sim_CLKKs.shape[0], binner.nbin))
        for i, c in enumerate(sim_CLKKs):
            sim_CLKKs_binned[i] = binner(prefactor * c) / transfer_func
            sim_CLKKs_div_uL2[i] = prefactor * c / transfer_func
        outputs["CL_KK_cov_sim_binned"] = np.cov(sim_CLKKs_binned.T)
        print(outputs["CL_KK_cov_sim_binned"].shape)
        CL_KK_cov_use = np.cov(sim_CLKKs_div_uL2.T)
        CK_KK_cov_binned_use = outputs["CL_KK_cov_sim_binned"].copy()
        outputs["CK_KK_cov_sim_binned"] = CK_KK_cov_binned_use
        
    #theory errors for comparison
    Ls = np.arange(binner.lmax+1)
    clKK_ksz = prefactor * auto_outputs["Cl_KK_ksz_theory"]
    outputs["CL_KK_ksz_theory"] = clKK_ksz
    outputs["CL_KK_ksz_theory_binned"] = binner(clKK_ksz)
    clKK2 = (theory_N0_wnoise[:binner.lmax+1] + 
             clKK_ksz[:binner.lmax+1])**2
    var_CLKK = 2/(2*Ls+1)/w1 * clKK2
    outputs["var_CLKK_theory"] = var_CLKK
    var_CLKK_binned = np.zeros_like(binner.bin_mids)
    for i in range(binner.nbin):
        l_inds = np.arange(binner.bin_lims[i], binner.bin_lims[i+1])
        wl = (2*Ls[l_inds]+1)
        var_CLKK_binned[i] = (
            (wl**2 * var_CLKK[l_inds]).sum() / (wl.sum())**2
        )
    outputs["var_CLKK_theory_binned"] = var_CLKK_binned
    if CL_KK_cov_use is None:
        CL_KK_cov_use = np.diag(var_CLKK)
        CK_KK_cov_binned_use = np.diag(var_CLKK_binned)
    
    #And do amplitude fit while we're at it 
    bias_over_sig = (CL_KK-clKK_ksz)[:binner.lmax+1]/np.sqrt(np.diag(CL_KK_cov_use))[:binner.lmax+1]
    outputs["bias_over_sig"] = bias_over_sig
    outputs["frac_diff"] = (CL_KK/clKK_ksz-1)
    
    denom = np.sum( (clKK_ksz*clKK_ksz)[binner.lmin:binner.lmax+1] / 
                   np.diag(CL_KK_cov_use)[binner.lmin:binner.lmax+1] )
    num = np.sum( (CL_KK * clKK_ksz)[binner.lmin:binner.lmax+1] / 
                 np.diag(CL_KK_cov_use)[binner.lmin:binner.lmax+1] )
    outputs["A_ksz"]= num/denom
    outputs["A_ksz_err"] = np.sqrt(1./denom)
    
    #also check binned amplitude
    denom_binned = np.sum( binner(clKK_ksz)**2 / np.diag(CK_KK_cov_binned_use) )
    num_binned = np.sum( binner(clKK_ksz)*CL_KK_binned / np.diag(CK_KK_cov_binned_use) )
    outputs["A_ksz_binned"] = num_binned/denom_binned
    outputs["A_ksz_binned_err"] = np.sqrt(1./denom_binned)
    
    return outputs

def get_rdn0_scatter(rdn0_array, binner):
    binned_rdn0_array = np.zeros((rdn0_array.shape[0], binner.nbin))
    for i,a in enumerate(rdn0_array):
        binned_rdn0_array[i] = binner(a)
    return np.std(binned_rdn0_array, axis=0)

def plot_clKK(ax, outputs_dir, binner, mask=None, color=None, label=None, 
              lpower=0, raw_auto=False, yscale="linear", clKK_signal=None, 
              do_err=True, nsim_meanfield=64, meanfield_dir=None, plot_rdn0=False, plot_mcn0=False, rdn0_file="rdn0_outputs.pkl",
              add_to_err=None, mult_offset=None, add_offset=None, plot_meanfield=False, 
              use_raw_auto_for_err=False, divide_by_uL=True, plot_theory=True,
             est="qe", marker='o', raw_auto_marker='s', plot_raw_auto=False, plot_meanfield_correction=False,
             sim_CLKKs_file=None, no_cross=False, plot_scaled_theory=False, check_auto=True,
             transfer_func=1., use_alt_norm=False, use_r24_norm=False):

    #mask=enmap.read_map(mask_file)
    if mask is not None:
        try:
            w4 = maps.wfactor(4, mask)
            w1 = maps.wfactor(1, mask)
        except AttributeError:
            mask = enmap.read_map(mask)
            w4 = maps.wfactor(4, mask)
            w1 = maps.wfactor(1, mask)
    else:
        w1, w4 = None, None

    if sim_CLKKs_file is not None:
        sim_CLKKs = np.load(sim_CLKKs_file)
    else:
        sim_CLKKs = None
    
    cl_kk_stuff = get_CLKK_stuff(outputs_dir, rdn0_file, binner, w1=w1, w4=w4, nsim_meanfield=nsim_meanfield,
                                 meanfield_dir=meanfield_dir, use_mcn0=False,
            est=est, sim_CLKKs=sim_CLKKs, no_cross=no_cross, divide_by_uL=divide_by_uL, lpower=lpower,
                                check_auto=check_auto, transfer_func=transfer_func, use_alt_norm=use_alt_norm,
                                use_r24_norm=use_r24_norm)
    if sim_CLKKs is None:
        CL_KK_err_binned = np.sqrt(cl_kk_stuff["var_CLKK_theory_binned"])
    else:
        CL_KK_err_binned = np.sqrt(np.diag(cl_kk_stuff["CK_KK_cov_sim_binned"]))
        
    x_vals = binner.bin_mids
    if mult_offset is not None:
        x_vals *= mult_offset
    if add_offset is not None:
        x_vals += add_offset
        
    label += "\n $A_\mathrm{A16} = %.1f \pm %.1f$"%(
        cl_kk_stuff["A_ksz_binned"], cl_kk_stuff["A_ksz_binned_err"])
        
    if yscale=="log":
        is_pos = cl_kk_stuff["CL_KK_binned"] > 0
        ax.errorbar(x_vals[is_pos], (cl_kk_stuff["CL_KK_binned"])[is_pos], 
                    yerr=(CL_KK_err_binned)[is_pos], fmt=marker,color=color, label=label)
        ax.errorbar(x_vals[~is_pos], -(cl_kk_stuff["CL_KK_binned"])[~is_pos], 
                        yerr=(CL_KK_err_binned)[~is_pos], fmt=marker, mfc='none', color=color)
    else:
        ax.errorbar(x_vals, cl_kk_stuff["CL_KK_binned"], 
                    yerr=CL_KK_err_binned, fmt=marker,color=color, label=label)
        
    if plot_theory:
        ax.plot(binner.bin_mids, 
                cl_kk_stuff["CL_KK_ksz_theory_binned"], 
                linestyle="-", color="k")
        
    if plot_scaled_theory:
        ax.plot(binner.bin_mids, cl_kk_stuff["CL_KK_ksz_theory_binned"]*cl_kk_stuff["A_ksz_binned"],
               linestyle="--", color="k")
            
    if plot_rdn0:
        ax.plot(binner.bin_mids, cl_kk_stuff["rdn0_binned"],
                linestyle="--", color=color)
        if yscale == "log":
            ax.plot(binner.bin_mids, -cl_kk_stuff["rdn0_binned"],
                    linestyle="--", color=color)
            
    if plot_mcn0:
        ax.plot(binner.bin_mids, cl_kk_stuff["mcn0_binned"],
                linestyle="-.", color=color)

    if plot_meanfield:
        ax.plot(binner.bin_mids, cl_kk_stuff["meanfield_auto_binned"], color=color, linestyle=":")
            
    if plot_meanfield_correction:
        print("meanfield correction:", cl_kk_stuff["meanfield_correction_binned"])
        ax.plot(binner.bin_mids, cl_kk_stuff["meanfield_correction_binned"], color=color, linestyle=(0, (3, 1, 1, 1)))
            
    if plot_raw_auto:
        ax.plot(x_vals, cl_kk_stuff["CL_KK_raw_binned"], raw_auto_marker, mfc="none", color=color)
            
    if yscale is not None:
        ax.set_yscale(yscale)
    return cl_kk_stuff
    
def plot_rdn0(ax, outputs_dir, binner, color=None, label=None, 
              lfac=2, tag=None, scatter=False, linestyle='-', yscale="linear",
              mcn0_linestyle=None):


    #read N0 output
    n0_output_file = opj("../scripts/",outputs_dir, "rdn0_outputs.pkl")
    with open(n0_output_file, "rb") as f:
        rdn0_outputs = pickle.load(f)
        
    if scatter:
        rdn0_scatter = get_rdn0_scatter(rdn0_outputs["rdn0"],
                                        binner)
        mcn0_scatter = get_rdn0_scatter(rdn0_outputs["mcn0"],
                                        binner)
        ax.plot(binner.bin_mids, binner.bin_mids**lfac*rdn0_scatter, '-', color=color, label=label)
        #ax.plot(binner.bin_mids, binner.bin_mids**lfac*mcn0_scatter, '--', color=color, label=label)
    else:
        ax.plot(binner.bin_mids, binner.bin_mids**lfac*binner(rdn0_outputs["rdn0"].mean(axis=0)), linestyle=linestyle, color=color, label=label)
        if mcn0_linestyle is not None:
            ax.plot(binner.bin_mids, binner.bin_mids**lfac*binner(rdn0_outputs["mcn0"].mean(axis=0)), linestyle=mcn0_linestyle, color=color, label=label)
        #ax.plot(binner.bin_mids, -binner.bin_mids**lfac*binner(rdn0_outputs["mcn0"].mean(axis=0)), '--', color=color, label=label)
    ax.set_yscale(yscale)
    
    
def plot_meanfield(ax, outputs_dir, meanfield_dir, binner, color=None, label=None,
                  lfac=2, linestyle='-'):
    
    with open(opj(outputs_dir, meanfield_dir, "Ks_ab_mean.pkl"), "rb") as f:
        Ks_ab_mf = pickle.load(f)
    with open(opj(outputs_dir, meanfield_dir, "Ks_cd_mean.pkl"), "rb") as f:
        Ks_cd_mf = pickle.load(f)

    CL_KK_meanfield = split_phi_to_cl(Ks_ab_mf,Ks_cd_mf)
    
    CL_KK_meanfield_binned = binner(CL_KK_meanfield)
    
    ax.plot(binner.bin_mids, binner.bin_mids**lfac*CL_KK_meanfield_binned)
    
def plot_e2e_sims(axs, output_path_template, meanfield_dir, rdn0_file, binner, est="qe", nsim=64):
    ax,ax_hist=axs
    frac_diffs=[]
    bias_over_sigs=[]
    amps = []
    cl_kks = []

    nsim_read=0
    for irot in range(nsim):
        print("isim:",irot)
        d=output_path_template%irot
        try:
            with open(opj(d, "auto_outputs.pkl"), "rb") as f:
                pickle.load(f)
        except Exception as e:
            print(e)
            print("skipping irot=%d"%irot)
            continue
        nsim_read+=1

        cl_kk_stuff = get_CLKK_stuff(
                d, meanfield_dir, rdn0_file, w1, w4, binner, use_mcn0=True,
                est=est
        )
        cl_kks.append(cl_kk_stuff["CL_KK"])
        bias_over_sigs.append( cl_kk_stuff["bias_over_sig"] )
        frac_diffs.append( cl_kk_stuff["frac_diff"] )

        #ax.plot(binner.bin_mids, CL_KK / binner(auto_outputs["Cl_KK_ksz_theory"])-1)
        ax.plot(binner.bin_mids, binner(cl_kk_stuff["bias_over_sig"]))

        amps.append(cl_kk_stuff["A_ksz"])

        print("A_ksz = %.6f +/- %.6f"%(cl_kk_stuff["A_ksz"], cl_kk_stuff["A_ksz_err"]))
        print("binned A_ksz = %.6f +/- %.6f"%(cl_kk_stuff["A_ksz_binned"], cl_kk_stuff["A_ksz_binned_err"]))


    #ax.plot(binner.bin_mids, binner.bin_mids**2*binner(auto_outputs["Cl_KK_ksz_theory"]), color="k")
    frac_diffs = np.array(frac_diffs)
    bias_over_sigs = np.array(bias_over_sigs)

    mean_frac_diff = frac_diffs.mean(axis=0)
    mean_frac_diff_err = np.std(frac_diffs, axis=0)/np.sqrt(nsim_read-1)

    #ax.fill_between(binner.bin_mids, mean_frac_diff-mean_frac_diff_err, mean_frac_diff+mean_frac_diff_err, 
    #               color="k", alpha=0.25)

    ax_hist.hist(np.array(amps)-1)
    ax_hist.set_xlabel("$A_{ksz} - 1$")
    A_mean, A_err = np.mean(amps), np.std(amps)/np.sqrt(nsim)
    ax_hist.set_title(r"$\bar{A}_{ksz} = %.3f \pm %.3f, \sigma(A)=%.3f$"%(A_mean, A_err, np.std(amps)))

    #ax.set_yscale('symlog', linthreshy=10.)
    ax.set_xscale('log')
    ax.set_xlabel("$L$")
    ax.set_ylabel("bias/sigma in $C_L^{KK}$")
    
    return amps, A_mean, A_err, cl_kks
