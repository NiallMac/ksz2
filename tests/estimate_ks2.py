from __future__ import print_function
import os,sys
from os.path import join as opj
from orphics import maps,io,cosmology,stats,pixcov
from pixell import enmap,curvedsky,utils,enplot
from solenspipe.bias import mcrdn0
import numpy as np
import healpy as hp
from falafel import qe
import falafel.utils as futils
import matplotlib.pyplot as plt
from orphics import maps
from websky_model import WebSky
import astropy.io.fits as afits
import pytempura
import numpy as np
from scipy.signal import savgol_filter
import errno

output_dir="estimate_ks2_output"
def safe_mkdir(d):
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise(e)
safe_mkdir(output_dir)

def get_disable_mpi():
    try:
        disable_mpi_env = os.environ['DISABLE_MPI']
        disable_mpi = True if disable_mpi_env.lower().strip() == "true" else False
    except:
        disable_mpi = False
    return disable_mpi

lmin=3000
lmax=5000
mlmax=6000
ells = np.arange(mlmax+1)

#Read alms and get (smooth) Cl for filter
#and rdn0
alm_file = 'alms_4e3_2048_50_50_ksz.fits'
alms=hp.fitsfunc.read_alm(alm_file)
alm_lmax=hp.Alm.getlmax(len(alms))
if alm_lmax>mlmax:
    alms = futils.change_alm_lmax(
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

#Get qfunc and normalization
#profile is Cl**0.5
profile = cl_smooth**0.5
print(np.any(np.isnan(profile)))
px = qe.pixelization(nside=4096)
ucls,tcls = futils.get_theory_dicts(grad=False)
norm = pytempura.get_norms(
    ['src'], ucls, tcls,
    lmin, lmax, k_ellmax=mlmax,
    profile=profile)['src']
norm[0]=0.

z = np.zeros_like(alms)
def filter_alms(alms):
    return futils.isotropic_filter(
        (alms,z,z), tcls, lmin, lmax,
        ignore_te=True)[0]
    
def qfunc(X, Y):
    k = qe.qe_source(px,mlmax,profile=profile,
                     fTalm=Y,xfTalm=X)
    return curvedsky.almxfl(k,norm)

def get_sim_alm(seed):
    #generate alms
    #seed has form (icov, iset, inoise)
    #we just need one number - the following
    #should ensure we get unique realizations
    #for rdn0
    seed = seed[1]*9999+seed[2]
    sim_alms = curvedsky.rand_alm(cl_smooth,
                                  seed=seed)
    #filter and return
    return filter_alms(sim_alms)

do_recon=False
alms_filtered = filter_alms(alms)

if do_recon:
    k2_recon = qfunc(alms_filtered,alms_filtered)
    #save
    hp.fitsfunc.write_alm(opj(
        output_dir, 'k2_recon_lmin%d_lmax%d.fits'%(lmin,lmax)),
        k2_recon, overwrite=True)
    cl = curvedsky.alm2cl(k2_recon)
    np.save(opj(
        output_dir, 'cl_recon_lmin%d_lmax%d.npy'%(lmin,lmax)),
        cl)
    print(cl)
    ells = np.arange(len(cl))
    fig,ax=plt.subplots()
    lmin_plot, lmax_plot = 10, 200
    ax.plot(ells[lmin_plot:lmax_plot+1], cl[lmin_plot:lmax_plot+1])
    fig.savefig(opj(
        output_dir, 'cl_ks2_raw.png')
                )
    
else:
    powfunc = lambda x,y: curvedsky.alm2cl(x,y)
    nsim_n0=40
    rdn0,mcn0 = mcrdn0(
        0, get_sim_alm, powfunc, nsim_n0, 
        qfunc, qfunc2=None, Xdat=alms_filtered, use_mpi=True)
    np.save(
        opj(output_dir, "rdn0_nsim%d"%nsim_n0),
        rdn0)
    np.save(
        opj(output_dir, "mcn0_nsim%d"%nsim_n0),
        mcn0)
