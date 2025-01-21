#!/usr/bin/env python
# coding: utf-8

# Run the kSZ 4-point estimator on the Amber simulations

# In[2]:


from os.path import join as opj
import os
from ksz4.cross import four_split_K, split_phi_to_cl, mcrdn0_s4
from ksz4.reconstruction import setup_recon, setup_ABCD_recon, get_cl_smooth
from pixell import curvedsky, enmap
from scipy.signal import savgol_filter
from cmbsky import safe_mkdir, get_disable_mpi, ClBinner
from falafel import utils, qe
import healpy as hp
import yaml
import argparse
from orphics import maps, mpi
import numpy as np
import pickle
from string import Template
from pytempura import noise_spec
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

disable_mpi = get_disable_mpi()
if not disable_mpi:
    from mpi4py import MPI

if not disable_mpi:
    comm = MPI.COMM_WORLD
    rank,size = comm.Get_rank(), comm.Get_size()
else:
    rank,size = 0,1

# - First thing I'm going to do is convert the maps to alms so we don't have to do that every time (comment out save_alms below)

# In[3]:


sim_dir = "/global/cfs/cdirs/cmb/data/generic/extragalactic/amber/"
output_alm_dir = "/global/cfs/projectdirs/act/data/maccrann/amber"

sim_map_files = glob.glob(opj(sim_dir, "output*", "cmb", "map_ksz_nside=4096.fits"))
sim_tags = [f.split("/")[-3] for f in sim_map_files]
lmax_alm=8000
sim_alm_files = [opj(output_alm_dir, sim_tag+"ksz_alm_lmax%d.fits"%(lmax_alm)) for sim_tag in sim_tags]
print(sim_alm_files[0])


# In[4]:


def save_alms():
    safe_mkdir(output_alm_dir)
    for f_map,f_alm in zip(sim_map_files, sim_alm_files):
        print(f_map)
        m = hp.read_map(f_map)
        alm = hp.map2alm(m, lmax=lmax_alm)
        hp.write_alm(f_alm, alm, overwrite=True)

#Uncomment the below if this is the first time running!!!
#save_alms()
#do_stuff


# - First thing to decide is the filters to use. We generally want to use Cl_KSZ^0.5 / Cl_total
# - Probably less confusing to use the same filters for all the different simulations, so start by 
# using the Alvarez Cl_KSZ.

# In[5]:

ksz_alm=hp.read_alm("../tests/alms_4e3_2048_50_50_ksz.fits")
cl_ksz = get_cl_smooth(ksz_alm)
#cl_ksz_raw = hp.alm2cl( hp.read_alm(sim_alm_files[0]) )
#smooth this
#cl_ksz = savgol_filter(cl_ksz_raw, 301, 2)
#negative values in this causes nans, so I'm going to 
#set low ls to some constant - we don't use these anyway
#cl_ksz[:1000] = cl_ksz[1000]


# In[6]:



ls = np.arange(len(cl_ksz))
lfac=ls*(ls+1)/2/np.pi

fig,ax=plt.subplots()
ax.plot(ls, lfac*cl_ksz)
ax.set_yscale('log')


# Few more options to set here:
# - Noise level (let's assume white noise)
# - lmin and lmax 
# - Think that's it really
# - oh and beam

# In[7]:


noise_sigma = 1.  #muK arcmin
beam_fwhm = 1.5 #arcmin
lmin = 3000
lmax = 5000
mlmax = 6000
px=qe.pixelization(nside=4096)

tcmb = 2.725e6
# - Get beam, noise and total Cl

# In[8]:


ells = np.arange(mlmax+1)
beam = maps.gauss_beam(ells, beam_fwhm)
Nl_tt = (noise_sigma*np.pi/180./60.)**2./beam**2
nells = {"TT":Nl_tt, "EE":2*Nl_tt, "BB":2*Nl_tt}
_,tcls = utils.get_theory_dicts(grad=True, nells=nells, lmax=mlmax)
Cl_tot_theory = tcls["TT"][:mlmax+1] + cl_ksz[:mlmax+1] #add kSZ to total Cl

#Could also use total Cl from data...
#with open("/pscratch/sd/m/maccrann/ksz_outputs/output_hilc_hilc_hilc_hilc_dr6v4_v4_lmax5000_mask60sk_gausssims/auto_outputs.pkl","rb") as f:
#    auto_outputs = pickle.load(f)
#Cl_tot = auto_outputs["cltot_A"]
Cl_tot = Cl_tot_theory


# In[ ]:


#setup the estimators etc.
#I'm using the most general setup here (which allows for 4 different maps).
#But here we're just using the same map in each leg hence all those Cl_tot
#arguments are the same. 
recon_setup = setup_ABCD_recon(px, lmin, lmax, mlmax,
                      cl_ksz[:mlmax+1], Cl_tot, Cl_tot,
                      Cl_tot, Cl_tot,
                      Cl_tot, Cl_tot,
                      Cl_tot, Cl_tot, do_lh=True,
                      do_psh=False)


# recon_setup is a dictionary that contains various things (including the "qfuncs") we need to run the estimator on the data, and als
# funtions for getting theory N0s. So now we're ready to read in maps and run on them.

# In[9]:

CL_KK_stuff = {}

#K_outdir="/global/cfs/projectdirs/act/data/maccrann/amber_Ks/amber_Ks_22.02.24"
K_outdir="/global/cfs/projectdirs/act/data/maccrann/amber_Ks/amber_Ks_21.01.25"
safe_mkdir(K_outdir)

n_sim_to_run = len(sim_tags) #set to something else if e.g. we just wantt to run a couple


if rank==0:
    #Also run Alvarez
    ksz_alm_f = recon_setup["filter_A"](ksz_alm)
    K_alvarez = recon_setup["qfunc_K_AB"](ksz_alm_f, ksz_alm_f)
    CL_KK_alvarez = curvedsky.alm2cl(K_alvarez, K_alvarez) - recon_setup["get_fg_trispectrum_N0_ABCD"](cl_ksz,cl_ksz,cl_ksz,cl_ksz)
    CL_KK_stuff["CL_KK_alvarez"] = CL_KK_alvarez
    CL_KK_stuff["Cl_ksz_alvarez"] = cl_ksz

for i,(sim_tag, alm_file) in enumerate(zip(sim_tags[:n_sim_to_run], sim_alm_files)):
    if rank>=n_sim_to_run:
        continue
    if i%size != rank:
        continue
        
    print(sim_tag)
    CL_KK_stuff[sim_tag] = {}
    #read in map, convert to alms, and filter
    alm = hp.read_alm(alm_file)
    #fix units 
    alm *= tcmb
    cl_sim = get_cl_smooth(alm)
    alm_Af = recon_setup["filter_A"](alm) #note only need to use filter_A since all legs the same here
    
    K = recon_setup["qfunc_K_AB"](alm_Af, alm_Af)
    CL_KK_stuff[sim_tag]["CL_KK_raw"] = curvedsky.alm2cl(K,K)

    trispectrum_N0 = recon_setup["get_fg_trispectrum_N0_ABCD"](cl_sim, cl_sim, cl_sim, cl_sim)
    CL_KK_stuff[sim_tag]["trispectrum_N0"] = trispectrum_N0
    CL_KK_stuff[sim_tag]['N0'] = recon_setup["N0_ABCD_K"]
    CL_KK_stuff[sim_tag]["CL_KK"] = (CL_KK_stuff[sim_tag]["CL_KK_raw"] - CL_KK_stuff[sim_tag]["trispectrum_N0"])/recon_setup["profile"]**2
    CL_KK_stuff[sim_tag]["profile"] = recon_setup["profile"]
    CL_KK_stuff[sim_tag]["Cl_ksz"] = cl_sim
    
    
    #Also save K for Derby
    filename = (opj(K_outdir, "K_%s.fits"%sim_tag)).strip()
    print("saving K to %s"%filename)
    hp.write_alm(filename, K, overwrite=True)
    
if rank==0:
    
    safe_mkdir("amber_plots")

    #collect
    n_collected=1
    while n_collected<size:
        rec = comm.recv(source=MPI.ANY_SOURCE)
        for key in rec.keys():
            print("adding sim %s"%key)
            CL_KK_stuff[key] = rec[key]
        n_collected+=1

    import pickle
    with open(opj(K_outdir, "amber_auto_data_test.pkl"), "wb") as f:
        pickle.dump(CL_KK_stuff, f)

        
    print( CL_KK_stuff["output_z=8.00_D=4.00_A=3.00_M=1E8_l=5.00"]["CL_KK_raw"] )
    
    fig,ax=plt.subplots()

    binner = ClBinner(lmin=5, lmax=300, nbin=10, log=True)

    ax.plot(binner.bin_mids, binner.bin_mids**2 * binner(CL_KK_alvarez), color="k", label="Alvarez")
    
    profile = recon_setup["profile"]
    for sim_tag in sim_tags[:n_sim_to_run]:

        print(sim_tag)
        ax.plot(binner.bin_mids, binner.bin_mids**2*binner(CL_KK_stuff[sim_tag]["CL_KK"]),
               label=sim_tag)
        #ax.plot(binner.bin_mids, binner.bin_mids**2*binner(CL_KK_stuff[sim_tag]["CL_KK_raw"]), 
        #       label=sim_tag)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$C_L^{KK}$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(K_outdir+"/"+"CLKK_amber.png", dpi=200)

else:
    comm.send(CL_KK_stuff, dest=0)
    
# In[ ]:




