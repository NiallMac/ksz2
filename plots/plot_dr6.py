#!/usr/bin/env python
# coding: utf-8

# In[67]:


#Run the kSZ 4-point function
#
#
from os.path import join as opj
import os
os.environ["DISABLE_MPI"]="true"
from ksz4.cross import four_split_K, split_phi_to_cl
from ksz4.reconstruction import setup_recon, setup_ABCD_recon, get_cl_fg_smooth
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
import matplotlib
matplotlib.use('Agg')
import pickle


# In[68]:


binner = ClBinner(lmin=10, lmax=200, nbin=10)


# In[69]:


#also add theory
outputs_dir='/pscratch/sd/m/maccrann/cmb/fg_outputs'
def get_fg_terms(tag, recon_tag, freq):
    fg_term_file = opj(outputs_dir, tag, 
                       "ksz2_fg_terms_%s/fg_terms_%s.npy"%(
                           recon_tag, freq)
                      )
    return np.load(fg_term_file)
fg_terms = get_fg_terms("allfgs_nemo-wdr6dn_tsz-nemo-model-snr5-mask-snr4-mr6-lowz0.1-mr30_ps-model-snr4_wplanck-ps2",
                        "lmax6000", "ilc_deproj-tsz-cib_ilc_ilc")
ksz_theory = fg_terms["cl_K_ksz"]


# #### read auto output

# In[61]:


#est_maps="90_90_90_90"

def get_rdn0_scatter(rdn0_array, binner):
    binned_rdn0_array = np.zeros((rdn0_array.shape[0], binner.nbin))
    for i,a in enumerate(rdn0_array):
        binned_rdn0_array[i] = binner(a)
    return np.std(binned_rdn0_array, axis=0)


def plot_clKK(ax, est_maps, binner, f_sky=0.3, color=None, label=None, lfac=2):

    #read auto outputs
    auto_outputs_file = "../scripts/test_%s/auto_outputs.pkl"%est_maps
    with open(auto_outputs_file, "rb") as f:
        auto_outputs = pickle.load(f)

    #read N0 output
    n0_output_file = "../scripts/test_%s/rdn0_outputs.pkl"%est_maps
    with open(n0_output_file, "rb") as f:
        rdn0_outputs = pickle.load(f)
        
    mcn0_scatter = get_rdn0_scatter(rdn0_outputs["rdn0"],
                                    binner)
    print(mcn0_scatter)
    
    cl_KK_binned = binner(auto_outputs["cl_KK_raw"]-rdn0_outputs["rdn0"].mean(axis=0))
    cl_KK_err_binned = get_rdn0_scatter(rdn0_outputs["rdn0"], binner)
    is_pos=cl_KK_binned>0
    ax.errorbar(binner.bin_mids[is_pos], (binner.bin_mids**lfac*cl_KK_binned)[is_pos], 
                yerr=cl_KK_err_binned[is_pos], fmt='o',color=color, label=label)
    ax.errorbar(binner.bin_mids[~is_pos], -(binner.bin_mids**lfac*cl_KK_binned)[~is_pos], 
                    yerr=cl_KK_err_binned[~is_pos], fmt='o', mfc='none', color=color)


# In[65]:


#get_ipython().run_line_magic('matplotlib', 'inline')
fig,ax=plt.subplots()
lfac=2
plot_clKK(ax, "90_90_90_90", binner, color='C0', label="90 only", lfac=lfac)
plot_clKK(ax, "hilc_hilc_hilc_hilc", binner, color='C1', label="ilc only", lfac=lfac)
plot_clKK(ax, "hilc_hilc-tszandcibd_hilc_hilc", binner, color='C2', label="icii", lfac=lfac)

#plot Alvarez theory
ax.plot(binner.bin_mids, (binner.bin_mids**lfac)*binner(ksz_theory), color='k', label="kSZ theory")
ax.set_yscale('log')

fig.savefig("clKK_dr6_new.png")
exit()


# In[41]:


#### read rdn0 output


# In[53]:


#errorbars seem very off. compare to previous maps (with proper sims)
rdn0_output_file_old="../bin/output_lmin3000_lmax4000/rdn0_outputs.pkl"
with open(rdn0_output_file_old,"rb") as f:
    rdn0_old=pickle.load(f)


# In[66]:


get_ipython().run_line_magic('matplotlib', 'inline')

fig,ax=plt.subplots()

ax.plot((rdn0_old["mcn0"].std(axis=0))[:200])
ax.plot((rdn0_outputs["mcn0"].std(axis=0))[:200])
ax.set_yscale('log')


# In[42]:


n0_output_file = "../scripts/test_%s/rdn0_outputs.pkl"%est_maps
with open(n0_output_file, "rb") as f:
    rdn0_outputs = pickle.load(f)
print(rdn0_outputs.keys())


# In[ ]:





# In[43]:


print(rdn0_outputs['mcn0'].shape)


# In[44]:


#also add theory
outputs_dir='/pscratch/sd/m/maccrann/cmb/fg_outputs'
def get_fg_terms(tag, recon_tag, freq):
    fg_term_file = opj(outputs_dir, tag, 
                       "ksz2_fg_terms_%s/fg_terms_%s.npy"%(
                           recon_tag, freq)
                      )
    return np.load(fg_term_file)
fg_terms = get_fg_terms("allfgs_nemo-wdr6dn_tsz-nemo-model-snr5-mask-snr4-mr6-lowz0.1-mr30_ps-model-snr4_wplanck-ps2",
                        "lmax6000", "ilc_deproj-tsz-cib_ilc_ilc")
ksz_theory = fg_terms["cl_K_ksz"]


# In[45]:


print(auto_outputs.keys())

def get_rdn0_scatter(rdn0_array, binner):
    binned_rdn0_array = np.zeros((rdn0_array.shape[0], binner.nbin))
    for i,a in enumerate(rdn0_array):
        binned_rdn0_array[i] = binner(a)
    return np.std(binned_rdn0_array, axis=0)


# In[46]:


get_ipython().run_line_magic('matplotlib', 'inline')

fig,ax=plt.subplots()

cl_KK = auto_outputs["cl_KK_raw"]-rdn0_outputs["rdn0"].mean(axis=0)

cl_KK_err = 2*get_rdn0_scatter(rdn0_outputs["rdn0"],binner)**2 / (2*binner.bin_mids+1) / 0.27
print(rdn0_outputs["rdn0"])
ax.errorbar(binner.bin_mids, binner(cl_KK), yerr=cl_KK_err, color='C0', label="C_L^KK")
ax.plot(binner.bin_mids, binner(rdn0_outputs["rdn0"].mean(axis=0)), color='C1', label="RDNO")
ax.plot(binner.bin_mids, binner(rdn0_outputs["mcn0"].mean(axis=0)), color='C2', label="MCN0")
#ax.plot(binner.bin_mids, -binner(cl_KK), '--', color='C0', label="C_L^KK")
#ax.plot(binner.bin_mids, binner(rdn0_outputs["rdn0"].mean(axis=0)), label="Gaussian sim rdn0")
ax.plot(binner.bin_mids, -binner(rdn0_outputs["rdn0"].mean(axis=0)), '--', color='C1')
#ax.plot(binner.bin_mids, binner(rdn0_outputs["mcn0"].mean(axis=0)), label="Gaussian sim mcn0")
#ax.plot(binner.bin_mids, -binner(rdn0_outputs["mcn0"].mean(axis=0)), '--', color='C2')
ax.plot(binner.bin_mids, binner(auto_outputs["N0"]), color='C3',label="theory N0")
ax.plot(binner.bin_mids, binner(auto_outputs["N0_nonoise"]), color='C4', label="theory N0 no noise")
ax.plot(binner.bin_mids, binner(ksz_theory), color='C5', label="kSZ theory")
ax.set_yscale('log')
ax.legend()


# In[ ]:





# In[ ]:




