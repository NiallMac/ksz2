#!/usr/bin/env python
# coding: utf-8

# Look at relations between CIB at different frequencies

# In[13]:


#Plots
import pickle
import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
import os
#os.environ['DISABLE_MPI']="true"
from os.path import join as opj
import numpy as np
#from solenspipe import weighted_bin1D
#from prepare_websky_map import safe_mkdir
#import healpy as hp


# In[14]:


freqs = ["0095", "0145", "0545"]
smoothing_scales = [1., 5., 20.] #arcmin


# In[15]:


websky_dir = "/global/cscratch1/sd/maccrann/cmb/websky/"
CONVERSION_FACTORS = {"CIB" : 
                      {"0093" : 4.6831e3, "0100" : 4.1877e3, "0145" : 2.6320e3, "0545" : 1.7508e4},
                      "Y" : 
                      {"0093" : -4.2840e6, "0100": -4.1103e6, "0145" : -2.8355e6},
}


# In[16]:


cib_145 = hp.read_map(opj(websky_dir, "cib_nu0145.fits"))
t_145 = cib_145*CONVERSION_FACTORS["CIB"]["0145"]
cib_545 = hp.read_map(opj(websky_dir, "cib_nu0545.fits"))
t_545 = cib_545*CONVERSION_FACTORS["CIB"]["0545"]


# In[17]:


def get_mask(cib_map, flux_threshold):
    #cib_map is flux density in MJy
    cib_flux = cib_map * 1.e9 * hp.nside2pixarea(4096)
    return cib_flux < flux_threshold


# In[18]:


lmax=6000
mask_thresh_145 = 7.
mask_thresh_545 = 350.

cib_mask_145 = get_mask(cib_145, mask_thresh_145)
cib_mask_545 = get_mask(cib_545, mask_thresh_545)


# In[21]:


def get_rl(map1, map2, lmax, mask=None, sub_mean=True):
    if mask is None:
        mask = np.ones_like(map1)
    if sub_mean:
        map1 = map1 - map1.mean()
        map2 = map2 - map2.mean()
    cl11 = hp.anafast(map1*mask, lmax=lmax)
    cl22 = hp.anafast(map2*mask, lmax=lmax)
    return hp.anafast(map1*mask, map2*mask, lmax=lmax) / np.sqrt(cl11*cl22)


# In[ ]:


rl_nomask = get_rl(t_145, t_545, lmax)
rl_mask_145 = get_rl(t_145, t_545, lmax, mask=cib_mask_145)
rl_mask_545 = get_rl(t_145, t_545, lmax, mask=cib_mask_545)


# In[ ]:


#get_ipython().run_line_magic('matplotlib', 'inline')
fig,ax = plt.subplots()
ells = np.arange(lmax+1)
ax.plot(ells, rl_nomask, label='no mask')
ax.plot(ells, rl_mask_145, label='>%d mJY at 145GHz masked'%mask_thresh_145)
ax.plot(ells, rl_mask_545, label='>%d mJY at 545GHz masked'%mask_thresh_545)
ax.legend()
ax.set_xlabel(r"$L$")
ax.set_ylabel(r"$C_L^{12} / \sqrt{C_L^{11}C_L{22}}$")
ax.set_ylim([0.5,1.])
ax.set_title("145/545 GHz cross-correlation coefficient")
fig.savefig('RL_cib_145-545.png')
