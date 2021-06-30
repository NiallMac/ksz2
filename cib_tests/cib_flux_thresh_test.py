#!/usr/bin/env python
# coding: utf-8

# Look at relations between CIB at different frequencies

#Plots
import pickle
import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
import os
from os.path import join as opj
import numpy as np
import healpy as hp


freqs = ["0095", "0145", "0545"]

websky_dir = "/global/cscratch1/sd/maccrann/cmb/websky/"
CONVERSION_FACTORS = {"CIB" : 
                      {"0093" : 4.6831e3, "0100" : 4.1877e3, "0145" : 2.6320e3, "0545" : 1.7508e4},
                      "Y" : 
                      {"0093" : -4.2840e6, "0100": -4.1103e6, "0145" : -2.8355e6},
}
cib_145 = hp.read_map(opj(websky_dir, "cib_nu0145.fits"))
t_145 = cib_145*CONVERSION_FACTORS["CIB"]["0145"]
cib_545 = hp.read_map(opj(websky_dir, "cib_nu0545.fits"))
t_545 = cib_545*CONVERSION_FACTORS["CIB"]["0545"]

def get_mask(cib_map, flux_threshold):
    #cib_map is flux density in MJy
    cib_flux = cib_map * 1.e9 * hp.nside2pixarea(4096)
    return cib_flux < flux_threshold

flux_thresholds_545 = np.arange(50,1000,4)
frac_high_545 = []
frac_high_masked_145 = []
for f in flux_thresholds_545:
    mask_545 = get_mask(cib_545, f)
    f_545 = float((~mask_545).sum())/len(mask_545)
    frac_high_545.append(f_545)
    percentile_145 = np.percentile(cib_145, (1-f_545)*100)
    high_145 = cib_145 > percentile_145
    f_145 = float((high_145 & (~mask_545)).sum()) / (high_145).sum()
    print(f_145)
    frac_high_masked_145.append(f_145)

fig,ax = plt.subplots()
ax.plot(flux_thresholds_545, frac_high_masked)
fig.savefig('cib_flux_thresh_test.png')
