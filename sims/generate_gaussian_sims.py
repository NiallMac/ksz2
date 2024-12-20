#Read in data alms and get auto and cross Cls
#Generate Gaussian sims, apply k-space mask and survey mask
#Re-measure Cls to get transfer function
#Boost input Cls and re-generate Gaussian sims
#Plots
import pickle
import healpy as hp
import os
os.environ['DISABLE_MPI']="true"
from os.path import join as opj
import numpy as np
from collections import OrderedDict
from pixell import curvedsky,enmap,utils,enplot
from falafel import utils as futils
from cmbsky import ClBinner, safe_mkdir
import healpy as hp
from string import Template
import argparse
from scipy.signal import savgol_filter
from orphics import maps
from solenspipe.utility import kspace_mask
import pickle

def parse_args():
    parser = argparse.ArgumentParser(
        prog='Generate Gaussian sims',
        description="Generate Gaussian sims"
    )
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--data_template_path", "-d",
                        type=str, default="/pscratch/sd/m/maccrann/cmb/act_dr6/coadd_data/kcoadd_f_${freq}_lmax7000_split${split}.fits")
    parser.add_argument("--mask_file", default="/global/cfs/projectdirs/act/data/maccrann/dr6/act_mask_20220316_GAL060_rms_60.00sk.fits")
    parser.add_argument("--lmax", type=int, default=5000)
    parser.add_argument("--freqs", nargs="*", default=["90","150"])
    parser.add_argument("--sg_window", type=int, default=101)
    parser.add_argument("--sg_order", type=int, default=2)
    parser.add_argument("--apply_mask", action="store_true", default=False)
    parser.add_argument("--save_masked_alms", action="store_true", default=False)
    parser.add_argument("--nsim_cal", type=int, default=10)
    parser.add_argument("--nsim", type=int, default=100)
    parser.add_argument("--mpi", action="store_true", default=False)
    parser.add_argument("--use_cmb_below_ell", type=int, default=1000)
    parser.add_argument("--apply_beam_before_mask", type=float, default=None)
    return parser.parse_args()

NSPLIT=4

def main():

    args=parse_args()
    
    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank,size = comm.Get_rank(), comm.Get_size()
    else:
        print("Not using MPI!!!!")
        comm = None
        rank,size = 0,1
    
    if rank==0:
        safe_mkdir(args.output_dir)
    
    data_template_path = Template(args.data_template_path)
    print("args.data_template_path:", args.data_template_path)

    mask=enmap.read_map(args.mask_file)
    print("reading mask from %s"%args.mask_file)
    w2 = maps.wfactor(2, mask)
    z = enmap.zeros(shape=mask.shape, wcs=mask.wcs)

    freqs=args.freqs

    Cl_dict = {}

    def mask_alms(alms, mask):
        lmax = hp.Alm.getlmax(len(alms[0]))
        z = enmap.zeros(mask.shape, mask.wcs)
        return [curvedsky.map2alm(curvedsky.alm2map(alm, z, tweak=True) * mask, tweak=True, lmax=lmax) for alm in alms]
    
    print("getting data Cls")
    for i in range(len(freqs)):
        freq_i = freqs[i]
        print("freq_i:",freq_i)
        print("reading alms from", data_template_path.substitute(freq=freq_i, split=0))
        alms_i = [hp.read_alm(data_template_path.substitute(freq=freq_i, split=split)) for split in range(NSPLIT)]
        if args.apply_mask:
            print("masking data alms")
            alms_i = mask_alms(alms_i, mask)
            if args.save_masked_alms:
                if rank==0:
                    for split in range(NSPLIT):
                        hp.write_alm(opj(args.output_dir, "data_alm_%s_split%d.fits"%(freq_i,split)), alms_i[split], overwrite=True)
        alms_i = [futils.change_alm_lmax(alm, args.lmax) for alm in alms_i]
        print("read alms")

        for j in range(i,len(freqs)):
            freq_j = freqs[j]
            print("freq_j:",freq_j)
            if i!=j:
                alms_j = [hp.read_alm(data_template_path.substitute(freq=freq_j, split=split)) for split in range(NSPLIT)]
                if args.apply_mask:
                    print("masking data alms")
                    alms_j = mask_alms(alms_j, mask)
                if args.save_masked_alms:
                    if rank==0:
                        for split in range(NSPLIT):
                            hp.write_alm(opj(args.output_dir, "data_alm_%s_split%d.fits"%(freq_j,split)), alms_j[split], overwrite=True)
                alms_j = [futils.change_alm_lmax(alm, args.lmax) for alm in alms_j]
            else:
                alms_j=alms_i

            #we want to get just signal parts, 
            cl_splits = []
            for split_1 in range(NSPLIT):
                for split_2 in range(NSPLIT):
                    if split_2==split_1:
                        continue
                    cl_ij_12 = curvedsky.alm2cl(alms_i[split_1], alms_j[split_2])
                    cl_splits.append(cl_ij_12)
            cl_splits = np.array(cl_splits)
            cl_mean = np.mean(cl_splits, axis=0)
            cl_smooth = savgol_filter(np.mean(cl_splits, axis=0), args.sg_window, args.sg_order) / w2
            cl_smooth[cl_smooth<0.] = 0.
            Cl_dict[freq_i,freq_j] = cl_smooth
            Cl_dict[freq_j,freq_i] = cl_smooth

    with open(opj(args.output_dir, "cl_orig.pkl"), "wb") as f:
        pickle.dump(Cl_dict, f)
            
    #Ok, now we need to generate some number of simulations
    cov = np.zeros((len(freqs), len(freqs), args.lmax+1))
    for i,freq_i in enumerate(freqs):
        for j,freq_j in enumerate(freqs):
            cov[i,j,:] = Cl_dict[(freq_i,freq_j)]

    print("generating %d transfer function sims"%(args.nsim_cal))
    Cl_dict_cal = {}
    for i,freq_i in enumerate(freqs):
        for j,freq_j in enumerate(freqs):
            Cl_dict_cal[freq_i,freq_j] = []
    for isim in range(args.nsim_cal):
        if size>1:
            if isim%size != rank:
                continue
        sim_alms = curvedsky.rand_alm(cov, seed=1234+isim)
        if args.apply_beam_before_mask:
            ells = np.arange(mlmax+1)
            beam = maps.gauss_beam(ells, beam_fwhm)
        if len(freqs)==0:
            sim_alms = [sim_alms]
        #generate masked maps
        sim_maps = [curvedsky.alm2map(a, enmap.zeros(shape=mask.shape, wcs=mask.wcs), tweak=True)*mask 
                    for a in sim_alms]
        #k-space masking
        def apod(m, width):
            return enmap.apod(m,[width,0]) 

        sim_maps = [enmap.apply_window(apod(s, 10), pow=1.0) for s in sim_maps]
        sim_maps_kspace_masked = []
        for m in sim_maps:
            sim_maps_kspace_masked.append(
                kspace_mask(m, deconvolve=True)
                )
        #back to alms
        sim_alms = [curvedsky.map2alm(m, lmax=args.lmax, tweak=True)
                    for m in sim_maps_kspace_masked]

        for i,freq_i in enumerate(freqs):
            for j,freq_j in enumerate(freqs):
                cl_raw = curvedsky.alm2cl(
                    sim_alms[i], sim_alms[j])
                Cl_dict_cal[freq_i,freq_j].append(
                    savgol_filter(cl_raw, args.sg_window, args.sg_order)/ w2)
    if rank==0:
        #collect
        n_collected=1
        while n_collected<size:
            cls_to_add = comm.recv(source=MPI.ANY_SOURCE)
            for key in Cl_dict_cal.keys():
                Cl_dict_cal[key] += cls_to_add[key]
            n_collected+=1
        #convert to arrays
        for key in Cl_dict_cal:
            Cl_dict_cal[key] = (np.array(Cl_dict_cal[key])).mean(axis=0)
            
        with open(opj(args.output_dir, "cl_cal.pkl"), "wb") as f:
            pickle.dump(Cl_dict_cal, f)

    else:
        comm.send(Cl_dict_cal, dest=0)

    if comm is not None:
        comm.Barrier()
    
    if size>1:
        Cl_dict_cal = comm.bcast(Cl_dict_cal, root=0)
        
    #Ok now get transfer function-corrected Cls
    Cl_dict_boosted = {}
    for freq_i in freqs:
        inv_tfii = Cl_dict[freq_i,freq_i] / Cl_dict_cal[freq_i,freq_i]
        for freq_j in freqs:
            inv_tfjj = Cl_dict[freq_j,freq_j] / Cl_dict_cal[freq_j,freq_j]
            cl_orig = Cl_dict[freq_i,freq_j]
            Cl_dict_boosted[freq_i,freq_j] = (
                cl_orig * np.sqrt(inv_tfii*inv_tfjj)
            )
            
    if rank==0:
        with open(opj(args.output_dir, "cl_boosted.pkl"), "wb") as f:
            pickle.dump(Cl_dict_boosted, f)
            
    #And now, finally, we can generate the simulations
    #Ok, now we need to generate some number of simulations
    cov_boosted = np.zeros((len(freqs), len(freqs), args.lmax+1))
    cmb_theory_cl = (futils.get_theory_dicts(grad=True, lmax=args.lmax+1)[1])["TT"]
    for i,freq_i in enumerate(freqs):
        for j,freq_j in enumerate(freqs):
            if args.use_cmb_below_ell > 0:
                cov_boosted[i,j,:args.use_cmb_below_ell] = cmb_theory_cl[:args.use_cmb_below_ell]
                cov_boosted[i,j,args.use_cmb_below_ell:] = Cl_dict_boosted[(freq_i,freq_j)][args.use_cmb_below_ell:]
            else:
                cl_use = Cl_dict_boosted[(freq_i,freq_j)]
                cl_use[cl_use<0] = 0.
                cov_boosted[i,j] = cl_use
            
    for isim in range(args.nsim):
        if size>1:
            if isim%size != rank:
                continue
        sim_alms = curvedsky.rand_alm(cov_boosted, seed=1234+isim)
        if len(freqs)==0:
            sim_alms = [sim_alms]
            
        #generate masked maps
        sim_maps = [curvedsky.alm2map(a, enmap.zeros(shape=mask.shape, wcs=mask.wcs), tweak=True)*mask
                    for a in sim_alms]
        #k-space masking
        def apod(m, width):
            return enmap.apod(m,[width,0]) 

        sim_maps = [enmap.apply_window(apod(s, 10), pow=1.0) for s in sim_maps]
        sim_maps_kspace_masked = []
        for m in sim_maps:
            sim_maps_kspace_masked.append(
                kspace_mask(m, deconvolve=True)
                )
        #back to alms
        sim_alms = [curvedsky.map2alm(m, lmax=args.lmax, tweak=True)
                    for m in sim_maps_kspace_masked]
        
        for freq,alm in zip(freqs, sim_alms):
            filename = opj(args.output_dir, "%s_isim%05d.fits"%(freq,isim))
            print("rank %d writing alm to %s"%(rank, filename))
            hp.write_alm(filename, alm, overwrite=True)
    
            

if __name__=="__main__":
    main()

