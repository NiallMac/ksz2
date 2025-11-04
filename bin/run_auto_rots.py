#Run auto measurement on simulations,
#using a rotated version of the input kSZ signal
#for each realisation
from os.path import join as opj, dirname
import os

#import sys
#sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ksz4.cross import four_split_K, split_phi_to_cl, mcrdn0_s4
#from ksz4.reconstruction import  setup_ABCD_recon, get_cl_smooth
from pixell import curvedsky, enmap
from scipy.signal import savgol_filter
from cmbsky import safe_mkdir, get_disable_mpi
from falafel import utils, qe
import healpy as hp
import yaml
import argparse
from orphics import maps, mpi
import numpy as np
import pickle
from string import Template
from pytempura import noise_spec

disable_mpi = get_disable_mpi()
if not disable_mpi:
    from mpi4py import MPI
    comm = mpi.MPI.COMM_WORLD
else:
    comm = None

#DEFAULT_CL_RKSZ=np.load(opj(dirname(__file__), "../tests/cl_4e3_2048_50_50_ksz.npy"))

#with open(opj(dirname(__file__),"../run_auto_defaults.yml"),'rb') as f:
#    DEFAULTS=yaml.safe_load(f)
    
from run_auto import *
DEFAULTS["nrot"] = 5
DEFAULTS["irot_start"] = 0
DEFAULTS["no_planck"] = False

def main():

    #setup mpi if we want it                                                                                                                                                                                                
    if not disable_mpi:
        comm = MPI.COMM_WORLD
        rank,size = comm.Get_rank(), comm.Get_size()
    else:
        comm = None
        rank,size = 0,1

    #get args                                                                                                                                                                                                               
    args,_,_ = get_config()
    data_template_path_orig = args.data_template_path
    output_dir_orig = args.output_dir
    
    for irot in range(args.irot_start, args.irot_start + args.nrot):
        if size>1:
            if irot%size != rank:
                continue
        print("rank %d doing irot: %d"%(rank, irot))

        act_seed=irot+1
        planck_seed=irot+200

        print(args.data_template_path)
        if args.no_planck:
            args.data_template_path = data_template_path_orig%("%05d"%act_seed, "%d"%irot)
        else:
            args.data_template_path = data_template_path_orig%("%d"%planck_seed, "%05d"%act_seed, "%d"%irot)
        args.output_dir = output_dir_orig + "_rot%d"%irot
        print(args.data_template_path)
        
        if rank==0:
            print("args:")
            print(args)
        args, setup, data_alms_f = do_setup(args,
                                            verbose = (rank==0))
    
        outputs = {}
        if args.ksz_theory_alm_file is not None:
            ksz_theory_alm = utils.change_alm_lmax(
                hp.read_alm(args.ksz_theory_alm_file), args.mlmax)
        else:
            ksz_theory_alm = None

        """
            cl_ksz_smooth = get_cl_smooth(ksz_theory_alm)
            #I'm going to assume this is from a full-sky nside=4096 map, in which
            #case, we need to re-run the setup
            px_ksz = qe.pixelization(nside=4096)

            #for key in ["cltot_A","cltot_B","cltot_C","cltot_D","cltot_AC","cltot_BD","cltot_AD","cltot_BC"]:
            #    assert np.all(np.isclose(setup[key], eval(key)))

            setup_ksz = setup_ABCD_recon(
                px_ksz, setup["lmin"], setup["lmax"], setup["mlmax"],
                setup["cl_rksz"],
                setup["cltot_A"], setup["cltot_B"], setup["cltot_C"], setup["cltot_D"],
                setup["cltot_AC"], setup["cltot_BD"], setup["cltot_AD"], setup["cltot_BC"],
                do_lh=True, do_psh=True)                

            ksz_theory_alms_f = [
                f(ksz_theory_alm) for f in [
                    setup_ksz["filter_A"], setup_ksz["filter_B"], setup_ksz["filter_C"], setup_ksz["filter_D"]]
                ]

            K_ksz_AB = setup_ksz["qfunc_K_AB"](ksz_theory_alms_f[0],
                                           ksz_theory_alms_f[1])
            K_ksz_CD = setup_ksz["qfunc_K_CD"](ksz_theory_alms_f[2],
                                           ksz_theory_alms_f[3])
            Cl_KK_ksz_raw = curvedsky.alm2cl(K_ksz_AB, K_ksz_CD)
            ksz_N0 = setup_ksz["get_fg_trispectrum_N0_ABCD"](cl_ksz_smooth, cl_ksz_smooth, cl_ksz_smooth, cl_ksz_smooth)
            outputs["Cl_KK_ksz_theory_raw"] = Cl_KK_ksz_raw
            outputs["N0_ksz_theory"] = ksz_N0
            outputs["Cl_KK_ksz_theory"] = Cl_KK_ksz_raw-ksz_N0

            if args.do_lh:
                K_ksz_AB = setup_ksz["qfunc_K_AB_lh"](ksz_theory_alms_f[0],
                                               ksz_theory_alms_f[1])
                K_ksz_CD = setup_ksz["qfunc_K_CD_lh"](ksz_theory_alms_f[2],
                                               ksz_theory_alms_f[3])
                Cl_KK_ksz_lh_raw = curvedsky.alm2cl(K_ksz_AB, K_ksz_CD)
                ksz_N0_lh = setup_ksz["get_fg_trispectrum_N0_ABCD_lh"](cl_ksz_smooth, cl_ksz_smooth, cl_ksz_smooth, cl_ksz_smooth)
                outputs["Cl_KK_ksz_theory_raw_lh"] = Cl_KK_ksz_lh_raw
                outputs["N0_ksz_theory_lh"] = ksz_N0_lh
                outputs["Cl_KK_ksz_theory_lh"] = Cl_KK_ksz_lh_raw - ksz_N0_lh

            if args.do_psh:
                K_ksz_AB = setup_ksz["qfunc_K_AB_psh"](ksz_theory_alms_f[0],
                                               ksz_theory_alms_f[1])
                K_ksz_CD = setup_ksz["qfunc_K_CD_psh"](ksz_theory_alms_f[2],
                                               ksz_theory_alms_f[3])
                Cl_KK_ksz_psh_raw = curvedsky.alm2cl(K_ksz_AB, K_ksz_CD)
                ksz_N0_psh = setup_ksz["get_fg_trispectrum_N0_ABCD_psh"](cl_ksz_smooth, cl_ksz_smooth, cl_ksz_smooth, cl_ksz_smooth)
                outputs["Cl_KK_ksz_theory_raw_psh"] = Cl_KK_ksz_psh_raw
                outputs["N0_ksz_theory_psh"] = ksz_N0_psh
                outputs["Cl_KK_ksz_theory_psh"] = Cl_KK_ksz_psh_raw - ksz_N0_psh
        """

        #Now run on data
        auto_outputs = run_ClKK(args, setup, data_alms_f,
                                comm=None, do_lh=args.do_lh, do_psh=args.do_psh,
                                ksz_theory_alm=ksz_theory_alm)
        outputs.update(auto_outputs)
        #outputs["cl_KK_raw"] = auto_outputs["cl_abcd"]
        #outputs["cl_KK_lh_raw"] = auto_outputs["cl_abcd_lh"]
        #outputs["cl_KK_lh_raw"] = auto_outputs["cl_abcd_psh"]
        outputs["N0"] = setup["N0_ABCD_K"]
        if args.do_lh:
            outputs["N0_lh"] = setup["N0_ABCD_K_lh"]
        if args.do_psh:
            outputs["N0_psh"] = setup["N0_ABCD_K_psh"]
        outputs["N0_nonoise"] = setup["N0_K_nonoise"]
        outputs["norm_K_AB"] = setup["norm_K_AB"]
        outputs["norm_K_CD"] = setup["norm_K_CD"]

        outputs["lmin"] = setup["lmin"]
        outputs["lmax"] = setup["lmax"]
        outputs["profile"] = setup["profile"]


        #save pkl
        output_filename = opj(args.output_dir, "auto_outputs.pkl")
        print("rank %d writing output for rot %d to %s"%(rank, irot, output_filename)) 
        with open(output_filename, 'wb') as f:
            pickle.dump(outputs, f)
    
    return 0

if __name__=="__main__":
    main()
