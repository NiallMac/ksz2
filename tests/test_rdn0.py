"""
Test rdn0
"""

from os.path import join as opj
from ksz4.cross import four_split_K, split_phi_to_cl, mcrdn0_s4
from ksz4.reconstruction import setup_recon, setup_ABCD_recon, get_cl_fg_smooth
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

disable_mpi = get_disable_mpi()
if not disable_mpi:
    from mpi4py import MPI
    comm = mpi.MPI.COMM_WORLD

NSPLIT=4
MASK_PATH="/global/homes/m/maccrann/cmb/lensing/code/so-lenspipe/bin/planck/act_mask_20220316_GAL060_rms_70.00_d2.fits"
DATA_DIR="/global/cscratch1/sd/maccrann/cmb/act_dr6/ilc_cldata_smooth-301-2_modelsub_v1"

#Read in kSZ alms
ksz_alm = utils.change_alm_lmax(hp.fitsfunc.read_alm("../tests/alms_4e3_2048_50_50_ksz.fits"),
                                6000)
cl_rksz = get_cl_fg_smooth(ksz_alm)

def get_sim_alm_dr6_hilc(map_name, sim_seed, data_dir="/global/cscratch1/sd/maccrann/cmb/act_dr6/ilc_cldata_smooth-301-2_modelsub_v1",
                         mlmax=None):
    map_filenames = [opj("sim_planck%03d_act%02d_%05d"%(sim_seed),
                                   "%s_split%d.fits"%(map_name, split))
                     for split in range(NSPLIT)]
    map_paths = [opj(data_dir, f) for f in map_filenames]
    alms = []
    for p in map_paths:
        alm = hp.read_alm(p)
        if mlmax is not None:
            alm = utils.change_alm_lmax(alm, mlmax)
        alms.append(alm)
    return alms

def get_data_Cltot_dict(alm_dict,
                   sg_window=301, sg_order=2,
                   w2=None):
    #just do a straight average of splits
    #and use Savgol filter
    map_names = list(alm_dict.keys())
    cl_dict = {}
    for i,map_i in enumerate(map_names):
        mean_alm_i = np.average(np.array(alm_dict[map_i]), axis=0)
        assert mean_alm_i.shape == (alm_dict[map_i][0]).shape
        for j,map_j in enumerate(map_names[i:]):
            mean_alm_j = np.average(np.array(alm_dict[map_j]), axis=0)
            cl = savgol_filter(
                curvedsky.alm2cl(mean_alm_i, mean_alm_j),
                sg_window, sg_order)
            if w2 is not None:
                cl /= w2
            cl_dict[(map_i,map_j)] = cl
            cl_dict[(map_j,map_i)] = cl
    return cl_dict

def main(use_mpi=False, nsim_rdn0 = 1):

    
    mlmax=2000
    lmin=1000
    lmax=1500
    
    total_Cl_from_data=True

    #get w2 and w4
    mask = enmap.read_map(MASK_PATH)
    px = qe.pixelization(shape=mask.shape,wcs=mask.wcs)
    #w2 = maps.wfactor(2, mask)
    #w4 = maps.wfactor(4, mask)
    w2 = 0.2758258447605825
    w4 = 0.2708454607428685
    print("w2:",w2)
    print("w4:",w4)

    #setup mpi if we want it
    if not disable_mpi:
        comm = MPI.COMM_WORLD
        rank,size = comm.Get_rank(), comm.Get_size()
    else:
        rank,size = 0,1
    
    #read in alms for data
    est_maps_read = ["hilc", "hilc-tszandcibd"]

    #since we're testing rdn0, use one of the sims as "data"
    data_alms = {}
    data_seed = (399, 0, 200)
    for m in set(est_maps_read):
        data_alms[m] = get_sim_alm_dr6_hilc(
            m, data_seed, 
            data_dir=DATA_DIR,
            mlmax=mlmax)
    
    print(data_alms)
    
    #get total Cls...hmm - maybe easiest to
    #estimate these from input maps to start with,
    #but also add an estimate from sims option
    if total_Cl_from_data:
        #get from data
        cltot_dict = get_data_Cltot_dict(data_alms,
                   sg_window=301, sg_order=2,
                w2=w2)
    else:
        pass
        #get from sims
    
    #do setup
    #decide on lmin, lmax etc.
    #and what combination of maps to use
    #est_maps = recon_config["est_maps"]
    n_alm = len( data_alms[ list(data_alms.keys())[0] ][0] )
    print("mlmax:",mlmax)

    """
    cltot_A = cltot_dict[(est_maps[0], est_maps[0])][:mlmax+1]
    cltot_B = cltot_dict[(est_maps[1], est_maps[1])][:mlmax+1]
    cltot_C = cltot_dict[(est_maps[2], est_maps[2])][:mlmax+1]
    cltot_D = cltot_dict[(est_maps[3], est_maps[3])][:mlmax+1]
    
    cltot_Y = cltot_dict[("hilc-tszandcibd", "hilc-tszandcibd")][:mlmax+1]
    cltot_XY = cltot_dict[("hilc", "hilc-tszandcibd")][:mlmax+1]
    
    """
    
    
    def run_test(est_maps, nsim_rdn0, use_mpi=False):
    
        px = qe.pixelization(shape=mask.shape,wcs=mask.wcs)
        setup = setup_ABCD_recon(
            px, lmin, lmax, mlmax,
            cl_rksz[:mlmax+1], 
            cltot_dict[(est_maps[0], est_maps[0])][:mlmax+1], #cltot_A
            cltot_dict[(est_maps[1], est_maps[1])][:mlmax+1], #cltot_B
            cltot_dict[(est_maps[2], est_maps[2])][:mlmax+1], #cltot_C
            cltot_dict[(est_maps[3], est_maps[3])][:mlmax+1], #cltot_D
            cltot_dict[(est_maps[0], est_maps[2])][:mlmax+1], #cltot_AC
            cltot_dict[(est_maps[1], est_maps[3])][:mlmax+1], #cltot_BD
            cltot_dict[(est_maps[0], est_maps[3])][:mlmax+1], #cltot_AD
            cltot_dict[(est_maps[1], est_maps[2])][:mlmax+1],#cltot_BC
            do_lh=True, do_psh=False)
        
        #run some measurements
        #filter
        data_alms_Af = [setup["filter_A"](
            data_alms[est_maps[0]][split])
                        for split in range(NSPLIT)]
        data_alms_Bf = [setup["filter_B"](
            data_alms[est_maps[1]][split])
                        for split in range(NSPLIT)]
        if est_maps[2]==est_maps[0]:
            data_alms_Cf = data_alms_Af.copy()
        if est_maps[3]==est_maps[0]:
            data_alms_Df = data_alms_Af.copy()

        def run_rdn0(setup, nsim, use_mpi=False):
            #get sim alm functions
            def get_sim_alms_A(i):
                alms = get_sim_alm_dr6_hilc(
                    est_maps[0], (200+i,0,i+1),
                    data_dir=DATA_DIR, mlmax=mlmax)
                alms_f = [setup["filter_A"](alm) for alm in alms]
                return alms_f

            def get_sim_alms_B(i):
                alms = get_sim_alm_dr6_hilc(
                    est_maps[1], (200+i,0,i+1),
                    data_dir=DATA_DIR, mlmax=mlmax)
                alms_f = [setup["filter_B"](alm) for alm in alms]
                return alms_f

            if est_maps[2]==est_maps[0]:
                get_sim_alms_C = None
            else:
                raise NotImplementedError("")
            if est_maps[3]==est_maps[0]:
                get_sim_alms_D = None
            else:
                raise NotImplementedError("")

            """
            Copy the function definition here to help get 
            the arguments right.
            mcrdn0_s4(nsims, power, qfunc_AB, split_K_func,
                  get_sim_alms_A, data_split_alms_A,
                  get_sim_alms_B=None, get_sim_alms_C=None, get_sim_alms_D=None,
                  data_split_alms_B=None, data_split_alms_C=None,data_split_alms_D=None,
                  qfunc_CD=None,
                  use_mpi=True, verbose=True, skip_rd=False, power_mcn0=None):
            """

            rdn0,mcn0 = mcrdn0_s4(nsim_rdn0, split_phi_to_cl,
                                  setup["qfunc_K_AB"],
                                  four_split_K,
                                  get_sim_alms_A, data_alms_Af,
                                  get_sim_alms_B = get_sim_alms_B,
                                  data_split_alms_B = data_alms_Bf,
                                  qfunc_CD=setup["qfunc_K_CD"],
                                  use_mpi=use_mpi)

            rdn0 /= w4
            mcn0 /= w4

            return rdn0, mcn0


        #run rdn0
        rdn0, mcn0 = run_rdn0(setup, nsim_rdn0, use_mpi=use_mpi)
        outputs = {}
        outputs["rdn0"] = rdn0
        outputs["mcn0"] = mcn0
        outputs["theory_N0"] = setup["N0_ABCD_K"]
        outputs["theory_N0_lh"] = setup["N0_ABCD_K_lh"]
        return outputs
    
    def save_rdn0_stuff(filename, stuff):
        #save pkl
        with open(filename, 'wb') as f:
            pickle.dump(stuff, f)    

    est_maps_test = [["hilc"]*4,
                     ["hilc", "hilc-tszandcibd", "hilc", "hilc"]
                     ]
    for est_maps in est_maps_test:
        print("running rdn0 test for maps: ", est_maps)
        rdn0_stuff = run_test(["hilc"]*4, nsim_rdn0, use_mpi=use_mpi)
        filename = "rdn0_test_%s.pkl"%("_".join(est_maps))
        print("saving output to %s"%filename)
        save_rdn0_stuff(filename, rdn0_stuff)


            
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="run tests")
    parser.add_argument("-m", "--mpi", action='store_true',
                        help="use_mpi")
    parser.add_argument("-n", "--nsim_rdn0", type=int, default=2)
    args = parser.parse_args()
    main(use_mpi=args.mpi, nsim_rdn0=args.nsim_rdn0)
