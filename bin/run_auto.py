#Run the kSZ 4-point function
#
#
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

with open("../run_auto_defaults.yml",'rb') as f:
    DEFAULTS=yaml.load(f)

def get_config(description="Do 4pt measurement"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("output_dir", type=str)

    defaults = {}
    bool_keys = []
    for key,val in DEFAULTS.items():
        nargs=None
        if val=="iNone":
            t,val = int, None
        elif val=="fNone":
            t,val = float, None
        elif val=="sNone":
            t,val = str, None
        elif val=="liNone":
            t,val,nargs = int,None,'*'
        elif val=="lsNone":
            t,val,nargs = str,None,'*'
        elif type(val) is bool:
            t = str
            bool_keys.append(key)
        else:
            t = type(val)
        defaults[key] = val
        parser.add_argument("--%s"%key, type=t,
                            nargs=nargs)

    #This parser will have optional arguments set to
    #None if not set. So if that's the case, replace
    #with the default value
    args_dict=vars(parser.parse_args())
    for key in bool_keys:
        if args_dict[key] is not None:
            if args_dict[key]=='True':
                args_dict[key]=True
            elif args_dict[key]=='False':
                args_dict[key]=False
            else:
                raise ValueError('%s must be True or False'%key)

    output_dir = args_dict.pop("output_dir")
    """
    config_file = opj(args_dict['output_dir'], 'config_from_file.yml')
    with open(config_file,"rb") as f:
        try:
            config = yaml.load(f)
        except KeyError:
            config = {}
    """
    config = {}
    config['output_dir'] = output_dir
    for key,val in args_dict.items():
        if key in config:
            if val is not None:
                config[key] = val
        else:
            if val is not None:
                config[key] = val
            else:                
                config[key] = defaults[key]
    #I think most convenient to return
    #a namespace
    from argparse import Namespace
    config_namespace = Namespace(**config)
    return config_namespace

def get_alms_dr6_hilc(data_dir="/global/cscratch1/sd/maccrann/cmb/act_dr6/ilc_cldata_smooth-301-2_v1",
                      map_names=["hilc", "hilc-tszandcibd"], sim_seed=None, mlmax=None):

    #get splits
    alms = {}
    for map_name in map_names:
        alms[map_name] = []
        for split in range(NSPLIT):
            if sim_seed is None:
                map_filename = "%s_split%d.fits"%(map_name, split)
                map_path = opj(data_dir, map_filename)
            else:
                map_filename = opj("sim_planck%03d_act%02d_%05d"%(sim_seed),
                                   "%s_split%d.fits"%(map_name, split))
                map_path = opj(data_dir, map_filename)
            alm = hp.read_alm(map_path)
            if mlmax is not None:
                alm = utils.change_alm_lmax(alm, mlmax)
            alms[map_name].append(alm)
            
    return alms

def get_sim_alm_dr6_hilc(map_name, sim_seed, data_dir="/global/cscratch1/sd/maccrann/cmb/act_dr6/ilc_cldata_smooth-301-2_v1",
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

def main():

    #setup mpi if we want it
    if not disable_mpi:
        comm = MPI.COMM_WORLD
        rank,size = comm.Get_rank(), comm.Get_size()
    else:
        rank,size = 0,1
    
    #get args
    args = get_config()
    if rank==0:
        print("args:")
        print(args)
    recon_config = vars(args)
    safe_mkdir(args.output_dir)

    #get w2 and w4
    mask = enmap.read_map(MASK_PATH)
    px = qe.pixelization(shape=mask.shape,wcs=mask.wcs)
    #w2 = maps.wfactor(2, mask)
    #w4 = maps.wfactor(4, mask)
    w2 = 0.2758258447605825
    w4 = 0.2708454607428685
    if rank==0:
        print("w2:",w2)
        print("w4:",w4)

    
    #read in alms for data
    est_maps = (args.est_maps.strip()).split("_")
    data_alms = get_alms_dr6_hilc(mlmax=args.mlmax,
                                  map_names=list(set(est_maps)))

    #get total Cls...hmm - maybe easiest to
    #estimate these from input maps to start with,
    #but also add an estimate from sims option
    if args.total_Cl_from_data:
        #get from data
        cltot_dict = get_data_Cltot_dict(data_alms,
                   sg_window=301, sg_order=2,
                w2=w2)
        #save total Cls
        with open(opj(args.output_dir,
                      "total_Cl_from_data.pkl"), "wb") as f:
            pickle.dump(cltot_dict, f)
    else:
        pass
        #get from sims
    
    #do setup
    #decide on lmin, lmax etc.
    #and what combination of maps to use
    #est_maps = recon_config["est_maps"]
    lmin, lmax = recon_config["K_lmin"], recon_config["K_lmax"]

    n_alm = len( data_alms[ list(data_alms.keys())[0] ][0] )
    if recon_config["mlmax"] is None:
        mlmax = hp.Alm.getlmax(n_alm)
    else:
        mlmax = args.mlmax
    
    if rank==0:
        print("lmin,lmax:", lmin,lmax)
        print("mlmax:",mlmax)

    """
    cltot_A = cltot_dict[(est_maps[0], est_maps[0])][:mlmax+1]
    cltot_B = cltot_dict[(est_maps[1], est_maps[1])][:mlmax+1]
    cltot_C = cltot_dict[(est_maps[2], est_maps[2])][:mlmax+1]
    cltot_D = cltot_dict[(est_maps[3], est_maps[3])][:mlmax+1]
    
    cltot_Y = cltot_dict[("hilc-tszandcibd", "hilc-tszandcibd")][:mlmax+1]
    cltot_XY = cltot_dict[("hilc", "hilc-tszandcibd")][:mlmax+1]
    """

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
    """
    alm_hilc_f = [iici_setup["filter_A"](data_alms["hilc"][split])
                  for split in range(NSPLIT)]
    alm_tszandcibd_f = [iici_setup["filter_B"](data_alms["hilc-tszandcibd"][split])
                        for split in range(NSPLIT)]
    data_alms_Af = alm_hilc_f
    data_alms_Bf = alm_tszandcibd_f
    """
    assert est_maps[2]==est_maps[0]
    assert est_maps[3]==est_maps[0]
    data_alms_Cf = data_alms_Af
    data_alms_Df = data_alms_Af


    def run_ClKK():
        #K from Q(ilc, ilc-tszandcibd)
        print("running K_AB")
        Ks_ab = four_split_K(
            setup["qfunc_K_AB"],
            data_alms_Af[0], data_alms_Af[1],
            data_alms_Af[2], data_alms_Af[3],
            Xdatp_0=data_alms_Bf[0],
            Xdatp_1=data_alms_Bf[1],
            Xdatp_2=data_alms_Bf[2],
            Xdatp_3=data_alms_Bf[3],
            )
        Ks_ab_lh = four_split_K(
            setup["qfunc_K_AB_lh"],
            data_alms_Af[0], data_alms_Af[1],
            data_alms_Af[2], data_alms_Af[3],
            Xdatp_0=data_alms_Bf[0],
            Xdatp_1=data_alms_Bf[1],
            Xdatp_2=data_alms_Bf[2],
            Xdatp_3=data_alms_Bf[3],
            )

        #K from Q(ilc,ilc)
        print("running K_CD")
        Ks_cd = four_split_K(
            setup["qfunc_K_CD"],
            data_alms_Af[0], data_alms_Af[1],
            data_alms_Af[2], data_alms_Af[3],
            )
        Ks_cd_lh = four_split_K(
            setup["qfunc_K_CD_lh"],
            data_alms_Af[0], data_alms_Af[1],
            data_alms_Af[2], data_alms_Af[3],
            )

        print("getting CL_KK")
        cl_abcd = split_phi_to_cl(Ks_ab, Ks_cd)
        cl_abcd_lh = split_phi_to_cl(Ks_ab_lh, Ks_cd_lh)
        #apply w4
        cl_abcd /= w4
        cl_abcd_lh /= w4
        return cl_abcd, cl_abcd_lh
        print("done")

    if args.do_auto:
        cl_iici, cl_iici_lh = run_ClKK()
        outputs = {}
        outputs["cl_KK_raw"] = cl_iici
        outputs["cl_KK_lh_raw"] = cl_iici_lh
        outputs["N0"] = setup["N0_ABCD_K"]
        outputs["N0_lh"] = setup["N0_ABCD_K_lh"]
        #save pkl
        with open(opj(args.output_dir, "auto_outputs.pkl"), 'wb') as f:
            pickle.dump(outputs, f)

            
    def run_rdn0(setup, nsim, use_mpi=False,
                 est=None):
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
        if est=="lh":
            qfunc_AB = setup["qfunc_K_AB_lh"]
            qfunc_CD = setup["qfunc_K_CD_lh"]
        else:
            qfunc_AB = setup["qfunc_K_AB"]
            qfunc_CD = setup["qfunc_K_CD"]
        rdn0,mcn0 = mcrdn0_s4(nsim, split_phi_to_cl,
                              qfunc_AB,
                              four_split_K,
                              get_sim_alms_A, data_alms_Af,
                              get_sim_alms_B = get_sim_alms_B,
                              data_split_alms_B = data_alms_Bf,
                              qfunc_CD=qfunc_CD,
                              use_mpi=use_mpi)

        rdn0 /= w4
        mcn0 /= w4

        return rdn0, mcn0

    if args.do_rdn0:
        rdn0, mcn0 = run_rdn0(setup, args.nsim_rdn0, use_mpi=args.use_mpi)
        outputs = {}
        outputs["rdn0"] = rdn0
        outputs["mcn0"] = mcn0

        if args.do_lh:
            rdn0_lh, mcn0_lh = run_rdn0(setup, args.nsim_rdn0, use_mpi=args.use_mpi,
                                  est="lh")
            outputs["rdn0_lh"] = rdn0_lh
            outputs["mcn0_lh"] = mcn0_lh            
        
        outputs["theory_N0"] = setup["N0_ABCD_K"]
        outputs["theory_N0_lh"] = setup["N0_ABCD_K_lh"]
        #save pkl
        rdn0_file = opj(args.output_dir, "rdn0_outputs.pkl")
        print("saving rdn0 outputs to %s"%rdn0_file)
        with open(rdn0_file, 'wb') as f:
            pickle.dump(outputs, f)
    
    #run rdn0
    """
    rdn0, mcn0 = run_rdn0(setup, nsim_rdn0, use_mpi=use_mpi)
    outputs = {}
    outputs["rdn0"] = rdn0
    outputs["mcn0"] = mcn0
    outputs["theory_N0"] = iici_setup["N0_ABCD_K"]
    outputs["theory_N0_lh"] = iici_setup["N0_ABCD_K_lh"]
    return outputs
            
            
    def run_rdn0(nsim):
        #get sim alm functions
        def get_sim_alms_A(i):
            alms = get_sim_alm_dr6_hilc(
                est_maps[0], (200+i,0,i+1),
                data_dir=DATA_DIR, mlmax=mlmax)
            alms_f = [iici_setup["filter_A"](alm) for alm in alms]
            return alms_f
        
        def get_sim_alms_B(i):
            alms = get_sim_alm_dr6_hilc(
                est_maps[1], (200+i,0,i+1),
                data_dir=DATA_DIR, mlmax=mlmax)
            alms_f = [iici_setup["filter_B"](alm) for alm in alms]
            return alms_f

        if est_maps[2]==est_maps[0]:
            get_sim_alms_C = None
        else:
            raise NotImplementedError("")
        if est_maps[3]==est_maps[0]:
            get_sim_alms_D = None
        else:
            raise NotImplementedError("")


        rdn0,mcn0 = mcrdn0_s4(args.nsim_rdn0, split_phi_to_cl,
                              iici_setup["qfunc_K_AB"],
                              four_split_K,
                              get_sim_alms_A, data_alms_Af,
                              get_sim_alms_B = get_sim_alms_B,
                              data_split_alms_B = data_alms_Bf,
                              qfunc_CD=iici_setup["qfunc_K_CD"],
                              use_mpi=args.use_mpi)

        rdn0 /= w4
        mcn0 /= w4

        return rdn0, mcn0

    if args.do_rdn0:
        rdn0, mcn0 = run_rdn0(args.nsim_rdn0)
        outputs = {}
        outputs["rdn0"] = rdn0
        outputs["mcn0"] = mcn0
        outputs["theory_N0"] = iici_setup["N0_ABCD_K"]
        outputs["theory_N0_lh"] = iici_setup["N0_ABCD_K_lh"]
        #save pkl
        rdn0_file = opj(args.output_dir, "rdn0_outputs.pkl")
        print("saving rdn0 outputs to %s"%rdn0_file)
        with open(rdn0_file, 'wb') as f:
            pickle.dump(outputs, f)
    """


    

if __name__=="__main__":
    main()
