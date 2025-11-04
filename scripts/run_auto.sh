#!/bin/bash

#source  /global/cfs/projectdirs/act/data/maccrann/lenspipe_new/bin/activate
source /global/cfs/projectdirs/act/data/maccrann/lenspipe_py3.13/bin/activate
export DISABLE_MPI=false
export MPI4PY_RC_RECV_MPROBE=0

#tag=dr6v3_v7000_actonly_lmax5000_gausssims
#tag=dr6v3_v7000_lmax4000_gausssims
#tag=dr6v3_v7000_lmin4000_lmax5000_gausssims
#tag=dr6v3_v7000_lmin2000_lmax3000_gausssims

#for tag in dr6v4_v4_lmax4000_mask60sk_noisysims_sourcethreshmask
#for tag in dr6v3_v7000_lmax5000_gausssims_lh dr6v3_v7000_lmax5000_gausssims_psh
#for tag in dr6v4_v4_lmax4000_mask60sk_nocross_fullsims
for tag in dr6v4_v4_lmax4000_mask60sk_noisysims_re-run
do
    
config=configs/${tag}.yml

outdir="/pscratch/sd/m/maccrann/ksz_outputs/"

for est_maps in hilc_hilc_hilc_hilc  # hilc_hilc_hilc_hilc # hilc_hilc-tszandcibd_hilc_hilc #hilc_hilc_hilc_hilc
#for est_maps in hilc_hilc-tszandcibd_hilc_hilc hilc_hilc_hilc_hilc
#for est_maps in hilc_hilc-tszandcibd_hilc_hilc-tszandcibd
#for est_maps in ilc_ilc_ilc_ilc
do
    echo $est_maps
    echo `date`

    #auto
    cmd="srun -u -l -n 8 python ../bin/run_auto.py ${outdir}/output_${est_maps}_${tag} -c $config --use_mpi True --est_maps $est_maps"
    echo $cmd
    $cmd

    #rdn0 and meanfield
    cmd="srun -u -l -n 32 python ../bin/run_auto.py ${outdir}/output_${est_maps}_${tag} -c $config --use_mpi True --do_rdn0 True --do_meanfield True  --skip_auto True --est_maps $est_maps"
    echo $cmd
    $cmd

    echo `date`
done
done
