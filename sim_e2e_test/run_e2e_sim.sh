#!/bin/bash

#submit e2e sims
#for the first sim, we want to run Gaussian realisations, run auto, and also run rdn0 and meanfield
#then we just run autos

#source /global/common/software/act/maccrann/lenspipe/bin/activate
source /global/cfs/projectdirs/act/data/maccrann/lenspipe_py3.13/bin/activate
export DISABLE_MPI=false

for config in  auto_config_lmax4000 # auto_config_lmax5000_lh auto_config_lmax6000 
do
    
    config_file=${config}.yml
    for est_maps in hilc_hilc_hilc_hilc
    do
	
get_sim_cov_extra_args="--meanfield_dir mean_field_nsim64 --rdn0_file rdn0_outputs_nsim32.pkl"
#sim_dir="/pscratch/sd/m/maccrann/cmb/act_dr6/e2e_sims/hilc_ilc_cldata_smooth-301-2_v4_lmax6000_60skmask_lmax7000"
sim_dir="/pscratch/sd/m/maccrann/cmb/act_dr6/e2e_sims/hilc_ilc_cldata_smooth-301-2_v4_lmax6000_60skmask_31102025_lmax7000"
data_template_path="${sim_dir}/sim_planck%s_act00_%s/\${freq}_split\${split}_wksz_rot%s.fits"
data_template_path_for_rot0="${sim_dir}/sim_planck200_act00_00001/\${freq}_split\${split}_wksz_rot0.fits"
mask="/global/cfs/projectdirs/act/data/synced_maps/DR6_lensing/masks/act_mask_20220316_GAL060_rms_60.00sk.fits"

nrot=128

outdir_base="/pscratch/sd/m/maccrann/ksz_outputs/sim_e2e_test/output_${config}_${est_maps}"
outdir="${outdir_base}/output"

#run Gaussians
freq="hilc"

logfile=job_outputs/${config}_${est_maps}.log
rm $logfile
echo "writing output to $logfile"

#make gaussian sims
echo "running Gaussian sims" >> $logfile 2>&1
gaussian_sim_dir="${sim_dir}/gaussian_sims"
cmd="srun -u -l -n 16 python /global/homes/m/maccrann/cmb/ksz2/sims/generate_gaussian_sims.py $gaussian_sim_dir --data_template_path ${data_template_path_for_rot0} --freq $freq --mpi --nsim 128 --mask $mask"

#echo $cmd >> $logfile 2>&1
#$cmd >> $logfile 2>&1

echo $est_maps >> $logfile 2>&1
echo `date` >> $logfile 2>&1

sim_template_path="${gaussian_sim_dir}/\${freq}_isim\${actseed}.fits"
extra_args="--mask $mask --data_template_path $data_template_path_for_rot0 --sim_template_path $sim_template_path"

#rdn0 and mean-field
cmd="srun -u -l -n 32 python ../bin/run_auto.py ${outdir}_rot0 -c ${config}.yml --use_mpi True --do_rdn0 True --do_meanfield True --skip_auto True --est_maps $est_maps $extra_args"
#echo $cmd >> $logfile 2>&1
#$cmd >> $logfile 2>&1

#remove mean-field realisations
rm -rf ${outdir}_rot0/mean_field_nsim*/Ks_ab_sim*
rm -rf ${outdir}_rot0/mean_field_nsim*/Ks_cd_sim*

echo "running auto realizations"
echo `date` >> $logfile 2>&1
#now run other realizations
cmd="srun -u -l -n 16 python ../bin/run_auto_rots.py ${outdir} --data_template_path $data_template_path --mask $mask -c ${config}.yml --est_maps $est_maps --nrot $nrot"
#echo $cmd >> $logfile 2>&1
#$cmd >> $logfile 2>&1
#echo `date` >> $logfile 2>&1

#and get CL_KK realizations
cmd="srun -u -l -n 8 python get_sim_cov.py output_${config}_${est_maps} --mpi --nsim $nrot $get_sim_cov_extra_args"
echo $cmd >> $logfile 2>&1
$cmd >> $logfile 2>&1

done
done
