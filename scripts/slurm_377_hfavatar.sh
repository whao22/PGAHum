#!/bin/bash
#SBATCH --job-name=hf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=256G
#SBATCH --partition=gpujl
#SBATCH --gres=gpu:2
#SBATCH --nodelist=node27

# show currrent status
echo Start time is `date`
echo Directory is $PWD 
echo This job runs on the following nodes: $SLURM_JOB_NODELIST
conda info --envs
gpustat

################################################################
##################### CUSTOM SCRIPTS START #####################

user_comment='mvs_winner_wdeltasdf'
exp_comment=`date +%s`
run_name="CoreView_377_${exp_comment}_slurm_${user_comment}"
base_exp_dir="exp/${run_name}"
conf="confs/hfavatar-zju/ZJUMOCAP-377-4gpus.conf"
echo The base experiment directory is ${base_exp_dir}.

python train.py \
    --conf ${conf} \
    --base_exp_dir ${base_exp_dir} \
    --run_name ${run_name}

###################### CUSTOM SCRIPTS END ######################
################################################################

echo Current time is `date`
echo Job completed!