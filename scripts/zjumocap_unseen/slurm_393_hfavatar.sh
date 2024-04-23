#!/bin/bash
#SBATCH --job-name=393
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=71
#SBATCH --mem=400G
#SBATCH --partition=gpujl
#SBATCH --gres=gpu:4

# show currrent status
echo Start time is `date`
echo Directory is $PWD 
echo This job runs on the following nodes: $SLURM_JOB_NODELIST
conda info --envs
gpustat

################################################################
##################### CUSTOM SCRIPTS START #####################

user_comment='mvs_1_1_3_true_woinner'
exp_comment=`date +%s`
run_name="CoreView_393_${exp_comment}_slurm_${user_comment}"
base_exp_dir="exp/${run_name}"
conf="confs/hfavatar-zjumocap/ZJUMOCAP-393-4gpus.conf"
echo The base experiment directory is ${base_exp_dir}.

python train.py \
    --conf ${conf} \
    --base_exp_dir ${base_exp_dir} \
    --run_name ${run_name}

###################### CUSTOM SCRIPTS END ######################
################################################################
ps -ef | grep wangyubo | grep wandb | grep -v grep | awk '{print $2}' | xargs kill -9
echo Current time is `date`
echo Job completed!
