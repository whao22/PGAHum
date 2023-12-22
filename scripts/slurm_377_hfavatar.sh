#!/bin/bash
#SBATCH --job-name=hfavatar # 作业名称 
#SBATCH --ntasks=1 # 任务数为1 
#SBATCH --cpus-per-task=72
#SBATCH --mem=128G
#SBATCH --partition=gpujl
#SBATCH --gres=gpu:4 # 如果需要，使用1个GPU

# show currrent status
echo Start time is `date`
echo Directory is $PWD 
echo This job runs on the following nodes: $SLURM_JOB_NODELIST
conda info --envs
gpustat

################################################################
##################### CUSTOM SCRIPTS START #####################

# user_comment=''
# user_comment='w/_inner-w/_init_sdf-rgb50-lpip5-eik0.1'
user_comment='w/_inner-w/_init_sdf-w/_tri'
exp_comment=`date +%s`
run_name="CoreView_377_${exp_comment}_slurm_${user_comment}"
base_exp_dir="exp/${run_name}"
echo The base experiment directory is ${base_exp_dir}.

python train.py \
    --conf confs/hfavatar-zju/ZJUMOCAP-377-mono-4gpus.conf  \
    --base_exp_dir ${base_exp_dir} \
    --run_name ${run_name}

###################### CUSTOM SCRIPTS END ######################
################################################################

echo Current time is `date`
echo Job completed!