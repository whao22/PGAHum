#!/bin/bash
#SBATCH --job-name=HF
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=gpujl
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node04

####SBATCH --exclude=node04,node14,node15

# show currrent status
echo Start time is `date`
echo Directory is $PWD 
echo This job runs on the following nodes: $SLURM_JOB_NODELIST
conda info --envs
gpustat

################################################################
##################### CUSTOM SCRIPTS START #####################

echo "CUSTOM SCRIPTS"

###################### CUSTOM SCRIPTS END ######################
################################################################

echo Current time is `date`
echo Job completed!