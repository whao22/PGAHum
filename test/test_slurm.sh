#!/bin/bash
#SBATCH --job-name=hf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=256G
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

echo "CUSTOM SCRIPTS"

###################### CUSTOM SCRIPTS END ######################
################################################################

echo Current time is `date`
echo Job completed!