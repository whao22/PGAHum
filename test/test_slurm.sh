#!/bin/bash
#SBATCH --job-name=test
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=gpujl
#SBATCH --gres=gpu:4
#SBATCH --output=output.txt

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