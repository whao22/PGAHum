#!/bin/bash

#SBATCH --partition=gpujl

# show currrent status
export SLURM_OUTPUT_DIR=./output
echo Time is `date`
echo Directory is $PWD 
echo This job runs on the following nodes: $SLURM_JOB_NODELIST
conda info --envs
gpustat

# print complete
echo completed jobs