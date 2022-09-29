#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output=/u/erdos/students/xcui32/cnslab/counter.out
#SBATCH --nodelist=node002
#SBATCH --job-name=counter



module purg
module load gcc5 cuda10.1
module load openmpi/cuda/64
module load pytorch-py36-cuda10.1-gcc
module load ml-pythondeps-py36-cuda10.1-gcc


python3 /u/erdos/students/xcui32/cnslab/count_imagenet.py

