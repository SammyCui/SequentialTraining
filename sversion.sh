#!/bin/bash
#SBATCH --job-name=printversion
#SBATCH --mail-user=xcui32@fordham.edu
#SBATCH --nodelist=node002
#SBATCH --output=/u/erdos/students/xcui32/cnslab/version.out

module purg
module load gcc5 cuda10.1
module load openmpi/cuda/64
module load pytorch-py36-cuda10.1-gcc
module load ml-pythondeps-py36-cuda10.1-gcc


python3 /u/erdos/students/xcui32/cnslab/get_version.py

