#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#BATCH --gres=gpu:1
#SBATCH --output=/u/erdos/students/xcui32/cnslab/results/console.out

module purg
module load xmltodict
module load gcc5 cuda10.1
module load openmpi/cuda/64
module load pytorch-py36-cuda10.1-gcc
module load ml-pythondeps-py36-cuda10.1-gcc
python test2.py