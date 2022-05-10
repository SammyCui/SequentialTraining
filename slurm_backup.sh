#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=VOCAlexnet8
#SBATCH --mail-user=xcui32@fordham.edu
#SBATCH --output=/u/erdos/students/xcui32/cnslab/results/VOCAlexnet8_v3/console.out

module purg
module load gcc5 cuda10.1
module load openmpi/cuda/64
module load pytorch-py36-cuda10.1-gcc
module load ml-pythondeps-py36-cuda10.1-gcc/3.3.0

export CUDA_VISIBLE_DEVICES=0

python3 /u/erdos/students/xcui32/cnslab/main.py