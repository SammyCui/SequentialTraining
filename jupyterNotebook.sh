#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=15GB
#SBATCH --output=/u/erdos/students/xcui32/cnslab/results/console.out

module purge
module load jupyter
module load gcc5 cuda10.1
module load openmpi/cuda/64
module load pytorch-py36-cuda10.1-gcc
module load ml-pythondeps-py36-cuda10.1-gcc

cat /etc/hosts
jupyter notebook --no-browser --port=7777
