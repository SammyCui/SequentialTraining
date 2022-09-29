#!/bin/bash
#SBATCH -N 3
#SBATCH -n 3
#SBATCH --cpus-per-task=6
#BATCH --gres=gpu:1
#SBATCH --job-name=VOC8classAllgrey20Class
#SBATCH --mail-user=xcui32@fordham.edu
#SBATCH --output=/u/erdos/students/xcui32/cnslab/results/consoleVOCAll20Classgrey.out

module purg
module load gcc5 cuda10.1
module load openmpi/cuda/64
module load pytorch-py36-cuda10.1-gcc
module load ml-pythondeps-py36-cuda10.1-gcc
python3 /u/erdos/students/xcui32/cnslab/run.py

