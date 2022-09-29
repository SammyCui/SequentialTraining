#!/bin/bash

#SBATCH --job-name=getBrightness
#SBATCH --mail-user=xcui32@fordham.edu
#SBATCH --output=/u/erdos/students/xcui32/cnslab/brightness.out

module purg
module load gcc5 cuda10.1
module load openmpi/cuda/64
module load pytorch-py36-cuda10.1-gcc
module load ml-pythondeps-py36-cuda10.1-gcc
python3 /u/erdos/students/xcui32/cnslab/get_brightness_stat.py /u/erdos/students/xcui32/cnslab/datasets/VOC2012/VOC2012_filtered/train/root
