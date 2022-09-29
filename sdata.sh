#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --job-name=imgntadj
#SBATCH --mail-user=xcui32@fordham.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/u/erdos/students/xcui32/cnslab/results/imgnt-5000R18BlackCURlrAdjust/console.out

module purg
module load gcc5 cuda10.1
module load openmpi/cuda/64
module load pytorch-py36-cuda10.1-gcc
module load ml-pythondeps-py36-cuda10.1-gcc

python3 data_utils.py /u/erdos/cnslab/imagenet-bndbox/bndbox/ /u/erdos/cnslab/imagenet/n01742172/n01742172_9999.JPEG
