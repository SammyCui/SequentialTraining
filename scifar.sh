#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --job-name=ci10r18
#SBATCH --nodelist=node002
#SBATCH --mail-user=xcui32@fordham.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/u/erdos/students/xcui32/cnslab/results/Cifar10R18BlackBenchmark/console.out

module purg
module load gcc5 cuda10.1
module load openmpi/cuda/64
module load pytorch-py36-cuda10.1-gcc
module load ml-pythondeps-py36-cuda10.1-gcc


python3 /u/erdos/students/xcui32/cnslab/cifar.py

