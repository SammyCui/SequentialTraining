#!/bin/bash
#SBATCH -N 3
#SBATCH -n 3
#SBATCH --cpus-per-task=6
#BATCH --gres=gpu:1
#SBATCH --mail-user=xcui32@fordham.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=test_xla
#SBATCH --output=/u/erdos/students/xcui32/cnslab/xla_test.out

module purge
module load gcc5 cuda10.1
module load openmpi/cuda/64
module load pytorch-py36-cuda10.1-gcc
module load ml-pythondeps-py36-cuda10.1-gcc/3.3.0
module load pytorch-py37-cuda10.2-gcc8
python3 /u/erdos/students/xcui32/cnslab/test5.py
