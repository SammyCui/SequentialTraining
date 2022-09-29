#!/bin/bash
#SBATCH --output=/u/erdos/students/xcui32/cnslab/test.out

module purg
module load openmpi/cuda/64
module load ml-pythondeps-py36-cuda10.1-gcc/3.3.0
module load gcc5 cuda10.1
module load pytorch-py36-cuda10.1-gcc

pip3 install numpy
python3 test_gpu.py
