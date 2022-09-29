#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --nodelist=node002
#SBATCH --job-name=voc20r50
#SBATCH --mail-user=xcui32@fordham.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/u/erdos/students/xcui32/cnslab/results/VOC20-5000R50test.out

module purg
module load gcc5 cuda10.1
module load openmpi/cuda/64
module load pytorch-py36-cuda10.1-gcc
module load ml-pythondeps-py36-cuda10.1-gcc


python3 /u/erdos/students/xcui32/cnslab/test_voc.py --lr 0.1 --epoch 400 --model resnet50 --num_workers 16 --lr_patience 3

