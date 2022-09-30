#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --nodelist=node002
#SBATCH --job-name=ci10r18
#SBATCH --mail-user=xcui32@fordham.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/u/erdos/students/xcui32/SequentialTraining/results/Cifar10R18test.out

module purg
module load gcc5 cuda10.1
module load openmpi/cuda/64
module load pytorch-py36-cuda10.1-gcc
module load ml-pythondeps-py36-cuda10.1-gcc


python3 /u/erdos/students/xcui32/SequentialTraining/main.py --regimens all --dataset_name VOC --train_size --test_size --num_classes --cls_to_use --input_size '0.2,0.4,0.6,0.8,1' --sizes 150 --resize_method long --lr 0.1 --epoch 200 --model resnet18 --lr_patience 3 --min_lr 0.00001 --n_folds 5 --n_folds_to_use 5 --save_progress_ckpt False --save_result_ckpt False --result_path ./results/VOC
