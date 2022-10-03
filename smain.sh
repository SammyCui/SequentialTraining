#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --nodelist=node002
#SBATCH --job-name=ImagenetR18
#SBATCH --mail-user=xcui32@fordham.edu
#SBATCH --mail-type=END
#SBATCH --output=/u/erdos/students/xcui32/SequentialTraining/results/ImagenetR18.out

module purg
module load gcc5 cuda10.1
module load openmpi/cuda/64
module load pytorch-py36-cuda10.1-gcc
module load ml-pythondeps-py36-cuda10.1-gcc


python3 /u/erdos/students/xcui32/SequentialTraining/main.py --regimens as_is --dataset_name Imagenet --num_samples_to_use  --test_size 0.2 --num_classes --cls_to_use --sizes '1' --input_size 150 --resize_method long --lr 0.1 --epoch 400 --model resnet18 --lr_patience 5 --early_stop_patience 20 --min_lr 0.00001 --n_folds 5 --n_folds_to_use 1 --save_progress_ckpt False --save_result_ckpt False --result_path ./results/ImagenetR18
