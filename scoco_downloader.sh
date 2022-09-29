#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=cocodownload
#SBATCH --output=/u/erdos/students/xcui32/cnslab/coco_downloader.out
#SBATCH --nodelist=node002


module purg
module load gcc5 cuda10.1
module load openmpi/cuda/64
module load pytorch-py36-cuda10.1-gcc
module load ml-pythondeps-py36-cuda10.1-gcc


python3 /u/erdos/students/xcui32/cnslab/coco_downloader.py --annotation_path /u/erdos/cnslab/coco/annotations/instances_train2017.json --img_root /u/erdos/cnslab/coco/images --classes None --images_per_class 600 --n_classes 40 
