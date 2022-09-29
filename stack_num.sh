#!/bin/bash

#SBATCH --exclude=node[001] 

#SBATCH --mail-user=yliu707@fordham.edu

#SBATCH --mail-type=ALL

#SBATCH --gres=gpu:1

#SBATCH --output=stack_num.txt

#SBATCH --mem=160gb

#SBATCH --time=7-00:00:00


#load modules


#module load keras-py36-cuda10.1-gcc/2.3.1



#module load ml-pythondeps-py36-cuda10.1-gcc/3.3.0



#module load cudnn7.6-cuda10.1/7.6.5.32



#module load cuda10.1/toolkit/10.1.243

#module load cuda11.0/toolkit/11.0.3               



#module load shared tensorflow2-py36-cuda10.1-gcc/2.0.0

#module load tensorflow2-py36-cuda10.1-gcc/2.0.0 ml-pythondeps-py37-cuda10.2-gcc8/4.5.3

#module load openmpi/cuda/64/3.1.4

#module load ml-pythondeps-py36-cuda10.1-gcc/3.3.0 
#module load python36
module load ml-pythondeps-py37-cuda10.1-gcc


#module load cuda10.1/toolkit/10.1.243



#module load cudnn7.6-cuda10.1/7.6.5.32



#module load keras-py36-cuda10.1-gcc/2.3.1



#module load cm-ml-python3deps/3.3.0  



#module load tensorflow2-py36-cuda10.1-gcc/2.0.0

module load tensorflow2-py37-cuda10.1-gcc


#module load openmpi/cuda/64/3.1.4                                    



#module load openmpi/gcc/64/1.10.7


#module load keras-py36-cuda10.1-gcc/2.3.1           
#module load keras-py36-cuda10.1-gcc
#module load ml-pythondeps-py36-cuda10.1-gcc/3.3.0   
#module load keras-py37-cuda10.1-gcc/2.3.1 

#module load ml-pythondeps-py37-cuda10.2-gcc8/4.5.3  
#module load ml-pythondeps-py37-cuda11.2-gcc8/4.5.3  





#module load cuda10.1/toolkit

#module load cuda10.2/toolkit
module load cuda11.0/toolkit 
#module load cuda11.2/toolkit


#module load shared tensorflow2-py36-cuda10.1-gcc
#module load tensorflow-py36-cuda10.1-gcc/1.15.3

#module load tensorflow2-py37-cuda10.2-gcc8/2.4.1    

#module load openmpi-geib-cuda11.2-gcc8


#module load cudnn7.6-cuda10.2
#module load cudnn8.0-cuda11.0
#module load cudnn7.6-cuda10.1
#module load cudnn8.1-cuda11.2






# Parameters:

# 1 data directory 

# 2 output directory 

# 3 model directory



# this is to test rotation predictions



python3.7  Ensemble_stacking_num.py \

