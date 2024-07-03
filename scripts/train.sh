#!/bin/bash

#SBATCH -J "train"

#SBATCH -A saumc
#SBATCH -p a100q

#SBATCH -n 64
#SBATCH -N 1
#SBATCH --gres=gpu:1

module load ANACONDA/Anaconda3-2024.02-1-python-3.11
module load cuda/cuda-11.7-a100q

python3 train.py
