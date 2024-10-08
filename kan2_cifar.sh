#!/bin/bash

#$ -M zyuan2@nd.edu   # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -pe smp 1         # Specify parallel environment and legal core size
#$ -q gpu@@yye7_lab        # Run on the GPU cluster
#$ -l gpu_card=1     # Run on 1 GPU card
#$ -N umamba703      # Specify job name

# # 定义 Miniconda 的路径
# CONDA_PATH=/afs/crc.nd.edu/user/z/zyuan2/miniconda3

# # 激活 Conda 环境
# source $CONDA_PATH/bin/activate

# conda env list
# export CUDA_VISIBLE_DEVICES=3
# module load conda
conda activate base

export PATH=$PATH:/afs/crc.nd.edu/user/z/zyuan2/.local/bin

python main.py --data-set CIFAR
