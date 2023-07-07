#!/bin/bash

# Assign class name to variable
CLASS_NAME="cable"

# CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 train.py --seed=42 --workers=8 \
#   --outdir=workdir/runs/finetune_cifar/$CLASS_NAME/ \
#   --batch=64 --batch-gpu=2 --lr=0.0003 \
#   --arch ncsnpp --precond=ve --transfer=workdir/pretrained_models/baseline-cifar10-32x32-uncond-ve.pkl \
#   --data=/DATA/Users/amahmood/GDrive/MVTec_AD/$CLASS_NAME \
#   --resolution=256 --augment 0.08 --cond 0 \
#   --tick=1 --snap=25 --dump=50 --duration=1

CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nproc_per_node=2 train.py --seed=42 --workers=1 \
  --outdir=workdir/runs/finetune_cifar/$CLASS_NAME/  \
  --batch=96 --batch-gpu=24 --lr=0.0003 \
  --arch ncsnpp --precond=ve --resume=workdir/runs/finetune_cifar/cable/00013-mvtec_cable_128-uncond-ncsnpp-ve-gpus1-batch96-fp32/training-state-007484.pt \
  --data=/DATA/Users/amahmood/GDrive/MVTec_AD/$CLASS_NAME \
  --resolution=128 --augment 0.05 --cond 0 \
  --tick=1 --snap=50 --dump=100 --duration=20 #--fp16=1 --ls=100

# CUDA_VISIBLE_DEVICES=1 python train.py --seed=42 --workers=2 \
#   --outdir=workdir/runs/finetune_cifar/$CLASS_NAME/  \
#   --batch=96 --batch-gpu=24 --lr=0.0003 \
#   --arch ncsnpp --precond=ve --resume=workdir/runs/finetune_cifar/cable/00013-mvtec_cable_128-uncond-ncsnpp-ve-gpus1-batch96-fp32/training-state-007484.pt \
#   --data=/DATA/Users/amahmood/GDrive/MVTec_AD/$CLASS_NAME \
#   --resolution=128 --augment 0.05 --cond 0 \
#   --tick=1 --snap=50 --dump=100 --duration=20 #--fp16=1 --ls=100