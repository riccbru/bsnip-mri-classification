#!/bin/bash
#SBATCH --job-name=bsnip_3dcnn_v5
#SBATCH --output=logs/bsnip_3dcnn_v5_%j.out
#SBATCH --error=logs/bsnip_3dcnn_v5_%j.err
#SBATCH --partition=gpu2
#SBATCH --account=pi-bonnietfleming
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:30:00

source ~/.bashrc
conda activate adni_rcc

# 1. Training con LR adeguato e Global Average Pooling
python train_3dcnn.py \
    --exp-name bsnip_3dcnn_v5 \
    --use-gap \
    --augment \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-4 \
    --seed 42 \
    --device cuda

# 2. Generazione Grad-CAM automatica
python gradcam_bsnip.py --exp-name bsnip_3dcnn_v5 --num-samples 20
