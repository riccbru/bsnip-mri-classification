#!/bin/bash
#SBATCH --job-name=bsnip_3dcnn_v4
#SBATCH --output=logs/train_v4_%j.out
#SBATCH --error=logs/train_v4_%j.err
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

# 1. Training + salvataggio automatico curve e metriche
python train_3dcnn.py \
    --exp-name bsnip_3dcnn_v4 \
    --augment \
    --weighted-loss \
    --epochs 50 \
    --batch-size 4 \
    --lr 2e-5 \
    --seed 42 \
    --device cuda

# 2. Generazione automatica mappe Grad-CAM
python gradcam_bsnip.py --exp-name bsnip_3dcnn_v4 --num-samples 10
