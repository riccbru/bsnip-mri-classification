#!/bin/bash
#SBATCH --job-name=bsnip_cnn_v6_5fold
#SBATCH --output=logs/bsnip_cnn_v6_5fold_%j.out
#SBATCH --error=logs/bsnip_3dcnn_v6_5fold_%j.err
#SBATCH --partition=gpu2
#SBATCH --account=pi-bonnietfleming
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:30:00

source ~/.bashrc
conda activate adni_rcc

# Execute a complete 5-Fold Cross-Validation
python train_3dcnn.py \
    --exp-name bsnip_cnn_v6_5fold \
    --use-gap \
    --augment \
    --cv-folds 5 \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-4 \
    --seed 42 \
    --device cuda
