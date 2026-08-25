#!/bin/bash
#SBATCH --job-name=bsnip_v3_gap
#SBATCH --output=logs/bsnip_v2_gap_%j.out
#SBATCH --error=logs/bsnip_v3_gap_%j.err
#SBATCH --partition=gpu2
#SBATCH --account=pi-bonnietfleming
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

source ~/.bashrc
conda activate adni_rcc

python /home/ricbruno/bsnip-mri-classification/train_3dcnn.py \
    --seed 42 \
    --lr 2e-5 \
    --epochs 50 \
    --device cuda \
    --batch-size 4 \
    --log-csv training_log_bsnip_v3_gap.csv \
    --checkpoint-path best_bsnip_v3_gap.pth

