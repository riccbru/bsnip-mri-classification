#!/bin/bash
#SBATCH --job-name=bsnip_3dcnn
#SBATCH --output=train_bsnip_slurm_%j.out
#SBATCH --error=train_bsnip_slurm_%j.err
#SBATCH --partition=gpu2
#SBATCH --account=pi-bonnietfleming
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

source ~/.bashrc
conda activate adni_rcc

python train_3dcnn.py --epochs 50 --batch-size 8 --lr 1e-4 --seed 42 --device cuda