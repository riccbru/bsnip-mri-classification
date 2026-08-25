#!/bin/bash
#SBATCH --job-name=bsnip_3dcnn_v7_grid_search
#SBATCH --output=logs/bsnip_3dcnn_v7_grid_search_%j.out
#SBATCH --error=logs/bsnip_3dcnn_v7_grid_search_%j.err
#SBATCH --partition=gpu2
#SBATCH --account=pi-bonnietfleming
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=18:00:00

source ~/.bashrc
conda activate adni_rcc

# 2D grid search: lr x weight_decay, 9 configs, each a 5-fold CV run of
# train_3dcnn.py (GAP architecture). Safe to resubmit if this job hits the
# 18h wall time before finishing all 9 — run_grid_search.py skips any
# configuration whose runs/bsnip_3dcnn_v7_grid/lr_*_wd_*/cv_summary.json
# already exists, so a re-submitted job resumes rather than restarting.
python run_grid_search.py
