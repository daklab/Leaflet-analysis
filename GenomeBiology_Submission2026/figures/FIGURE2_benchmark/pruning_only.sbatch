#!/bin/bash
#SBATCH --job-name=prune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/FIGURE2_benchmark/pruning_only.log
cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/FIGURE2_benchmark
/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python -u -c "
import compute_baselines_and_pruning as m
" 2>/dev/null || /gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python -u compute_baselines_and_pruning.py
