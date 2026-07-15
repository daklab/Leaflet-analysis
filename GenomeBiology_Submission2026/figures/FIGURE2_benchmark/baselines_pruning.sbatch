#!/bin/bash
#SBATCH --job-name=baseln
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/FIGURE2_benchmark/baselines_pruning.log
cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/FIGURE2_benchmark
/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python -u compute_baselines_and_pruning.py
