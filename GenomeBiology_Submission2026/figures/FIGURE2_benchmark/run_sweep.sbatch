#!/bin/bash
#SBATCH --job-name=sim_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/FIGURE2_benchmark/sweep.log
cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/FIGURE2_benchmark
# -u = unbuffered, so progress STREAMS to sweep.log instead of appearing only at the end.
# Each fit is appended to sweep_results.csv as it finishes, and re-running resumes.
/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python -u run_sweep.py
