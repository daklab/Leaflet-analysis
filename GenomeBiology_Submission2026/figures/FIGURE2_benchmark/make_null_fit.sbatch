#!/bin/bash
#SBATCH --job-name=nullfit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/FIGURE2_benchmark/make_null_fit.log
cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/FIGURE2_benchmark
/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python -u make_null_fit.py
