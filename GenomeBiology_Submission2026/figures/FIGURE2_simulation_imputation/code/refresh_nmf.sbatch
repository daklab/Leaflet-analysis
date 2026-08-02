#!/bin/bash
#SBATCH --job-name=nmf_fix
#SBATCH --partition=cpu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/FIGURE2_simulation_imputation/logs/refresh_nmf.log
cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/FIGURE2_simulation_imputation
/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python -u code/refresh_nmf.py
echo "REFRESH_NMF DONE"
