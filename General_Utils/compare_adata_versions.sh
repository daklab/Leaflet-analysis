#!/bin/bash
#SBATCH --job-name=compare_adata
#SBATCH --partition=bigmem
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/commons/home/kisaev/Leaflet-analysis/General_Utils/compare_adata_%j.out
#SBATCH --error=/gpfs/commons/home/kisaev/Leaflet-analysis/General_Utils/compare_adata_%j.err

source /gpfs/commons/home/kisaev/miniconda3/etc/profile.d/conda.sh
conda activate LeafletSC

python /gpfs/commons/home/kisaev/Leaflet-analysis/General_Utils/compare_adata_versions.py
