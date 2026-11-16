#!/bin/bash
#SBATCH --job-name=figs_validation
#SBATCH --partition=cpu
#SBATCH --exclude=ne1dg7-[001-011]
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_bioinformatics/paper_figures/logs/figs_validation_%j.log
# Validation candidate figures (Karin 2026-11-16, item 3): figure AN panel b candidate (check 17) and three supplementary
# figures (checks 16/16b/17, simulations 15/15b, scaling). Memory: earlier figure build peak 0.6 GB + 30 %, rounded up.
source /gpfs/commons/home/kisaev/miniconda3/etc/profile.d/conda.sh
conda activate LeafletSC
cd /gpfs/commons/home/kisaev/Leaflet-analysis-bioinformatics/Bioinformatics2026/paper/figures
python build_validation_figures.py
