#!/bin/bash
#SBATCH -J make_mudata
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH -p bigmem,cpu
#SBATCH -o /gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/EasySci/MultiVI/make_mudata_%j.out

/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python /gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/EasySci/MultiVI/01_make_mudata.py
