#!/bin/bash
# merge_junctions.sh
#SBATCH --job-name=junction_merge
#SBATCH --output=logs/junction_merge_%j.out
#SBATCH --error=logs/junction_merge_%j.err
#SBATCH --mem=32G
#SBATCH -p bigmem,cpu,dev
#SBATCH --cpus-per-task=4

conda activate LeafletSC

SCRIPT_PATH=/gpfs/commons/home/kisaev/Leaflet-analysis/split_process_merge_slurm_junctions.py
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/SpliceVI/202510/RH/ATSEmap/junction_processing_20251022
cd $WD

# Merge results
python $SCRIPT_PATH \
    --mode merge \
    --output-dir results \
    --merge-output results/final_junctions.pkl


