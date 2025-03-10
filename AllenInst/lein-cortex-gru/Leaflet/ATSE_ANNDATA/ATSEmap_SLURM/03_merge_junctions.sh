#!/bin/bash
# merge_junctions.sh
#SBATCH --job-name=junction_merge
#SBATCH --output=logs/junction_merge_%j.out
#SBATCH --error=logs/junction_merge_%j.err
#SBATCH --mem=200G
#SBATCH -p bigmem
#SBATCH --cpus-per-task=4

SCRIPT_PATH=/gpfs/commons/home/kisaev/Leaflet-analysis/split_process_merge_slurm_junctions.py
WD=/gpfs/commons/datasets/controlled/BRAIN_NeMO/lein-human-cortex/junction_processing_20250304

# Create base directory with today's date
cd $WD

# Merge results
python $SCRIPT_PATH \
    --mode merge \
    --output-dir results \
    --merge-output results/final_junctions.pkl