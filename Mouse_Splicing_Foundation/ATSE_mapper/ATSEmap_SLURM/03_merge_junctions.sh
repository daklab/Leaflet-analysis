#!/bin/bash
# merge_junctions.sh
#SBATCH --job-name=junction_merge
#SBATCH --output=logs/junction_merge_%j.out
#SBATCH --error=logs/junction_merge_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu,dev,bigmem

SCRIPT_PATH=/gpfs/commons/home/kisaev/Leaflet-analysis/split_process_merge_slurm_junctions.py
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper/junction_processing_20250622
cd $WD

# Merge results
python $SCRIPT_PATH \
    --mode merge \
    --output-dir results \
    --merge-output results/final_junctions.pkl