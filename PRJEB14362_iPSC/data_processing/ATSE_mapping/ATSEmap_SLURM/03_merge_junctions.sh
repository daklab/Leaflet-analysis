#!/bin/bash
#SBATCH --job-name=junction_merge
#SBATCH --output=logs/junction_merge_%j.out
#SBATCH --error=logs/junction_merge_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

SCRIPT_PATH=/gpfs/commons/home/kisaev/Leaflet-analysis/split_process_merge_slurm_junctions.py
WD=/gpfs/commons/projects/knowles_singlecell_splicing/PRJEB14362/LeafletFA/ATSEs/022025/junction_processing_20250210

# Create base directory with today's date
cd $WD
echo "Currently in the directory: $(pwd)"

# Merge results
python $SCRIPT_PATH \
    --mode merge \
    --output-dir results \
    --merge-output results/final_junctions.pkl