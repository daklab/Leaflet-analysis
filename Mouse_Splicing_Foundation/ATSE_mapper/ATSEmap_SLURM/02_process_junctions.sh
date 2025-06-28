#!/bin/bash
#SBATCH --job-name=junction_proc
#SBATCH --output=logs/junction_%A_%a.out
#SBATCH --error=logs/junction_%A_%a.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-100%50
#SBATCH -p cpu

SCRIPT_PATH=/gpfs/commons/home/kisaev/Leaflet-analysis/split_process_merge_slurm_junctions.py
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper/junction_processing_20250622

# Create base directory with today's date
cd $WD

# Print debug information
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Will process file: chunks/chunk_${SLURM_ARRAY_TASK_ID}.txt"

# Check if chunk file exists
if [ ! -f "chunks/chunk_${SLURM_ARRAY_TASK_ID}.txt" ]; then
    echo "Error: Chunk file chunks/chunk_${SLURM_ARRAY_TASK_ID}.txt does not exist"
    exit 1
fi

# Process chunk
python $SCRIPT_PATH \
    --mode process \
    --chunk-file "chunks/chunk_${SLURM_ARRAY_TASK_ID}.txt" \
    --output-dir results