#!/bin/bash
#SBATCH --job-name=junction_proc
#SBATCH --output=logs/junction_%A_%a.out
#SBATCH --error=logs/junction_%A_%a.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-99%30

SCRIPT_PATH=/gpfs/commons/home/kisaev/Leaflet-analysis/split_process_merge_slurm_junctions.py
WD=/gpfs/commons/projects/knowles_singlecell_splicing/PRJEB14362/LeafletFA/ATSEs/022025/junction_processing_20250210

# Create base directory with today's date
cd $WD

echo "Currently in the directory: $(pwd)"

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