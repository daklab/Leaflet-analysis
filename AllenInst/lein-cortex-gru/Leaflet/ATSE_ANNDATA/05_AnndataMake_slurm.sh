#!/bin/bash
#SBATCH --job-name=chunk_anndata
#SBATCH --output=logs/chunkAdata_%A_%a.out
#SBATCH --error=logs/chunkAdata_%A_%a.err
#SBATCH --mem=200G
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=4
#SBATCH --array=0-99%32

SCRIPT_PATH=/gpfs/commons/home/kisaev/Leaflet-analysis/AllenInst/lein-cortex-gru/Leaflet/ATSE_ANNDATA/05_AnndataMake.py
WD=/gpfs/commons/datasets/controlled/BRAIN_NeMO/lein-human-cortex/junction_processing_20250304
OUTPUT_DIR=/gpfs/commons/datasets/controlled/BRAIN_NeMO/lein-human-cortex/junction_processing_20250304/anndatas
CHUNK_DIR=/gpfs/commons/datasets/controlled/BRAIN_NeMO/lein-human-cortex/junction_processing_20250304/chunks

# Create base directory with today's date
cd $WD

# Print debug information
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Will process file: ${CHUNK_DIR}/chunk_${SLURM_ARRAY_TASK_ID}.txt"

# Process the specific chunk based on array task ID
python $SCRIPT_PATH \
  --chunk-file "${CHUNK_DIR}/chunk_${SLURM_ARRAY_TASK_ID}.txt" \
  --output-dir $OUTPUT_DIR \
  --chunk-id ${SLURM_ARRAY_TASK_ID}