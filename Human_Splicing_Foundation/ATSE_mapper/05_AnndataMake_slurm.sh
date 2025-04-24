#!/bin/bash
#SBATCH --job-name=chunk_anndata
#SBATCH --output=logs/chunkAdata_%A_%a.out
#SBATCH --error=logs/chunkAdata_%A_%a.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-99%32

SCRIPT_PATH=/gpfs/commons/home/kisaev/Leaflet-analysis/Human_Splicing_Foundation/ATSE_mapper/05_AnndataMake.py
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/ATSE_mapper/junction_processing_20250421
OUTPUT_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/ATSE_mapper/junction_processing_20250421/anndatas
CHUNK_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/ATSE_mapper/junction_processing_20250421/chunks

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