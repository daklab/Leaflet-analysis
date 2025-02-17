#!/bin/bash
#SBATCH --job-name=chunk_anndata
#SBATCH --output=logs/Adata_%A_%a.out
#SBATCH --error=logs/Adata_%A_%a.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-99%64
#SBATCH --partition=bigmem

SCRIPT_PATH=/gpfs/commons/home/kisaev/Leaflet-analysis/PRJEB14362_iPSC/data_processing/ATSE_mapping/05_AnndataMake.py
WD=/gpfs/commons/projects/knowles_singlecell_splicing/PRJEB14362/LeafletFA/ATSEs/022025/junction_processing_20250210/
OUTPUT_DIR=/gpfs/commons/projects/knowles_singlecell_splicing/PRJEB14362/LeafletFA/ATSEs/022025/junction_processing_20250210/anndatas
CHUNK_DIR=${WD}/chunks/

# Create base directory with today's date
cd $WD

# Check if OUTPUT_DIR exists, if not create it
if [ ! -d $OUTPUT_DIR ]; then
  mkdir -p $OUTPUT_DIR
fi

# Print debug information
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Will process file: ${CHUNK_DIR}/chunk_${SLURM_ARRAY_TASK_ID}.txt"

# Process the specific chunk based on array task ID
python $SCRIPT_PATH \
  --chunk-file "${CHUNK_DIR}/chunk_${SLURM_ARRAY_TASK_ID}.txt" \
  --output-dir $OUTPUT_DIR \
  --chunk-id ${SLURM_ARRAY_TASK_ID}