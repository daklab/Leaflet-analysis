#!/bin/bash
#SBATCH --job-name=chunk_anndata
#SBATCH --output=logs/chunkAdata_%A_%a.out
#SBATCH --error=logs/chunkAdata_%A_%a.err
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-99%32
#SBATCH --partition=cpu,dev,bigmem

SCRIPT_PATH=/gpfs/commons/home/kisaev/Leaflet-analysis/EasySci/LeafletFA/data_processing/ATSE_ANNDATA/RH/05_AnndataMake.py
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/ATSEmap/SpliceVI/RH/output/junction_processing_20250917
OUTPUT_DIR=${WD}/anndatas
CHUNK_DIR=${WD}/chunks

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

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/ATSEmap/SpliceVI/RH/output/junction_processing_20250917
# sbatch /gpfs/commons/home/kisaev/Leaflet-analysis/EasySci/LeafletFA/data_processing/ATSE_ANNDATA/RH/05_AnndataMake_slurm.sh