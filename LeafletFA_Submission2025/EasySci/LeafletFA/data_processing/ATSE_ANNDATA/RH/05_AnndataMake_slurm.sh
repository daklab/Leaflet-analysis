#!/bin/bash
#SBATCH --job-name=chunk_anndata
#SBATCH --output=logs/chunkAdata_%A_%a.out
#SBATCH --error=logs/chunkAdata_%A_%a.err
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-99%32
#SBATCH --partition=cpu,dev,bigmem

SCRIPT_PATH=/gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/EasySci/LeafletFA/data_processing/ATSE_ANNDATA/RH/05_AnndataMake.py
export METACELL_SUFFIX=${METACELL_SUFFIX:-""}
export PROC_DATE=${PROC_DATE:-"20260318"}
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/SpliceVI${METACELL_SUFFIX}/ATSEmap/junction_processing_${PROC_DATE}
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

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/SpliceVI/20260210/RH/ATSEmap/junction_processing_20260213
# sbatch /gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/EasySci/LeafletFA/data_processing/ATSE_ANNDATA/RH/05_AnndataMake_slurm.sh