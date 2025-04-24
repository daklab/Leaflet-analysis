#!/bin/bash
#SBATCH --job-name=chunk_anndata
#SBATCH --output=logs/chunkAdata_%A_%a.out
#SBATCH --error=logs/chunkAdata_%A_%a.err
#SBATCH --mem=300G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-99%64
#SBATCH -p bigmem

SCRIPT_PATH=/gpfs/commons/home/kisaev/Leaflet-analysis/TabulaSenis/ATSEs/05_AnndataMake.py
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/ATSEmap/output/junction_processing_20250324
OUTPUT_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/ATSEmap/output/junction_processing_20250324/anndatas
CHUNK_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/ATSEmap/output/junction_processing_20250324/chunks

# Create base directory with today's date
cd $WD

# if $OUTPUT_DIR does not exist, create it
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