#!/bin/bash
#SBATCH --job-name=junction_proc
#SBATCH --output=logs/junction_%A_%a.out
#SBATCH --error=logs/junction_%A_%a.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH -p bigmem,cpu,dev
# NOTE: set --array at submit time, e.g. sbatch --array=0-141%30

# Set METACELL_SUFFIX and PROC_DATE before submitting
SUFFIX="${METACELL_SUFFIX:-}"
PROC_DATE="${PROC_DATE:-$(date +%Y%m%d)}"
BASE="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/SpliceVI${SUFFIX}"

SCRIPT_PATH=/gpfs/commons/home/kisaev/Leaflet-analysis/General_Utils/split_process_merge_slurm_junctions.py
WD="$BASE/ATSEmap/junction_processing_${PROC_DATE}"
cd "$WD"

echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Will process file: chunks/chunk_${SLURM_ARRAY_TASK_ID}.txt"

if [ ! -f "chunks/chunk_${SLURM_ARRAY_TASK_ID}.txt" ]; then
    echo "Error: Chunk file chunks/chunk_${SLURM_ARRAY_TASK_ID}.txt does not exist"
    exit 1
fi

python $SCRIPT_PATH \
    --mode process \
    --chunk-file "chunks/chunk_${SLURM_ARRAY_TASK_ID}.txt" \
    --output-dir results

# Usage:
#   METACELL_SUFFIX=_500 PROC_DATE=20260317 sbatch --array=0-141%30 02_process_junctions.sh