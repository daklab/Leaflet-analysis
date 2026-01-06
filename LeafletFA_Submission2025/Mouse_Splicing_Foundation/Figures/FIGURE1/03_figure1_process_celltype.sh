#!/bin/bash
#SBATCH --job-name=fig1_ct
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu,bigmem,dev
#SBATCH --output=logs/figure1_ct_%A_%a.out
#SBATCH --error=logs/figure1_ct_%A_%a.err

# This script processes a single cell type (called by SLURM array job)
# Each array task processes one cell type from the list

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# Get today's date
TODAY=$(date +%Y%m%d)

# Define paths
BASE_DIR="/gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/Mouse_Splicing_Foundation"
SCRIPT_DIR="/gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/Mouse_Splicing_Foundation/Figures/FIGURE1/"
OUTPUT_DIR="/gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/Mouse_Splicing_Foundation/Figures/FIGURE1"
TODAY=$(date +%Y%m%d)
CELLTYPES_FILE="${OUTPUT_DIR}/figure1_${TODAY}/cell_types_to_process.txt"

# Check if cell types file exists
if [ ! -f "$CELLTYPES_FILE" ]; then
    echo "ERROR: Cell types file not found at $CELLTYPES_FILE"
    exit 1
fi

# Get the cell type for this array task
# sed -n "${SLURM_ARRAY_TASK_ID}p" gets the Nth line from the file
CELLTYPE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CELLTYPES_FILE")

if [ -z "$CELLTYPE" ]; then
    echo "ERROR: Could not read cell type for array task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo ""
echo "=========================================="
echo "Processing Cell Type: $CELLTYPE"
echo "=========================================="
echo ""

# Run Python script for this cell type
python ${SCRIPT_DIR}/02_figure1_process_celltype.py \
    --celltype "$CELLTYPE" \
    --today "$TODAY"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Completed processing $CELLTYPE"
else
    echo "FAILED: Error processing $CELLTYPE (exit code: $EXIT_CODE)"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE