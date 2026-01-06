#!/bin/bash
# Master submission script for parallel cell type processing

# Setup directories
conda activate LeafletSC 
BASE_DIR="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
SCRIPT_DIR="/gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/Mouse_Splicing_Foundation/Figures/FIGURE1"
OUTPUT_DIR="/gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/Mouse_Splicing_Foundation/Figures/FIGURE1"
TODAY=$(date +%Y%m%d)

cd $OUTPUT_DIR
mkdir -p logs
mkdir -p ${OUTPUT_DIR}/figure1_${TODAY}/intermediate

# First, run preprocessing to get the list of cell types
echo "Step 1: Running preprocessing to identify cell types..."
sbatch --job-name=fig1_prep \
       --time=4:00:00 \
       --mem=300G \
       --partition=cpu,bigmem \
       --cpus-per-task=4 \
       --output=logs/figure1_prep_%j.out \
       --error=logs/figure1_prep_%j.err \
       --wrap "python ${SCRIPT_DIR}/01_figure1_preprocess.py"

# Check if cell types file was created
CELLTYPES_FILE="${OUTPUT_DIR}/figure1_${TODAY}/cell_types_to_process.txt"
if [ ! -f "$CELLTYPES_FILE" ]; then
    echo "ERROR: Cell types file not found at $CELLTYPES_FILE"
    exit 1
fi

# Count cell types
N_CELLTYPES=$(wc -l < "$CELLTYPES_FILE")
echo "Found $N_CELLTYPES cell types to process"

# Submit array job for all cell types
echo "Step 2: Submitting array job for $N_CELLTYPES cell types..."
ARRAY_JOB=$(sbatch --parsable \
       --job-name=fig1_ct \
       --time=12:00:00 \
       --mem=50G \
       --partition=cpu,bigmem,dev \
       --cpus-per-task=4 \
       --array=1-${N_CELLTYPES}%50 \
       --output=logs/figure1_ct_%A_%a.out \
       --error=logs/figure1_ct_%A_%a.err \
       ${SCRIPT_DIR}/03_figure1_process_celltype.sh)

echo "Submitted array job: $ARRAY_JOB"

# Submit merging job that depends on array job completion
echo "Step 3: Submitting merge job (depends on array completion)..."
sbatch --job-name=fig1_merge \
       --time=4:00:00 \
       --mem=100G \
       --partition=cpu,bigmem \
       --cpus-per-task=4 \
       --output=logs/figure1_merge_%j.out \
       --error=logs/figure1_merge_%j.err \
       --wrap "python ${SCRIPT_DIR}/05_figure1_merge_results.py"

echo "Pipeline submitted successfully!"
echo "Monitor with: squeue -u $USER | grep fig1"