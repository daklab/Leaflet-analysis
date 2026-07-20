#!/bin/bash
# 05_make_anndata_slurm.sh — submit one SLURM array job per chunk
#
# Usage: bash 05_make_anndata_slurm.sh <config_yaml> <processing_dir> <atse_file>
#
# Example:
#   PROC=/gpfs/.../MOUSE_SPLICING_FOUNDATION/ATSE_mapper/junction_processing_20260510
#   ATSE=/gpfs/.../ATSE_files/MOUSE_SMARTSEQ2_ATSE_FILE_unanno_also_2026-05-10_12-00-00.txt.gz
#   bash 05_make_anndata_slurm.sh ../../config/mouse_smartseq2.yaml $PROC $ATSE

set -euo pipefail

CONFIG=${1:?"Usage: $0 <config> <proc_dir> <atse_file>"}
PROC_DIR=${2:?}
ATSE_FILE=${3:?}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SCRIPT="$SCRIPT_DIR/05_make_anndata.py"

ANNDATA_DIR="$PROC_DIR/anndatas"
JUNCTIONS_PKL="$PROC_DIR/results/final_junctions.pkl"
CHUNKS_DIR="$PROC_DIR/chunks"
LOGS_DIR="$PROC_DIR/logs"

mkdir -p "$ANNDATA_DIR" "$LOGS_DIR"

N_CHUNKS=$(ls "$CHUNKS_DIR"/*.txt | wc -l)
MAX_IDX=$((N_CHUNKS - 1))

echo "Submitting AnnData jobs for $N_CHUNKS chunks (array 0-$MAX_IDX)..."

sbatch \
    --job-name=make_adata \
    --output="$LOGS_DIR/adata_%A_%a.out" \
    --error="$LOGS_DIR/adata_%A_%a.err" \
    --mem=64G \
    --cpus-per-task=4 \
    --partition=cpu,bigmem \
    --array="0-${MAX_IDX}%30" \
    --wrap="python $SCRIPT \
        --config $CONFIG \
        --chunk-file $CHUNKS_DIR/chunk_\${SLURM_ARRAY_TASK_ID}.txt \
        --atse-file $ATSE_FILE \
        --junctions-pkl $JUNCTIONS_PKL \
        --output-dir $ANNDATA_DIR \
        --chunk-id \${SLURM_ARRAY_TASK_ID}"

echo ""
echo "After all jobs complete, run:"
echo "  python 06_merge_anndatas.py --input-dir $ANNDATA_DIR --output-file $ANNDATA_DIR/merged_anndata.h5ad"
echo ""
echo "Then verify counts with:"
echo "  python ../../tests/test_count_audit.py $ANNDATA_DIR/merged_anndata.h5ad $ANNDATA_DIR"
