#!/bin/bash
# 02_process.sh — SLURM array job: process one chunk of junction files
#
# Submit with:
#   N_CHUNKS=100
#   sbatch --array=0-$((N_CHUNKS-1))%50 \
#       pipelines/01_junctions/02_process.sh $CONFIG $PROC_DIR
#
#SBATCH --job-name=junc_proc
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu,bigmem

set -euo pipefail

CONFIG=${1:-}
PROC_DIR=${2:-}

if [[ -z "$CONFIG" || -z "$PROC_DIR" ]]; then
    echo "ERROR: Usage: $0 <config_yaml> <processing_dir>"
    exit 1
fi

SPLIT_SCRIPT=/gpfs/commons/home/kisaev/Leaflet-analysis/General_Utils/split_process_merge_slurm_junctions.py

# Read ATSEmapper params from config
read_cfg() {
    python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c$1)"
}

MIN_INTRON=$(read_cfg "['atsemapper']['min_intron']")
MAX_INTRON=$(read_cfg "['atsemapper']['max_intron']")
MIN_READS=$(read_cfg "['atsemapper']['min_junc_reads']")
MIN_CELLS=$(read_cfg "['atsemapper']['min_num_cells']")
BATCH_SIZE=$(read_cfg "['atsemapper']['batch_size']")
NUM_WORKERS=$(read_cfg "['atsemapper']['num_workers']")
SEQ_TYPE=$(read_cfg "['anndata']['sequencing_type']")

cd "$PROC_DIR"

CHUNK_FILE="chunks/chunk_${SLURM_ARRAY_TASK_ID}.txt"
if [[ ! -f "$CHUNK_FILE" ]]; then
    echo "ERROR: Chunk file not found: $CHUNK_FILE"
    exit 1
fi

echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Processing: $CHUNK_FILE"
echo "Params: min_reads=$MIN_READS min_cells=$MIN_CELLS sequencing_type=$SEQ_TYPE"

python3 "$SPLIT_SCRIPT" \
    --mode process \
    --chunk-file "$CHUNK_FILE" \
    --output-dir results \
    --min-intron   "$MIN_INTRON" \
    --max-intron   "$MAX_INTRON" \
    --min-reads    "$MIN_READS" \
    --min-cells    "$MIN_CELLS" \
    --batch-size   "$BATCH_SIZE" \
    --num-workers  "$NUM_WORKERS" \
    --sequencing-type "$SEQ_TYPE"

echo "Done: chunk $SLURM_ARRAY_TASK_ID"
