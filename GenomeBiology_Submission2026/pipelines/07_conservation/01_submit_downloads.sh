#!/bin/bash
# Submit GSE138734 fasterq-dump commands as a SLURM array.
#
# Usage:
#   1. Generate commands:
#        python 00_download_gse138734.py GSE138734_ffq.json \
#            --outdir $FASTQ_DIR --manifest $FASTQ_DIR/manifest.csv \
#            > $FASTQ_DIR/download_commands.sh
#   2. Submit:
#        bash 01_submit_downloads.sh $FASTQ_DIR/download_commands.sh $LOG_DIR
#
# Each SLURM task runs one fasterq-dump + pigz block (one sample).
# Lines are grouped by blank lines (5 lines per sample).

set -euo pipefail

COMMANDS_FILE=${1:?Usage: $0 <commands_file> <log_dir>}
LOG_DIR=${2:?Usage: $0 <commands_file> <log_dir>}

mkdir -p "$LOG_DIR"

# Split into per-sample temp scripts
TMP_DIR=$(mktemp -d)
SAMPLE_IDX=0

# Each sample block is 5 lines + 1 blank line
while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ -z "$line" ]]; then
        SAMPLE_IDX=$((SAMPLE_IDX + 1))
        continue
    fi
    SCRIPT="$TMP_DIR/sample_${SAMPLE_IDX}.sh"
    # Strip trailing backslash-space continuation (it's cosmetic in the output)
    echo "${line% \\}" >> "$SCRIPT"
done < "$COMMANDS_FILE"

N_SAMPLES=$((SAMPLE_IDX))
echo "Submitting array of $N_SAMPLES download jobs"

sbatch \
    --array=0-$((N_SAMPLES - 1))%10 \
    --job-name=gse138734_dl \
    --partition=cpu \
    --cpus-per-task=8 \
    --mem=16G \
    --time=4:00:00 \
    --output="${LOG_DIR}/dl_%A_%a.out" \
    --error="${LOG_DIR}/dl_%A_%a.err" \
    --wrap "bash ${TMP_DIR}/sample_\${SLURM_ARRAY_TASK_ID}.sh"

echo "Logs → $LOG_DIR"
echo "Temp scripts → $TMP_DIR  (safe to delete after jobs complete)"
