#!/bin/bash
# Run gene-level read counting on all DT metacell BAMs and build AnnData.
# Requires: featureCounts (subread), Python with anndata, pandas, scipy, pyyaml.
# make sure to first activate python3ENV conda environment and load subread module: module load subread
# Usage: ./run_DT_gene_counts.sh [config.yaml]
#        Or submit via SLURM (see below).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/config.yaml}"
OUTPUT_DIR=""
DT_BAM_DIR=""
GTF_FILE=""
THREADS=8

# Parse config (simple YAML: key: value)
if [[ -f "$CONFIG" ]]; then
    while IFS= read -r line; do
        [[ "$line" =~ ^#.*$ ]] && continue
        if [[ "$line" =~ ^([^:]+):[[:space:]]*(.*)$ ]]; then
            key="${BASH_REMATCH[1]}"
            val="${BASH_REMATCH[2]//\"/}"
            case "$key" in
                dt_bam_dir)   DT_BAM_DIR="$val" ;;
                gtf_file)     GTF_FILE="$val" ;;
                output_dir)   OUTPUT_DIR="$val" ;;
                threads)      THREADS="$val" ;;
            esac
        fi
    done < "$CONFIG"
fi

if [[ -z "$DT_BAM_DIR" || -z "$GTF_FILE" || -z "$OUTPUT_DIR" ]]; then
    echo "ERROR: Missing dt_bam_dir, gtf_file, or output_dir in config. Edit config.yaml."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Require samtools for name-sorting (needed for proper paired-end counting)
if ! command -v samtools &>/dev/null; then
    echo "ERROR: samtools not found. Load module, e.g. module load samtools"
    exit 1
fi

# Discover DT metacell BAMs: each subdir of DT_BAM_DIR has <metacell>.bam
# Name-sort each BAM so read pairs are adjacent (featureCounts -p works correctly)
BAM_LIST="${OUTPUT_DIR}/bam_list.txt"
METACELL_LIST="${OUTPUT_DIR}/metacell_ids.txt"
> "$BAM_LIST"
> "$METACELL_LIST"

for dir in "$DT_BAM_DIR"/*/; do
    [[ -d "$dir" ]] || continue
    metacell=$(basename "$dir")
    bam="${dir}${metacell}.bam"
    name_sorted_bam="${dir}${metacell}.name_sorted.bam"
    if [[ -f "$bam" && -s "$bam" ]]; then
        if [[ ! -f "$name_sorted_bam" || "$name_sorted_bam" -ot "$bam" ]]; then
            echo "Name-sorting: $bam"
            samtools sort -n -@ "$THREADS" "$bam" -o "$name_sorted_bam"
        fi
        echo "$name_sorted_bam" >> "$BAM_LIST"
        echo "$metacell" >> "$METACELL_LIST"
    fi
done

n_bams=$(wc -l < "$BAM_LIST")
echo "Found $n_bams DT metacell BAMs."

if [[ "$n_bams" -eq 0 ]]; then
    echo "ERROR: No BAMs found under $DT_BAM_DIR (expected <dt_bam_dir>/<metacell>/<metacell>.bam)"
    exit 1
fi

# Check for featureCounts (subread)
if ! command -v featureCounts &>/dev/null; then
    echo "ERROR: featureCounts not found. Load subread module or install subread."
    echo "  module load subread   # or equivalent"
    exit 1
fi

# Run featureCounts on all BAMs at once (one matrix output)
COUNTS_TSV="${OUTPUT_DIR}/gene_counts.tsv"
COUNTS_SUMMARY="${OUTPUT_DIR}/gene_counts.tsv.summary"

if [[ -f "$COUNTS_TSV" && -s "$COUNTS_TSV" ]]; then
    echo "gene_counts.tsv already exists, skipping featureCounts."
else
    echo "Running featureCounts (paired-end: counting fragments, not individual reads)..."
    # -p: count fragments (read pairs). Without it you get "The reads are assigned on the single-end mode" and double-count.
    featureCounts \
        -T "$THREADS" \
        -p \
        --countReadPairs \
        -B \
        -t exon \
        -g gene_id \
        -a "$GTF_FILE" \
        -o "$COUNTS_TSV" \
        --primary \
        -s 0 \
        $(cat "$BAM_LIST")
    if [[ ! -f "$COUNTS_TSV" ]]; then
        echo "ERROR: featureCounts did not produce $COUNTS_TSV"
        exit 1
    fi
fi

echo "Converting counts to AnnData..."
python3 "${SCRIPT_DIR}/counts_to_anndata.py" \
    --counts "$COUNTS_TSV" \
    --bam-list "$BAM_LIST" \
    --metacell-ids "$METACELL_LIST" \
    --output "${OUTPUT_DIR}/DT_metacell_expression.h5ad"

echo "Done. AnnData: ${OUTPUT_DIR}/DT_metacell_expression.h5ad"
