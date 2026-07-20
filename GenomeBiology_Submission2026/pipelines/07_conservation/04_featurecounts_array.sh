#!/bin/bash
# 04_featurecounts_array.sh — gene-level counts from existing STAR BAMs (SLURM array).
#
# STAR was run WITHOUT --quantMode GeneCounts, so no ReadsPerGene.out.tab exist.
# This quantifies genes with featureCounts over the same gencode.v45 GTF used for
# the STAR index / ATSE mapping, so GE and splicing share one annotation.
#
# Restricted to the 2,228 PRIMARY CELL TYPE samples (sample_type == "cell type"),
# matching the splice AnnData. Bulk mixed "tissue" samples are excluded; cancer /
# normal cell lines were never downloaded.
#
# Usage:
#   # 1) build the BAM list first (writes $WD/bams_celltype_only.txt):
#   python 04_prepare_bam_list.py
#   # 2) submit (2,228 BAMs / BATCH_SIZE=50 -> 45 tasks, indices 0-44):
#   sbatch --array=0-44%15 04_featurecounts_array.sh
#
# Output: $OUT_DIR/counts_batch_<TASKID>.txt  (+ .summary)
# Then:   python 05_build_ge_anndata.py
#
#SBATCH --job-name=fcounts
#SBATCH --output=logs/fcounts_%A_%a.out
#SBATCH --error=logs/fcounts_%A_%a.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
# bigmem QOS has a 500G min-memory floor on this cluster -> cpu only (see WORKFLOW D3)
#SBATCH --partition=cpu

set -euo pipefail

WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/RNA_ATLAS/GSE138734
GTF=/gpfs/commons/datasets/controlled/BRAIN_NeMO/human-reference/gencode/gencode.v45.primary_assembly.annotation.gtf
BAM_LIST=$WD/bams_celltype_only.txt
OUT_DIR=$WD/gene_expression/featurecounts
BATCH_SIZE=50
THREADS=8

# Strandedness: library is fr-firststrand (RF) -> featureCounts -s 2 (reverse).
# Verify once before the full run (see 04_check_strandedness.sh).
STRAND=2

mkdir -p "$OUT_DIR" logs

if [[ ! -f "$BAM_LIST" ]]; then
    echo "ERROR: BAM list not found: $BAM_LIST  (run 04_prepare_bam_list.py first)"
    exit 1
fi

TASK=${SLURM_ARRAY_TASK_ID:?must be run as a SLURM array job}
START=$(( TASK * BATCH_SIZE + 1 ))
END=$(( START + BATCH_SIZE - 1 ))

# Select this task's slice of BAMs (sed range is inclusive; tail past EOF is empty)
mapfile -t BAMS < <(sed -n "${START},${END}p" "$BAM_LIST")

if [[ ${#BAMS[@]} -eq 0 ]]; then
    echo "No BAMs for task $TASK (lines $START-$END) — nothing to do."
    exit 0
fi

OUT=$OUT_DIR/counts_batch_${TASK}.txt

# Idempotent: skip if this batch already finished (lets you resubmit failed tasks)
if [[ -s "$OUT" && -s "${OUT}.summary" ]]; then
    echo "Batch $TASK already done: $OUT — skipping."
    exit 0
fi

echo "=== featureCounts batch $TASK: ${#BAMS[@]} BAMs (lines $START-$END) ==="
echo "GTF:    $GTF"
echo "Strand: -s $STRAND (fr-firststrand / reverse)"
echo "Out:    $OUT"

module load Subread/2.0.4-GCC-11.3.0

# -p --countReadPairs : paired-end, count FRAGMENTS (not reads)
# -s 2                : reverse-stranded (fr-firststrand)
# -t exon -g gene_id  : sum exonic reads per gene (standard gene-level quant)
# --primary           : count primary alignments only (avoids double-counting multimappers)
featureCounts \
    -a "$GTF" \
    -o "$OUT" \
    -p --countReadPairs \
    -s "$STRAND" \
    -t exon \
    -g gene_id \
    --primary \
    -T "$THREADS" \
    "${BAMS[@]}"

echo "Batch $TASK done -> $OUT"
