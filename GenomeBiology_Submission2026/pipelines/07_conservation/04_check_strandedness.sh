#!/bin/bash
# 04_check_strandedness.sh — VERIFY the featureCounts -s setting before the full run.
#
# The GSE138734 libraries are described as fr-firststrand (RF) => -s 2 (reverse).
# A wrong -s silently discards most fragments, so confirm empirically on one BAM:
# run -s 0 (unstranded), -s 1 (forward), -s 2 (reverse) and compare "Assigned" rates.
#
# Expect for a correct fr-firststrand library:
#   -s 2  -> HIGHEST assigned % (the true setting)
#   -s 1  -> much lower (reads assigned to the wrong strand)
#   -s 0  -> similar to or slightly above -s 2 (ignores strand, may double-count overlaps)
# If -s 1 wins instead, the library is fr-secondstrand => change STRAND=1 in
# 04_featurecounts_array.sh. If -s 0 ~ -s 1 ~ -s 2, the library is unstranded => -s 0.
#
# Usage:  bash 04_check_strandedness.sh [SRR_ACCESSION]
#         (runs inline, ~a few minutes on one BAM; safe on an interactive node)

set -euo pipefail

WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/RNA_ATLAS/GSE138734
GTF=/gpfs/commons/datasets/controlled/BRAIN_NeMO/human-reference/gencode/gencode.v45.primary_assembly.annotation.gtf
SRR=${1:-SRR10261826}
BAM=$WD/bams/$SRR/Aligned.sortedByCoord.out.bam
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

[[ -f "$BAM" ]] || { echo "ERROR: no BAM at $BAM"; exit 1; }

module load Subread/2.0.4-GCC-11.3.0

echo "=== Strandedness check on $SRR ==="
echo "BAM: $BAM"
echo ""

for S in 0 1 2; do
    featureCounts -a "$GTF" -o "$TMP/s${S}.txt" \
        -p --countReadPairs -s "$S" -t exon -g gene_id --primary -T 4 \
        "$BAM" > "$TMP/s${S}.log" 2>&1

    # .summary: column 2 holds counts; Assigned is row 1
    assigned=$(awk 'NR==2 {print $2}' "$TMP/s${S}.txt.summary")
    total=$(awk 'NR>1 {sum+=$2} END {print sum}' "$TMP/s${S}.txt.summary")
    pct=$(awk -v a="$assigned" -v t="$total" 'BEGIN {printf "%.1f", (t>0 ? 100*a/t : 0)}')
    printf "  -s %d : assigned %'d / %'d fragments  (%s%%)\n" "$S" "$assigned" "$total" "$pct"
done

echo ""
echo "Pick the -s with the highest assigned% (expect -s 2 for fr-firststrand)."
echo "Then set STRAND= that value in 04_featurecounts_array.sh"
