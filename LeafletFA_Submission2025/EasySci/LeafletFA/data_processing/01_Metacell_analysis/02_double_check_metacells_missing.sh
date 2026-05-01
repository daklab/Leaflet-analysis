#!/bin/bash
# Verify all expected metacells from CSV have a merged RH BAM in the output.
# Handles a single date dir (e.g. SpliceVI_500/20260301) or a base dir (SpliceVI_500)
# that contains multiple date subdirs — scans all of them combined.
# Writes missing_rh_from_csv.txt to the most recent date dir for use by 03_remake_missing.sh.
# Usage: bash 02_double_check_metacells_missing.sh <DIR> [CSV_FILE]
#   DIR: date dir (e.g. SpliceVI_500/20260301) or base dir (e.g. SpliceVI_500)
#   CSV_FILE: optional; defaults based on METACELL_SUFFIX env var

SUFFIX="${METACELL_SUFFIX:-}"
DIR="${1:-}"
CSV_FILE="${2:-${METACELL_CSV:-/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/DT_w_RH_cells_anndata_meta${SUFFIX}.tsv}}"

if [[ -z "$DIR" || ! -d "$DIR" ]]; then
    echo "Usage: $0 <DIR> [CSV_FILE]" >&2
    echo "  DIR: date dir (SpliceVI_500/20260301) or base dir (SpliceVI_500)" >&2
    exit 1
fi

if [[ ! -f "$CSV_FILE" ]]; then
    echo "Error: CSV not found: $CSV_FILE" >&2
    echo "Set METACELL_SUFFIX (e.g. _500) or METACELL_CSV (full path)." >&2
    exit 1
fi

# Determine where to search for RH dirs and where to write output files
if [[ -d "$DIR/RH" ]]; then
    # Single date dir passed directly
    TARGET_DIR="$DIR"
    mapfile -t RH_SEARCH_DIRS < <(echo "$DIR/RH")
else
    # Base dir: find all date subdirs that contain an RH folder
    mapfile -t RH_SEARCH_DIRS < <(find "$DIR" -maxdepth 2 -type d -name "RH" | sort)
    if [[ ${#RH_SEARCH_DIRS[@]} -eq 0 ]]; then
        echo "Error: No RH directories found under $DIR" >&2
        exit 1
    fi
    # Write outputs to the most recent date subdir
    TARGET_DIR=$(find "$DIR" -maxdepth 1 -mindepth 1 -type d | sort | tail -1)
    echo "Scanning ${#RH_SEARCH_DIRS[@]} date dir(s):"
    printf '  %s\n' "${RH_SEARCH_DIRS[@]}"
fi

MISSING_OUT="$TARGET_DIR/missing_rh_from_csv.txt"
CSV_CLUSTERS="$TARGET_DIR/csv_clusters.txt"
RH_WITH_BAMS="$TARGET_DIR/rh_with_bams.txt"

# Expected metacells from CSV column 5 (Main_cluster_name_wkmeans), normalizing / -> _
tail -n +2 "$CSV_FILE" | cut -d',' -f5 | tr '/' '_' | sort -u > "$CSV_CLUSTERS"
N_EXPECTED=$(wc -l < "$CSV_CLUSTERS")
echo "Expected metacells from CSV: $N_EXPECTED"

# Metacells with an RH merged BAM on disk across all scanned dirs
find "${RH_SEARCH_DIRS[@]}" -name "*.bam" 2>/dev/null | awk -F'/' '{print $(NF-1)}' | sort -u > "$RH_WITH_BAMS"
N_FOUND=$(wc -l < "$RH_WITH_BAMS")
echo "Metacells with RH BAM on disk: $N_FOUND"

# CSV clusters missing an RH BAM (comm requires both inputs sorted)
comm -23 "$CSV_CLUSTERS" "$RH_WITH_BAMS" > "$MISSING_OUT"
N_MISSING=$(wc -l < "$MISSING_OUT")
echo "Missing RH BAMs: $N_MISSING"
echo "Missing list written to: $MISSING_OUT"

if [[ "$N_MISSING" -gt 0 ]]; then
    echo ""
    echo "Next: update --array=1-${N_MISSING} in 03_remake_missing.sh, then:"
    echo "  METACELL_SUFFIX=${SUFFIX} METACELL_OUTPUT_DIR=${TARGET_DIR} sbatch 03_remake_missing.sh"
fi
