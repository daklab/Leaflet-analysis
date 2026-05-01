#!/bin/bash
# Wrapper: compute number of metacells from the TSV and submit 01_MetaCellMakePseudobulk_RH_DT.sh with --array=1-N.
# Usage:
#   ./run_MetaCellMakePseudobulk_RH_DT.sh              # default TSV (no suffix)
#   METACELL_SUFFIX=_1000 ./run_MetaCellMakePseudobulk_RH_DT.sh
#   METACELL_SUFFIX=_2000 ./run_MetaCellMakePseudobulk_RH_DT.sh
#   METACELL_CSV=/path/to/custom.tsv ./run_MetaCellMakePseudobulk_RH_DT.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUFFIX="${METACELL_SUFFIX:-}"
CSV_FILE="${METACELL_CSV:-/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/DT_w_RH_cells_anndata_meta${SUFFIX}.tsv}"

if [[ ! -f "$CSV_FILE" ]]; then
  echo "Error: TSV not found: $CSV_FILE" >&2
  echo "Set METACELL_SUFFIX (e.g. _1000) or METACELL_CSV (full path)." >&2
  exit 1
fi

N=$(tail -n +2 "$CSV_FILE" | cut -d',' -f5 | sort -u | wc -l)
if [[ "$N" -lt 1 ]]; then
  echo "Error: No metacells found in $CSV_FILE" >&2
  exit 1
fi

MAX_CONCURRENT="${METACELL_MAX_CONCURRENT:-128}"

echo "TSV: $CSV_FILE"
echo "Metacells (array size): $N  max concurrent: $MAX_CONCURRENT"
echo "Submitting: sbatch --array=1-${N}%${MAX_CONCURRENT} $SCRIPT_DIR/01_MetaCellMakePseudobulk_RH_DT.sh"
sbatch --array=1-${N}%${MAX_CONCURRENT} "$SCRIPT_DIR/01_MetaCellMakePseudobulk_RH_DT.sh"
