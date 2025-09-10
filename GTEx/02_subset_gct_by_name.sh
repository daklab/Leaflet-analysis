#!/usr/bin/env bash
# Subset a GTEx GCT.gz by Name column using an ID whitelist.
# Usage:
#   ./subset_gct_by_name.sh GTEx_Analysis_v10_STARv2.7.10a_junctions.gct.gz ids.txt out_subset.gct.gz
#
# where ids.txt contains one junction ID per line (exactly as in the GCT "Name" column),
# e.g. produced by: cut -f7 gtex_junctions_coordinates_atse_overlap.tsv | sort -u > ids.txt

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <in.gct.gz> <ids.txt> <out.gct.gz>"
  exit 1
fi

IN_GCT="$1"
IDS_FILE="$2"
OUT_GCT="$3"

# Use pigz if available
if command -v pigz >/dev/null 2>&1; then
  ZCAT="pigz -dc"
  GZIP="pigz -c"
else
  ZCAT="gzip -dc"
  GZIP="gzip -c"
fi

TMP_DIR="${TMPDIR:-/tmp}"
TMP_ROWS="$(mktemp "$TMP_DIR/gct_rows.XXXXXX.tsv")"
trap 'rm -f "$TMP_ROWS"' EXIT

# 1) Stream and collect matching rows to TMP_ROWS; also capture header + ncols
#    - Load ids into awk hash from IDS_FILE
#    - Determine Name column index from header (line 3)
#    - Keep original version line (1), ncols from line 2, and header (3)
VERSION_LINE=""
HEADER_LINE=""
NCOLS=""
NAME_IDX=""

LC_ALL=C $ZCAT "$IN_GCT" | awk -v FS='\t' -v OFS='\t' -v ids="$IDS_FILE" '
  BEGIN {
    # load whitelist
    while ((getline line < ids) > 0) {
      keep[line] = 1
    }
    close(ids)
    row_count = 0
  }
  NR==1 { version_line = $0; next }
  NR==2 {
    # format: nrows \t ncols
    ncols = $2;
    next
  }
  NR==3 {
    header_line = $0;
    for (i=1; i<=NF; i++) {
      if ($i == "Name") { name_i = i; break }
    }
    if (name_i == 0) {
      print "ERROR: Could not find Name column in header." > "/dev/stderr"
      exit 2
    }
    next
  }
  NR>=4 {
    key = $name_i;
    if (key in keep) {
      print $0 > "'"$TMP_ROWS"'"
      row_count++
    }
    next
  }
  END {
    # stash for printing later via shell
    print version_line      > "/dev/stderr"
    print header_line       > "/dev/stderr"
    print "NCOLS=" ncols    > "/dev/stderr"
    print "NROWS=" row_count > "/dev/stderr"
  }
' 2> >(tee "$TMP_DIR/gct_subset_meta.$$" >&2)

# 2) Recover meta from stderr tee
VERSION_LINE=$(sed -n '1p' "$TMP_DIR/gct_subset_meta.$$")
HEADER_LINE=$(sed -n '2p' "$TMP_DIR/gct_subset_meta.$$")
NCOLS=$(sed -n 's/^NCOLS=\(.*\)/\1/p' "$TMP_DIR/gct_subset_meta.$$")
NROWS=$(sed -n 's/^NROWS=\(.*\)/\1/p' "$TMP_DIR/gct_subset_meta.$$")
rm -f "$TMP_DIR/gct_subset_meta.$$"

if [[ -z "${NCOLS:-}" || -z "${NROWS:-}" ]]; then
  echo "ERROR: Failed to derive NCOLS or NROWS." >&2
  exit 3
fi

# 3) Assemble valid GCT: line1 version; line2 nrows\tncols; line3 header; then matched rows
#    Compress to OUT_GCT
{
  printf "%s\n" "$VERSION_LINE"
  printf "%s\t%s\n" "$NROWS" "$NCOLS"
  printf "%s\n" "$HEADER_LINE"
  cat "$TMP_ROWS"
} | $GZIP > "$OUT_GCT"

echo "Wrote subset GCT.gz with $NROWS rows (of original ncols=$NCOLS): $OUT_GCT"

# ---------------------------------------------------------------------------
# slurm submission
# ---------------------------------------------------------------------------

# cd /gpfs/commons/groups/knowles_lab/Karin/data/GTEx/v10

# First get ids
# cut -f7 gtex_junctions_coordinates_atse_overlap.tsv | sort -u > ids.txt
# wc -l ids.txt   # number of junctions you’ll keep

#script=/gpfs/commons/home/kisaev/Leaflet-analysis/GTEx/02_subset_gct_by_name.sh
#sbatch -J gct_subset --mem=100G -p dev,cpu,bigmem --wrap "\
#bash $script \
#  /gpfs/commons/groups/knowles_lab/Karin/data/GTEx/v10/GTEx_Analysis_v10_STARv2.7.10a_junctions.gct.gz \
#  /gpfs/commons/groups/knowles_lab/Karin/data/GTEx/v10/ids.txt \
#  /gpfs/commons/groups/knowles_lab/Karin/data/GTEx/v10/GTEx_subset_ATSE_overlap.gct.gz"
