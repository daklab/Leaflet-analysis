#!/bin/bash
# 01_split.sh — discover all junction files and split into SLURM chunks
#
# Usage: bash 01_split.sh <config_yaml>
#   e.g. bash 01_split.sh ../../config/mouse_smartseq2.yaml
#
#SBATCH --job-name=junc_split
#SBATCH --output=logs/junc_split_%j.out
#SBATCH --error=logs/junc_split_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cpu,dev

set -euo pipefail

CONFIG=${1:-}
if [[ -z "$CONFIG" ]]; then
    echo "ERROR: must pass config YAML as first argument"
    exit 1
fi

# Resolve to absolute path before any cd
CONFIG=$(realpath "$CONFIG")

# Parse YAML fields we need (requires python/pyyaml)
parse_yaml() {
    python3 - <<EOF
import yaml, sys
with open("$CONFIG") as f:
    cfg = yaml.safe_load(f)
print(cfg["$1"])
EOF
}

WD=$(parse_yaml wd)
N_CHUNKS=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['junction_processing']['n_chunks'])")
TODAY=$(date +%Y%m%d)
SPLIT_SCRIPT=/gpfs/commons/home/kisaev/Leaflet-analysis/General_Utils/split_process_merge_slurm_junctions.py

cd "$WD"
mkdir -p ATSE_mapper/junction_processing_${TODAY}/{logs,chunks,results}
cd ATSE_mapper/junction_processing_${TODAY}

echo "=== Junction Discovery: $TODAY ==="
echo "Config: $CONFIG"
echo "WD: $WD"

# ── Discover files per source ─────────────────────────────────────────────────
COMBINED_LIST="junction_files_combined_${TODAY}.txt"
> "$COMBINED_LIST"

python3 - <<EOF
import yaml, os, glob

with open("$CONFIG") as f:
    cfg = yaml.safe_load(f)

wd = cfg["wd"]
combined = "$WD/ATSE_mapper/junction_processing_${TODAY}/$COMBINED_LIST"

total = 0
with open(combined, "w") as out:
    for src_name, src in cfg["sources"].items():
        if "junction_dir" in src:
            pattern = os.path.join(src["junction_dir"], src.get("junction_pattern", "*.bed"))
            files = sorted(glob.glob(pattern))
            print(f"  {src_name}: {len(files)} junction files found")
        elif "junction_filelist" in src:
            with open(src["junction_filelist"]) as f:
                files = [l.strip() for l in f if l.strip()]
            print(f"  {src_name}: {len(files)} junction files from pre-collected list")
        else:
            print(f"  WARNING: {src_name} has no junction_dir or junction_filelist — skipping")
            continue

        for fp in files:
            out.write(fp + "\n")
        total += len(files)

print(f"Total junction files: {total}")
print(f"Written to: {combined}")
EOF

TOTAL=$(wc -l < "$COMBINED_LIST")
echo "Total junction files: $TOTAL"

# ── Split into chunks ─────────────────────────────────────────────────────────
python3 "$SPLIT_SCRIPT" \
    --mode split \
    --input-file "$COMBINED_LIST" \
    --chunks "$N_CHUNKS" \
    --output-dir chunks

CHUNK_COUNT=$(ls chunks/ | wc -l)
echo "Created $CHUNK_COUNT chunks (requested $N_CHUNKS)"
echo ""
echo "Next: sbatch 02_process.sh $CONFIG"
