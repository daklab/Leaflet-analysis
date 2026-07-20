#!/bin/bash
# 03_merge.sh — merge all processed chunk pkl files into final_junctions.pkl
#
# Usage: sbatch 03_merge.sh <config_yaml> <processing_dir>
#
#SBATCH --job-name=junc_merge
#SBATCH --output=logs/junc_merge_%j.out
#SBATCH --error=logs/junc_merge_%j.err
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
# NOTE: bigmem QOS has a 500G minimum-memory floor on this cluster; requesting
# <500G there is hard-rejected with (QOSMinMemory). This job asks for 128G, so
# use cpu only. If a merge ever needs >500G, use: --mem=500G -p bigmem
#SBATCH --partition=cpu

set -euo pipefail

CONFIG=${1:-}
PROC_DIR=${2:-}

if [[ -z "$CONFIG" || -z "$PROC_DIR" ]]; then
    echo "ERROR: Usage: $0 <config_yaml> <processing_dir>"
    exit 1
fi

SPLIT_SCRIPT=/gpfs/commons/home/kisaev/Leaflet-analysis/General_Utils/split_process_merge_slurm_junctions.py

read_cfg() {
    python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c$1)"
}

MIN_READS=$(read_cfg "['atsemapper']['min_junc_reads']")
MIN_CELLS=$(read_cfg "['atsemapper']['min_num_cells']")

echo "QC thresholds: min_reads=$MIN_READS  min_cells=$MIN_CELLS"

cd "$PROC_DIR"

# Check all expected chunks produced a result
N_CHUNKS=$(ls chunks/*.txt 2>/dev/null | wc -l)
N_RESULTS=$(ls results/*_results.pkl 2>/dev/null | wc -l)
echo "Chunks: $N_CHUNKS  |  Results: $N_RESULTS"

if [[ $N_RESULTS -lt $N_CHUNKS ]]; then
    echo "WARNING: $((N_CHUNKS - N_RESULTS)) chunks have no result file."
    echo "         Check SLURM logs before proceeding."
    MISSING=$(comm -23 \
        <(ls chunks/*.txt | xargs -I{} basename {} .txt | sort) \
        <(ls results/*_results.pkl | xargs -I{} basename {} _results.pkl | sed 's/chunk_//' | sort))
    echo "Missing chunks: $MISSING"
fi

python3 "$SPLIT_SCRIPT" \
    --mode merge \
    --output-dir results \
    --merge-output results/final_junctions.pkl \
    --min-reads    "$MIN_READS" \
    --min-cells    "$MIN_CELLS"

echo ""
python3 - <<'EOF'
import pickle
with open("results/final_junctions.pkl", "rb") as f:
    jd = pickle.load(f)
print(f"Total unique junctions in final pkl: {len(jd):,}")
EOF

echo ""
echo "Next: python 04_run_atsemapper.py <config_yaml> <processing_dir>"
