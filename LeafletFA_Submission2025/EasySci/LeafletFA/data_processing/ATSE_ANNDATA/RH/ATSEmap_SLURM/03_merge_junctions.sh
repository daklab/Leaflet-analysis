#!/bin/bash
# merge_junctions.sh
#SBATCH --job-name=junction_merge
#SBATCH --output=logs/junction_merge_%j.out
#SBATCH --error=logs/junction_merge_%j.err
#SBATCH --mem=32G
#SBATCH -p bigmem,cpu,dev
#SBATCH --cpus-per-task=4

conda activate LeafletSC

# Set METACELL_SUFFIX and PROC_DATE before submitting
SUFFIX="${METACELL_SUFFIX:-}"
PROC_DATE="${PROC_DATE:-$(date +%Y%m%d)}"
BASE="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/SpliceVI${SUFFIX}"

SCRIPT_PATH=/gpfs/commons/home/kisaev/Leaflet-analysis/General_Utils/split_process_merge_slurm_junctions.py
WD="$BASE/ATSEmap/junction_processing_${PROC_DATE}"
cd "$WD"

python $SCRIPT_PATH \
    --mode merge \
    --output-dir results \
    --merge-output results/final_junctions.pkl

# Usage:
#   METACELL_SUFFIX=_500 PROC_DATE=20260317 sbatch 03_merge_junctions.sh
