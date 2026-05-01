#!/bin/bash
# split_junctions.sh
#SBATCH --job-name=junction_split
#SBATCH --output=logs/junction_split_%j.out
#SBATCH --error=logs/junction_split_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

conda activate LeafletSC

# Set METACELL_SUFFIX before submitting (e.g. _500, _1000, _2000, _4000)
SUFFIX="${METACELL_SUFFIX:-}"
BASE="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/SpliceVI${SUFFIX}"

SCRIPT_PATH=/gpfs/commons/home/kisaev/Leaflet-analysis/General_Utils/split_process_merge_slurm_junctions.py
JUNCTION_FILES="$BASE/junctions"
WD="$BASE/ATSEmap"

if [[ ! -d "$JUNCTION_FILES" ]]; then
    echo "Error: junctions dir not found: $JUNCTION_FILES" >&2
    exit 1
fi

mkdir -p "$WD"
cd "$WD"

# Build list of junction files
find "$JUNCTION_FILES" -path "*/junctions_with_barcodes.bed" > "$WD/junction_files.txt"
INPUT_FILE="$WD/junction_files.txt"
echo "Number of files in INPUT_FILE: $(wc -l $INPUT_FILE)"

# Create base directory with today's date
BASE_DIR="junction_processing_$(date +%Y%m%d)"
mkdir -p "$BASE_DIR"/{logs,chunks,results}
cd "$BASE_DIR"

mkdir -p chunks

# Run the split job
python $SCRIPT_PATH \
    --mode split \
    --input-file "$INPUT_FILE" \
    --chunks 100 \
    --output-dir chunks

# Verify chunks were created
echo "Number of chunks:"
ls chunks/ | wc -l

# Usage:
#   METACELL_SUFFIX=_500 sbatch 01_split_junctions.sh
#   METACELL_SUFFIX=_1000 sbatch 01_split_junctions.sh