#!/bin/bash
# split_junctions.sh
#SBATCH --job-name=junction_split
#SBATCH --output=logs/junction_split_%j.out
#SBATCH --error=logs/junction_split_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

conda activate LeafletSC

SCRIPT_PATH=/gpfs/commons/home/kisaev/Leaflet-analysis/split_process_merge_slurm_junctions.py
JUNCTION_FILES=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/junctions/SpliceVI/202510/RH
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/SpliceVI/202510/RH/ATSEmap

# Create base directory with today's date
# create output directory if it doesn't exist
if [ ! -d "$WD" ]; then
    mkdir -p $WD
fi
cd $WD

# in WD, make junction_files.txt with all the *_junctions_with_barcodes.bed files found in JUNCTION_FILES
find $JUNCTION_FILES -path "*/junctions_with_barcodes.bed" > $WD/junction_files.txt
INPUT_FILE=$WD/junction_files.txt
# print number of files in INPUT_FILE
echo "Number of files in INPUT_FILE: $(wc -l $INPUT_FILE)"

# Create base directory with today's date
BASE_DIR="junction_processing_$(date +%Y%m%d)"
mkdir -p $BASE_DIR/{logs,chunks,results}
cd $BASE_DIR

# Ensure chunks directory exists
mkdir -p chunks

# Run the split job
python $SCRIPT_PATH \
    --mode split \
    --input-file $INPUT_FILE \
    --chunks 100 \
    --output-dir chunks

# Verify chunks were created
echo "Number of chunks:"
ls chunks/ | wc -l