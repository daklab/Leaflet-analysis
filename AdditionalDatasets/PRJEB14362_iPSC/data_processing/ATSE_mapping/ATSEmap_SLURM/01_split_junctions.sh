#!/bin/bash
# split_junctions.sh
#SBATCH --job-name=junction_split
#SBATCH --output=logs/junction_split_%j.out
#SBATCH --error=logs/junction_split_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

conda activate LeafletSC

SCRIPT_PATH=/gpfs/commons/home/kisaev/Leaflet-analysis/split_process_merge_slurm_junctions.py
INPUT_FILE=/gpfs/commons/projects/knowles_singlecell_splicing/PRJEB14362/LeafletFA/ATSEs/022025/junction_files.txt
WD=/gpfs/commons/projects/knowles_singlecell_splicing/PRJEB14362/LeafletFA/ATSEs/022025/output

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