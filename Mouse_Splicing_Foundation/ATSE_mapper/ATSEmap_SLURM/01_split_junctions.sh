#!/bin/bash
# split_junctions.sh
#SBATCH --job-name=junction_split
#SBATCH --output=logs/junction_split_%j.out
#SBATCH --error=logs/junction_split_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

conda activate LeafletSC

# Path to the script
SCRIPT_PATH=/gpfs/commons/home/kisaev/Leaflet-analysis/split_process_merge_slurm_junctions.py

# Working directory
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper 
cd $WD

# Path to the junction files
TMS=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/ATSEmap/output/junction_files.txt # already contains all junctions paths in Tabula Muris Senis
# copy TMS file to $WD/junctions_files_TMS.txt
cp $TMS $WD/junction_files_TMS.txt

# Allen Brain 
JUNCTION_FILES_AB=/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/junctions # allen brain nuclei 
find $JUNCTION_FILES_AB -name "*_junctions_with_barcodes.bed" > $WD/junction_files_AB.txt

# Junction file to save everything to - add date to this file to avoid overwriting and make sure it's in $WD 
junction_files_list=$WD/junction_files_$(date +%Y%m%d).txt
cat $WD/junction_files_AB.txt $WD/junction_files_TMS.txt > $junction_files_list
INPUT_FILE=$junction_files_list

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
    --chunks 1000 \
    --output-dir chunks

# Verify chunks were created
echo "Number of chunks:"
ls chunks/ | wc -l