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

# Path to the junction files
JUNCTION_FILES_AB=/gpfs/commons/datasets/controlled/BRAIN_NeMO/lein-human-cortex/Leaflet # allen brain nuclei 
JUNCTION_FILES_TS=/gpfs/commons/projects/CZI-tabula-sapiens/LeafletFA # tabula sapiens single cell 

# Working directory
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/ATSE_mapper 
cd $WD

# in WD, make junction_files.txt with all the *_junctions_with_barcodes.bed files found in JUNCTION_FILES
find $JUNCTION_FILES_AB -name "*_junctions_with_barcodes.bed" > $WD/junction_files_AB.txt
find $JUNCTION_FILES_TS -name "*_junctions_with_barcodes.bed" > $WD/junction_files_TS.txt

# Update junction file for TS and AB using this script /gpfs/commons/home/kisaev/Leaflet-analysis/Human_Splicing_Foundation/tabula_sapien_extract_BAM.ipynb
# So that only use the cells that Tabula Sapiens endedup using (had way more raw data...)
clean_TS_juncs=$WD/junction_files_TS_subset.txt
clean_AB_juncs=$WD/junction_files_AB_subset.txt

# Junction file to save everything to - add date to this file to avoid overwriting and make sure it's in $WD 
junction_files_list=$WD/junction_files_$(date +%Y%m%d).txt

cat $clean_AB_juncs $clean_TS_juncs > $junction_files_list
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
    --chunks 100 \
    --output-dir chunks

# Verify chunks were created
echo "Number of chunks:"
ls chunks/ | wc -l