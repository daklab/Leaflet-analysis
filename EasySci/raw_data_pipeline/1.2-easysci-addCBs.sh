#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 6
#SBATCH --mem=64000M
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -J easysci_pseudo # <-- name of job
#SBATCH --array=1-31  # <-- number of cell_type folders in /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load samtools

cells=/gpfs/commons/groups/knowles_lab/data/sc/rockefeller_2022/cell_ids_to_type_conversion.txt
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci/

cd $WD 

# Get list of folders inside EasySci to cycle through with arrays
folders=($(ls -d $WD/*/)) # This will create an array of folder paths

# Get the folder for this array task
current_folder=${folders[$SLURM_ARRAY_TASK_ID - 1]}
echo $current_folder

# Navigate into the current folder
cd $current_folder

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 1. make a folder for random hexamers 
mkdir -p random_hexamers
# Move all the RH bam files into there 
for file in *RH*.bam; do
    mv "$file" random_hexamers/
done

echo Done moving all RH files into their folders!

# make a directory for polyDT 
mkdir -p polyDT
# Move all the DT bam files into there
for file in *DT*.bam; do
    mv "$file" polyDT/
done

echo Done moving all DT files into their folders!

# 2. Add cell barcodes to the read names of the BAM files so easier to run through regtools later (keep RT and RH seperate for now)
bam_files=($(find $random_hexamers -name "*RH*.bam"))

# Loop through every BAM file in the subfolder and obtain its cell id, cell type (via cells file) and make symlink to output directory
for bam_file in "${bam_files[@]}"; do
    # Get the cell id from the BAM file name
    cell_id=$(basename "$bam_file" .RH.bam)
    #echo $cell_id

    # Get the cell type for the cell_id
    cell_type=$(grep -w "$cell_id" $cells | cut -f3)

    # Add cell tag 
    samtools view -h $bam_file | awk -v cb=$cell_id -F '\t' 'BEGIN {OFS="\t"} {$NF = $NF"\tCB:Z:"cb; print}' | samtools view -bS - > $current_folder/random_hexamers/${cell_id}.RH.CB.bam
done

echo Done adding cell barcodes to random hexamer BAM files!

# Now do the same for polyDT
bam_files=($(find $polyDT -name "*DT*.bam"))

# Loop through every BAM file in the subfolder and obtain its cell id, cell type (via cells file) and make symlink to output directory
for bam_file in "${bam_files[@]}"; do
    # Get the cell id from the BAM file name
    cell_id=$(basename "$bam_file" .DT.bam)
    #echo $cell_id

    # Get the cell type for the cell_id
    cell_type=$(grep -w "$cell_id" $cells | cut -f3)

    # Add cell tag 
    samtools view -h $bam_file | awk -v cb=$cell_id -F '\t' 'BEGIN {OFS="\t"} {$NF = $NF"\tCB:Z:"cb; print}' | samtools view -bS - > $current_folder/polyDT/${cell_id}.DT.CB.bam
done

echo Done adding cell barcodes to DT BAM files!

# Go into random hexamers folder and remove any files that have DT in them 
cd $current_folder/random_hexamers
for file in *DT*; do
    if [ -f "$file" ]; then
        rm "$file"
    fi
done