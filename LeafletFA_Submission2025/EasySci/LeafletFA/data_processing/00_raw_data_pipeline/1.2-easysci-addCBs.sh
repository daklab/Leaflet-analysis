#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 6
#SBATCH --mem=64000M
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -J easysci_pseudo # <-- name of job
#SBATCH --array=1-33  # <-- number of cell_type folders in /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load samtools

cells=/gpfs/commons/groups/knowles_lab/data/sc/rockefeller_2022/cell_ids_to_type_conversion.txt
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/

cd $WD 

# Get list of folders inside EasySci to cycle through with arrays
folders=($(ls -d $WD/*/)) # This will create an array of folder paths

# Get the folder for this array task
current_folder=${folders[$SLURM_ARRAY_TASK_ID - 1]}
echo $current_folder

# make sure it's not slurm or genome_files folder if it is then exit 
if [[ $current_folder == *"slurm"* ]] || [[ $current_folder == *"genome_files"* ]]; then
    echo "This is a slurm or genome_files folder, exiting"
    exit 1
fi

# Navigate into the current folder
cd $current_folder

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 1. Add cell barcodes to the read names of the BAM files so easier to run through regtools later (keep RT and RH seperate for now)
bam_files=($(find $RH -name "*RH*.bam"))

# Loop through every BAM file in the subfolder and obtain its cell id, cell type (via cells file) and make symlink to output directory
for bam_file in "${bam_files[@]}"; do
    # Get the cell id from the BAM file name
    cell_id=$(basename "$bam_file" .RH.bam)
    echo $cell_id

    # Get the cell type for the cell_id
    cell_type=$(grep -w "$cell_id" $cells | cut -f3)
    echo $cell_type
    # Add cell tag 
    samtools view -h $bam_file | awk -v cb=$cell_id -F '\t' 'BEGIN {OFS="\t"} {$NF = $NF"\tCB:Z:"cb; print}' | samtools view -bS - > $current_folder/RH/${cell_id}.RH.CB.bam
done

echo Done adding cell barcodes to random hexamer BAM files!

# Now do the same for polyDT
bam_files=($(find $DT -name "*DT*.bam"))

# Loop through every BAM file in the subfolder and obtain its cell id, cell type (via cells file) and make symlink to output directory
for bam_file in "${bam_files[@]}"; do
    # Get the cell id from the BAM file name
    cell_id=$(basename "$bam_file" .DT.bam)
    #echo $cell_id

    # Get the cell type for the cell_id
    cell_type=$(grep -w "$cell_id" $cells | cut -f3)

    # Add cell tag 
    samtools view -h $bam_file | awk -v cb=$cell_id -F '\t' 'BEGIN {OFS="\t"} {$NF = $NF"\tCB:Z:"cb; print}' | samtools view -bS - > $current_folder/DT/${cell_id}.DT.CB.bam
done

echo Done adding cell barcodes to DT BAM files!
