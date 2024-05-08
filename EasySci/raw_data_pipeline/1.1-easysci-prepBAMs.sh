#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 6
#SBATCH --mem=10000M
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -J easysci_pseudo # <-- name of job
#SBATCH --array=1-1000%128  # <-- number of subfolders in /gpfs/commons/groups/knowles_lab/data/sc/rockefeller_2022/random_hexamer and polyDT

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load samtools

WD=/gpfs/commons/groups/knowles_lab/data/sc/rockefeller_2022
cells=/gpfs/commons/groups/knowles_lab/data/sc/rockefeller_2022/cell_ids_to_type_conversion.txt
output_dir=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/

# Calculate the index folder
index_folder=$((SLURM_ARRAY_TASK_ID - 1))

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# First do random hexamer
subfolder="$WD/random_hexamer/subfolder_$index_folder"
echo Working on random hexamer!

echo "Index Folder: $index_folder"
echo "Subfolder: $subfolder"

# Get list of all BAM files inside the subfolder
bam_files=($(find $subfolder -name "*.bam"))

# Loop through every BAM file in the subfolder and obtain its cell id, cell type (via cells file) and make symlink to output directory
for bam_file in "${bam_files[@]}"; do
    # Get the cell id from the BAM file name
    cell_id=$(basename "$bam_file" .bam)
    echo $cell_id

    # Get the cell type for the cell_id
    # find the row in the file cells, where the first column is cell_id
    cell_type=$(grep -w "$cell_id" $cells | cut -f3)

    # Make a directory for the cell type if it doesn't exist in the output directory
    if [ ! -d "$output_dir/$cell_type" ]; then
        mkdir -p "$output_dir/$cell_type"
    fi 

    # make a directory within called RH 
    if [ ! -d "$output_dir/$cell_type/RH" ]; then
        mkdir -p "$output_dir/$cell_type/RH"
    fi 

    # Check if symlink already exists and if not then make it 
    if [ ! -f "$output_dir/$cell_type/RH/$cell_id.RH.bam" ]; then
        ln -s "$bam_file" "$output_dir/$cell_type/RH/$cell_id.RH.bam"
        echo "Made symlink for $cell_id"
    fi
done

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Second do polyDT
subfolder="$WD/shortDT/subfolder_$index_folder"
echo Working on polyDT!

echo "Index Folder: $index_folder"
echo "Subfolder: $subfolder"

# Get list of all BAM files inside the subfolder
bam_files=($(find $subfolder -name "*.bam"))

# Loop through every BAM file in the subfolder and obtain its cell id, cell type (via cells file) and make symlink to output directory
for bam_file in "${bam_files[@]}"; do
    # Get the cell id from the BAM file name
    cell_id=$(basename "$bam_file" .bam)
    echo $cell_id

    # Get the cell type for the cell_id
    # find the row in the file cells, where the first column is cell_id
    cell_type=$(grep -w "$cell_id" $cells | cut -f3)

    # Make a directory for the cell type if it doesn't exist in the output directory
    if [ ! -d "$output_dir/$cell_type" ]; then
        mkdir -p "$output_dir/$cell_type"
    fi

    # make a directory within called DT
    if [ ! -d "$output_dir/$cell_type/DT" ]; then
        mkdir -p "$output_dir/$cell_type/DT"
    fi

    # Check if symlink already exists and if not then make it 
    if [ ! -f "$output_dir/$cell_type/DT/$cell_id.DT.bam" ]; then
        ln -s "$bam_file" "$output_dir/$cell_type/DT/$cell_id.DT.bam"
        echo "Made symlink for $cell_id"
    fi
done
