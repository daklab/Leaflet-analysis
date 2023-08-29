#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 4
#SBATCH --mem=32000M
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -J moving_easysci # <-- name of job
#SBATCH --array=1-31  # <-- number of cell_type folders in /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load samtools

cells=/gpfs/commons/groups/knowles_lab/data/sc/rockefeller_2022/cells_wmain_cluster_type_sex.txt
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci/

cd $WD 

# Get list of folders inside EasySci to cycle through with arrays
folders=($(ls -d $WD/*/)) # This will create an array of folder paths

# Get the folder for this array task
current_folder=${folders[$SLURM_ARRAY_TASK_ID - 1]}
echo $current_folder

# List of folders to exclude
excluded_folders=(
  "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci//Cerebellum_granule_neurons/"
  "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci//Cortical_projection_neurons_1/"
  "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci//Interbrain_and_midbrain_neurons_1/"
  "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci//Interbrain_and_midbrain_neurons_2/"
  "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci//Oligodendrocytes/"
  "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci//Striatal_neurons_1/"
)

# Check if $current_folder is in the list of excluded folders
if [[ " ${excluded_folders[*]} " =~ " $current_folder " ]]; then
  echo "Folder is excluded. Exiting."
  exit 1
fi

# Navigate into the current folder
cd "$current_folder"

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 1. Go into random_hexamer folder and remove all *RH.CB.RH.CB* files 
cd random_hexamers  

# Get a list of all the files with the extension RH.CB.bam.RH.CB.bam
files=($(find -name "*RH.CB.bam.RH.CB.bam"))

# Iterate through the list of files and remove them
for file in ${files[@]}; do
    echo $file 
    rm $file
done

echo Done removing RH.CB.bam.RH.CB.bam files

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Combine files into pseudobulk based on sex_Type to break them up 
# Using the file: $cells, figure out which Type_sex combinations exist for the cell type we are looking at and then move the BAM files accordingly 
# extract cell type from $current_folder as last item after the last "/" 
cleaned_path="${current_folder%/}"   # Remove trailing slashes
cell_type="${cleaned_path##*/}"
echo "$cell_type"

mkdir 21mo_female
mkdir 21mo_male 
mkdir 3mo_female 
mkdir 3mo_male 
mkdir 5xFAD_female 
mkdir 5xFAD_male 
mkdir 6mo_female 
mkdir 6mo_male 
mkdir APOE4_female  
mkdir APOE4_male  

# Now go through each BAM file in directory and move it to the correct folder based on its corresponding type_sex 
# Get list of all BAM files inside the subfolder
bam_files=($(find -name "*RH.CB.bam"))

# Loop through every BAM file in the subfolder and obtain its cell id, cell type (via cells file) and make symlink to output directory
for bam_file in "${bam_files[@]}"; do
    # Get the cell id from the BAM file name
    cell_id=$(basename "$bam_file" .RH.CB.bam)
    #echo $cell_id

    # Get the cell type for the cell_id
    # find the row in the file cells, where the first column is cell_id
    cell_type=$(grep -w "$cell_id" $cells | cut -f3)
    mouse_type=$(grep -w "$cell_id" $cells | cut -f4)
    sex=$(grep -w "$cell_id" $cells | cut -f5)
    dir_move_to="${mouse_type}_${sex}"
    mv $bam_file $dir_move_to
done

echo Done moving RH.CB.bam files

# Now go back to DT 
cd .. 
cd polyDT
mkdir 21mo_female
mkdir 21mo_male 
mkdir 3mo_female 
mkdir 3mo_male 
mkdir 5xFAD_female 
mkdir 5xFAD_male 
mkdir 6mo_female 
mkdir 6mo_male 
mkdir APOE4_female  
mkdir APOE4_male  

bam_files=($(find -name "*DT.CB.bam"))

# Loop through every BAM file in the subfolder and obtain its cell id, cell type (via cells file) and make symlink to output directory
for bam_file in "${bam_files[@]}"; do
    # Get the cell id from the BAM file name
    cell_id=$(basename "$bam_file" .DT.CB.bam)
    #echo $cell_id

    # Get the cell type for the cell_id
    # find the row in the file cells, where the first column is cell_id
    cell_type=$(grep -w "$cell_id" $cells | cut -f3)
    mouse_type=$(grep -w "$cell_id" $cells | cut -f4)
    sex=$(grep -w "$cell_id" $cells | cut -f5)
    dir_move_to="${mouse_type}_${sex}"
    mv $bam_file $dir_move_to
done

echo Done moving DT.CB.bam files
