#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 4
#SBATCH --mem=32000M
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -J moving_easysci # <-- name of job
#SBATCH --array=1-35  # <-- number of cell_type folders in /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load samtools

cells=/gpfs/commons/groups/knowles_lab/data/sc/rockefeller_2022/cells_wmain_cluster_type_sex.txt
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/

output_new=/gpfs/commons/projects/knowles_singlecell_splicing/EasySci/

cd $WD 

# Get list of folders inside EasySci2024 to cycle through with arrays
folders=($(ls -d $WD/*/)) # This will create an array of folder paths
 
# Get the folder for this array task
current_folder=${folders[$SLURM_ARRAY_TASK_ID - 1]}
echo $current_folder

current_folder=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024//Cerebellum_granule_neurons/
echo $current_folder

# make sure it's not slurm or genome_files folder if it is then exit 
if [[ $current_folder == *"slurm"* ]] || [[ $current_folder == *"genome_files"* ]]; then
    echo "This is a slurm or genome_files folder, exiting"
    exit 1
fi

# also make sure it's not the "EasySci-RNA-mouse-brain" folder otherwise exit 
if [[ $current_folder == *"EasySci-RNA-mouse-brain"* ]]; then
    echo "This is the EasySci-RNA-mouse-brain folder, exiting"
    exit 1
fi

# also make sure it's not the Seacells folder
if [[ $current_folder == *"Seacells"* ]]; then
    echo "This is the Seacells folder, exiting"
    exit 1
fi

# Navigate into the current folder
cd $current_folder

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 1. Go into random_hexamer folder and remove all *RH.CB.RH.CB* files 
cd RH  

# Combine files into pseudobulk based on sex_Type to break them up 
# Using the file: $cells, figure out which Type_sex combinations exist for the cell type we are looking at and then move the BAM files accordingly 
# extract cell type from $current_folder as last item after the last "/" 
cleaned_path="${current_folder%/}"   # Remove trailing slashes
cell_type="${cleaned_path##*/}"
echo "$cell_type"

# in the output_new folder make a new directory for cell_type and RH and navigate to it
mkdir -p $output_new/$cell_type/RH

# now make the directories for the different mouse types within $output_new/$cell_type/RH/
cd $output_new/$cell_type/RH
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

# get the list of bam files from $current_folder/RH/
cd $current_folder/RH
bam_files=($(find -name "*RH.CB.bam"))

# Loop through every BAM file in the subfolder and obtain its cell id, cell type (via cells file) and make symlink to output directory
for bam_file in "${bam_files[@]}"; do
    # Get the cell id from the BAM file name
    cell_id=$(basename "$bam_file" .RH.CB.bam)
    echo $cell_id
    # Get the cell type for the cell_id
    # find the row in the file cells, where the first column is cell_id
    cell_type=$(grep -w "$cell_id" $cells | cut -f3)
    mouse_type=$(grep -w "$cell_id" $cells | cut -f4)
    sex=$(grep -w "$cell_id" $cells | cut -f5)
    dir_move_to="$output_new/$cell_type/RH/${mouse_type}_${sex}"
    mv $bam_file $dir_move_to
done

echo Done moving RH.CB.bam files

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

cleaned_path="${current_folder%/}"   # Remove trailing slashes
cell_type="${cleaned_path##*/}"
echo "$cell_type"

# 2. Go into DT folder 
cd $current_folder/DT
# in the output_new folder make a new directory for cell_type and RH and navigate to it
mkdir -p $output_new/$cell_type/DT

cd $output_new/$cell_type/DT
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

cd $current_folder/DT
bam_files=($(find -name "*DT.CB.bam"))

# Loop through every BAM file in the subfolder and obtain its cell id, cell type (via cells file) and make symlink to output directory
for bam_file in "${bam_files[@]}"; do
    # Get the cell id from the BAM file name
    cell_id=$(basename "$bam_file" .DT.CB.bam)
    # echo $cell_id
    # Get the cell type for the cell_id
    # find the row in the file cells, where the first column is cell_id
    cell_type=$(grep -w "$cell_id" $cells | cut -f3)
    mouse_type=$(grep -w "$cell_id" $cells | cut -f4)
    sex=$(grep -w "$cell_id" $cells | cut -f5)
    dir_move_to="$output_new/$cell_type/DT/${mouse_type}_${sex}"
    mv $bam_file $dir_move_to
done

echo Done moving DT.CB.bam files