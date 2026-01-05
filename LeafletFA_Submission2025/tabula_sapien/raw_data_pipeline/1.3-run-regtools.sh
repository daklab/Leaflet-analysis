#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 6
#SBATCH --mem=40000M
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -J cell_juncs # <-- name of job
#SBATCH --array=0-12 # <-- number of jobs to run (number of unique cell ID samples)

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load samtools
module load sambamba

input_dir=/commons/projects/CZI-tabula-sapiens/SS2_data_processing
cd $input_dir

# Get a list of all folders in /commons/projects/CZI-tabula-sapiens/SS2_data_processing; this will be arrayed (one job per folder)
#ls -d */ > donor_folders.txt

# NOTE --> IM NOT CURRENTLY RUNNING THIS BECAUSE IT WILL GENERATE HUGE BAM FILES 
# WILL ATTEMPT TO RUN REGTOOLS ON EACH FILE INSTEAD

# Get the folder name for the current job
donor_name=$(sed -n "$(($SLURM_ARRAY_TASK_ID + 1))p" donor_folders.txt)
cd $donor_name
echo "The donor name is $donor_name"

output_dir="/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions"
regtools_run=/gpfs/commons/home/kisaev/regtools/build/regtools

# Loop through each organ tissue folder
organ_folders=($(ls -d */))

for organ_folder in "${organ_folders[@]}"; do
    cd "$organ_folder"
    echo "The organ tissue folder is $organ_folder"

    # Set the output pseudobulk BAM file name
    organ_folder="${organ_folder//\//}"
    donor_name="${donor_name//\//}"

    # Going to run Regtools on each cell in this folder
    all_cells=($(ls *.Aligned.out.sorted.bam.CB.bam))
    mkdir $output_dir/$donor_name
    mkdir $output_dir/$donor_name/$organ_folder
    for cell in "${all_cells[@]}"; do
        echo "extracting junctions for $cell"
        cell_name="${cell//.Aligned.out.sorted.bam.CB.bam/}"
        # Set the output pseudobulk BAM file name
        output_file="${output_dir}/${donor_name}/${organ_folder}/${cell_name}.juncs"
        output_barcodes="${output_dir}/${donor_name}/${organ_folder}/${cell_name}.barcodes"
        output_juncswbarcodes="${output_dir}/${donor_name}/${organ_folder}/${cell_name}.juncswbarcodes"
        # otherwise run regtools if file doesn't exist
        echo Extracting junctions with regtools!
        $regtools_run junctions extract -a 6 -m 50 -M 500000 $cell -o $output_file -s XS -b $output_barcodes
        paste --delimiters='\t' $output_file $output_barcodes > $output_juncswbarcodes
        echo "done extracting junctions for $cell"
    done
    cd ..
done
