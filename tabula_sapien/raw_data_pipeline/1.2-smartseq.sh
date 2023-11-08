#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 6
#SBATCH --mem=40000M
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -J pseudo_BAM # <-- name of job
#SBATCH --array=0-12 # <-- number of jobs to run (number of unique cell ID samples)

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load samtools

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

output_dir="/commons/projects/CZI-tabula-sapiens/SS2_psuedobulk_BAM"

# Loop through each organ tissue folder
organ_folders=($(ls -d */))

for organ_folder in "${organ_folders[@]}"; do
    cd "$organ_folder"
    echo "The organ tissue folder is $organ_folder"

    # Set the output pseudobulk BAM file name
    organ_folder="${organ_folder//\//}"
    donor_name="${donor_name//\//}"
    output_file="${output_dir}/${donor_name}_${organ_folder}_pseudobulk.bam"

    echo "The pseudobulk merged file will be $output_file"

    # making a pseudobulk file for each organ tissue folder
    samtools merge "${output_file}" *.Aligned.out.sorted.bam.CB.bam

    # index it 
    samtools index "${output_file}"
    cd ..
done
