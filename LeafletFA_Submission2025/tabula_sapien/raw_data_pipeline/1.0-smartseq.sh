#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 6
#SBATCH --mem=10000M
#SBATCH -t 1-00:00 # Runtime in D-HH:MM
#SBATCH -J TS_symlinks # <-- name of job

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load samtools

input_dir=/gpfs/commons/datasets/controlled/CZI/tabula-sapiens/AWS_data/alignment-gencode/SS2
output_dir=/commons/projects/CZI-tabula-sapiens/SS2_data_processing

# Define the SS2 sample file used in final version of tabula sapien adata file 
input_file="/commons/projects/CZI-tabula-sapiens/SS2_data_processing/ss2_samples.txt"

# Read the lines of the input file and submit an array job for each donor
IFS=$'\t' # Set tab as the delimiter
line_number=0

while read -r donor_id organ_tissue bam_file_suffix donor_id_orig; do
    # Get the directory name for the donor_id
    
    echo "The donor is $donor_id"
    echo "The Pilot number is '$donor_id_orig'"

    # Create the donor directory if it doesn't exist
    mkdir -p ${output_dir}/${donor_id}

    # Create the organ tissue subdirectory within the donor directory if it doesn't exist
    subdirectory=${output_dir}/${donor_id}/${organ_tissue}
    mkdir -p "$subdirectory"

    # Find the matching original raw BAM file
    orig_bam=$(ls "${input_dir}/${donor_id_orig}/" | grep "${bam_file_suffix}" | grep '\.bam$')
    orig_bai=$(ls "${input_dir}/${donor_id_orig}/" | grep "${bam_file_suffix}" | grep '\.bai$')

    # Create a symlink for each matching file in the subdirectory
    ln -s "${input_dir}/${donor_id_orig}/${orig_bam}" "${subdirectory}/${orig_bam}"
    ln -s "${input_dir}/${donor_id_orig}/${orig_bai}" "${subdirectory}/${orig_bai}"

    #line_number=$((line_number + 1))
done < "$input_file"

