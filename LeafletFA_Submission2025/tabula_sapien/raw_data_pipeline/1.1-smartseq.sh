#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 6
#SBATCH --mem=30000M
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -J merge_BAM # <-- name of job
#SBATCH --array=0-12 # <-- number of jobs to run (number of unique cell ID samples)

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load samtools

input_dir=/commons/projects/CZI-tabula-sapiens/SS2_data_processing
cd $input_dir

# Get a list of all folders in /commons/projects/CZI-tabula-sapiens/SS2_data_processing; this will be arrayed (one job per folder)
#ls -d */ > donor_folders.txt

# Get the folder name for the current job
donor_name=$(sed -n "$(($SLURM_ARRAY_TASK_ID + 1))p" donor_folders.txt)
cd $donor_name
echo "The donor name is $donor_name"

# Loop through each organ tissue folder
organ_folders=($(ls -d */))
for organ_folder in "${organ_folders[@]}"; do
    cd "$organ_folder"
    echo "The organ tissue folder is $organ_folder"
    # Loop through each BAM file in the current organ tissue folder
    for bam_file in *.Aligned.out.sorted.bam; do
        # Extract the cell ID from the BAM file name (everything before .Aligned.out.sorted.bam)
        cell_id=$(basename "$bam_file" .Aligned.out.sorted.bam)
        #check if $bam_file.CB.bam does not exist and if so, add the cell ID to the BAM file
        if [ ! -f "$bam_file.CB.bam" ]; then
            echo "Adding cell ID to $bam_file"
            samtools view -h $bam_file | awk -v cb=$cell_id -F '\t' 'BEGIN {OFS="\t"} {$NF = $NF"\tCB:Z:"cb; print}' | samtools view -bS - > $bam_file.CB.bam
            # index this new file 
             samtools index $bam_file.CB.bam
        else
            echo "$bam_file.CB.bam already exists"
            continue
        fi 
    done
    echo "Finished adding cell IDs to BAM files in $organ_folder"
    cd ..
echo "Finished adding cell IDs to BAM files in $donor_name"
done
