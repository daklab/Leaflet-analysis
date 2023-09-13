#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 4
#SBATCH --mem=64000M
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -J pseudobulk # <-- name of job

#####SBATCH --array=1-31  # <-- number of cell_type folders in /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci

# load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load samtools
module load sambamba

# run on just astrocytes for now -> make pseudobulk first within each subgroup 
cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci/Astrocytes/random_hexamers

# Define the list of subgroups
subgroups=("21mo_female" "21mo_male" "3mo_female" "3mo_male" "5xFAD_female" "5xFAD_male" "6mo_female" "6mo_male" "APOE4_female" "APOE4_male")

# Specify other variables
TISSUE="Astrocytes"
random_hexamers="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci/${TISSUE}/random_hexamers"
output_dir="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci/Pseudobulks"

# Iterate through each subgroup
for subgroup in "${subgroups[@]}"; do
    echo "Processing subgroup ${subgroup}..."
    output_file="${output_dir}/${TISSUE}_${subgroup}.pseudobulk.bam"
    echo $output_file
    
    # Merge BAM files for the current subgroup and tissue
    samtools merge ${output_file} ${random_hexamers}/${subgroup}/*.CB.bam -f 
    echo "Merged BAM for ${subgroup} tissue ${TISSUE} into ${output_file}"

    # Optionally index the merged BAM file
    sambamba sort -o ${output_file}.sorted.bam ${output_file}
    echo "Sorted and indexed BAM for ${subgroup} tissue ${TISSUE} into ${output_file}.sorted.bam"
done

echo "All BAM files merged successfully!"