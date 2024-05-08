#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 4
#SBATCH --mem=64000M
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -J pseudobulk # <-- name of job
#SBATCH --array=1-31  # <-- 31 predefined cell types total

# load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load samtools
module load sambamba

# file with sample names (n=32 cell types)
samples="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci/regions_samples.txt" 
output_new=/gpfs/commons/projects/knowles_singlecell_splicing/EasySci/

# make file with all the cell types in the EasySci2024 folder
samples=$output_new/cell_types.txt

# extract SAMPLE from samples using row number from SLURM_ARRAY_TASK_ID
SAMPLE=$(awk "NR==$SLURM_ARRAY_TASK_ID" "$samples")

# print sample being processed 
echo "Processing sample ${SAMPLE}..."

output_dir_logs=/gpfs/commons/projects/knowles_singlecell_splicing/EasySci/Pseudobulks/logs

# create log file for sample
log_file="${output_dir_logs}/${SAMPLE}_pseudobulk.log"
echo "Log file for ${SAMPLE} is ${log_file}"

# Echo that now processing RH 
echo "Now processing RH files for ${SAMPLE}..." >> $log_file

# Directories
input_dir=$output_new/${SAMPLE}/RH
output_dir=$output_new/Pseudobulks/RH

# Ensure output directory exists
mkdir -p "$output_dir"

# Define the list of subgroups (based on your data)
subgroups=("21mo_female" "21mo_male" "3mo_female" "3mo_male" "5xFAD_female" "5xFAD_male" "6mo_female" "6mo_male" "APOE4_female" "APOE4_male")

# Iterate through each subgroup
for subgroup in "${subgroups[@]}"; do
    echo "Processing subgroup ${subgroup}..."
    # Save echo message to log_file
    echo "Processing subgroup ${subgroup}..." >> $log_file
    output_file="${output_dir}/${SAMPLE}_${subgroup}.pseudobulk.bam"
    
    # Merge BAM files for the current subgroup
    # Watch out for bash: /nfs/sw/samtools/samtools-1.9/bin/samtools: Argument list too long 
    
    #samtools merge ${output_file} ${input_dir}/${subgroup}/*.CB.bam -f
    find ${input_dir}/${subgroup} -name '*.CB.bam' -print0 | xargs -0 samtools merge ${output_file} -f

    echo "Merged BAM for ${subgroup} into ${output_file}"
    echo "Merged BAM for ${subgroup} into ${output_file}" >> $log_file

    # Sort and index the merged BAM file
    sambamba sort -o "${output_file}.sorted.bam" ${output_file}
    echo "Sorted and indexed BAM for ${subgroup} into ${output_file}.sorted.bam"
    echo "Sorted and indexed BAM for ${subgroup} into ${output_file}.sorted.bam" >> $log_file

    # Print the number of reads in the merged BAM file 
    echo "Number of reads in ${output_file}.sorted.bam:" >> $log_file
    samtools view -c "${output_file}.sorted.bam" >> $log_file

    # Print number of cells in the merged BAM file using CB tag
    echo "Number of cells in ${output_file}.sorted.bam:" >> $log_file
    samtools view "${output_file}.sorted.bam" | cut -f 1 | sort | uniq | wc -l >> $log_file

    # Remove the unsorted BAM file
    rm ${output_file}
done

echo "All RH BAM files for $SAMPLE merged and indexed successfully!" >> $log_file

## processing DT reads

# Echo that now processing DT
echo "Now processing DT files for ${SAMPLE}..." >> $log_file

# Directories
input_dir=$output_new/${SAMPLE}/DT
output_dir=$output_new/Pseudobulks/DT

# Ensure output directory exists
mkdir -p "$output_dir"

# Iterate through each subgroup
for subgroup in "${subgroups[@]}"; do
    echo "Processing subgroup ${subgroup}..."
    # Save echo message to log_file
    echo "Processing subgroup ${subgroup}..." >> $log_file
    output_file="${output_dir}/${SAMPLE}_${subgroup}.pseudobulk.bam"
    
    # Merge BAM files for the current subgroup
    #samtools merge ${output_file} ${input_dir}/${subgroup}/*.CB.bam -f
    find ${input_dir}/${subgroup} -name '*.CB.bam' -print0 | xargs -0 samtools merge ${output_file} -f

    echo "Merged BAM for ${subgroup} into ${output_file}"
    echo "Merged BAM for ${subgroup} into ${output_file}" >> $log_file

    # Sort and index the merged BAM file
    sambamba sort -o "${output_file}.sorted.bam" ${output_file}
    echo "Sorted and indexed BAM for ${subgroup} into ${output_file}.sorted.bam"
    echo "Sorted and indexed BAM for ${subgroup} into ${output_file}.sorted.bam" >> $log_file

    # Print the number of reads in the merged BAM file 
    echo "Number of reads in ${output_file}.sorted.bam:" >> $log_file
    samtools view -c "${output_file}.sorted.bam" >> $log_file

    # Print number of cells in the merged BAM file using CB tag
    echo "Number of cells in ${output_file}.sorted.bam:" >> $log_file
    samtools view "${output_file}.sorted.bam" | cut -f 1 | sort | uniq | wc -l >> $log_file

    # Remove the unsorted BAM file
    rm ${output_file}
done