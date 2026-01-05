#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 4
#SBATCH --mem=32000M
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -J get_junctions # <-- name of job
#SBATCH --array=1-310%14  # <-- number of files in the input directory RH and DT

# load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load samtools
module load sambamba

# Specify other variables
input_dir_RH="/gpfs/commons/projects/knowles_singlecell_splicing/EasySci/Pseudobulks/RH"
input_dir_DT="/gpfs/commons/projects/knowles_singlecell_splicing/EasySci/Pseudobulks/DT"

# Calculate the number of files in the input directory RH and DT
num_files_RH=$(ls $input_dir_RH | wc -l)
num_files_DT=$(ls $input_dir_DT | wc -l)

# Print the number of files in the input directory RH and DT
echo "Number of files in the input directory RH: $num_files_RH"
echo "Number of files in the input directory DT: $num_files_DT"

output_dir="/gpfs/commons/projects/knowles_singlecell_splicing/EasySci/Leaflet/junctions"

# Make directory in $output_dir for RH and for DT
mkdir -p $output_dir/RH
mkdir -p $output_dir/DT

regtools_run=/gpfs/commons/home/kisaev/regtools/build/regtools

## ----------------- Get junctions for RH pseudobulks ----------------- ##

cd $input_dir_RH

# Get list of all BAM files in the input directory and use array index to get the file name
bam_files=($(ls *.pseudobulk.bam.sorted.bam))

# Use the array index to get the file name for this job
bam=${bam_files[$SLURM_ARRAY_TASK_ID-1]}
echo "Working on $bam"

# Figure out what the actual sample name is everything before .pseudobulk.bam.sorted.bam
sample_name=$(echo $bam | cut -d'.' -f1)
echo $sample_name

# Set the output pseudobulk BAM file name
output_file_RH="${output_dir}/RH/${sample_name}.juncs"
output_barcodes_RH="${output_dir}/RH/${sample_name}.barcodes"
output_juncswbarcodes_RH="${output_dir}/RH/${sample_name}.juncswbarcodes"

"echo Extracting junctions with regtools!"

$regtools_run junctions extract -a 6 -m 50 -M 500000 $bam -o $output_file_RH -s XS -b $output_barcodes_RH
paste --delimiters='\t' $output_file_RH $output_barcodes_RH > $output_juncswbarcodes_RH
echo "Finished getting junctions for all RH pseudobulks!"

## ----------------- Get junctions for DT pseudobulks ----------------- ##

cd $input_dir_DT

# Get list of all BAM files in the input directory and use array index to get the file name
bam_files=($(ls *.pseudobulk.bam.sorted.bam))

# Use the array index to get the file name for this job
bam=${bam_files[$SLURM_ARRAY_TASK_ID-1]}
echo "Working on $bam"

# Figure out what the actual sample name is everything before .pseudobulk.bam.sorted.bam
sample_name=$(echo $bam | cut -d'.' -f1)
echo $sample_name

# Set the output pseudobulk BAM file name
output_file_DT="${output_dir}/DT/${sample_name}.juncs"
output_barcodes_DT="${output_dir}/DT/${sample_name}.barcodes"
output_juncswbarcodes_DT="${output_dir}/DT/${sample_name}.juncswbarcodes"

"echo Extracting junctions with regtools!"

$regtools_run junctions extract -a 6 -m 50 -M 500000 $bam -o $output_file_DT -s XS -b $output_barcodes_DT
paste --delimiters='\t' $output_file_DT $output_barcodes_DT > $output_juncswbarcodes_DT
echo "Finished getting junctions for all DT pseudobulks!"