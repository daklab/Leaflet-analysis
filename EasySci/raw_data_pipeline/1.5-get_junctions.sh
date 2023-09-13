#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 4
#SBATCH --mem=64000M
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -J get_junctions # <-- name of job

#####SBATCH --array=1-31  # <-- number of cell_type folders in /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci

# load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load samtools
module load sambamba

# Specify other variables
TISSUE="Astrocytes"
input_dir="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci/Pseudobulks"
output_dir="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci/junctions"
regtools_run=/gpfs/commons/home/kisaev/regtools/build/regtools

cd $input_dir

# Get list of all BAM files and iterate through each one 
bam_files=($(ls *sorted.bam))

for bam in "${bam_files[@]}"; do
    echo $bam
    # figure out what the actual sample name is everything before .pseudobulk.bam.sorted.bam
    sample_name=$(echo $bam | cut -d'.' -f1)
    echo $sample_name
    # Set the output pseudobulk BAM file name
    output_file="${output_dir}/${sample_name}.juncs"
    output_barcodes="${output_dir}/${sample_name}.barcodes"
    output_juncswbarcodes="${output_dir}/${sample_name}.juncswbarcodes"
    echo Extracting junctions with regtools!
    $regtools_run junctions extract -a 6 -m 50 -M 500000 $bam -o $output_file -s XS -b $output_barcodes
    paste --delimiters='\t' $output_file $output_barcodes > $output_juncswbarcodes
done

echo "Finished getting junctions for all pseudobulks!"