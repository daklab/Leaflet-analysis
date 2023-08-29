#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p bigmem
#SBATCH -c 6
#SBATCH --mem=300000M
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -J split_easysci # <-- name of job

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load samtools

# Get list of bam files to run array jobs
#input_bams=/gpfs/commons/groups/knowles_lab/data/sc/rockefeller_2022
#wd=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci/symlinks
# Create symlinks in wd for the input_bams 
#ln -s ${input_bams}/random_hexamer/ ${wd} # something here didn't work... need to fix this after files are done being moved 
#ln -s ${input_bams}/shortDT/ ${wd}

WD=/gpfs/commons/groups/knowles_lab/data/sc/rockefeller_2022

# Let's first create a bunch of folders to split the files into so we are not dealing with just one giant directory

# Number of subfolders
num_subfolders=1000

# Create subfolders random hexamers 
#for ((i=0; i<num_subfolders; i++)); do
#    mkdir -p "$WD/random_hexamer/subfolder_$i"
#done

## Create subfolders shortDT
#for ((i=0; i<num_subfolders; i++)); do
#    mkdir -p "$WD/shortDT/subfolder_$i"
#done

# List all files in the source directory random hexamers
files=("$WD"/random_hexamer/*.bam)
num_files=${#files[@]}  # Get the number of files

# Loop through files and distribute randomly
for file in "${files[@]}"; do
    random_subfolder=$((RANDOM % num_subfolders))
    destination_subfolder="$WD/random_hexamer/subfolder_$random_subfolder"
    mv "$file" "$destination_subfolder/"
done
echo "Files randomly distributed into subfolders for random hexamers."

# Loop through files and distribute randomly shortDT
files=("$WD"/shortDT/*.bam)

# Loop through files and distribute randomly
for file in "${files[@]}"; do
    random_subfolder=$((RANDOM % num_subfolders))
    destination_subfolder="$WD/shortDT/subfolder_$random_subfolder"
    mv "$file" "$destination_subfolder/"
done