#!/bin/bash
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -J SRA_download
#SBATCH -c 1
#SBATCH --mem=128G
#SBATCH -t 5-00:00 # Runtime in D-HH:MM

# Load any required modules here
module load sratoolkit

output_dir="/commons/projects/knowles_singlecell_splicing/allen-brain/neocortical_mouse_2018/data"
sra_accession_file="/commons/projects/knowles_singlecell_splicing/allen-brain/neocortical_mouse_2018/GSE115746_accession_table.csv"

# Loop through each line in the accession file
while IFS= read -r line; do

    sra_run=$(echo "$line" | awk '{print $2}')

    echo "Downloading $sra_run"
    
    # Create directory for each SRA Run ID
    mkdir -p "$output_dir/$sra_run"

    # Download the SRA files using fastq-dump into the created directory
    fastq-dump --outdir "$output_dir/$sra_run" --gzip --split-files $sra_run

    echo "Downloaded $sra_run"

done < $sra_accession_file
echo "All downloads complete"