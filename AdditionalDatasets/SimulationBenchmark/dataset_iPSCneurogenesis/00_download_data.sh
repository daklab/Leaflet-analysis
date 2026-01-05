#!/bin/bash
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -J SRA_download
#SBATCH -c 1
#SBATCH --mem=128G
#SBATCH -t 5-00:00 # Runtime in D-HH:MM

# Load any required modules here
module load sratoolkit

output_dir="/gpfs/commons/projects/knowles_singlecell_splicing/GSE85908/suppfiles/data/SRA"

# Read the SRAs from the file
sra_file="/gpfs/commons/projects/knowles_singlecell_splicing/GSE85908/suppfiles/SRR_Acc_List_iPSCneuro.txt"

# Loop through each SRA and download
while read -r sra; do
    if [[ ! -z "$sra" ]]; then
        echo "Downloading $sra"
        
        # Create directory for each SRA Run ID
        mkdir -p "$output_dir/$sra"

        # Download the SRA files using fastq-dump into the created directory
        fastq-dump --outdir "$output_dir/$sra" --gzip --split-files "$sra"
    fi
done < "$sra_file"
