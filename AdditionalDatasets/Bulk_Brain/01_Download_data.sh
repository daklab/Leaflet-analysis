#!/bin/bash
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -J BULKSRA_download
#SBATCH -c 1
#SBATCH --mem=128G
#SBATCH -t 5-00:00 # Runtime in D-HH:MM

# Load any required modules here
module load sratoolkit

output_dir="/commons/projects/knowles_singlecell_splicing/bulk_whole_mouse/data"

# Array of SRAs
sras=(
    SRR26643415
    SRR26643416
    SRR26643417
    SRR26643418
    SRR26643419
    SRR26643420
    SRR26643421
    SRR26643422
)

# Loop through each SRA and download
for sra in "${sras[@]}"; do

    echo "Downloading $sra"
    
    # Create directory for each SRA Run ID
    mkdir -p "$output_dir/$sra"

    # Download the SRA files using fastq-dump into the created directory
    fastq-dump --outdir "$output_dir/$sra" --gzip --split-files $sra
done