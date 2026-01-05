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

output_dir="/gpfs/commons/projects/knowles_singlecell_splicing/10X_cortex_2020/data"

# Array of SRAs
sras=(
    SRR9169236
    SRR9170684	
    SRR9170686	
    SRR9170687	
    SRR9170691
    SRR9170692	
    SRR9170693
    SRR9170886	
    SRR9169228
    SRR9169229
    SRR9169230
    SRR9169231
    SRR9169233
    SRR9169234
    SRR9169235
    SRR9169236
)

# Loop through each SRA and download
for sra in "${sras[@]}"; do

    echo "Downloading $sra"
    
    # Create directory for each SRA Run ID
    mkdir -p "$output_dir/$sra"

    # Download the SRA files using fastq-dump into the created directory
    fastq-dump --outdir "$output_dir/$sra" --gzip --split-files $sra
done