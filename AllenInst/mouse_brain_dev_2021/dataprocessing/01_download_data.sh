#!/bin/bash
#
#SBATCH -J SRA_download
#SBATCH --mem=128G
#SBATCH -t 5-00:00 # Runtime in D-HH:MM

# conda activate python3ENV

# Load any required modules here
module load sratoolkit

output_dir="/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/SRA/data"

# Read the SRAs from the file
sra_file="SRR_Acc_List.txt"

# Function to download one SRA
download_sra() {
    sra=$1
    sra_dir="$output_dir/$sra"
    
    # Check if the directory exists and is non-empty
    if [ -d "$sra_dir" ] && [ "$(ls -A "$sra_dir")" ]; then
        echo "Directory for $sra already exists and is not empty, skipping download."
    else
        echo "Downloading $sra"
        
        # Create directory if it doesn't exist
        mkdir -p "$sra_dir"

        # Download the SRA files using fastq-dump into the created directory
        fastq-dump --outdir "$sra_dir" --gzip --split-files "$sra"
    fi
}

export -f download_sra
export output_dir

# Use xargs to run 30 downloads in parallel
cat "$sra_file" | xargs -n 1 -P 30 -I {} bash -c 'download_sra "$@"' _ {}

# scp SRR_Acc_List.txt kisaev@pe2cc3-042://gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/SRA

