#!/bin/bash
#SBATCH -J SRA_download
#SBATCH --mem=250G
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH --cpus-per-task=20  # Match the maximum CPUs available per node
#SBATCH --ntasks=1          # Single task using all CPUs
#SBATCH -p bigmem

# conda activate python3ENV

# Load the necessary modules
module load sratoolkit

# Define variables
output_dir="/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/SRA/data"
sra_file="/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/SRA/SRR_Acc_List.txt"

# Ensure the output directory exists
mkdir -p "$output_dir"

# Function to download a single SRA
download_sra() {
    local sra=$1
    local sra_dir="$output_dir/$sra"

    if [ -d "$sra_dir" ] && [ "$(ls -A "$sra_dir")" ]; then
        echo "Directory for $sra already exists and is not empty. Skipping download."
    else
        echo "Downloading $sra"
        mkdir -p "$sra_dir"

        # Use fastq-dump with gzip
        fastq-dump --outdir "$sra_dir" --gzip --split-files "$sra" || {
            echo "Download failed for $sra. Retrying..."
            rm -rf "$sra_dir"
            return 1
        }
    fi
}

# Export necessary variables and functions for parallel processing
export -f download_sra
export output_dir

# Use GNU Parallel for better parallelization and error handling
parallel --jobs 30 download_sra :::: "$sra_file"

# scp SRR_Acc_List.txt kisaev@pe2cc3-042://gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/SRA

# check top 50 most recent folders to ensure both fastq files are present 
# ls -lt --group-directories-first | grep '^d' | head -n 50