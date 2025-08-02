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

mkdir -p "$output_dir"

download_sra() {
    local sra=$1
    local sra_dir="$output_dir/$sra"
    local fq1="$sra_dir/${sra}_1.fastq.gz"
    local fq2="$sra_dir/${sra}_2.fastq.gz"

    # Check if both files exist and are non-empty
    if [ -s "$fq1" ] && [ -s "$fq2" ]; then
        echo "$sra already downloaded. Skipping."
        return 0
    fi

    echo "Downloading $sra"
    rm -rf "$sra_dir"
    mkdir -p "$sra_dir"

    # Attempt download
    if ! fastq-dump --outdir "$sra_dir" --gzip --split-files "$sra"; then
        echo "Download failed for $sra"
        rm -rf "$sra_dir"
        return 1
    fi

    # Verify files again after download
    if [ ! -s "$fq1" ] || [ ! -s "$fq2" ]; then
        echo "Incomplete download for $sra — retrying..."
        rm -rf "$sra_dir"
        return 1
    fi
}

export -f download_sra
export output_dir

parallel --jobs 20 download_sra :::: "$sra_file"

# scp SRR_Acc_List.txt kisaev@pe2cc3-042://gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/SRA

# check top 50 most recent folders to ensure both fastq files are present 
# ls -lt --group-directories-first | grep '^d' | head -n 5
# cd /gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/SRA/ 
# sbatch /gpfs/commons/home/kisaev/Leaflet-analysis/AllenInst/mouse_brain_dev_2021/dataprocessing/01_download_data.sh 