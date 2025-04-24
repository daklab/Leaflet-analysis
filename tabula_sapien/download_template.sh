#!/bin/sh
#SBATCH -N 1 
#SBATCH --mem=40000M
#SBATCH -t 5-00:00 
#SBATCH --job-name=AWS_${1}  # Corrected Job Name Expansion

module purge
module load awscli

# Check if an argument was provided
if [ -z "$1" ]; then
    echo "Error: No dataset name provided. Usage: sbatch download_tsp_v2.sh TSP1"
    exit 1
fi

# Argument (dataset name, e.g., TSP1, TSP2, ..., TSPN10)
DATASET="$1"

# Define base S3 path
S3_BASE="s3://czb-tabula-sapiens/TabulaSapiens_v2/${DATASET}/alignment_gencode41/smartseq/"
LOCAL_DIR="/gpfs/commons/datasets/controlled/CZI/tabula-sapiens/TabulaSapiens_v2/SS2/${DATASET}"

# Create local directory (cleaner structure)
mkdir -p "$LOCAL_DIR"
cd "$LOCAL_DIR"

echo "Downloading from $S3_BASE"

# Find only the relevant BAM and TXT files inside 'per/' subdirectories
aws s3 ls "$S3_BASE" --recursive | awk '{print $NF}' | grep -E "/per/.+(Aligned.sorted.out.bam|htseq-count.txt)$" | while read -r file_path; do
    echo "Processing: $file_path"

    # Extract a cleaner local directory structure
    CLEANED_PATH=$(echo "$file_path" | sed "s#TabulaSapiens_v2/${DATASET}/alignment_gencode41/smartseq/##")
    LOCAL_PATH="$LOCAL_DIR/$(dirname "$CLEANED_PATH")"
    FILENAME=$(basename "$file_path")

    # Ensure local directory exists
    mkdir -p "$LOCAL_PATH"

    # Check if file already exists
    if [ -f "$LOCAL_PATH/$FILENAME" ]; then
        echo "Skipping (already downloaded): $LOCAL_PATH/$FILENAME"
    else
        echo "Downloading: $file_path"
        aws s3 cp "s3://czb-tabula-sapiens/$file_path" "$LOCAL_PATH/"
    fi
done

echo "Download completed for $DATASET"

# ALL DONE!! 