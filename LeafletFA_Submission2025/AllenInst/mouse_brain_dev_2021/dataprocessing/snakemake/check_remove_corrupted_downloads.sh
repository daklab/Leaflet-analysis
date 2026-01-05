#!/bin/bash
#SBATCH --job-name=check_fastq   # Job name
#SBATCH --output=check_fastq.out  # Output log file
#SBATCH --error=check_fastq.err   # Error log file
#SBATCH --time=72:00:00           # Time limit (2 hours)
#SBATCH --mem=32G                  # Memory allocation (8GB)

# Load necessary modules if required (example)
module load gzip

# Define the base directory where SRR folders are located
BASE_DIR="/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/SRA/data"

# Log file to store corrupted files
LOG_FILE="/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/SRA/"
LOG_FILE="$LOG_FILE/$(date +%Y%m%d)_corrupted_fastq_list.txt"
echo "$LOG_FILE"

# Initialize log file
echo "Corrupted FASTQ files detected and deleted on $(date)" > "$LOG_FILE"

# Use find to locate all FASTQ files and check them in parallel
find "$BASE_DIR" -type f -name "*.fastq.gz" | \
xargs -P 8 -I {} sh -c 'gzip -t "{}" 2>/dev/null || echo "{}"'

# Collect list of corrupted files
CORRUPTED_FILES=$(find "$BASE_DIR" -type f -name "*.fastq.gz" | xargs -P 8 -I {} sh -c 'gzip -t "{}" 2>/dev/null || echo "{}"')

# Log corrupted files
echo "$CORRUPTED_FILES" >> "$LOG_FILE"

# Remove directories containing corrupted files
echo "$CORRUPTED_FILES" | sed 's|/[^/]*$||' | sort -u | tee -a "$LOG_FILE" | xargs -P 4 rm -rf

echo "Check complete. List of deleted files saved in $LOG_FILE."

# cd /gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/SRA 
# sbatch /gpfs/commons/home/kisaev/Leaflet-analysis/AllenInst/mouse_brain_dev_2021/dataprocessing/snakemake/check_remove_corrupted_downloads.sh 