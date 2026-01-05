#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 6
#SBATCH --mem=30000M
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -J PBMC_sra2 # <-- name of job
#SBATCH --array=1-410%10

# Download the SRA files from the SRA accession numbers in the PBMC1 and PBMC2 sample files
# This data is from the following paper:
# https://www.nature.com/articles/s41587-020-0465-8
# Specifically all raw data for PBMC1 and PBMC2 samples

# Load the SRA toolkit
module load sratoolkit 
cd /gpfs/commons/groups/knowles_lab/data/sc/pbmc
mkdir PBMC2
cd PBMC2

# Define the file names
PBMC2_FILE="/gpfs/commons/groups/knowles_lab/data/sc/pbmc/PBMC2_sample.tsv"

# Calculate line to process (add 1 because of header)
LINE_NUM=$(($SLURM_ARRAY_TASK_ID + 1))

# Read SRA accession number from the specific line
SRA_ACCESSION=$(sed -n "${LINE_NUM}p" "$PBMC2_FILE" | cut -f1)

echo "Downloading $SRA_ACCESSION..."
prefetch "$SRA_ACCESSION"
echo "$SRA_ACCESSION download complete."

echo "Job $SLURM_ARRAY_TASK_ID for sample $SRA_ACCESSION complete."