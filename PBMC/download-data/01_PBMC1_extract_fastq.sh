#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 6
#SBATCH --mem=30000M
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -J PBMC_sra # <-- name of job
#SBATCH --array=1-408%10

# Download the SRA files from the SRA accession numbers in the PBMC1 and PBMC2 sample files
# This data is from the following paper:
# https://www.nature.com/articles/s41587-020-0465-8
# Specifically all raw data for PBMC1 and PBMC2 samples

# Load the SRA toolkit
module load sratoolkit 
cd /gpfs/commons/groups/knowles_lab/data/sc/pbmc
output_dir=/gpfs/commons/groups/knowles_lab/data/sc/pbmc/fastq_files/PBMC1
cd PBMC1

# Get list of SRA based on folder names 
SRRs=($(ls -d SRR*))

# Use SLURM_ARRAY_TASK_ID to pick the SRA accession
SRA_ACCESSION=${SRRs[$SLURM_ARRAY_TASK_ID-1]}

echo "Processing $SRA_ACCESSION..."

# Go into the folder and download the SRA file
cd $SRA_ACCESSION

# Get the SRA file name
sra_file=$(ls *.sra)

echo "Extracting FASTQ from $SRA_ACCESSION..."
echo "The SRA file is $sra_file"

# Download the SRA file as fastq files 
fasterq-dump "$sra_file" --split-files --outdir ${output_dir}/"$SRA_ACCESSION"

echo "$SRA_ACCESSION FASTQ extraction complete."
echo "Job $SLURM_ARRAY_TASK_ID for sample $SRA_ACCESSION complete."

# At the end --> double check slurm files --> it seems some accessions were finished downloading completely in the first job  