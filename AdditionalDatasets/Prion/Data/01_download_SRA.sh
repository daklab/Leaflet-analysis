#!/bin/bash
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -J SRA_download
#SBATCH -c 1
#SBATCH --mem=128G
#SBATCH -t 5-00:00 # Runtime in D-HH:MM

# Navigate to the directory where the data will be downloaded
WD=/gpfs/commons/projects/knowles_singlecell_splicing/prion_disease_model

module load sratoolkit

# Read accessions from file
accession_file=$WD/"SRR_Acc_List_Prion"

# In working directory, make new directory for fastq files
mkdir -p $WD/fastq
cd $WD/fastq

# Check if file exists
if [ ! -f "$accession_file" ]; then
    echo "Error: Accession file $accession_file not found."
    exit 1
fi

# Download each accession
while IFS= read -r accession || [ -n "$accession" ]; do
    # Make a directory for each accession
    mkdir -p $accession
    echo "Downloading $accession..."
    # Make sure to download files into the directory for each accession
    cd $accession
    fastq-dump --split-files "$accession"
    cd ..
done < "$accession_file"

echo "Download complete."
