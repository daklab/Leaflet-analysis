#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 6
#SBATCH --mem=30000M
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -J LGN # <-- name of job
#SBATCH --array=1-5552%10

# Download the SRA files from the SRA accession numbers --> https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE182211
# Title	Single-cell and single-nucleus RNA-seq uncovers shared and distinct axes of variation in dorsal LGN neurons in mice, non-human primates, and humans
# Organisms	Macaca fascicularis; Macaca nemestrina; Homo sapiens; Mus musculus

# Load the SRA toolkit
module load sratoolkit 
cd /gpfs/commons/groups/knowles_lab/data/sc/allen-brain/LGN_mouse_human_monkey
mkdir SRA
cd SRA

# Define the file names
V1_FILE="/gpfs/commons/groups/knowles_lab/data/sc/allen-brain/LGN_mouse_human_monkey/GSE182211_first_volume.tsv"

#[done once]
#V2_FILE="/gpfs/commons/groups/knowles_lab/data/sc/allen-brain/LGN_mouse_human_monkey/GSE182211_second_volume.tsv"
# Remove header from V2_file and append to V1_file
#tail -n +2 "$V2_FILE" >> "$V1_FILE"

# print number of lines in the file $V1_FILE -1 (to account for the header) to get number of samples
NUM_SAMPLES=$(($(wc -l < "$V1_FILE") - 1))
echo "Number of samples: $NUM_SAMPLES"

# Calculate line to process (add 1 because of header)
LINE_NUM=$(($SLURM_ARRAY_TASK_ID + 1))

# Stop running if LINE_NUM is 1 
if [ $LINE_NUM -eq 1 ]; then
    echo "Skipping header line."
    exit 0
fi

# Read SRA accession number from the specific line
SRA_ACCESSION=$(sed -n "${LINE_NUM}p" "$V1_FILE" | cut -f1)

echo "Downloading $SRA_ACCESSION..."

# Error handling placeholder for prefetch
if prefetch "$SRA_ACCESSION"; then
    echo "$SRA_ACCESSION download complete."
else
    echo "Error downloading $SRA_ACCESSION."
    # Handle error, e.g., log to a file or retry
fi

echo "Job $SLURM_ARRAY_TASK_ID for sample $SRA_ACCESSION complete."