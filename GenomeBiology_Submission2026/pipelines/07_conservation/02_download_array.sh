#!/bin/bash
#SBATCH --job-name=gse138734_dl
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/RNA_ATLAS/GSE138734/fastqs/logs/dl_%A_%a.out
#SBATCH --error=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/RNA_ATLAS/GSE138734/fastqs/logs/dl_%A_%a.err

set -euo pipefail

module load SRA-Toolkit/3.1.1-gompi-2023b

PIGZ=/gpfs/commons/home/kisaev/miniconda3/envs/spipe/bin/pigz
SRR_LIST=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/RNA_ATLAS/GSE138734/fastqs/srr_list.txt
FASTQ_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/RNA_ATLAS/GSE138734/fastqs

SRR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SRR_LIST")

if [[ -z "$SRR" ]]; then
    echo "ERROR: empty SRR for task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

OUTD="$FASTQ_DIR/$SRR"
mkdir -p "$OUTD"

echo "Downloading $SRR → $OUTD"
fasterq-dump "$SRR" --outdir "$OUTD" --temp "$OUTD" --split-files --threads 8

mv "$OUTD/${SRR}_1.fastq" "$OUTD/${SRR}_R1.fastq"
mv "$OUTD/${SRR}_2.fastq" "$OUTD/${SRR}_R2.fastq"

$PIGZ -p 8 "$OUTD/${SRR}_R1.fastq" "$OUTD/${SRR}_R2.fastq"
echo "Done: $SRR"
