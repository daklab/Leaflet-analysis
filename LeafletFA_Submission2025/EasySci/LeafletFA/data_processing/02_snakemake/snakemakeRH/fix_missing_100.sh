#!/bin/bash
#SBATCH -J fix_100
#SBATCH -p cpu,dev
#SBATCH --mem=4G
#SBATCH --time=1:00:00
#SBATCH --array=0-2395%32
#SBATCH -o /gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/EasySci/LeafletFA/data_processing/02_snakemake/snakemakeRH/fix_100_logs/slurm-%A_%a.out

module load samtools

REGTOOLS="/gpfs/commons/home/kisaev/regtools/build/regtools"
BASE="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/SpliceVI_100"
MISSING_FILE="/gpfs/commons/home/kisaev/missing_100_clusters.txt"

# Get cluster name for this array task
CLUSTER=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MISSING_FILE")
echo "Processing: $CLUSTER"

BAM="$BASE/combined_RH/$CLUSTER/${CLUSTER}_CB.bam"
OUTDIR="$BASE/junctions/$CLUSTER"

# Skip if already done
if [[ -f "$OUTDIR/junctions.bed" ]]; then
    echo "Already exists, skipping"
    exit 0
fi

# Index _CB.bam if needed
if [[ ! -f "${BAM}.bai" ]]; then
    samtools index "$BAM"
fi

# Extract junctions
mkdir -p "$OUTDIR"
$REGTOOLS junctions extract -a 6 -m 50 -M 500000 "$BAM" -o "$OUTDIR/junctions.bed" -s XS -b "$OUTDIR/barcodes.txt"
paste --delimiters=$'\t' "$OUTDIR/junctions.bed" "$OUTDIR/barcodes.txt" > "$OUTDIR/junctions_with_barcodes.bed"

# Generate summary
total_reads=$(samtools view -c -F 4 "$BAM")
spliced_reads=$(awk '{ sum += $5 } END { print (sum == "" ? 0 : sum) }' "$OUTDIR/junctions.bed")
unspliced_reads=$((total_reads - spliced_reads))
echo -e "Sample\tTotal_Reads\tSpliced_Reads\tUnspliced_Reads" > "$BASE/junctions/${CLUSTER}_splicing_summary.tsv"
echo -e "$CLUSTER\t$total_reads\t$spliced_reads\t$unspliced_reads" >> "$BASE/junctions/${CLUSTER}_splicing_summary.tsv"

echo "Done: $CLUSTER"
