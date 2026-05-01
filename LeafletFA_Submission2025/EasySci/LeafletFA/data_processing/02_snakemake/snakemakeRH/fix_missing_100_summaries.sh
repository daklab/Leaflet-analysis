#!/bin/bash
#SBATCH -J fix_100_sum
#SBATCH -p cpu,dev
#SBATCH --mem=2G
#SBATCH --time=0:30:00
#SBATCH --array=0-8730%200
#SBATCH -o /gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/EasySci/LeafletFA/data_processing/02_snakemake/snakemakeRH/fix_100_logs/sum-%A_%a.out

module load samtools

BASE="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/SpliceVI_100"
MISSING_FILE="/gpfs/commons/home/kisaev/missing_100_summaries.txt"

CLUSTER=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MISSING_FILE")

# Skip if summary already exists
if [[ -f "$BASE/junctions/${CLUSTER}_splicing_summary.tsv" ]]; then
    exit 0
fi

BAM="$BASE/combined_RH/$CLUSTER/${CLUSTER}_CB.bam"
JUNC_BED="$BASE/junctions/$CLUSTER/junctions.bed"

total_reads=$(samtools view -c -F 4 "$BAM")
spliced_reads=$(awk '{ sum += $5 } END { print (sum == "" ? 0 : sum) }' "$JUNC_BED")
unspliced_reads=$((total_reads - spliced_reads))
echo -e "Sample\tTotal_Reads\tSpliced_Reads\tUnspliced_Reads" > "$BASE/junctions/${CLUSTER}_splicing_summary.tsv"
echo -e "$CLUSTER\t$total_reads\t$spliced_reads\t$unspliced_reads" >> "$BASE/junctions/${CLUSTER}_splicing_summary.tsv"
