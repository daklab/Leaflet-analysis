#!/bin/bash
# Build pseudobulk BAMs per metacell for EasySci RH (random hexamer) reads.
# Metacell definitions from 00_MetaCellAnalysis.py (DT-based clustering).
#SBATCH -J RH_pseudobulk
#SBATCH --mem=64G
#SBATCH -t 0-03:00 # Runtime in D-HH:MM
#SBATCH -p cpu,bigmem,dev
#SBATCH --array=1-4931%128 # number of metacells; %128 = max 128 concurrent
#SBATCH -o /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/slurm/%A_%a.out
#SBATCH -e /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/slurm/%A_%a.err

#conda activate python3ENV 
module load samtools

# Input variables. Set METACELL_SUFFIX for multiple runs (e.g. _1000, _2000, _4000).
SUFFIX="${METACELL_SUFFIX:-}"
CSV_FILE="${METACELL_CSV:-/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/DT_w_RH_cells_anndata_meta${SUFFIX}.tsv}"
ROOT_DIR="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/"
OUTPUT_BASE="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/SpliceVI${SUFFIX}"
OUTPUT_DIR="$OUTPUT_BASE" 

# Make a new directory for the date in OUTPUT_DIR
# Make data year month day 
DATE=$(date +%Y%m%d)
OUTPUT_DIR="$OUTPUT_DIR/$DATE"
mkdir -p "$OUTPUT_DIR"

# Make subdir for RH
RH_DIR="$OUTPUT_DIR/RH"
mkdir -p "$RH_DIR"

tail -n +2 "$CSV_FILE" | cut -d',' -f5 | sort | uniq > "$OUTPUT_DIR/cluster_list.txt"

# Get the cluster name corresponding to this task ID (original for comparison)
cluster_original=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$OUTPUT_DIR/cluster_list.txt")
echo "1/5 cluster: $cluster_original"

# If "/" is found within the cluster name, replace it with "_" for paths
cluster=$(echo "$cluster_original" | tr '/' '_')

# Process RH only (DT anndatas already exist)
for cell_type in RH; do
    type_dir="$RH_DIR"
    cluster_dir="$type_dir/$cluster"
    mkdir -p "$cluster_dir"
    merged_bam="$cluster_dir/${cluster}.bam"

    # Skip if merged BAM already exists and is non-empty
    if [[ -f "$merged_bam" && -s "$merged_bam" ]]; then
        echo "2/5 already exists: $merged_bam"
        continue
    fi

    # TSV column 24 = RH_cell_BAM_file_name
    bam_col=24
    sorted_bam_list=()
    while IFS=',' read -r -a fields; do
        main_cluster_wkmeans="${fields[4]}"
        bam_file="${fields[$bam_col]}"
        main_cluster="${fields[3]}"
        if [[ "$main_cluster_wkmeans" == "$cluster_original" ]]; then
            bam_path="$ROOT_DIR/${main_cluster}/${cell_type}/${bam_file}"
            if [[ -f "$bam_path" ]]; then
                sorted_bam="$cluster_dir/$(basename "$bam_file" .bam).sorted.bam"
                if [[ ! -f "$sorted_bam" ]]; then
                    samtools sort -o "$sorted_bam" "$bam_path"
                fi
                sorted_bam_list+=("$sorted_bam")
            fi
        fi
    done < <(tail -n +2 "$CSV_FILE")
    echo "2/5 found ${#sorted_bam_list[@]} BAM files"

    # Merge and index sorted BAM files
    if [[ ${#sorted_bam_list[@]} -gt 0 ]]; then
        echo "3/5 merging -> $merged_bam"
        samtools merge "$merged_bam" "${sorted_bam_list[@]}"
        samtools index "$merged_bam"
        if [[ -f "$merged_bam.bai" ]]; then
            echo "4/5 indexed ok"
        else
            echo "4/5 ERROR: index not found"
        fi
        for sorted_bam in "${sorted_bam_list[@]}"; do
            rm "$sorted_bam"
        done
    else
        echo "3/5 no BAM files found, skipping"
    fi
done

echo "5/5 done: $cluster"

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/slurm
# make new slurm dir
# mkdir 02092026
# cd 02092026
# Prefer wrapper (computes array size N from TSV): ./run_MetaCellMakePseudobulk_RH_DT.sh
# Or: METACELL_SUFFIX=_1000 ./run_MetaCellMakePseudobulk_RH_DT.sh
# sbatch 01_MetaCellMakePseudobulk_RH_DT.sh

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/SpliceVI
# Outputs: RH/<cluster>/<cluster>.bam and DT/<cluster>/<cluster>.bam per metacell
# how many values from DT_w_RH_cells_anndata_meta.tsv are missing in the directory 5th column?

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/SpliceVI/20260210
# summarize which metacells are present in RH and DT and which are not found in both 
# find DT -name "*.bam" | cut -d'/' -f2 | sort -u > dt_with_bams.txt
# find RH -name "*.bam" | cut -d'/' -f2 | sort -u > rh_with_bams.txt
# comm -23 dt_with_bams.txt rh_with_bams.txt
# comm -13 dt_with_bams.txt rh_with_bams.txt 
# comm -12 dt_with_bams.txt rh_with_bams.txt 
# comm -123 --total dt_with_bams.txt rh_with_bams.txt
# comm -13 dt_with_bams.txt rh_with_bams.txt > missing_from_dt.txt

# --- Find RH files (cluster dirs) that are NOT in the CSV ---
# Run from the run date dir, e.g. SpliceVI/20260210. Set CSV path:
# CSV_FILE="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/DT_w_RH_cells_anndata_meta.tsv"
# Clusters in CSV (column 5), same normalization as script (slash -> underscore):
# tail -n +2 "$CSV_FILE" | cut -d',' -f5 | tr '/' '_' | sort -u > csv_clusters.txt
# RH cluster dirs that have bams on disk:
# find RH -name "*.bam" | cut -d'/' -f2 | sort -u > rh_with_bams.txt
# RH clusters on disk that are NOT in the CSV:
# comm -23 rh_with_bams.txt csv_clusters.txt
# (optional) save to file:
# comm -23 rh_with_bams.txt csv_clusters.txt > rh_not_in_csv.txt

# --- Find CSV clusters that are MISSING RH files (use csv_clusters.txt = 4931 as reference) ---
# CSV clusters that have no RH BAM on disk:
# comm -23 csv_clusters.txt rh_with_bams.txt
# (optional) save to file:
# comm -23 csv_clusters.txt rh_with_bams.txt > missing_rh_from_csv.txt