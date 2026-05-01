#!/bin/bash
#SBATCH -J Retry_BAMs
#SBATCH --mem=96G
#SBATCH -t 0-08:00
#SBATCH -p cpu,bigmem
#SBATCH --array=1-3972   # N = number of lines in missing_rh_from_csv.txt (wc -l)
#SBATCH -o /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/slurm/%A_%a.out
#SBATCH -e /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/slurm/%A_%a.err

# conda activate python3ENV
module load samtools

# Only write logs for the first 5 tasks; suppress output for the rest
if [[ "${SLURM_ARRAY_TASK_ID:-1}" -gt 5 ]]; then
    exec >/dev/null 2>&1
fi

# --- Input variables. Set METACELL_SUFFIX (e.g. _500) and METACELL_OUTPUT_DIR for the run. ---
SUFFIX="${METACELL_SUFFIX:-}"
CSV_FILE="${METACELL_CSV:-/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/DT_w_RH_cells_anndata_meta${SUFFIX}.tsv}"
ROOT_DIR="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/"
OUTPUT_DIR="${METACELL_OUTPUT_DIR:-/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/SpliceVI${SUFFIX}/20260210}"

# Use the list of CSV clusters missing RH BAMs (written by 02_double_check_metacells_missing.sh)
RETRY_LIST="$OUTPUT_DIR/missing_rh_from_csv.txt"
RH_DIR="$OUTPUT_DIR/RH"

# Get the cluster name for this task ID
cluster_original=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$RETRY_LIST")
echo "Processing cluster: $cluster_original"

# Replace "/" with "_" for paths; CSV may have / so we need both for matching
cluster=$(echo "$cluster_original" | tr '/' '_')
cluster_original_slash=$(echo "$cluster_original" | tr '_' '/')

cluster_dir="$RH_DIR/$cluster"
mkdir -p "$cluster_dir"
merged_bam="$cluster_dir/${cluster}.bam"

# Skip if non-empty
if [[ -f "$merged_bam" && -s "$merged_bam" ]]; then
    echo "[RH] Merged BAM already exists. Skipping."
    exit 0
fi

sorted_bam_list=()
while IFS=',' read -r -a fields; do
    main_cluster_wkmeans="${fields[4]}"
    bam_file="${fields[24]}"   # RH_cell_BAM_file_name (col 25, 0-indexed)
    main_cluster="${fields[3]}"

    if [[ "$main_cluster_wkmeans" == "$cluster_original" || "$main_cluster_wkmeans" == "$cluster_original_slash" ]]; then
        bam_path="$ROOT_DIR/${main_cluster}/RH/${bam_file}"
        if [[ -f "$bam_path" ]]; then
            sorted_bam="$cluster_dir/$(basename "$bam_file" .bam).sorted.bam"
            if [[ ! -f "$sorted_bam" ]]; then
                samtools sort -o "$sorted_bam" "$bam_path"
            fi
            sorted_bam_list+=("$sorted_bam")
        fi
    fi
done < <(tail -n +2 "$CSV_FILE")

# Merge and index
if [[ ${#sorted_bam_list[@]} -gt 0 ]]; then
    echo "[RH] Merging ${#sorted_bam_list[@]} files -> $merged_bam"
    samtools merge -f "$merged_bam" "${sorted_bam_list[@]}"
    samtools index "$merged_bam"
    for sorted_bam in "${sorted_bam_list[@]}"; do
        rm "$sorted_bam"
    done
else
    echo "[RH] No BAM files found for $cluster. Skipping."
fi

echo "Finished: $cluster"
