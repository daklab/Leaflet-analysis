#!/bin/bash
#SBATCH -J RH_pseudobulk
#SBATCH --mem=64G
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH --array=1-23498

#conda activate python3ENV 
module load samtools

# Input variables
CSV_FILE="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/RH_cells.csv"   
ROOT_DIR="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/" 
OUTPUT_DIR="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells" 

# make subdir for RH with date 
RH_DIR="$OUTPUT_DIR/RH"

mkdir -p "$RH_DIR"
echo "Output directory: $RH_DIR"

# Make OUTPUT_DIR if it does not exist
mkdir -p "$RH_DIR"

tail -n +2 "$CSV_FILE" | cut -d',' -f5 | sort | uniq > $RH_DIR/RH_cluster_list.txt

# Get the cluster name corresponding to this task ID
cluster=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $RH_DIR/RH_cluster_list.txt)
echo "Processing cluster: $cluster"

# If "/" is found within the cluster name, replace it with "_"
cluster=$(echo "$cluster" | tr '/' '_')

# Create output directory for the cluster
cluster_dir="$RH_DIR/$cluster"
mkdir -p "$cluster_dir"

# Collect and process BAM files for the cluster
sorted_bam_list=()
while IFS=',' read -r sample type primer main_cluster main_cluster_wkmeans bam_file; do
    if [[ "$main_cluster_wkmeans" == "$cluster" ]]; then
        bam_path="$ROOT_DIR/${main_cluster}/RH/${bam_file}"
        if [[ -f "$bam_path" ]]; then
            # Sort the BAM file
            sorted_bam="$cluster_dir/$(basename "$bam_file" .bam).sorted.bam"
            if [[ ! -f "$sorted_bam" ]]; then
                echo "Sorting BAM file: $bam_path -> $sorted_bam"
                samtools sort -o "$sorted_bam" "$bam_path"
            fi
            sorted_bam_list+=("$sorted_bam")
        else
            echo "Warning: BAM file not found: $bam_path"
        fi
    fi
done < <(tail -n +2 "$CSV_FILE")

# Merge and index sorted BAM files
merged_bam="$cluster_dir/${cluster}.bam"
if [[ ${#sorted_bam_list[@]} -gt 0 ]]; then
    echo "Merging ${#sorted_bam_list[@]} sorted BAM files into $merged_bam"
    samtools merge "$merged_bam" "${sorted_bam_list[@]}"
    echo "Indexing merged BAM file: $merged_bam"
    samtools index "$merged_bam"

    # Ensure index was created
    if [[ -f "$merged_bam.bai" ]]; then
        echo "Index file $merged_bam.bai created successfully."
    else
        echo "Error: Index file $merged_bam.bai not found."
    fi

    # Clean up intermediate sorted BAM files
    echo "Cleaning up sorted BAM files..."
    for sorted_bam in "${sorted_bam_list[@]}"; do
        rm "$sorted_bam"
    done
else
    echo "No BAM files found for cluster $cluster. Skipping."
fi

echo "Finished processing cluster: $cluster"