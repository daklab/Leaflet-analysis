#!/bin/bash
# split_junctions.sh
#SBATCH --job-name=junction_split
#SBATCH --output=logs/junction_split_%j.out
#SBATCH --error=logs/junction_split_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Activate conda environment
conda activate LeafletSC

# Configuration
TODAY=$(date +%Y%m%d)
echo "Starting junction processing on: $(date)"
echo "Date suffix: $TODAY"

# Path to the script
SCRIPT_PATH=/gpfs/commons/home/kisaev/Leaflet-analysis/General_Utils/split_process_merge_slurm_junctions.py

# Path to the junction files
JUNCTION_FILES_AB=/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/junctions # allen brain nuclei

# Tabula Muris Senis - already contains all junction paths
# Made via /gpfs/commons/home/kisaev/Leaflet-analysis/TabulaSenis/00_collect_junction_files_to_file.py
TMS_SOURCE=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/ATSEmap/output/junction_files.txt

# Working directory
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper

cd $WD
echo "Working directory: $(pwd)"

# ============================================================================
# STEP 1: DISCOVER ALL FILES (UNFILTERED)
# ============================================================================
echo ""
echo "=== STEP 1: DISCOVERING ALL FILES ==="

# Find Allen Brain junction files
echo "Finding Allen Brain junction files..."
find $JUNCTION_FILES_AB -name "*_junctions_with_barcodes.bed" > $WD/junction_files_AB_${TODAY}_all.txt

# Copy Tabula Muris Senis junction file list (already contains all junction paths)
echo "Using pre-collected Tabula Muris Senis junction file list..."
if [[ ! -f "$TMS_SOURCE" ]]; then
    echo "ERROR: TMS junction file list not found: $TMS_SOURCE"
    exit 1
fi
cp $TMS_SOURCE $WD/junction_files_TMS_${TODAY}_all.txt

# Count discovered files
AB_ALL_COUNT=$(wc -l < $WD/junction_files_AB_${TODAY}_all.txt)
TMS_ALL_COUNT=$(wc -l < $WD/junction_files_TMS_${TODAY}_all.txt)

echo "Discovered junction files:"
echo "  Allen Brain: $AB_ALL_COUNT files"
echo "  Tabula Muris Senis: $TMS_ALL_COUNT files (pre-collected)"

# Find all splicing summary files for Allen Brain
echo ""
echo "Finding splicing summary files..."
find $JUNCTION_FILES_AB -name "*_splicing_summary.tsv" > $WD/splicing_summary_files_AB_${TODAY}_all.txt

AB_SPLICE_ALL_COUNT=$(wc -l < $WD/splicing_summary_files_AB_${TODAY}_all.txt)

echo "Discovered splicing summary files:"
echo "  Allen Brain: $AB_SPLICE_ALL_COUNT files"

# ============================================================================
# STEP 2: CREATE FILTERED SPLICING SUMMARY FILE LISTS
# ============================================================================
echo ""
echo "=== STEP 2: FILTERING SPLICING SUMMARY FILES ==="

# Function to extract cell ID from junction file path and find corresponding splicing summary
function create_filtered_splicing_list() {
    local junction_list=$1
    local output_list=$2
    local dataset_name=$3
    
    echo "Creating filtered splicing summary list for $dataset_name..."
    
    # Clear output file
    > $output_list
    
    local found_count=0
    local missing_count=0
    
    while IFS= read -r junction_path; do
        # Extract cell ID from junction file path
        local junction_basename=$(basename "$junction_path")
        local cell_id=${junction_basename%_junctions_with_barcodes.bed}
        
        # Construct expected splicing summary path
        local junction_dir=$(dirname "$junction_path")
        local splicing_path="${junction_dir}/${cell_id}_splicing_summary.tsv"
        
        if [[ -f "$splicing_path" ]]; then
            echo "$splicing_path" >> $output_list
            ((found_count++))
        else
            ((missing_count++))
            if [[ $missing_count -le 3 ]]; then
                echo "  WARNING: Missing splicing file: $splicing_path"
            fi
        fi
    done < "$junction_list"
    
    echo "  Found: $found_count splicing summary files"
    echo "  Missing: $missing_count splicing summary files"
    
    if [[ $missing_count -gt 3 ]]; then
        echo "  ... and $(($missing_count - 3)) more missing files"
    fi
}

# Create filtered splicing summary file lists for Allen Brain
clean_AB_splicing=$WD/splicing_summary_files_AB_${TODAY}_subset.txt
create_filtered_splicing_list "$WD/junction_files_AB_${TODAY}_all.txt" "$clean_AB_splicing" "Allen Brain"

# Create filtered splicing summary file lists for Tabula Muris Senis
clean_TMS_splicing=$WD/splicing_summary_files_TMS_${TODAY}_subset.txt
create_filtered_splicing_list "$WD/junction_files_TMS_${TODAY}_all.txt" "$clean_TMS_splicing" "Tabula Muris Senis"

# ============================================================================
# STEP 3: COMPILE ALL SPLICING SUMMARIES INTO ONE FILE
# ============================================================================
echo ""
echo "=== STEP 3: COMPILING SPLICING SUMMARIES ==="

# Combined splicing summary output
COMBINED_SPLICING_SUMMARY=$WD/combined_splicing_summary_${TODAY}.tsv

echo "Compiling all splicing summaries into: $COMBINED_SPLICING_SUMMARY"

# Function to compile splicing summaries
function compile_splicing_summaries() {
    local splicing_list=$1
    local dataset_name=$2
    local is_first=$3
    
    echo "Processing $dataset_name splicing summaries..."
    
    local file_count=0
    local processed_count=0
    local error_count=0
    
    while IFS= read -r splicing_path; do
        ((file_count++))
        
        if [[ ! -f "$splicing_path" ]]; then
            echo "  WARNING: File not found: $splicing_path"
            ((error_count++))
            continue
        fi
        
        # Extract cell ID from file path
        local basename=$(basename "$splicing_path")
        local cell_id=${basename%_splicing_summary.tsv}
        
        # Read the splicing summary file and add cell_id and dataset columns
        if [[ $is_first == "true" && $processed_count -eq 0 ]]; then
            # Add header for first file
            echo -e "cell_id\tdataset\t$(head -n1 "$splicing_path")" >> $COMBINED_SPLICING_SUMMARY
        fi
        
        # Add data rows (skip header)
        tail -n +2 "$splicing_path" | while IFS= read -r line; do
            echo -e "${cell_id}\t${dataset_name}\t${line}" >> $COMBINED_SPLICING_SUMMARY
        done
        
        ((processed_count++))
        
        # Progress indicator
        if [[ $((processed_count % 1000)) -eq 0 ]]; then
            echo "    Processed $processed_count/$file_count files..."
        fi
        
    done < "$splicing_list"
    
    echo "  $dataset_name summary: $processed_count files processed, $error_count errors"
}

# Clear the combined file
> $COMBINED_SPLICING_SUMMARY

# Process Allen Brain first (to create header)
if [[ -f "$clean_AB_splicing" && -s "$clean_AB_splicing" ]]; then
    compile_splicing_summaries "$clean_AB_splicing" "allen_brain" "true"
else
    echo "WARNING: No Allen Brain splicing files to process"
fi

# Process Tabula Muris Senis (append to existing file)
if [[ -f "$clean_TMS_splicing" && -s "$clean_TMS_splicing" ]]; then
    compile_splicing_summaries "$clean_TMS_splicing" "tabula_muris_senis" "false"
else
    echo "WARNING: No Tabula Muris Senis splicing files to process"
fi

# Check final compiled file
if [[ -f "$COMBINED_SPLICING_SUMMARY" ]]; then
    TOTAL_ROWS=$(wc -l < $COMBINED_SPLICING_SUMMARY)
    FILE_SIZE=$(du -h "$COMBINED_SPLICING_SUMMARY" | cut -f1)
    echo ""
    echo "Combined splicing summary created successfully:"
    echo "  File: $COMBINED_SPLICING_SUMMARY"
    echo "  Total rows: $TOTAL_ROWS (including header)"
    echo "  File size: $FILE_SIZE"
    
    # Show first few lines as verification
    echo ""
    echo "First 5 lines of combined file:"
    head -n 5 "$COMBINED_SPLICING_SUMMARY"
else
    echo "ERROR: Failed to create combined splicing summary file"
fi

# ============================================================================
# STEP 4: CREATE COMBINED JUNCTION FILE LIST
# ============================================================================
echo ""
echo "=== STEP 4: CREATING COMBINED JUNCTION LIST ==="

# Junction file to save everything to
junction_files_list=$WD/junction_files_combined_${TODAY}.txt

echo "Creating combined junction file list: $junction_files_list"

# Combine junction files
cat $WD/junction_files_AB_${TODAY}_all.txt $WD/junction_files_TMS_${TODAY}_all.txt > $junction_files_list

# Verify combined file
TOTAL_JUNCTION_COUNT=$(wc -l < $junction_files_list)
echo "Combined junction files: $TOTAL_JUNCTION_COUNT total"
echo "  Allen Brain: $AB_ALL_COUNT"
echo "  Tabula Muris Senis: $TMS_ALL_COUNT"
echo "  Sum check: $((AB_ALL_COUNT + TMS_ALL_COUNT)) (should match total)"

# Verify no duplicates
UNIQUE_JUNCTION_COUNT=$(sort $junction_files_list | uniq | wc -l)
if [[ $TOTAL_JUNCTION_COUNT -eq $UNIQUE_JUNCTION_COUNT ]]; then
    echo "✓ No duplicate junction files found"
else
    DUPLICATE_COUNT=$((TOTAL_JUNCTION_COUNT - UNIQUE_JUNCTION_COUNT))
    echo "WARNING: $DUPLICATE_COUNT duplicate junction files found"
fi

INPUT_FILE=$junction_files_list

# ============================================================================
# STEP 5: PREPARE FOR JUNCTION PROCESSING
# ============================================================================
echo ""
echo "=== STEP 5: PREPARING FOR JUNCTION PROCESSING ==="

# Create base directory with today's date
BASE_DIR="junction_processing_${TODAY}"
echo "Creating processing directory: $BASE_DIR"

mkdir -p $BASE_DIR/{logs,chunks,results}
cd $BASE_DIR

echo "Changed to directory: $(pwd)"

# Ensure chunks directory exists
mkdir -p chunks

# Verify script exists
if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "ERROR: Processing script not found: $SCRIPT_PATH"
    exit 1
fi

echo "Using processing script: $SCRIPT_PATH"

# ============================================================================
# STEP 6: RUN THE SPLIT JOB
# ============================================================================
echo ""
echo "=== STEP 6: RUNNING JUNCTION SPLIT ==="

# Run the split job
echo "Running split command with 100 chunks..."
python $SCRIPT_PATH \
    --mode split \
    --input-file $INPUT_FILE \
    --chunks 100 \
    --output-dir chunks

# Verify chunks were created
CHUNK_COUNT=$(ls chunks/ 2>/dev/null | wc -l)
echo ""
echo "Split job completed:"
echo "  Number of chunks created: $CHUNK_COUNT"
echo "  Expected chunks: 100"

if [[ $CHUNK_COUNT -eq 100 ]]; then
    echo "✓ Split job successful - all chunks created"
else
    echo "WARNING: Expected 100 chunks, but found $CHUNK_COUNT"
fi

# Show chunk sizes for verification
if [[ $CHUNK_COUNT -gt 0 ]]; then
    echo ""
    echo "Sample chunk sizes:"
    for chunk in $(ls chunks/ | head -3); do
        chunk_size=$(wc -l < chunks/$chunk)
        echo "  $chunk: $chunk_size files"
    done
    
    if [[ $CHUNK_COUNT -gt 3 ]]; then
        echo "  ... and $((CHUNK_COUNT - 3)) more chunks"
    fi
fi

# ============================================================================
# STEP 7: SUMMARY AND NEXT STEPS
# ============================================================================
echo ""
echo "=== PROCESSING SUMMARY ==="
echo "Date: $(date)"
echo "Working directory: $WD"
echo "Processing directory: $BASE_DIR"
echo ""
echo "Files created:"
echo "  Combined junction list: $INPUT_FILE ($TOTAL_JUNCTION_COUNT files)"
echo "  Combined splicing summary: $COMBINED_SPLICING_SUMMARY"
echo "  Filtered AB splicing list: $clean_AB_splicing"
echo "  Filtered TMS splicing list: $clean_TMS_splicing"
echo "  Processing chunks: $CHUNK_COUNT files in chunks/"
echo ""
echo "Next steps:"
echo "1. Submit chunk processing jobs using the created chunks"
echo "2. Merge results after all chunks complete"
echo "3. Use combined splicing summary for downstream analysis"
echo ""
echo "Junction split preparation complete!"