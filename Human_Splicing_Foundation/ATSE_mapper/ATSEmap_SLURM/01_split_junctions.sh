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
SCRIPT_PATH=/gpfs/commons/home/kisaev/Leaflet-analysis/split_process_merge_slurm_junctions.py

# Path to the junction files
JUNCTION_FILES_AB=/gpfs/commons/datasets/controlled/BRAIN_NeMO/lein-human-cortex/Leaflet # allen brain nuclei
JUNCTION_FILES_TS=/gpfs/commons/projects/CZI-tabula-sapiens/LeafletFA # tabula sapiens single cell

# Working directory
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/ATSE_mapper

cd $WD
echo "Working directory: $(pwd)"

# ============================================================================
# STEP 1: DISCOVER ALL FILES (UNFILTERED)
# ============================================================================
echo ""
echo "=== STEP 1: DISCOVERING ALL FILES ==="

# Find all junction files
echo "Finding junction files..."
find $JUNCTION_FILES_AB -name "*_junctions_with_barcodes.bed" > $WD/junction_files_AB_${TODAY}_all.txt
find $JUNCTION_FILES_TS -name "*_junctions_with_barcodes.bed" > $WD/junction_files_TS_${TODAY}_all.txt

# Count discovered files
AB_ALL_COUNT=$(wc -l < $WD/junction_files_AB_${TODAY}_all.txt)
TS_ALL_COUNT=$(wc -l < $WD/junction_files_TS_${TODAY}_all.txt)

echo "Discovered junction files:"
echo "  Allen Brain: $AB_ALL_COUNT files"
echo "  Tabula Sapiens: $TS_ALL_COUNT files"

# Find all splicing summary files
echo ""
echo "Finding splicing summary files..."
find $JUNCTION_FILES_AB -name "*_splicing_summary.tsv" > $WD/splicing_summary_files_AB_${TODAY}_all.txt
find $JUNCTION_FILES_TS -name "*_splicing_summary.tsv" > $WD/splicing_summary_files_TS_${TODAY}_all.txt

# Count discovered splicing files
AB_SPLICE_ALL_COUNT=$(wc -l < $WD/splicing_summary_files_AB_${TODAY}_all.txt)
TS_SPLICE_ALL_COUNT=$(wc -l < $WD/splicing_summary_files_TS_${TODAY}_all.txt)

echo "Discovered splicing summary files:"
echo "  Allen Brain: $AB_SPLICE_ALL_COUNT files"
echo "  Tabula Sapiens: $TS_SPLICE_ALL_COUNT files"

# ============================================================================
# STEP 2: USE FILTERED JUNCTION FILES FROM METADATA PROCESSING
# ============================================================================
echo ""
echo "=== STEP 2: USING FILTERED JUNCTION FILES ==="

# Use the filtered junction files created by the metadata processing script
# These contain only cells that have metadata
# These files came from /gpfs/commons/home/kisaev/Leaflet-analysis/Human_Splicing_Foundation/metadata/02_TS_vs_AB_make_shared_metadata.ipynb
clean_TS_juncs=$WD/junction_files_TS_subset.txt
clean_AB_juncs=$WD/junction_files_AB_subset.txt

# Verify filtered files exist
if [[ ! -f "$clean_AB_juncs" ]]; then
    echo "ERROR: Filtered Allen Brain junction file not found: $clean_AB_juncs"
    echo "Please run the metadata processing script first!"
    exit 1
fi

if [[ ! -f "$clean_TS_juncs" ]]; then
    echo "ERROR: Filtered Tabula Sapiens junction file not found: $clean_TS_juncs"
    echo "Please run the metadata processing script first!"
    exit 1
fi

# Count filtered files
AB_CLEAN_COUNT=$(wc -l < $clean_AB_juncs)
TS_CLEAN_COUNT=$(wc -l < $clean_TS_juncs)

echo "Using filtered junction files:"
echo "  Allen Brain: $AB_CLEAN_COUNT files (from $clean_AB_juncs)"
echo "  Tabula Sapiens: $TS_CLEAN_COUNT files (from $clean_TS_juncs)"

# Show filtering efficiency
AB_RETENTION=$(echo "scale=1; $AB_CLEAN_COUNT * 100 / $AB_ALL_COUNT" | bc -l)
TS_RETENTION=$(echo "scale=1; $TS_CLEAN_COUNT * 100 / $TS_ALL_COUNT" | bc -l)

echo "Filtering efficiency:"
echo "  Allen Brain: $AB_RETENTION% retention ($AB_CLEAN_COUNT/$AB_ALL_COUNT)"
echo "  Tabula Sapiens: $TS_RETENTION% retention ($TS_CLEAN_COUNT/$TS_ALL_COUNT)"

# ============================================================================
# STEP 3: CREATE FILTERED SPLICING SUMMARY FILE LISTS
# ============================================================================
echo ""
echo "=== STEP 3: FILTERING SPLICING SUMMARY FILES ==="

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

# Create filtered splicing summary file lists
clean_AB_splicing=$WD/splicing_summary_files_AB_subset.txt
clean_TS_splicing=$WD/splicing_summary_files_TS_subset.txt

create_filtered_splicing_list "$clean_AB_juncs" "$clean_AB_splicing" "Allen Brain"
create_filtered_splicing_list "$clean_TS_juncs" "$clean_TS_splicing" "Tabula Sapiens"

# ============================================================================
# STEP 4: COMPILE ALL SPLICING SUMMARIES INTO ONE FILE
# ============================================================================
echo ""
echo "=== STEP 4: COMPILING SPLICING SUMMARIES ==="

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

# Process Tabula Sapiens (append to existing file)
if [[ -f "$clean_TS_splicing" && -s "$clean_TS_splicing" ]]; then
    compile_splicing_summaries "$clean_TS_splicing" "tabula_sapiens" "false"
else
    echo "WARNING: No Tabula Sapiens splicing files to process"
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
# STEP 5: CREATE COMBINED JUNCTION FILE LIST
# ============================================================================
echo ""
echo "=== STEP 5: CREATING COMBINED JUNCTION LIST ==="

# Junction file to save everything to
junction_files_list=$WD/junction_files_combined_${TODAY}.txt

echo "Creating combined junction file list: $junction_files_list"

# Combine filtered junction files
cat $clean_AB_juncs $clean_TS_juncs > $junction_files_list

# Verify combined file
TOTAL_JUNCTION_COUNT=$(wc -l < $junction_files_list)
echo "Combined junction files: $TOTAL_JUNCTION_COUNT total"
echo "  Allen Brain: $AB_CLEAN_COUNT"
echo "  Tabula Sapiens: $TS_CLEAN_COUNT"
echo "  Sum check: $((AB_CLEAN_COUNT + TS_CLEAN_COUNT)) (should match total)"

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
# STEP 6: PREPARE FOR JUNCTION PROCESSING
# ============================================================================
echo ""
echo "=== STEP 6: PREPARING FOR JUNCTION PROCESSING ==="

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
# STEP 7: RUN THE SPLIT JOB
# ============================================================================
echo ""
echo "=== STEP 7: RUNNING JUNCTION SPLIT ==="

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
# STEP 8: SUMMARY AND NEXT STEPS
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
echo "  Filtered TS splicing list: $clean_TS_splicing"
echo "  Processing chunks: $CHUNK_COUNT files in chunks/"
echo ""
echo "Next steps:"
echo "1. Submit chunk processing jobs using the created chunks"
echo "2. Merge results after all chunks complete"
echo "3. Use combined splicing summary for downstream analysis"
echo ""
echo "Junction split preparation complete!"