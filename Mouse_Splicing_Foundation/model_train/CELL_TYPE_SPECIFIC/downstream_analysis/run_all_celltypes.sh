#!/bin/bash

# Simple bash script to run analysis on all cell types
# Usage: bash run_all_celltypes.sh

# Path to your analysis script
SCRIPT_PATH="/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/CELL_TYPE_SPECIFIC/downstream_analysis/evaluate_models.py"

# Cell types from your list
CELL_TYPES=(
    "BASAL_CELL"
    "B_CELL"
    "ENDOTHELIAL_CELL"
    "EPITHELIAL_CELL"
    "Excitatory_Neurons"
    "FIBROBLAST"
    "GLIAL_CELL"
    "GRANULOCYTE"
    "Inhibitory_Neurons"
    "INTESTINAL_CELL"
    "KERATINOCYTE"
    "MACROPHAGE"
    "MICROGLIA"
    "MONOCYTE"
    "MYELOID_IMMUNE_CELL"
    "PANCREATIC_CELL"
    "SKELETAL_MUSCLE_CELL"
    "SMOOTH_MUSCLE_CELL"
    "STEM_CELL"
    "T_CELL"
    "THYMOCYTE"
)

# Create logs directory
mkdir -p logs

echo "Starting analysis for ${#CELL_TYPES[@]} cell types..."
echo "Script: $SCRIPT_PATH"
echo "========================================"

# Track success/failure
SUCCESSFUL=()
FAILED=()

# Process each cell type
for i in "${!CELL_TYPES[@]}"; do
    CELL_TYPE="${CELL_TYPES[$i]}"
    echo ""
    echo "[$((i+1))/${#CELL_TYPES[@]}] Processing: $CELL_TYPE"
    echo "----------------------------------------"
    
    # Run the analysis
    if python "$SCRIPT_PATH" "$CELL_TYPE" > "logs/${CELL_TYPE}.log" 2>&1; then
        echo "✅ SUCCESS: $CELL_TYPE"
        SUCCESSFUL+=("$CELL_TYPE")
    else
        echo "❌ FAILED: $CELL_TYPE (check logs/${CELL_TYPE}.log)"
        FAILED+=("$CELL_TYPE")
    fi
done

echo ""
echo "========================================"
echo "BATCH ANALYSIS COMPLETE"
echo "========================================"
echo "Successful: ${#SUCCESSFUL[@]}"
echo "Failed: ${#FAILED[@]}"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed cell types:"
    for ct in "${FAILED[@]}"; do
        echo "  - $ct"
    done
fi

echo ""
echo "Check individual logs in the 'logs/' directory"