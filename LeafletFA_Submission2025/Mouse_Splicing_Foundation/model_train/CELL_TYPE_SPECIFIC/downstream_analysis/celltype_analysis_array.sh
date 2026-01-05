#!/bin/bash
#SBATCH --job-name=celltype_analysis
#SBATCH --array=0-20                    # 21 cell types (0-indexed)
#SBATCH --mem=200G                      # 200GB memory per job
#SBATCH --cpus-per-task=4              # Adjust if you need more CPUs
#SBATCH --time=24:00:00                # Max runtime (adjust as needed)
#SBATCH --output=logs/celltype_%A_%a.out
#SBATCH --error=logs/celltype_%A_%a.err
#SBATCH --partition=cpu,bigmem,dev                

# Create logs directory if it doesn't exist
mkdir -p logs

# Path to your analysis script
SCRIPT_PATH="/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/CELL_TYPE_SPECIFIC/downstream_analysis/evaluate_models.py"

# Cell types array (same order as your original script)
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

# Get the cell type for this array task
CELL_TYPE="${CELL_TYPES[$SLURM_ARRAY_TASK_ID]}"

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing cell type: $CELL_TYPE"
echo "Started at: $(date)"
echo "Running on node: $(hostname)"
echo "========================================"

# Run the analysis for this specific cell type
python "$SCRIPT_PATH" "$CELL_TYPE"

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ SUCCESS: $CELL_TYPE completed at $(date)"
else
    echo "❌ FAILED: $CELL_TYPE failed at $(date)"
    exit 1
fi