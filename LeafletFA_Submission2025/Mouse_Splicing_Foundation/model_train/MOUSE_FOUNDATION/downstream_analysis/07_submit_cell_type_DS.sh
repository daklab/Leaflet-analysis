#!/usr/bin/env bash
# --- paths ------------------------------------------------------------------
MODEL_TRAIN_DATE="2025-05-13"
BASE_DIR="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"

MODEL_OUTPUTS_DIR="${BASE_DIR}/Leaflet/leafletFAmodel/${MODEL_TRAIN_DATE}"
ATSE_ANNDATA_PATH="${BASE_DIR}/MODEL_INPUT/052025/MOUSE_SPLICING_FOUNDATION_Anndata_ATSE_counts_with_waypoints_20250513_073829.h5ad"
AGING_GENES_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/27857814"
RBP_FILE_PATH="/gpfs/commons/groups/knowles_lab/Karin/VanNostrand_2020_supptable1_41586_2020_2077_MOESM3_ESM.xlsx"

RESULTS_BASE_DIR="/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/results/${MODEL_TRAIN_DATE}"
TODAY=$(date +%Y-%m-%d)

script_py="/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/downstream_analysis/05_differential_splicing_analysis_cell_types.py"

cd "$MODEL_OUTPUTS_DIR/slurm"
conda activate LeafletSC

# --- analysis settings ------------------------------------------------------
PARAM_ID=1
N_CT=41

echo "Submitting param_id=${PARAM_ID} for ${N_CT} cell types"

# --- submit SLURM array -----------------------------------------------------
sbatch --array=0-$((N_CT-1)) \
       --mem=180G --cpus-per-task=16 -p cpu,bigmem --time=3-00:00:00 \
       -J ds_p${PARAM_ID}_ct%a \
       --wrap="\
OUTDIR=${RESULTS_BASE_DIR}/${TODAY}/param_${PARAM_ID}/ct_\${SLURM_ARRAY_TASK_ID}; \
mkdir -p \$OUTDIR; \
python ${script_py} \
       ${PARAM_ID} ${MODEL_OUTPUTS_DIR} \
       ${ATSE_ANNDATA_PATH} \
       ${AGING_GENES_PATH} ${RBP_FILE_PATH} \
       \$OUTDIR \${SLURM_ARRAY_TASK_ID}"
