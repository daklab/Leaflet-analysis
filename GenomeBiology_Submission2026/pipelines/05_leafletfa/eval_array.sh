#!/bin/bash
#SBATCH --job-name=lfa_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=ne1dg7-[001-011]
#SBATCH --mem=150G
#SBATCH --time=6:00:00
#SBATCH --array=0-35%3
#SBATCH --output=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/2026-07-18/eval_logs/eval_%A_%a.out
# Runs 04e (imputation) + 07c (structure) on the minibatch model for task $SLURM_ARRAY_TASK_ID.
# Only the 36 CORE tasks (0-35: K x prior x seed x init) are meaningful; 36-47 are ablations the
# minibatch trainer ignores. Array is 0-35.
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python
P=/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa
MD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/2026-07-18
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION
TRAIN=$WD/MODEL_INPUT/20260526/train_splice_model_ready_70pct_2026-06-17.h5ad
TEST=$WD/LEAFLETFA_MODELS/SMOKE_04E2/resampled/RESAMPLED_10_PERCENT_test_30_70_model_ready_combined_ge_splicing_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5mu
t=$SLURM_ARRAY_TASK_ID
OUT=$MD/eval_grouped/task_$t
mkdir -p $OUT/rs
ln -sf $TEST $OUT/rs/ 2>/dev/null
# adapt the minibatch fit -> pkl.gz the evals read
LEAFLET_ALLOW_CPU=1 $PY $P/mb_to_evalmodel.py $MD/mb_run_$t $OUT/model || exit 1
# 04e imputation (E-step lr=0.1, 500 ep -> converges ~250) + 07c structure
$PY -u $P/04e_evaluate_imputation_gpu.py $OUT/model $TRAIN $OUT/rs $OUT/04e_out.csv 4096 500 20 3000 0 20 pi 2>&1
$PY -u $P/07c_heldout_structure.py $OUT/model $TRAIN $TEST $OUT/07c_out.csv 4096 500 20 2>&1
