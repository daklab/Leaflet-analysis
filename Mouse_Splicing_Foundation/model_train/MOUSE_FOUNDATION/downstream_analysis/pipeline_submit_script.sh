#!/bin/bash
MODEL_TRAIN_DATE="2025-05-13"
BASE_DIR="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
MODEL_OUTPUTS_DIR="${BASE_DIR}/Leaflet/leafletFAmodel/${MODEL_TRAIN_DATE}"
ATSE_ANNDATA_PATH="${BASE_DIR}/MODEL_INPUT/052025/MOUSE_SPLICING_FOUNDATION_Anndata_ATSE_counts_with_waypoints_20250513_073829.h5ad"
GE_ANNDATA_scVI_PATH="${BASE_DIR}/scVI/ge_adata_with_both_scvi_models_2025-05-13.h5ad"
GE_ANNDATA_NMF_PATH="${BASE_DIR}/NMF/ge_adata_with_NMF_models_2025-05-16.h5ad"
AGING_GENES_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/27857814"
RBP_FILE_PATH="/gpfs/commons/groups/knowles_lab/Karin/VanNostrand_2020_supptable1_41586_2020_2077_MOESM3_ESM.xlsx"
RESULTS_BASE_DIR="/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/results/${MODEL_TRAIN_DATE}"
ATSE_FILE_PATH="${BASE_DIR}/ATSE_mapper/ATSE_files/MOUSE_FOUNDATION_ATSE_FILE_unanno_also_2025-04-26_19-55-26.txt.gz"

cd "$MODEL_OUTPUTS_DIR/slurm"
conda activate LeafletSC

# Get the number of rows in params.txt (-1 because of header)
num_rows=$(wc -l < $MODEL_OUTPUTS_DIR/parameter_combinations.csv)
num_rows=$((num_rows - 1))
echo "Number of models trained: $num_rows"

# Get today's date
TODAY=$(date +%Y-%m-%d)

# Submit each script for each param_id
# for i in 0; do
for ((i=0; i<=num_rows-1; i++)); do
    echo "Submitting script for param_id: $i"
    OUTPUT_DIR="${RESULTS_BASE_DIR}/${TODAY}/param_id_${i}"
    echo "Output directory: $OUTPUT_DIR"
    script1=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/downstream_analysis/01_evaluate_leafletFA_results.py 
    script2=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/downstream_analysis/02_leafletFA_regressions.py
    script3=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/downstream_analysis/03_differential_splicing_analysis.py
    script4=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/downstream_analysis/04_sanity_checks.py

    #sbatch --mem=450G -p cpu,bigmem -J "evaluate_leafletFA_results_$i" --wrap="python $script1 $i $MODEL_OUTPUTS_DIR $ATSE_ANNDATA_PATH $GE_ANNDATA_scVI_PATH $AGING_GENES_PATH $RBP_FILE_PATH $OUTPUT_DIR"
    #sbatch --mem=350G -p cpu,bigmem -J "leafletFA_regressions_$i" --wrap="python $script2 $i $MODEL_OUTPUTS_DIR $ATSE_ANNDATA_PATH $GE_ANNDATA_scVI_PATH $GE_ANNDATA_NMF_PATH $AGING_GENES_PATH $RBP_FILE_PATH $OUTPUT_DIR"
    sbatch --mem=220G -p cpu,bigmem -J "differential_splicing_analysis_$i" --wrap="python $script3 $i $MODEL_OUTPUTS_DIR $ATSE_ANNDATA_PATH $ATSE_FILE_PATH $GE_ANNDATA_scVI_PATH $GE_ANNDATA_NMF_PATH $AGING_GENES_PATH $RBP_FILE_PATH $OUTPUT_DIR"
    #sbatch --mem=400G -p cpu,bigmem -J "sanity_checks_$i" --wrap="python $script4 $i $MODEL_OUTPUTS_DIR $ATSE_ANNDATA_PATH $OUTPUT_DIR"

done

# After all jobs are done, run the following script to find the best model
# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel
# python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/downstream_analysis/05_summarize_models.py
# python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/downstream_analysis/06_summarize_models.py

# Just for testing differentially splicing analysis on param_id 1
i=1
echo "Submitting script for param_id: $i"
OUTPUT_DIR="${RESULTS_BASE_DIR}/${TODAY}/param_id_${i}"
echo "Output directory: $OUTPUT_DIR"

script3=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/downstream_analysis/03_differential_splicing_analysis.py

sbatch --mem=220G -p cpu,bigmem --time=5-00:00:00 --cpus-per-task=16 -J "differential_splicing_analysis_$i" --wrap="python $script3 $i $MODEL_OUTPUTS_DIR $ATSE_ANNDATA_PATH $ATSE_FILE_PATH $GE_ANNDATA_scVI_PATH $GE_ANNDATA_NMF_PATH $AGING_GENES_PATH $RBP_FILE_PATH $OUTPUT_DIR"
