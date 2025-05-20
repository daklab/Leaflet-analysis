# This is a bash script
MODEL_OUTPUTS_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel/2025-05-13
cd $MODEL_OUTPUTS_DIR
conda activate LeafletSC

# Get the number of rows in params.txt (-1 because of header)
num_rows=$(wc -l < $MODEL_OUTPUTS_DIR/parameter_combinations.csv)
num_rows=$((num_rows - 1))
echo "Number of models trained: $num_rows"

# Submit each script for each param_id
for ((i=0; i<=num_rows-1; i++)); do
    echo "Submitting script for param_id: $i"
    
    script1=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/downstream_analysis/01_evaluate_leafletFA_results.py 
    script2=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/downstream_analysis/02_leafletFA_regressions.py
    script3=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/downstream_analysis/03_differential_splicing_analysis.py
    script4=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/downstream_analysis/04_sanity_checks.py

    sbatch --mem=350G -p dev,cpu,bigmem -J "evaluate_leafletFA_results_$i" --wrap="python $script1 $i $MODEL_OUTPUTS_DIR"
    sbatch --mem=350G -p dev,cpu,bigmem -J "leafletFA_regressions_$i" --wrap="python $script2 $i $MODEL_OUTPUTS_DIR"
    #sbatch --mem=350G -p dev,cpu,bigmem -J "differential_splicing_analysis_$i" --wrap="python $script3 $i $MODEL_OUTPUTS_DIR"
    #sbatch --mem=350G -p dev,cpu,bigmem -J "sanity_checks_$i" --wrap="python $script4 $i $MODEL_OUTPUTS_DIR"

done



