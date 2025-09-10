#!/bin/bash

# Set variables
script=/gpfs/commons/home/kisaev/Leaflet-analysis/Multi_Species_Splicing_Foundation/joint_model_train/full_workflow/04_analyze_leafletfa.py
model_output=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/CROSS_SPECIES_AGING/Leaflet/model_combo_train/2025-09-04
anndata_file=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/CROSS_SPECIES_AGING/Leaflet/input_files/joint_anndata_20250903.h5ad
cd $model_output

conda activate LeafletSC

# Loop through all run directories
for run_dir in run_*; do
    # Extract param_id from directory name (e.g., "run_18" -> "18")
    param_id=${run_dir#run_}
    
    # Check if the model file exists
    if [ -f "${run_dir}/leafletfa_model.pkl.gz" ]; then
        output_dir=${model_output}/analysis_${run_dir}
        
        echo "Submitting job for param_id=${param_id}"
        
        sbatch --job-name=leaf_${param_id} \
               --partition=bigmem,cpu,dev \
               --mem=300G \
               --time=1-00:00:00 \
               --output=logs/leaflet_post_${param_id}_%j.out \
               --error=logs/leaflet_post_${param_id}_%j.err \
               --wrap="python $script $param_id $model_output $anndata_file $output_dir"
        
        # Optional: add a small delay to avoid overwhelming the scheduler
        sleep 0.5
    else
        echo "Skipping ${run_dir} - no model file found"
    fi
done