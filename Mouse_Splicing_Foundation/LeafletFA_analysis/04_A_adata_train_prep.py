#!/usr/bin/env python
"""
Simple script to extract splicing AnnData objects from MuData files
"""

import os
import mudata as mu
from datetime import datetime

# File paths
TRAIN_ADATA_PATH = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/train_70_30_model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5mu"
TEST_ADATA_PATH = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5mu"
OUTPUT_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025"

def extract_splicing_data():
    """Extract splicing modality from MuData files and save as AnnData objects"""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate timestamp for output files
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("Extracting splicing data from MuData files...")
    
    # Process train dataset
    print(f"\nProcessing train dataset...")
    train_mdata = mu.read_h5mu(TRAIN_ADATA_PATH)
    train_splicing = train_mdata["splicing"]
    
    # Print non-zero counts for each layer
    print(f"  Layers and non-zero counts:")
    for layer_name, layer_data in train_splicing.layers.items():
        nnz = layer_data.nnz if hasattr(layer_data, 'nnz') else (layer_data != 0).sum()
        print(f"    {layer_name}: {nnz:,} non-zero elements")
    
    train_output_path = os.path.join(OUTPUT_DIR, f"splicing_adata_train_{stamp}.h5ad")
    train_splicing.write_h5ad(train_output_path, compression='lzf')
    print(f"✓ Train splicing data saved: {train_output_path}")
    print(f"  Shape: {train_splicing.shape[0]:,} cells × {train_splicing.shape[1]:,} junctions")
    
    # Process test dataset
    print(f"\nProcessing test dataset...")
    test_mdata = mu.read_h5mu(TEST_ADATA_PATH)
    test_splicing = test_mdata["splicing"]
    
    # Print non-zero counts for each layer
    print(f"  Layers and non-zero counts:")
    for layer_name, layer_data in test_splicing.layers.items():
        nnz = layer_data.nnz if hasattr(layer_data, 'nnz') else (layer_data != 0).sum()
        print(f"    {layer_name}: {nnz:,} non-zero elements")
    
    test_output_path = os.path.join(OUTPUT_DIR, f"splicing_adata_test_{stamp}.h5ad")
    test_splicing.write_h5ad(test_output_path, compression='lzf')
    print(f"✓ Test splicing data saved: {test_output_path}")
    print(f"  Shape: {test_splicing.shape[0]:,} cells × {test_splicing.shape[1]:,} junctions")
    
    print(f"\n🎉 Extraction complete!")
    return train_output_path, test_output_path

# =============================================================================
# Script Entry Point
# =============================================================================

if __name__ == "__main__":
    train_path, test_path = extract_splicing_data()

"""
#Submit with:
conda activate LeafletSC
script=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/LeafletFA_analysis/04_imputation_analysis.py
cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_OUTPUT/
sbatch --job-name=leaflet_eval \
       --partition=bigmem,cpu \
       --mem=800G \
       --time=1-00:00:00 \
       --output=splicing_extraction_%j.out \
       --error=splicing_extraction_%j.err \
       --wrap="python $script"
"""