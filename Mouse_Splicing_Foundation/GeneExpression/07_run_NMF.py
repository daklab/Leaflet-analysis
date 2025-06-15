#!/usr/bin/env python
"""
NMF Model Training - Mouse Splicing Foundation

This script:
1. Loads gene expression data aligned with splicing data
2. Applies mini batch NMF dimensionality reduction
3. Saves the updated AnnData object for downstream clustering and visualization
"""

import os
import sys
import pandas as pd
import numpy as np
import anndata as ad
import datetime
import traceback
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.sparse import csr_matrix, issparse
from sklearn.decomposition import MiniBatchNMF

# Configuration
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/NMF"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}", flush=True)

# Input file path
GE_INPUT = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/062025/aligned_gene_expression_data_20250614_124502.h5ad"

# Define NMF model parameters
LINEAR_LATENT = 30
STANDARD_LATENT = 30
LINEAR_EPOCHS = 200
STANDARD_EPOCHS = 200
NMF_BATCH_SIZE = 1024 # Batch size for MiniBatchNMF

def load_data():
    """Load aligned gene expression data"""
    print("\n>> Loading gene expression data...")
    print(f"   ⚙️ Reading gene expression AnnData from {GE_INPUT}")
    ge_adata = ad.read_h5ad(GE_INPUT)
    print(f"   ✓ Loaded data with {ge_adata.n_obs} cells and {ge_adata.n_vars} genes")
    
    return ge_adata
    

def check_data_quality(adata, layer_name):
    """Check for NaN or Inf values in the specified layer of AnnData object."""
    print(f"\n>> Checking data quality for layer: {layer_name}...")
    if layer_name not in adata.layers:
        print(f"   Error: Layer '{layer_name}' not found in AnnData object.")
        sys.exit(1)

    data_matrix = adata.layers[layer_name]

    if isinstance(data_matrix, csr_matrix):
        data_matrix = data_matrix.toarray()

    nan_count = np.isnan(data_matrix).sum()
    inf_count = np.isinf(data_matrix).sum()

    if nan_count > 0:
        print(f"   Warning: Found {nan_count} NaN values in layer '{layer_name}'.")
    else:
        print(f"   No NaN values found in layer '{layer_name}'.")

    if inf_count > 0:
        print(f"   Warning: Found {inf_count} infinite values in layer '{layer_name}'.")
    else:
        print(f"   No infinite values found in layer '{layer_name}'.")

    if nan_count > 0 or inf_count > 0:
        print(f"   Consider handling these values (e.g., imputation, removal) before proceeding.")
    else:
        print(f"   Data quality check passed for layer '{layer_name}'.")


def train_mini_batch_NMF(ge_adata):
    """Train mini batch NMF model for gene expression data using sklearn's MiniBatchNMF"""
    print("\n>> Training MiniBatchNMF model...")
    
    layer_name = "predicted_log_norm_tms"
    if layer_name not in ge_adata.layers:
        print(f"  Error: Layer '{layer_name}' not found. Cannot proceed with NMF.")
        sys.exit(1)

    print(f"  Using layer '{layer_name}' for MiniBatchNMF.")
    _data_matrix_original = ge_adata.layers[layer_name]
    processed_data_matrix = None
    negative_values_found = False

    if issparse(_data_matrix_original):
        if _data_matrix_original.min() < 0:
            negative_values_found = True
            print("  Found negative values in the sparse input data layer. Using a zero-clipped version for NMF.")
            processed_data_matrix = _data_matrix_original.maximum(0)
        else:
            processed_data_matrix = _data_matrix_original 
    elif isinstance(_data_matrix_original, np.ndarray):
        if np.any(_data_matrix_original < 0):
            negative_values_found = True
            print("  Found negative values in the dense input data layer. Using a zero-clipped version for NMF.")
            processed_data_matrix = np.clip(_data_matrix_original, 0, None)
        else:
            processed_data_matrix = _data_matrix_original
    else:
        print(f"  Data matrix in layer '{layer_name}' is of an unexpected type: {type(_data_matrix_original)}. NMF might fail if it contains negative values or is incompatible.")
        processed_data_matrix = _data_matrix_original

    if negative_values_found:
        is_all_zero = False
        if issparse(processed_data_matrix):
            if processed_data_matrix.nnz == 0:
                is_all_zero = True
        elif isinstance(processed_data_matrix, np.ndarray):
            if np.all(processed_data_matrix == 0):
                is_all_zero = True
        
        if is_all_zero:
            print(f"   Error: Data matrix for layer '{layer_name}' became all zeros after clipping negative values. Cannot proceed with NMF.")
            sys.exit(1)
    
    data_matrix = processed_data_matrix # Use this variable for NMF

    # Linear MiniBatchNMF
    print(f"   Training Linear MiniBatchNMF with {LINEAR_LATENT} components, {LINEAR_EPOCHS} max_iter, batch_size {NMF_BATCH_SIZE}...")
    model_linear = MiniBatchNMF(
        n_components=LINEAR_LATENT,
        max_iter=LINEAR_EPOCHS,
        batch_size=NMF_BATCH_SIZE,
        random_state=0,
        init='nndsvda' # A good initialization for NMF
    )
    # Fit and transform
    # For MiniBatchNMF, fit_transform is done in batches.
    # We will iterate manually to show progress, though fit() can also be used.
    # However, for simplicity and directness with AnnData, let's use fit_transform directly on the whole data
    # if memory allows, or adapt to partial_fit if truly out-of-core for very large data is needed.
    # Assuming ge_adata.layers[layer_name] fits in memory for this step with sklearn.
    
    W_linear = model_linear.fit_transform(data_matrix)
    H_linear = model_linear.components_
    
    ge_adata.obsm['X_nmf_linear_mb'] = W_linear
    ge_adata.varm['nmf_linear_mb_components'] = H_linear.T # Store as n_genes x n_components

    print(f"   Linear MiniBatchNMF training complete.")
    print(f"   Results stored in ge_adata.obsm['X_nmf_linear_mb'] ({W_linear.shape}) and ge_adata.varm['nmf_linear_mb_components'] ({H_linear.T.shape})")

    # Standard MiniBatchNMF
    print(f"   Training Standard MiniBatchNMF with {STANDARD_LATENT} components, {STANDARD_EPOCHS} max_iter, batch_size {NMF_BATCH_SIZE}...")
    model_standard = MiniBatchNMF(
        n_components=STANDARD_LATENT,
        max_iter=STANDARD_EPOCHS,
        batch_size=NMF_BATCH_SIZE,
        random_state=0,
        init='nndsvda'
    )
    W_standard = model_standard.fit_transform(data_matrix)
    H_standard = model_standard.components_

    ge_adata.obsm['X_nmf_standard_mb'] = W_standard
    ge_adata.varm['nmf_standard_mb_components'] = H_standard.T # Store as n_genes x n_components

    print(f"   Standard MiniBatchNMF training complete.")
    print(f"   Results stored in ge_adata.obsm['X_nmf_standard_mb'] ({W_standard.shape}) and ge_adata.varm['nmf_standard_mb_components'] ({H_standard.T.shape})")

    return ge_adata

def save_results(ge_adata):
    """Save updated AnnData with NMF results"""
    print("\n>> Saving results...")
    
    try:
        # Define output filename
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        output_file = os.path.join(OUTPUT_DIR, f"ge_adata_with_NMF_model_{LINEAR_LATENT}_{NMF_BATCH_SIZE}_{today}.h5ad")
        
        # Save file
        print(f" Saving updated AnnData to {output_file}...")
        ge_adata.write_h5ad(output_file, compression="lzf")
        
        print(f" Results successfully saved to {output_file}")
        
        return True
        
    except Exception as e:
        print(f"   Error saving results: {str(e)}")
        traceback.print_exc()
        return False

# Main execution
print("\n========================================")
print("NMF Model Training - Mouse Splicing Foundation")
print("mini batch NMF model training...")
print("========================================\n")

# Load data
ge_adata = load_data()
check_data_quality(ge_adata, "predicted_log_norm_tms") # Using layer with predicted values for Allen Brain nuclei using TMS regression model

# Train mini batch NMF model
ge_adata = train_mini_batch_NMF(ge_adata)

# Save results to new AnnData object
save_results(ge_adata)

print("\n========================================")
print("NMF model training complete!")
print("Low-dimensional representations saved for downstream analysis")
print(f"Results saved to: {OUTPUT_DIR}")
print("========================================\n")

# conda activate LeafletSC
# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/NMF
# sbatch --mem=300G -p cpu,dev,bigmem -J "NMF_GE" --wrap "python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/GeneExpression/07_run_NMF.py"

