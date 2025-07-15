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
GE_INPUT = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/aligned_gene_expression_data_20250707_181219.h5ad"

# Define NMF model parameters
STANDARD_LATENT = 50
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

    """Train standard MiniBatchNMF model on HVGs using sklearn's MiniBatchNMF"""
    print("\n>> Training Standard MiniBatchNMF model...")

    layer_name = "predicted_log_norm_tms"
    if layer_name not in ge_adata.layers:
        print(f"  Error: Layer '{layer_name}' not found. Cannot proceed with NMF.")
        sys.exit(1)

    if "highly_variable" not in ge_adata.var.columns:
        print("  Error: 'highly_variable' column not found in .var. Run HVG selection first.")
        sys.exit(1)

    # Subset to HVGs
    hvg_mask = ge_adata.var["highly_variable"].values
    adata_hvg = ge_adata[:, hvg_mask]
    _data_matrix_original = adata_hvg.layers[layer_name]
    ge_adata.uns["hvg_mask_used_for_nmf"] = hvg_mask
    ge_adata.uns["nmf_standard_mb_params"] = {
        "n_components": STANDARD_LATENT,
        "max_iter": STANDARD_EPOCHS,
        "batch_size": NMF_BATCH_SIZE,
        "init": "nndsvda",
        "random_state": 0
    }
    
    print(f"  Using {hvg_mask.sum()} HVGs from layer '{layer_name}' for MiniBatchNMF.")

    # Handle negative values
    negative_values_found = False
    if issparse(_data_matrix_original):
        if _data_matrix_original.min() < 0:
            negative_values_found = True
            print("  Found negative values in sparse input. Clipping to zero.")
            processed_data_matrix = _data_matrix_original.maximum(0)
        else:
            processed_data_matrix = _data_matrix_original
    elif isinstance(_data_matrix_original, np.ndarray):
        if np.any(_data_matrix_original < 0):
            negative_values_found = True
            print("  Found negative values in dense input. Clipping to zero.")
            processed_data_matrix = np.clip(_data_matrix_original, 0, None)
        else:
            processed_data_matrix = _data_matrix_original
    else:
        print(f"  Unexpected data type for matrix: {type(_data_matrix_original)}")
        processed_data_matrix = _data_matrix_original

    # Check for all-zero input
    if negative_values_found:
        is_all_zero = (processed_data_matrix.nnz == 0) if issparse(processed_data_matrix) else np.all(processed_data_matrix == 0)
        if is_all_zero:
            print("  Error: All-zero matrix after clipping. Cannot run NMF.")
            sys.exit(1)

    data_matrix = processed_data_matrix

    # Train Standard MiniBatchNMF
    print(f"   Training with {STANDARD_LATENT} components, {STANDARD_EPOCHS} max_iter, batch size {NMF_BATCH_SIZE}...")
    model_standard = MiniBatchNMF(
        n_components=STANDARD_LATENT,
        max_iter=STANDARD_EPOCHS,
        batch_size=NMF_BATCH_SIZE,
        random_state=0,
        init='nndsvda'
    )

    W_standard = model_standard.fit_transform(data_matrix)
    H_standard = model_standard.components_

    # Store results
    ge_adata.obsm['X_nmf_standard_mb'] = W_standard

    H_standard_full = np.full((ge_adata.n_vars, STANDARD_LATENT), np.nan)
    H_standard_full[hvg_mask, :] = H_standard.T
    ge_adata.varm["nmf_standard_mb_components"] = H_standard_full

    print(f"   MiniBatchNMF training complete.")
    print(f"   Stored W → ge_adata.obsm['X_nmf_standard_mb']: {W_standard.shape}")
    print(f"   Stored H → ge_adata.varm['nmf_standard_mb_components']: {H_standard_full.shape}")

    return ge_adata


def save_results(ge_adata):
    """Save updated AnnData with NMF results"""
    print("\n>> Saving results...")
    
    try:
        # Define output filename
        today = datetime.datetime.now().strftime("%Y-%m-%d")        
        output_file = os.path.join(
            OUTPUT_DIR,
            f"ge_adata_with_NMF_standard_{STANDARD_LATENT}_{NMF_BATCH_SIZE}_{today}.h5ad"
            )
        
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
check_data_quality(ge_adata, "predicted_log_norm_tms") 

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

