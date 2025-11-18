#!/usr/bin/env python
"""
LeafletFA Test Data Preparation Script
Prepares masked test data by fixing sparsity patterns and saving processed data.
"""

import os
import sys
import numpy as np
import torch
import mudata as mu
from scipy import sparse

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
torch.set_default_tensor_type("torch.cuda.FloatTensor" if torch.cuda.is_available() else "torch.FloatTensor")
torch.manual_seed(0)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def fix_sparsity(ad):
    """Fix sparsity pattern mismatch"""
    junc = ad.layers["cell_by_junction_matrix"].tocoo()
    clust = ad.layers["cell_by_cluster_matrix"].tocoo()
    missing = set(zip(clust.row, clust.col)) - set(zip(junc.row, junc.col))
    
    if missing:
        mr, mc = zip(*missing)
        all_r = np.concatenate([junc.row, mr])
        all_c = np.concatenate([junc.col, mc])
        all_v = np.concatenate([junc.data, np.zeros(len(missing))])
        ad.layers["cell_by_junction_matrix"] = sparse.csr_matrix(
            (all_v, (all_r, all_c)), shape=junc.shape, dtype=junc.dtype)
    return ad

def fix_sparsity_fast(ad):
    """Fix sparsity pattern mismatch using vectorized operations"""
    print("Fixing sparsity patterns using vectorized operations...")
    # 1. Get dimensions
    n_rows, n_cols = ad.shape
    
    # 2. Get Coords
    junc = ad.layers["cell_by_junction_matrix"].tocoo()
    clust = ad.layers["cell_by_cluster_matrix"].tocoo()
    
    # 3. Convert (row, col) pairs to single linear indices
    # This keeps everything in fast NumPy arrays (int64)
    junc_linear = junc.row.astype(np.int64) * n_cols + junc.col
    clust_linear = clust.row.astype(np.int64) * n_cols + clust.col
    
    # 4. Find missing indices using optimized NumPy set difference
    # This avoids creating millions of Python tuples
    missing_linear = np.setdiff1d(clust_linear, junc_linear)
    
    if missing_linear.size > 0:
        # 5. Convert linear indices back to (row, col)
        mr = missing_linear // n_cols
        mc = missing_linear % n_cols
        
        # 6. Concatenate and rebuild (same as before)
        all_r = np.concatenate([junc.row, mr])
        all_c = np.concatenate([junc.col, mc])
        all_v = np.concatenate([junc.data, np.zeros(len(missing_linear))])
        
        ad.layers["cell_by_junction_matrix"] = sparse.csr_matrix(
            (all_v, (all_r, all_c)), shape=junc.shape, dtype=junc.dtype)
            
    return ad

# File paths
BASE_PATH = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/"
masked_datasets = ["MASKED_75_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5mu", 
    "MASKED_50_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5mu", 
    "MASKED_25_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5mu"]

for TEST_PATH in masked_datasets:
    TEST_PATH = os.path.join(BASE_PATH, TEST_PATH)
    TEST_PATH_MASKED = TEST_PATH.replace(".h5mu", "_sparsity_fixed.h5mu")
    print(f"Processing masked dataset: {TEST_PATH_MASKED}")
    ad_mdata = mu.read_h5mu(TEST_PATH)
    ad = ad_mdata["splicing"]
    print(f"Data shape of test data: {ad.shape}")
    print(f"Available layers: {list(ad.layers.keys())}")
    print(f"Observations preview:\n{ad.obs.head()}")
    print(f"Variables preview:\n{ad.var.head()}")
    # Fix sparsity
    print("Fixing sparsity patterns...")
    ad = fix_sparsity_fast(ad)
    print("Sparsity patterns fixed.")
    # Save the processed data
    print(f"Saving processed data to: {TEST_PATH_MASKED}")
    ad.write_h5ad(TEST_PATH_MASKED, compression='lzf')
    print(f"Data processing completed successfully for {TEST_PATH_MASKED}")

"""
#Submit with:
conda activate LeafletSC
script=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/LeafletFA_analysis/04_B_mask_data_test_prep.py
cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/
sbatch --job-name=mask_data_test_prep \
       --partition=bigmem,cpu \
       --mem=800G \
       --time=1-00:00:00 \
       --output=mask_data_test_prep_%j.out \
       --error=mask_data_test_prep_%j.err \
       --wrap="python $script"
"""