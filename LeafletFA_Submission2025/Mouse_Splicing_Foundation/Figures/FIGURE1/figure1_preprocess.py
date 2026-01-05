#!/usr/bin/env python
"""
FIGURE 1 - STEP 1: Preprocessing
Load data, calculate PSI, filter, and identify cell types for parallel processing
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
from datetime import datetime
from scipy.sparse import csr_matrix, issparse
import warnings
warnings.filterwarnings("ignore")

def calculate_psi_values(adata):
    """Calculate junction usage ratios (PSI values)"""
    print("Calculating PSI values...")
    A = adata.layers["cell_by_junction_matrix"].tocsr()
    B = adata.layers["cell_by_cluster_matrix"].tocsr()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_data = np.true_divide(A.data, B.data)
        valid_mask = np.isfinite(ratio_data)
        ratio_data[~valid_mask] = 0
    
    psi_matrix = csr_matrix((ratio_data, A.indices, A.indptr), shape=A.shape)
    adata.layers["psi"] = psi_matrix
    adata.layers["psi_valid"] = csr_matrix(
        (valid_mask.astype(float), A.indices, A.indptr), shape=A.shape
    )
    print("PSI calculation complete!")
    return adata


def preprocess_data(
    adata,
    min_age=3,
    young_ages=[3],
    old_ages=[18, 24],
    age_col="age",
    age_suffix="m",
    celltype_col="medium_cell_type",
    min_cells_per_age=50,
    min_ages_with_min_cells=2,
    min_cells_per_junction=10,
    min_detection_rate=0.01,
    verbose=True
):
    """Clean and prepare data with flexible filtering criteria"""
    filter_stats = {}
    initial_cells = adata.n_obs
    initial_junctions = adata.n_vars
    
    # Convert age to integers
    if pd.api.types.is_numeric_dtype(adata.obs[age_col]):
        adata.obs[age_col] = adata.obs[age_col].astype(int)
    else:
        adata.obs[age_col] = adata.obs[age_col].str.replace(age_suffix, "").astype(int)
    
    # Filter cells by age
    adata = adata[adata.obs[age_col] >= min_age].copy()
    
    if verbose:
        print(f"Age filter: {initial_cells:,} → {adata.n_obs:,} cells")
    
    # Define age groups
    def assign_age_group(age):
        if age in young_ages:
            return "young"
        elif age in old_ages:
            return "old"
        return "other"
    
    adata.obs["age_group"] = adata.obs[age_col].apply(assign_age_group)
    
    if verbose:
        print(f"\nAge group distribution:")
        for group, count in adata.obs['age_group'].value_counts().items():
            print(f"  {group}: {count:,} cells")
    
    # Filter cell types with sufficient cells
    ages_to_include = young_ages + old_ages
    adata_for_counting = adata[adata.obs[age_col].isin(ages_to_include)]
    counts = adata_for_counting.obs.groupby([celltype_col, age_col]).size().unstack(fill_value=0)
    valid_celltypes = counts.apply(
        lambda row: (row >= min_cells_per_age).sum() >= min_ages_with_min_cells,
        axis=1
    )
    valid_celltype_list = counts[valid_celltypes].index.tolist()
    adata = adata[adata.obs[celltype_col].isin(valid_celltype_list)].copy()
    
    if verbose:
        print(f"\nRetained {len(valid_celltype_list)} cell types with {adata.n_obs:,} cells")
    
    # Filter junctions by detection
    if "psi_valid" in adata.layers:
        valid_matrix = adata.layers["psi_valid"].tocsr()
        non_nan_counts = np.array(valid_matrix.sum(axis=0)).flatten()
    else:
        psi_matrix = adata.layers["psi"]
        non_nan_counts = np.array((psi_matrix != 0).sum(axis=0)).flatten()
    
    detection_rate = non_nan_counts / adata.n_obs
    valid_junctions = (non_nan_counts >= min_cells_per_junction) & \
                     (detection_rate >= min_detection_rate)
    
    adata = adata[:, valid_junctions].copy()
    
    if verbose:
        print(f"\nJunction filter: {initial_junctions:,} → {adata.n_vars:,} junctions")
        print(f"\n{'='*60}")
        print(f"FINAL DATASET:")
        print(f"  Cells: {adata.n_obs:,}")
        print(f"  Junctions: {adata.n_vars:,}")
        print(f"  Cell types: {len(valid_celltype_list)}")
        print(f"{'='*60}")
    
    return adata, filter_stats


def main():
    today = datetime.now().strftime("%Y%m%d")
    BASE_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
    output_dir = "/gpfs/commons/home/kisaev/Leaflet-analysis/Multi_Species_Splicing_Foundation/figure_paper"
    
    run_dir = f"{output_dir}/figure1_{today}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(f"{run_dir}/intermediate", exist_ok=True)
    
    print(f"Saving outputs to: {run_dir}")
    
    # File paths
    SPLICE_INPUT = f"{BASE_DIR}/MODEL_INPUT/102025/model_ready_aligned_splicing_data_20251009_024406.h5ad"
    
    print("\n" + "="*60)
    print("PREPROCESSING: Load and filter data")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    splice_adata = ad.read_h5ad(SPLICE_INPUT)
    print(f"Loaded {splice_adata.n_obs:,} cells and {splice_adata.n_vars:,} junctions")
    
    # Calculate PSI and preprocess
    print("\nCalculating PSI values...")
    splice_adata = calculate_psi_values(splice_adata)
    splice_adata.var["junction_id_index"] = range(splice_adata.shape[1])
    
    print("\nPreprocessing data...")
    splice_adata, stats = preprocess_data(splice_adata)
    splice_adata.var["junction_id_index"] = range(splice_adata.shape[1])
    
    # Create tissue-celltype combinations
    if 'tissue' in splice_adata.obs.columns and 'medium_cell_type' in splice_adata.obs.columns:
        splice_adata.obs['tissue_celltype'] = (
            splice_adata.obs['tissue'].astype(str) + '_' +
            splice_adata.obs['medium_cell_type'].astype(str)
        )
        print(f"\nCreated 'tissue_celltype' with {splice_adata.obs['tissue_celltype'].nunique()} combinations")
    
    splice_adata.var.reset_index(inplace=True)
    
    # Get list of cell types to process (with sufficient cells)
    print("\nIdentifying cell types for parallel processing...")
    cell_type_counts = splice_adata.obs.groupby(['tissue_celltype', 'age_group']).size().unstack(fill_value=0)
    valid_celltypes = cell_type_counts[
        (cell_type_counts.get('young', 0) >= 100) & 
        (cell_type_counts.get('old', 0) >= 100)
    ].index.tolist()
    
    print(f"Found {len(valid_celltypes)} cell types with >=100 cells in both young and old")
    
    # Show some examples
    print("\nExample cell types:")
    for i, ct in enumerate(valid_celltypes[:5]):
        young_n = cell_type_counts.loc[ct, 'young']
        old_n = cell_type_counts.loc[ct, 'old']
        print(f"  {i+1}. {ct}: young={young_n}, old={old_n}")
    if len(valid_celltypes) > 5:
        print(f"  ... and {len(valid_celltypes) - 5} more")
    
    # Save cell types list for array job
    celltypes_file = f"{run_dir}/cell_types_to_process.txt"
    with open(celltypes_file, 'w') as f:
        for ct in valid_celltypes:
            f.write(f"{ct}\n")
    print(f"\nSaved cell types to: {celltypes_file}")
    
    # Save preprocessed data
    print("\nSaving preprocessed data...")
    splice_adata.write_h5ad(f"{run_dir}/preprocessed_data_{today}.h5ad")
    print(f"Saved: {run_dir}/preprocessed_data_{today}.h5ad")
    
    # Save event IDs (as text file to avoid pickle issues)
    event_ids = splice_adata.var["event_id"].drop_duplicates().to_numpy()
    event_ids_file = f"{run_dir}/event_ids_{today}.txt"
    with open(event_ids_file, 'w') as f:
        for eid in event_ids:
            f.write(f"{eid}\n")
    print(f"Saved {len(event_ids):,} event IDs to {event_ids_file}")
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"Next step: Run array job to process {len(valid_celltypes)} cell types")
    print(f"Each cell type will process {len(event_ids):,} ATSEs")


if __name__ == "__main__":
    main()