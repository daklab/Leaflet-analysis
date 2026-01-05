#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to create pseudobulk samples from single-cell splicing data
and calculate PSI values (junction_counts/atse_counts) using the specified layers.
Focuses on cell_type grouping only and efficiently uses junction_id_index.
"""

# Standard libraries
import os
import sys
from datetime import datetime
import random
from tqdm import tqdm

# Data manipulation libraries
import numpy as np
import pandas as pd
import scipy.sparse
import anndata as ad

# ----- TO -DO! -----
# Change truly missing values where the ATSE wasn't at ALL detected in a given group of cells to NaN and color GREY in clustermap... 
# Make pseudobulk matrix via RAW data (not LeafletFA imputed values or anything)
# Align junctions with annotated isoform JUNCTIONS to label junctions with annotated vs novel isoform events...

# ----- Setup and configuration -----

# Timestamp for file naming
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# ----- Directory and file setup -----

# Output directory
output_dir = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/102025"
# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}", flush=True)

# ATSE file path
ATSE_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/ATSE_mapper/ATSE_files/HUMAN_FOUNDATION_ATSE_FILE_unanno_also_2025-09-21_05-02-35.txt.gz"
assert os.path.exists(ATSE_file), f"ATSE file does not exist: {ATSE_file}"

# Metadata file
metadata_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/human_metadata_combined.tsv"
assert os.path.exists(metadata_file), f"Metadata file does not exist: {metadata_file}"

# Load metadata
metadata = pd.read_csv(metadata_file, sep="\t")
print(metadata.head())

# Load ATSE data
atses = pd.read_csv(ATSE_file, sep="\t")
assert "event_id" in atses.columns, "'event_id' column missing from ATSE file"
assert len(atses) > 0, "ATSE file is empty"
print(f"The number of ATSEs in this dataset is {len(atses['event_id'].unique())}", flush=True)

# Splicing input file
input_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/102025/model_ready_aligned_splicing_data_20251009_023419.h5ad"
assert os.path.exists(input_file), f"Input file does not exist: {input_file}"

# Define which column to use for cell type grouping
cell_type_column = "medium_cell_type"

# ----- Load and preprocess data -----

# Read splicing AnnData
print("Loading splicing AnnData file...", flush=True)
splice_adata = ad.read_h5ad(input_file)

# Reset index and store original index
splice_adata.obs.reset_index(drop=True, inplace=True)
splice_adata.obs["cell_id_index"] = splice_adata.obs.index
print(f"The number of cells in the dataset is {splice_adata.shape[0]}", flush=True)

# Print dataset statistics
print("Dataset distribution:")
print(splice_adata.obs.dataset.value_counts())

print("\nTissue distribution:")
print(splice_adata.obs.tissue.value_counts())

# Assign sequencing technology based on source
splice_adata.obs["seqtech"] = "single_nuclei"
splice_adata.obs.loc[splice_adata.obs["dataset"] == "tabula_sapiens", "seqtech"] = "single_cell"
print("\nSequencing technology distribution:")
print(splice_adata.obs.seqtech.value_counts())

# ----- Create pseudobulk samples -----

# Check available columns for cell type grouping
print("\nAvailable columns in splice_adata.obs:")
print(splice_adata.obs.columns.tolist())

# Show distribution of cell types by dataset
print("\n=== Distribution of cell types by dataset ===")
cell_type_dataset_counts = pd.crosstab(
    splice_adata.obs[cell_type_column], 
    splice_adata.obs["dataset"], 
    margins=True, 
    margins_name="Total"
)
print(cell_type_dataset_counts)

# Save distribution to file
distribution_file = os.path.join(output_dir, f"cell_type_dataset_distribution_{timestamp}.csv")
cell_type_dataset_counts.to_csv(distribution_file)
print(f"Saved cell type distribution to: {distribution_file}")

# Function to create pseudobulk samples
def create_pseudobulk(adata, group_by, min_cells=50, layer=None):
    """
    Create pseudobulk samples by aggregating cells in the same group.
    
    Parameters:
    -----------
    adata : AnnData
        The input AnnData object
    group_by : str
        Column name in adata.obs to group cells by (e.g., 'cell_type')
    min_cells: int, optional
        Minimum number of cells per group to include in the pseudobulk sample
    layer : str, optional
        Layer to use for aggregation (if None, uses X)
        
    Returns:
    --------
    AnnData
        Pseudobulk AnnData object with summed read counts across cells in each group
    """
    print(f"Creating pseudobulk samples grouped by '{group_by}'")
    
    # Get unique groups
    unique_groups = adata.obs[group_by].unique()
    print(f"Found {len(unique_groups)} unique {group_by} groups")
    
    # Select data matrix (use specified layer or default to X)
    data_matrix = adata.layers[layer] if layer is not None and layer in adata.layers else adata.X
    print(f"Using {'layer ' + layer if layer else 'main expression matrix (X)'} for pseudobulk creation")
    
    # Initialize pseudobulk matrix (same type as input - sparse or dense)
    is_sparse = scipy.sparse.issparse(data_matrix)
    if is_sparse:
        pseudobulk_matrix = scipy.sparse.lil_matrix((len(unique_groups), adata.shape[1]), dtype=np.float32)
    else:
        pseudobulk_matrix = np.zeros((len(unique_groups), adata.shape[1]), dtype=np.float32)
    
    # Create observation dataframe to store metadata
    pseudobulk_obs = pd.DataFrame(index=unique_groups)
    pseudobulk_obs["cell_type"] = unique_groups  # Use cell_type as the column name for consistency
    pseudobulk_obs.index.name = "cell_type"
    
    # Add metadata columns we want to track
    meta_columns = ["dataset", "tissue", "age", "seqtech"]
    for col in meta_columns:
        if col in adata.obs.columns:
            pseudobulk_obs[col] = ""
    
    # Track number of cells per group
    pseudobulk_obs["n_cells"] = 0
    
    # Aggregate data for each group
    print(f"Aggregating data for {len(unique_groups)} pseudobulk samples...")
    
    # Process each group
    for i, group in enumerate(tqdm(unique_groups)):

        # Get indices of cells in this group
        cell_indices = np.where(adata.obs[group_by] == group)[0]
        n_cells = len(cell_indices)
        
        if n_cells == 0:
            print(f"Warning: No cells found for group {group}")
            continue

        if n_cells < min_cells:
            print(f"Warning: Only {n_cells} cells found for group {group}, skipping")
            continue
        
        # Update cell count
        pseudobulk_obs.loc[group, "n_cells"] = n_cells
        
        # Sum the counts across cells for this group
        if is_sparse:
            # For sparse matrix
            group_sum = data_matrix[cell_indices].sum(axis=0)
            # Convert to appropriate format
            if isinstance(group_sum, np.matrix):
                group_sum = np.array(group_sum).flatten()
            elif scipy.sparse.issparse(group_sum):
                group_sum = group_sum.toarray().flatten()
            
            # Assign to pseudobulk matrix
            pseudobulk_matrix[i, :] = scipy.sparse.csr_matrix(group_sum)
        else:
            # For dense matrix - simple sum
            pseudobulk_matrix[i, :] = np.sum(data_matrix[cell_indices], axis=0)
        
        # For each metadata column, use the most common value from cells in this group
        for col in meta_columns:
            if col in adata.obs.columns:
                value_counts = adata.obs.loc[cell_indices, col].value_counts()
                most_common = value_counts.index[0] if len(value_counts) > 0 else "Unknown"
                pseudobulk_obs.loc[group, col] = most_common
                
                # For dataset, also track percentage of cells from each source
                if col == "dataset":
                    for ds in adata.obs[col].unique():
                        count = value_counts.get(ds, 0)
                        percent = count / n_cells * 100 if n_cells > 0 else 0
                        pseudobulk_obs.loc[group, f"percent_{ds}"] = percent
    
    # Convert to CSR if matrix is sparse for efficiency
    if is_sparse and not isinstance(pseudobulk_matrix, scipy.sparse.csr_matrix):
        pseudobulk_matrix = pseudobulk_matrix.tocsr()
    
    # Preserve junction identifiers and indices
    var_df = adata.var.copy()
    
    # Ensure we have junction_id_index or similar for lookup
    if 'junction_id_index' not in var_df.columns:
        var_df['junction_id_index'] = np.arange(adata.shape[1])
    
    # Create AnnData object with the pseudobulk data
    pseudobulk_adata = ad.AnnData(
        X=pseudobulk_matrix,
        obs=pseudobulk_obs,
        var=var_df
    )
    
    # Preserve the original var index
    pseudobulk_adata.var.index = adata.var.index.copy()
    
    return pseudobulk_adata

# Check if the specified layers exist
junction_layer = "cell_by_junction_matrix"
atse_layer = "cell_by_cluster_matrix"

assert junction_layer in splice_adata.layers, f"Junction layer '{junction_layer}' not found in AnnData object"
assert atse_layer in splice_adata.layers, f"ATSE cluster layer '{atse_layer}' not found in AnnData object"

print(f"\nFound junction layer: {junction_layer}")
print(f"Found ATSE cluster layer: {atse_layer}")

# Create pseudobulk samples
print("\n=== Creating pseudobulk junction samples ===")
pseudobulk_junction = create_pseudobulk(splice_adata, cell_type_column, layer=junction_layer)
print(f"Created pseudobulk junction object with shape: {pseudobulk_junction.shape}")

print("\n=== Creating pseudobulk ATSE cluster samples ===")
pseudobulk_atse = create_pseudobulk(splice_adata, cell_type_column, layer=atse_layer)
print(f"Created pseudobulk ATSE object with shape: {pseudobulk_atse.shape}")

# ----- Add sanity check to confirm junction counts were correctly created -----

# Sample cell type 
cell_type_sample = pseudobulk_junction.obs.index[1]
print(f"Sampled cell type: {cell_type_sample}")

# sample junction_id_index 
junc_sample = pseudobulk_junction.var["junction_id_index"].sample(1)
junc_sample = junc_sample.values[0]
print(f"Sampled junction_id_index: {junc_sample}")

# Find ATSE event_id for this junction_id_index
atse_event_id = pseudobulk_junction.var[pseudobulk_junction.var["junction_id_index"] == junc_sample]["event_id"].values[0]
print(f"ATSE event_id: {atse_event_id}")

# Get sum of counts for this junction_id_index in raw data
# Find indices in splice_adata.obs that match cell_type_sample
cell_indices = np.where(splice_adata.obs[cell_type_column] == cell_type_sample)[0]
print(f"Found {len(cell_indices)} cells for cell type {cell_type_sample}")

# Get the counts for this junction_id_index in the raw data
junction_counts = splice_adata.layers[junction_layer][cell_indices, junc_sample]
print(f"Junction counts: {junction_counts}")
# print the sum of the counts
print(f"Sum of junction counts: {junction_counts.sum()}")

# Get sum of counts for this event_id in raw data
atse_counts = splice_adata.layers[atse_layer][cell_indices, junc_sample]
print(f"ATSE counts: {atse_counts}")
# print the sum of the counts
print(f"Sum of ATSE counts: {atse_counts.sum()}")

# Subset pseudobulk just to this cell type
pseudobulk_junction_cell_type = pseudobulk_junction[pseudobulk_junction.obs["cell_type"] == cell_type_sample]
print(f"Junction counts for this cell type: {pseudobulk_junction_cell_type.X[:, junc_sample].data}")

# Do the same for the ATSE counts
pseudobulk_atse_cell_type = pseudobulk_atse[pseudobulk_atse.obs["cell_type"] == cell_type_sample]
print(f"ATSE counts for this cell type: {pseudobulk_atse_cell_type.X[:, junc_sample].data}")

# ----- Calculate PSI values -----

def calculate_psi(junction_adata, atse_adata):
    """
    Calculate PSI (Percent Spliced In) values from junction and ATSE counts.
    
    Parameters:
    -----------
    junction_adata : AnnData
        AnnData object with junction counts
    atse_adata : AnnData
        AnnData object with ATSE counts
        
    Returns:
    --------
    AnnData
        AnnData object with PSI values
    """
    print("\n=== Calculating PSI values ===")
    
    # Create a copy of the junction AnnData to store PSI values
    psi_adata = junction_adata.copy()
    
    # Get the count matrices
    junction_counts = junction_adata.X
    atse_counts = atse_adata.X
    
    # Convert to dense if needed
    if scipy.sparse.issparse(junction_counts):
        junction_counts_dense = junction_counts.toarray()
    else:
        junction_counts_dense = junction_counts

    if scipy.sparse.issparse(atse_counts):
        atse_counts_dense = atse_counts.toarray()
    else:
        atse_counts_dense = atse_counts
    
    # Initialize PSI matrix with NaN values
    psi_values = np.full_like(junction_counts_dense, np.nan, dtype=float)
    
    # Calculate PSI where ATSE counts > 0
    nonzero_mask = atse_counts_dense > 0
    psi_values[nonzero_mask] = junction_counts_dense[nonzero_mask] / atse_counts_dense[nonzero_mask]
    
    # Store the raw counts in layers for reference
    psi_adata.layers["junction_counts"] = junction_counts
    psi_adata.layers["atse_counts"] = atse_counts
    
    # Store PSI values in the main matrix
    psi_adata.X = psi_values
    
    # Add metadata about PSI calculation
    psi_adata.uns["psi_info"] = {
        "calculation": "junction_counts / atse_counts",
        "created_timestamp": timestamp,
        "note": "PSI values are NaN where ATSE count is 0"
    }
    
    return psi_adata

# Calculate PSI values
pseudobulk_psi = calculate_psi(pseudobulk_junction, pseudobulk_atse)
# Suset to just cell_type_sample
pseudobulk_psi_cell_type = pseudobulk_psi[pseudobulk_psi.obs["cell_type"] == cell_type_sample]
print(pseudobulk_psi_cell_type.X[:, junc_sample]) 

# Print summary statistics
print("\n=== Pseudobulk PSI summary ===")
print(f"Number of pseudobulk samples: {pseudobulk_psi.shape[0]}")
print(f"Number of features (junctions): {pseudobulk_psi.shape[1]}")
print("Number of cells per pseudobulk sample:")
print(pseudobulk_psi.obs["n_cells"].describe())

# ----- Create long-format data frame -----

def create_long_format_df(psi_adata, atses):
    """
    Create a long-format DataFrame from the PSI AnnData object.
    
    Parameters:
    -----------
    psi_adata : AnnData
        AnnData object with PSI values
    atses : DataFrame
        DataFrame with ATSE information
        
    Returns:
    --------
    DataFrame
        Long-format DataFrame with PSI values
    """
    print("\n=== Creating long-format data frame ===")
    
    # Get data matrices
    psi_values = psi_adata.X
    junction_counts = psi_adata.layers["junction_counts"]
    atse_counts = psi_adata.layers["atse_counts"]
    
    # Convert to dense if needed
    if scipy.sparse.issparse(junction_counts):
        junction_counts_dense = junction_counts.toarray()
    else:
        junction_counts_dense = junction_counts
        
    if scipy.sparse.issparse(atse_counts):
        atse_counts_dense = atse_counts.toarray()
    else:
        atse_counts_dense = atse_counts
    
    # Create rows for the long format DataFrame
    rows = []
    for i, cell_type in enumerate(psi_adata.obs.index):
        n_cells = psi_adata.obs.loc[cell_type, "n_cells"]
        
        for j, junction_id in enumerate(psi_adata.var["junction_id"]):
            rows.append({
                "cell_type": cell_type,
                "junction_id": junction_id,
                "junction_id_index": psi_adata.var["junction_id_index"][j],
                "junction_count": junction_counts_dense[i, j],
                "atse_count": atse_counts_dense[i, j],
                "gene_symbol": psi_adata.var["gene_name"][j], 
                "psi": psi_values[i, j],
                "n_cells": n_cells
            })
    
    # Create the DataFrame
    long_df = pd.DataFrame(rows)
    
    # Merge ATSE information
    atses_subset = atses[["junction_id", "gene_id", "event_id"]].drop_duplicates()
    long_df = pd.merge(long_df, atses_subset, on="junction_id", how="left")
    
    return long_df

# Create long format DataFrame
long_df = create_long_format_df(pseudobulk_psi, atses)
long_df.sort_values(by="event_id", inplace=True)

# Save final version
final_file = os.path.join(output_dir, f"pseudobulk_final_{cell_type_column}_{timestamp}.csv.gz")
print(f"Saving final data to: {final_file}")
long_df.to_csv(final_file, index=False, compression="gzip")

print("\nPseudobulk creation and PSI calculation complete!")

# conda activate LeafletSC
# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/102025
# sbatch --mem=64G -p dev,cpu --wrap="python /gpfs/commons/home/kisaev/Leaflet-analysis/Human_Splicing_Foundation/RNA_gazers/01_make_human_pseudobulk_PSI_matrices.py"
