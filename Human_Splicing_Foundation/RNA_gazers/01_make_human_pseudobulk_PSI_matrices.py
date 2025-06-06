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

# Single-cell analysis
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
output_dir = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/062025"
# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}", flush=True)

# ATSE file path
ATSE_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/ATSE_mapper/ATSE_files/stella_gtf/TMS_atse_file_unanno_also_2025-05-11_06-23-05.txt.gz"
assert os.path.exists(ATSE_file), f"ATSE file does not exist: {ATSE_file}"

# Load ATSE data
atses = pd.read_csv(ATSE_file, sep="\t")
assert "event_id" in atses.columns, "'event_id' column missing from ATSE file"
assert len(atses) > 0, "ATSE file is empty"
print(f"The number of ATSEs in this dataset is {len(atses['event_id'].unique())}", flush=True)

# Splicing input file
input_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/052025/splice_adata_matched_2025-05-12.h5ad"
assert os.path.exists(input_file), f"Input file does not exist: {input_file}"

# Define which column to use for cell type grouping
cell_type_column = "broad_cell_type"

# ----- Load and preprocess data -----

# Read splicing AnnData
print("Loading splicing AnnData file...", flush=True)
splice_adata = ad.read_h5ad(input_file)

# Add parameter for testing with subset of cells
MAX_CELLS = 5000  # Set to None to use all cells
if MAX_CELLS is not None and MAX_CELLS < splice_adata.shape[0]:
    print(f"\nSubsampling to {MAX_CELLS} cells while maintaining cell type proportions...")
    
    # Get cell type proportions
    cell_type_props = splice_adata.obs[cell_type_column].value_counts(normalize=True)
    print(cell_type_props)
    
    # Calculate number of cells to sample per cell type
    cells_per_type = (cell_type_props * MAX_CELLS).round().astype(int)
    
    # Ensure we don't exceed MAX_CELLS
    while cells_per_type.sum() > MAX_CELLS:
        # Find the largest cell type and reduce by 1
        largest_type = cells_per_type.idxmax()
        cells_per_type[largest_type] -= 1
    
    # Sample cells for each cell type
    sampled_indices = []
    for cell_type, n_cells in cells_per_type.items():
        print(f"Sampling {n_cells} cells for cell type {cell_type}")
        type_indices = np.where(splice_adata.obs[cell_type_column] == cell_type)[0]
        if len(type_indices) > n_cells:
            sampled_indices.extend(np.random.choice(type_indices, n_cells, replace=False))
        else:
            sampled_indices.extend(type_indices)
    
    # Create subset of AnnData
    splice_adata = splice_adata[sampled_indices].copy()
    print(f"Created subset with {splice_adata.shape[0]} cells")
    print("\nCell type distribution in subset:")
    print(splice_adata.obs[cell_type_column].value_counts())

    # Verify layer data is properly reindexed
    print("\nVerifying layer data reindexing:")
    for layer in splice_adata.layers:
        print(f"Layer '{layer}' shape: {splice_adata.layers[layer].shape}")

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
def create_pseudobulk(adata, group_by, layer=None):
    """
    Create pseudobulk samples by aggregating cells in the same group.
    
    Parameters:
    -----------
    adata : AnnData
        The input AnnData object
    group_by : str
        Column name in adata.obs to group cells by (e.g., 'cell_type')
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

# ----- Add gene annotation to the long dataframe -----

def add_gene_symbols(long_df):
    """
    Add gene symbols to the long-format DataFrame.
    
    Parameters:
    -----------
    long_df : DataFrame
        Long-format DataFrame with gene_id column
        
    Returns:
    --------
    DataFrame
        DataFrame with added gene_symbol column
    """
    if "gene_id" not in long_df.columns:
        print("WARNING: gene_id column not found. Gene symbols will not be added.")
        return long_df
    
    try:
        from mygene import MyGeneInfo
        
        print("Using MyGeneInfo to map gene IDs to gene symbols...")
        # Get unique gene IDs to query
        unique_gene_ids = [gene_id.split('.')[0] for gene_id in long_df["gene_id"].dropna().unique()]
        
        if unique_gene_ids:
            mg = MyGeneInfo()
            lookup = mg.querymany(unique_gene_ids, scopes="ensembl.gene", fields="symbol", species="human")
            
            # Create mapping dictionary
            gene_map = {}
            for entry in lookup:
                if "query" in entry:
                    query = entry["query"]
                    if "symbol" in entry:
                        gene_map[query] = entry["symbol"]
                    else:
                        gene_map[query] = None
            
            # Function to map gene ID to symbol
            def map_gene_id_to_symbol(gene_id):
                if pd.isna(gene_id):
                    return None
                base_id = gene_id.split('.')[0] if '.' in gene_id else gene_id
                return gene_map.get(base_id, None)
            
            # Apply mapping
            long_df["gene_symbol"] = long_df["gene_id"].apply(map_gene_id_to_symbol)
            print(f"Added gene symbols for {sum(long_df['gene_symbol'].notna())} entries")
            
    except ImportError:
        print("WARNING: mygene module not available. Gene symbols will not be added.")
    except Exception as e:
        print(f"WARNING: Error in gene ID mapping: {e}")
    
    return long_df

# Add gene symbols to DataFrame
long_df = add_gene_symbols(long_df)
long_df.sort_values(by="event_id", inplace=True)

# Save final version
final_file = os.path.join(output_dir, f"pseudobulk_final_{cell_type_column}_{timestamp}.csv.gz")
print(f"Saving final data to: {final_file}")
long_df.to_csv(final_file, index=False, compression="gzip")

print("\nPseudobulk creation and PSI calculation complete!")

# ----- SANITY CHECK CODE BLOCK -----
print("\n===== RUNNING SANITY CHECK =====")

# 1. Choose a random cell type
cell_types = splice_adata.obs[cell_type_column].unique()
random_cell_type = random.choice(cell_types)
print(f"Selected random cell type: {random_cell_type}")

# 2. Choose a random ATSE event
event_ids = atses['event_id'].unique()
random_event = random.choice(event_ids)
print(f"Selected random event: {random_event}")

# 3. Get all junctions for this event
event_junctions = atses[atses['event_id'] == random_event]['junction_id'].unique()
print(f"Junctions in this event: {', '.join(event_junctions)}")

# 4. Get cell indices for the selected cell type
cell_indices = np.where(splice_adata.obs[cell_type_column] == random_cell_type)[0]
print(f"Found {len(cell_indices)} cells of type {random_cell_type}")

# 5. Calculate raw junction counts manually
print("\nCalculating raw junction counts from single-cell data...")
junction_layer = "cell_by_junction_matrix"

# Create a lookup dictionary of junction_id to junction_id_index
junction_id_to_idx = {}
if 'junction_id_index' in splice_adata.var.columns:
    # If we have junction_id_index column, use it directly
    for idx, row in splice_adata.var.iterrows():
        junction_id = row['junction_id'] if 'junction_id' in row else idx
        junction_id_to_idx[junction_id] = row['junction_id_index']
else:
    # Fallback: use position in the var index
    for idx, junction_id in enumerate(splice_adata.var.index):
        junction_id_to_idx[junction_id] = idx

# Sum up counts for each junction in this event
raw_junction_counts = {}
total_raw_junction_count = 0

for junction in event_junctions:
    if junction in junction_id_to_idx:
        # Get the position directly from the lookup
        junction_position = junction_id_to_idx[junction]
        
        # Extract counts for this junction across the selected cells
        if scipy.sparse.issparse(splice_adata.layers[junction_layer]):
            counts = splice_adata.layers[junction_layer][cell_indices, junction_position].sum()
            if scipy.sparse.issparse(counts):
                counts = counts.toarray()[0, 0]
        else:
            counts = np.sum(splice_adata.layers[junction_layer][cell_indices, junction_position])
        
        raw_junction_counts[junction] = counts
        total_raw_junction_count += counts
    else:
        print(f"Warning: Junction {junction} not found in var dataframe")

print("\nRaw junction counts from single-cell data:")
for junction, count in sorted(raw_junction_counts.items()):
    print(f"  {junction}: {count}")
print(f"Total raw junction count: {total_raw_junction_count}")

# 6. Get the pseudobulk counts from the long format data
print("\nRetrieving pseudobulk counts from results...")

# Filter the long format data for our cell type and event
pseudobulk_rows = long_df[
    (long_df['cell_type'] == random_cell_type) & 
    (long_df['event_id'] == random_event)
]

# Sum the junction counts
pseudobulk_junction_counts = {}
total_pseudobulk_junction_count = 0

for _, row in pseudobulk_rows.iterrows():
    junction = row['junction_id']
    count = row['junction_count']
    pseudobulk_junction_counts[junction] = count
    total_pseudobulk_junction_count += count

print("\nPseudobulk junction counts from results:")
for junction, count in sorted(pseudobulk_junction_counts.items()):
    print(f"  {junction}: {count}")
print(f"Total pseudobulk junction count: {total_pseudobulk_junction_count}")

# 7. Compare the total counts
print("\n=== COMPARISON ===")
print(f"Total raw junction count: {total_raw_junction_count}")
print(f"Total pseudobulk junction count: {total_pseudobulk_junction_count}")

# Calculate percent difference
if total_raw_junction_count > 0:
    percent_diff = abs(total_raw_junction_count - total_pseudobulk_junction_count) / total_raw_junction_count * 100
    print(f"Percent difference: {percent_diff:.4f}%")
    
    # Determine if the check passes
    if percent_diff < 0.01:  # Less than 0.01% difference
        print("\nSANITY CHECK PASSED! ✅")
        print("The pseudobulk junction counts match the raw sums.")
    else:
        print("\nSANITY CHECK FAILED! ❌")
        print("There's a significant difference between pseudobulk and raw counts.")
else:
    print("\nCannot calculate percent difference (raw count is zero).")

# Print a detailed comparison for each junction to help debug any differences
print("\n=== DETAILED COMPARISON BY JUNCTION ===")
all_junctions = sorted(set(list(raw_junction_counts.keys()) + list(pseudobulk_junction_counts.keys())))
for junction in all_junctions:
    raw = raw_junction_counts.get(junction, 0)
    pseudobulk = pseudobulk_junction_counts.get(junction, 0)
    diff = abs(raw - pseudobulk)
    if raw > 0:
        percent = (diff / raw) * 100
        print(f"{junction}: Raw={raw}, Pseudobulk={pseudobulk}, Diff={diff} ({percent:.4f}%)")
    else:
        print(f"{junction}: Raw={raw}, Pseudobulk={pseudobulk}, Diff={diff} (N/A%)")

print("\nSanity check complete.")

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/052025
# sbatch --mem=250G -p dev,cpu --wrap="python /gpfs/commons/home/kisaev/Leaflet-analysis/Human_Splicing_Foundation/RNA_gazers/01_make_human_pseudobulk_PSI_matrices.py"
