#!/usr/bin/env python
"""
Gene Expression Data Preparation for Mouse Splicing Foundation Dataset

This script processes gene expression data from Tabula Muris Senis and Allen Brain datasets:
1. Extracts gene transcript and intron length information from GTF files
2. Loads and preprocesses expression data from both datasets
3. Harmonizes gene names and cell metadata across datasets
4. Performs length normalization of gene expression values
5. Saves combined and processed datasets for downstream analysis
"""

import sys
import os
import datetime
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix
import numpy as np

# Add the directory containing the shared utils to the Python path
sys.path.append("/gpfs/commons/home/kisaev/Leaflet-analysis/Multi_Species_Splicing_Foundation/shared_utils")

# Import utility functions
from gene_processing import (
    extract_gene_transcript_info, 
    normalize_by_gene_length,
    safe_stringify_obs,
    preprocess_anndata,
    normalize_and_log_transform
)

# Set up logging
today = datetime.datetime.now().strftime("%Y-%m-%d")
print("="*80)
print(f"GENE EXPRESSION DATA PREPARATION - {today}")
print("="*80)

# Configuration
OUTPUT_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input file paths
GTF_FILE = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/genome_files/gencode.vM19/genes/genes.gtf"
DB_FILE = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/genome_files/GENCODE_vM19"
AB_METADATA = "/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/METADATA/metadata.csv"
AB_INTRONS = "/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/GeneExpression/expression_matrix_introns.csv"
AB_EXONS = "/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/GeneExpression/expression_matrix_exons.csv"
TMS_EXPRESSION = "/gpfs/commons/projects/knowles_singlecell_splicing/TabulaSenis/data/AWS/processed_for_scanpy/tabulamurissenisfacsofficialrawobj.h5ad"

# Step 1: Process gene annotation data
print("\n>> Processing gene annotation data...")
gene_info_df = extract_gene_transcript_info(GTF_FILE, DB_FILE)

# Step 2: Load Allen Brain data
print("\n>> Loading Allen Brain Atlas data...")
try:
    metadata_ab = pd.read_csv(AB_METADATA, low_memory=False)
    print(f"   ✓ Loaded metadata for {len(metadata_ab)} cells")
    
    print("   ⚙️ Loading intron expression data...")
    ab_adata_introns = sc.read_csv(AB_INTRONS)
    ab_adata_introns.obs["sample_name"] = ab_adata_introns.obs.index
    print(f"   ✓ Loaded intron expression data: {ab_adata_introns.shape}")
    
    print("   ⚙️ Loading exon expression data...")
    ab_adata_exons = sc.read_csv(AB_EXONS)
    ab_adata_exons.obs["sample_name"] = ab_adata_exons.obs.index
    print(f"   ✓ Loaded exon expression data: {ab_adata_exons.shape}")
except Exception as e:
    print(f"   ❌ Error loading Allen Brain data: {str(e)}")
    sys.exit(1)
    
# Step 3: Process Allen Brain data - make explicit copies
print("\n>> Processing Allen Brain data...")
ab_adata_introns = ab_adata_introns.copy()  # Explicit copy
ab_adata_exons = ab_adata_exons.copy()      # Explicit copy

ab_adata_introns = preprocess_anndata(
    ab_adata_introns, 
    metadata=metadata_ab, 
    dataset_label="allen_brain_introns", 
    metadata_key="sample_name"
)

ab_adata_exons = preprocess_anndata(
    ab_adata_exons, 
    metadata=metadata_ab, 
    dataset_label="allen_brain_exons", 
    metadata_key="sample_name"
)

# Step 4: Load Tabula Muris Senis data
print("\n>> Loading Tabula Muris Senis data...")
try:
    tms_adata = sc.read_h5ad(TMS_EXPRESSION)
    tms_adata = tms_adata.copy()  # Explicit copy
    tms_adata = preprocess_anndata(tms_adata, dataset_label="tabula_muris_senis")
    print(f"   ✓ Loaded TMS expression data: {tms_adata.shape}")
except Exception as e:
    print(f"   ❌ Error loading TMS data: {str(e)}")
    sys.exit(1)
    
# Step 5: Clean TMS cell IDs
print("\n>> Cleaning Tabula Muris Senis cell IDs...")
# Apply basic cleaning to all cell IDs
tms_adata.obs_names = pd.Index([
    cell_id.split(".mm10-plus-0-0")[0].split(".mus-2-1")[0] 
    for cell_id in tms_adata.obs_names
])
tms_adata.obs["cell_id"] = tms_adata.obs_names.values

# Special handling for 3m cells
if 'cell' in tms_adata.obs.columns:
    tms_adata.obs['cell_clean'] = tms_adata.obs['cell'].copy()
    if 'age' in tms_adata.obs.columns and (tms_adata.obs['age'] == '3m').any():
        mask = tms_adata.obs['age'] == '3m'
        tms_adata.obs.loc[mask, 'cell_clean'] = tms_adata.obs.loc[mask, 'cell'].str.replace('.', '_', 2)
    tms_adata.obs["cell_clean"] = tms_adata.obs["cell_clean"].str.split('_').str[:2].str.join('_')
print(f"   ✓ Cleaned cell IDs for {tms_adata.shape[0]} cells")

# Step 6: Harmonize gene information across datasets
print("\n>> Harmonizing genes across datasets...")

# Add gene info to all datasets
for adata, name in zip([tms_adata, ab_adata_introns, ab_adata_exons], 
                       ["TMS", "AB introns", "AB exons"]):
    print(f"   ⚙️ Adding gene length info to {name} dataset...")
    adata.var["gene_name"] = adata.var["gene_symbol"] if "gene_symbol" in adata.var.columns else adata.var.index
    adata.var["orig_index"] = adata.var.index  # store original index
    merged = adata.var.reset_index().merge(gene_info_df, on="gene_name", how="inner")
    if len(merged) < adata.shape[1]:
        print(f"   ⚠️ Lost {adata.shape[1] - len(merged)} genes during gene info merge")
    if "orig_index" in merged.columns:
        merged = merged.set_index("orig_index")
    adata._inplace_subset_var(merged.index)  # ensure dimensions match
    adata.var = merged
    print(f"   ✓ {name} now has {adata.shape[1]} genes with length information")

# Find common genes
common_genes = set(tms_adata.var["gene_name"]).intersection(
    ab_adata_introns.var["gene_name"]
).intersection(
    ab_adata_exons.var["gene_name"]
)
print(f"   ✓ Found {len(common_genes)} genes common to all datasets")

# Subset to common genes - make explicit copies
tms_adata = tms_adata[:, tms_adata.var["gene_name"].isin(common_genes)].copy()
ab_adata_introns = ab_adata_introns[:, ab_adata_introns.var["gene_name"].isin(common_genes)].copy()
ab_adata_exons = ab_adata_exons[:, ab_adata_exons.var["gene_name"].isin(common_genes)].copy()

# Sort by gene name
tms_adata = tms_adata[:, tms_adata.var.sort_values("gene_name").index].copy()
ab_adata_introns = ab_adata_introns[:, ab_adata_introns.var.sort_values("gene_name").index].copy()
ab_adata_exons = ab_adata_exons[:, ab_adata_exons.var.sort_values("gene_name").index].copy()

# Verification
gene_match_1 = all(tms_adata.var["gene_name"].values == ab_adata_exons.var["gene_name"].values)
gene_match_2 = all(ab_adata_introns.var["gene_name"].values == ab_adata_exons.var["gene_name"].values)
print(f"   {'✓' if gene_match_1 and gene_match_2 else '❌'} Gene alignment check: {'Passed' if gene_match_1 and gene_match_2 else 'Failed'}")

# Step 7: Ensure data is in sparse format
print("\n>> Converting to sparse matrix format...")
for adata, name in zip([ab_adata_introns, ab_adata_exons, tms_adata], 
                       ["AB introns", "AB exons", "TMS"]):
    if not isinstance(adata.X, csr_matrix):
        adata.X = csr_matrix(adata.X.copy())  # Make a copy before conversion
        print(f"   ✓ Converted {name} data to sparse format")

# Step 8: Clean metadata
print("\n>> Standardizing metadata across datasets...")

# Allen Brain introns
ab_adata_introns = ab_adata_introns.copy()  # Make a copy to avoid view issues
ab_adata_introns.obs["cell_id"] = ab_adata_introns.obs["exp_component_name"].copy()
ab_adata_introns.obs["cell_ontology_class"] = ab_adata_introns.obs["cell_type_alias_label"].copy()
ab_adata_introns.obs["tissue"] = ab_adata_introns.obs["exp_component_name"].copy()
ab_adata_introns.obs["age"] = "2m" 
ab_adata_introns.obs["mouse.id"] = ab_adata_introns.obs["external_donor_name_label"].copy()
ab_adata_introns.obs["subtissue"] = ab_adata_introns.obs["region_label"].copy()
ab_adata_introns.obs["sex"] = ab_adata_introns.obs["donor_sex_label"].copy()
# Make sure dataset column exists
ab_adata_introns.obs["dataset"] = "allen_brain_introns"
# Now subset columns
ab_adata_introns.obs = ab_adata_introns.obs[["cell_id", "age", "cell_ontology_class", 
                                             "mouse.id", "sex", "subtissue", "tissue", "dataset"]].copy()

# Allen Brain exons
ab_adata_exons = ab_adata_exons.copy()  # Make a copy to avoid view issues
ab_adata_exons.obs["cell_id"] = ab_adata_exons.obs["exp_component_name"].copy()
ab_adata_exons.obs["cell_ontology_class"] = ab_adata_exons.obs["cell_type_alias_label"].copy()
ab_adata_exons.obs["tissue"] = ab_adata_exons.obs["exp_component_name"].copy()
ab_adata_exons.obs["age"] = "2m" 
ab_adata_exons.obs["mouse.id"] = ab_adata_exons.obs["external_donor_name_label"].copy()
ab_adata_exons.obs["subtissue"] = ab_adata_exons.obs["region_label"].copy()
ab_adata_exons.obs["sex"] = ab_adata_exons.obs["donor_sex_label"].copy()
# Make sure dataset column exists
ab_adata_exons.obs["dataset"] = "allen_brain_exons"
# Now subset columns
ab_adata_exons.obs = ab_adata_exons.obs[["cell_id", "age", "cell_ontology_class", 
                                        "mouse.id", "sex", "subtissue", "tissue", "dataset"]].copy()

# TMS
tms_adata = tms_adata.copy()  # Make a copy to avoid view issues
# Ensure all required columns exist
keep_cols = ['cell_id', 'age', 'cell_ontology_class', 'mouse.id', 'sex', 'subtissue', 'tissue', 'dataset']
for col in keep_cols:
    if col not in tms_adata.obs.columns:
        print(f"   ⚠️ Column '{col}' not found in TMS data, adding empty column")
        tms_adata.obs[col] = "unknown"
        
tms_adata.obs = tms_adata.obs[keep_cols].copy()
print("   ✓ Standardized metadata columns across all datasets")

# Step 9: Apply length normalization
print("\n>> Performing gene length normalization...")
# For introns, use intron length instead of transcript length
ab_adata_introns.var["mean_transcript_length"] = ab_adata_introns.var["mean_intron_length"].copy()
normalize_by_gene_length(ab_adata_introns)
normalize_by_gene_length(ab_adata_exons)
normalize_by_gene_length(tms_adata)
print("   ✓ Applied length normalization to all datasets")

# Step 10: Library size normalization and log transformation
print("\n>> Performing library size normalization and log transformation...")
normalize_and_log_transform(ab_adata_introns)
normalize_and_log_transform(ab_adata_exons)
normalize_and_log_transform(tms_adata)
print("   ✓ Applied library size normalization and log transformation")

# Step 11: Final preparation for saving
print("\n>> Preparing datasets for saving...")

# Convert obs columns to strings for compatibility
ab_adata_introns = safe_stringify_obs(ab_adata_introns)
ab_adata_exons = safe_stringify_obs(ab_adata_exons)
tms_adata = safe_stringify_obs(tms_adata)

# Reset indices for consistency
tms_adata.var = tms_adata.var.reset_index(drop=True)
ab_adata_exons.var = ab_adata_exons.var.reset_index(drop=True)
tms_adata.obs = tms_adata.obs.reset_index(drop=True)

# Step 12: Create combined dataset
print("\n>> Creating combined TMS-Allen Brain dataset...")

# Ensure gene indices are strings
ab_adata_exons.var.index = ab_adata_exons.var.index.astype(str)
tms_adata.var.index = tms_adata.var.index.astype(str)

# Double-check gene alignment
common_genes = set(ab_adata_exons.var.index).intersection(set(tms_adata.var.index))
print(f"   ✓ Verified {len(common_genes)} genes with matching indices")

# Concatenate datasets
try:
    combined_adata = ad.concat(
        [ab_adata_exons, tms_adata],
        axis=0,  # concatenate cells (not genes)
        join="inner",  # require exact matching genes
        label="batch",  # new column in .obs to record original source
        keys=["allen_brain_exons", "tabula_muris_senis"],  # batch labels
        index_unique=None  # don't modify cell ids
    )
    
    # Make sure var info is consistent
    combined_adata.var = ab_adata_exons.var.copy()
    
    print(f"   ✓ Combined dataset contains {combined_adata.shape[0]} cells and {combined_adata.shape[1]} genes")
    print(f"   ✓ Batch distribution: {dict(combined_adata.obs['batch'].value_counts())}")
except Exception as e:
    print(f"   ❌ Error creating combined dataset: {str(e)}")
    sys.exit(1)

# Make sure introns data is in CSR format for saving
if hasattr(ab_adata_introns.X, "format") and ab_adata_introns.X.format == "coo":
    print("   ⚙️ Converting COO matrix to CSR format for saving...")
    ab_adata_introns.X = ab_adata_introns.X.tocsr()
    
# Also check any layers
for layer_name in ab_adata_introns.layers:
    if hasattr(ab_adata_introns.layers[layer_name], "format") and ab_adata_introns.layers[layer_name].format == "coo":
        print(f"   ⚙️ Converting layer '{layer_name}' from COO to CSR format...")
        ab_adata_introns.layers[layer_name] = ab_adata_introns.layers[layer_name].tocsr()

# Step 13: Save results
print("\n>> Saving processed datasets...")

def save_anndata(adata, filename, compress=True, overwrite=False):
    """
    Safely save AnnData object with error handling and file existence check
    
    Args:
        adata: AnnData object to save
        filename: Name of the file to save
        compress: Whether to use compression
        overwrite: Whether to overwrite existing file
        
    Returns:
        bool: True if file was saved or already exists, False otherwise
    """
    full_path = os.path.join(OUTPUT_DIR, filename)
    
    # Check if file already exists
    if not overwrite and os.path.exists(full_path):
        print(f"   ✓ File already exists: {full_path} (skipping)")
        return True
        
    try:
        compression = "lzf" if compress else None
        adata.write_h5ad(full_path, compression=compression)
        print(f"   ✓ Successfully saved to {full_path}")
        return True
    except Exception as e:
        print(f"   ❌ Error saving {filename}: {str(e)}")
        return False

# Define filenames
combined_filename = f"tms_ab_exons_combo_ge_adata_{today}.h5ad"
introns_filename = f"ab_adata_introns_{today}.h5ad"

# Save files, by default it won't overwrite existing files
combined_saved = save_anndata(combined_adata, combined_filename)
introns_saved = save_anndata(ab_adata_introns, introns_filename)

# Report final status
if combined_saved and introns_saved:
    status = "All files saved or already existed"
else:
    status = "Some files could not be saved"

print("\n" + "="*80)
print("PIPELINE COMPLETE")
print(f"{status} in {OUTPUT_DIR}")
print("="*80)

# to submit
# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/processed_data
# sbatch --job-name=prep_ge_data --mem=200G --partition dev,cpu,bigmem --wrap "python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/GeneExpression/01_prepare_expression_data.py"