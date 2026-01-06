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
from collections import defaultdict
import scipy.sparse as sp
import gffutils 
from tqdm import tqdm
import mygene

# Add the directory containing the shared utils to the Python path
sys.path.append("/gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/Multi_Species_Splicing_Foundation/shared_utils")

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
GTF_FILE = "/gpfs/commons/groups/knowles_lab/Megan/encode_pacbio/2025_mouse_longread/2025_mouse_collapse_GRCm38/all_samples_sp_collapse_all_chr_full.gtf"
DB_FILE = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper/genomes/lr_GRCm38.db"
AB_METADATA = "/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/METADATA/metadata.csv"

# These are raw data files:
AB_INTRONS = "/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/GeneExpression/expression_matrix_introns.csv"
AB_EXONS = "/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/GeneExpression/expression_matrix_exons.csv"
TMS_EXPRESSION = "/gpfs/commons/projects/knowles_singlecell_splicing/TabulaSenis/data/AWS/processed_for_scanpy/tabulamurissenisfacsofficialrawobj.h5ad"

# Step 1: Process gene annotation data
print("\n>> Processing gene annotation data...")

# %%
def validate_file_exists(filepath, description=""):
    """Validate that a file exists and is readable."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ {description} file not found: {filepath}")
    if not os.access(filepath, os.R_OK):
        raise PermissionError(f"❌ {description} file not readable: {filepath}")
    print(f"✅ {description} file found: {filepath}")
    return True

def extract_gene_transcript_info(gtf_file, db_file):
    """
    Parses a GENCODE GTF file to compute gene transcript information.
    
    Returns:
        DataFrame with gene_id, gene_name, mean_transcript_length, mean_intron_length, 
        num_transcripts, and transcript_biotypes.
    """
    print("\n=== EXTRACTING GENE TRANSCRIPT INFORMATION ===")
    
    # SANITY CHECK 1: Validate input files
    validate_file_exists(gtf_file, "GTF")
    
    # Load or create database
    if os.path.exists(db_file):
        print("✅ Using existing GTF database")
        db = gffutils.FeatureDB(db_file, keep_order=True)
        print("✅ Database loaded successfully!")
    else:
        print("⏳ Creating GTF database (this may take a few minutes)...")
        validate_file_exists(gtf_file, "GTF")
        db = gffutils.create_db(
            gtf_file,
            db_file,
            force=True,
            keep_order=True,
            disable_infer_transcripts=False,
            disable_infer_genes=True
        )
        print("✅ Database created successfully!")

    # Initialize data structures
    gene_exon_lengths = defaultdict(list)
    gene_intron_lengths = defaultdict(list)
    gene_names = {}
    gene_biotypes = defaultdict(set)
    transcript_counts = defaultdict(int)
    
    # Counters for sanity checks
    total_transcripts = 0
    skipped_transcripts = 0

    print("⏳ Processing transcripts to compute exon and intron lengths...")
    
    for transcript in tqdm(db.features_of_type("transcript"), desc="Processing Transcripts"):
        total_transcripts += 1
        
        # Extract transcript attributes
        gene_id = transcript.attributes.get("gene_id", [None])[0]
        gene_name = transcript.attributes.get("gene_name", ["unknown"])[0]
        transcript_biotype = transcript.attributes.get("transcript_type", ["unknown"])[0]
        
        if gene_id is None:
            print(f"⚠️  Skipping transcript without gene_id: {transcript.id}")
            skipped_transcripts += 1
            continue

        # Get exons for this transcript
        exons = list(db.children(transcript, featuretype="exon", order_by="start"))
        if len(exons) == 0:
            skipped_transcripts += 1
            continue

        # Calculate lengths
        exon_length = sum(exon.end - exon.start + 1 for exon in exons)
        transcript_start = min(exon.start for exon in exons)
        transcript_end = max(exon.end for exon in exons)
        transcript_span = transcript_end - transcript_start + 1
        intron_length = transcript_span - exon_length

        # Store data
        gene_exon_lengths[gene_id].append(exon_length)
        gene_intron_lengths[gene_id].append(max(0, intron_length))
        gene_names[gene_id] = gene_name
        gene_biotypes[gene_id].add(transcript_biotype)
        transcript_counts[gene_id] += 1

    # SANITY CHECK 2: Processing summary
    print(f"\n=== TRANSCRIPT PROCESSING SUMMARY ===")
    print(f"Total transcripts processed: {total_transcripts:,}")
    print(f"Transcripts skipped (no exons/gene_id): {skipped_transcripts:,}")
    print(f"Transcripts used: {total_transcripts - skipped_transcripts:,}")
    print(f"Unique genes found: {len(gene_exon_lengths):,}")

    if len(gene_exon_lengths) == 0:
        raise ValueError("❌ No valid genes found in GTF file!")

    # Create final DataFrame
    gene_ids = list(gene_exon_lengths.keys())
    gene_info_df = pd.DataFrame({
        "gene_id": gene_ids,
        "gene_name": [gene_names[g] for g in gene_ids],
        "mean_transcript_length": [sum(gene_exon_lengths[g]) / len(gene_exon_lengths[g]) for g in gene_ids],
        "mean_intron_length": [sum(gene_intron_lengths[g]) / len(gene_intron_lengths[g]) for g in gene_ids],
        "num_transcripts": [transcript_counts[g] for g in gene_ids],
        "transcript_biotypes": [", ".join(sorted(gene_biotypes[g])) for g in gene_ids]
    })

    # SANITY CHECK 3: Final DataFrame validation
    print(f"\n=== GENE INFO VALIDATION ===")
    print(f"Gene info DataFrame shape: {gene_info_df.shape}")
    print(f"Columns: {list(gene_info_df.columns)}")
    
    # Check for missing values
    missing_counts = gene_info_df.isnull().sum()
    if missing_counts.any():
        print(f"⚠️  Missing values found:\n{missing_counts[missing_counts > 0]}")
    else:
        print("✅ No missing values in gene info")
    
    # Basic statistics
    print(f"Mean transcript length range: {gene_info_df['mean_transcript_length'].min():.0f} - {gene_info_df['mean_transcript_length'].max():.0f}")
    print(f"Mean intron length range: {gene_info_df['mean_intron_length'].min():.0f} - {gene_info_df['mean_intron_length'].max():.0f}")
    print(f"Transcript count range: {gene_info_df['num_transcripts'].min()} - {gene_info_df['num_transcripts'].max()}")
    
    # Check for duplicates
    gene_id_dups = gene_info_df['gene_id'].duplicated().sum()
    gene_name_dups = gene_info_df['gene_name'].duplicated().sum()
    print(f"Duplicate gene_ids: {gene_id_dups}")
    print(f"Duplicate gene_names: {gene_name_dups}")
    
    if gene_id_dups > 0 or gene_name_dups > 0:
        print("⚠️  WARNING: Duplicates found in gene info!")

    return gene_info_df

gene_info_df = extract_gene_transcript_info(GTF_FILE, DB_FILE)

# Step 2: Load Allen Brain data
print("\n>> Loading Allen Brain Atlas data...")
try:
    metadata_ab = pd.read_csv(AB_METADATA, low_memory=False)
    print(f"   Loaded metadata for {len(metadata_ab)} cells")
    
    print("   Loading intron expression data...")
    ab_adata_introns = sc.read_csv(AB_INTRONS)
    ab_adata_introns.obs["sample_name"] = ab_adata_introns.obs.index
    print(f"   Loaded intron expression data: {ab_adata_introns.shape}")
    
    print("   Loading exon expression data...")
    ab_adata_exons = sc.read_csv(AB_EXONS)
    ab_adata_exons.obs["sample_name"] = ab_adata_exons.obs.index
    print(f"   Loaded exon expression data: {ab_adata_exons.shape}")
except Exception as e:
    print(f"   Error loading Allen Brain data: {str(e)}")
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
    print(f"   Loaded TMS expression data: {tms_adata.shape}")
except Exception as e:
    print(f"   Error loading TMS data: {str(e)}")
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

# Dictionary to store lost gene information
lost_genes_info = {}

# Initialize mygene
mg = mygene.MyGeneInfo()

# Create a clean gene_id column without version numbers
gene_info_df['gene_id_clean'] = gene_info_df['gene_id'].str.replace(r'\.\d+$', '', regex=True)

# Query gene info - request the gene biotype field
gene_ids = gene_info_df['gene_id_clean'].tolist()
results = mg.querymany(gene_ids, 
                       scopes='ensembl.gene', 
                       fields='symbol,type_of_gene',  # type_of_gene gives you the gene biotype
                       species='mouse', 
                       returnall=True)

# Process results
gene_info = pd.DataFrame(results['out'])

# Merge back to original dataframe
df_merged = gene_info_df.merge(
    gene_info[['query', 'symbol', 'type_of_gene']], 
    left_on='gene_id_clean',
    right_on='query', 
    how='left'
)

# Rename columns for clarity
df_merged = df_merged.rename(columns={
    'symbol': 'gene_name_new',
    'type_of_gene': 'gene_biotype'
})

# Drop the temporary columns if desired
df_merged = df_merged.drop(columns=['gene_id_clean', 'query'])

# Update the gene_name column with the new data where available
df_merged['gene_name'] = df_merged['gene_name_new'].fillna(df_merged['gene_name'])
df_merged = df_merged.drop(columns=['gene_name_new'])

# Let's remove any genes whose gene_name is unknown 
df_merged = df_merged[df_merged['gene_name'] != 'unknown']

# Remove transcript_biotype column
df_merged = df_merged.drop(columns=['transcript_biotypes'])

# remove genes that are not protein coding
df_merged = df_merged[df_merged['gene_biotype'] == 'protein-coding']
gene_info_df = df_merged.copy() 

# Add gene info to all datasets
for adata, name in zip([tms_adata, ab_adata_introns, ab_adata_exons], 
                       ["TMS", "AB introns", "AB exons"]):
    print(f"   Adding gene length info to {name} dataset...")
    adata.var["gene_name"] = adata.var["gene_symbol"] if "gene_symbol" in adata.var.columns else adata.var.index
    adata.var["orig_index"] = adata.var.index  # store original index

    # Track original genes
    original_genes = set(adata.var["gene_name"])
    
    # Perform merge
    merged = adata.var.reset_index().merge(gene_info_df, on="gene_name", how="inner")

    # Calculate lost genes
    if len(merged) < adata.shape[1]:
        merged_genes = set(merged["gene_name"])
        lost_genes = original_genes - merged_genes
        lost_genes_info[name] = {
            "total_before": len(original_genes),
            "total_after": len(merged_genes),
            "lost_genes": sorted(lost_genes),
            "lost_count": len(lost_genes)
        }
        print(f" Lost {adata.shape[1] - len(merged)} genes during gene info merge")
        print(f" Lost genes: {lost_genes_info[name]}")

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

# Check for and remove duplicate gene names
print("\n>> Checking for duplicate gene names...")
for adata, name in zip([tms_adata, ab_adata_introns, ab_adata_exons], 
                       ["TMS", "AB introns", "AB exons"]):
    n_duplicates = adata.var["gene_name"].duplicated().sum()
    if n_duplicates > 0:
        print(f"\n   {name}: Found {n_duplicates} duplicate gene names")
        duplicate_genes = adata.var[adata.var["gene_name"].duplicated(keep=False)].sort_values("gene_name")
        print(f"   Duplicate genes: {sorted(set(duplicate_genes['gene_name']))}")
    else:
        print(f"   {name}: No duplicate gene names found")

# Remove duplicate gene names - keep first occurrence
print("\n>> Removing duplicate gene names (keeping first occurrence)...")
tms_adata = tms_adata[:, ~tms_adata.var["gene_name"].duplicated(keep='first')].copy()
ab_adata_introns = ab_adata_introns[:, ~ab_adata_introns.var["gene_name"].duplicated(keep='first')].copy()
ab_adata_exons = ab_adata_exons[:, ~ab_adata_exons.var["gene_name"].duplicated(keep='first')].copy()

print(f"   ✓ After removing duplicates:")
print(f"     TMS: {tms_adata.shape[1]} genes")
print(f"     AB introns: {ab_adata_introns.shape[1]} genes")
print(f"     AB exons: {ab_adata_exons.shape[1]} genes")

# Sort by gene name
print("\n>> Sorting genes by name...")
tms_adata = tms_adata[:, tms_adata.var.sort_values("gene_name").index].copy()
ab_adata_introns = ab_adata_introns[:, ab_adata_introns.var.sort_values("gene_name").index].copy()
ab_adata_exons = ab_adata_exons[:, ab_adata_exons.var.sort_values("gene_name").index].copy()
print("   ✓ Gene sorting complete")

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
        print(f"   Column '{col}' not found in TMS data, adding empty column")
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
    print(f"   Error creating combined dataset: {str(e)}")
    sys.exit(1)

# Make sure introns data is in CSR format for saving
if hasattr(ab_adata_introns.X, "format") and ab_adata_introns.X.format == "coo":
    print("   Converting COO matrix to CSR format for saving...")
    ab_adata_introns.X = ab_adata_introns.X.tocsr()
    
# Also check any layers
for layer_name in ab_adata_introns.layers:
    if hasattr(ab_adata_introns.layers[layer_name], "format") and ab_adata_introns.layers[layer_name].format == "coo":
        print(f"   Converting layer '{layer_name}' from COO to CSR format...")
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
        print(f"   Successfully saved to {full_path}")
        return True
    except Exception as e:
        print(f"   Error saving {filename}: {str(e)}")
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