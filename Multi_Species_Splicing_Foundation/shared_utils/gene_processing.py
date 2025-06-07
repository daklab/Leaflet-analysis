# File: /gpfs/commons/home/kisaev/Leaflet-analysis/Multi_Species_Splicing_Foundation/shared_utils/gene_processing.py

"""
Utility functions for gene expression data processing in Leaflet-FA analysis.

This module contains shared functions for processing gene annotation data and 
normalizing gene expression across both mouse and human foundation datasets.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import gffutils
from scipy.sparse import csr_matrix

def extract_gene_transcript_info(gtf_file, db_file):
    """
    Parses a GENCODE GTF file to compute gene transcript and intron length information.
    
    Args:
        gtf_file (str): Path to the GTF file
        db_file (str): Path to store the gffutils database
        
    Returns:
        pd.DataFrame: Table with gene_id, gene_name, transcript lengths, and biotype info
    """
    print("\n>> Processing gene annotation data...")
    
    try:
        # Create database directory if it doesn't exist
        db_dir = os.path.dirname(db_file)
        if not os.path.exists(db_dir) and db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        if os.path.exists(db_file):
            print("   ✓ Using existing GTF database")
            db = gffutils.FeatureDB(db_file, keep_order=True)
        else:
            print("   ⚙️ Creating GTF database (this may take a few minutes)...")
            db = gffutils.create_db(
                gtf_file,
                db_file,
                force=True,
                keep_order=True,
                disable_infer_transcripts=False,
                disable_infer_genes=True
            )
            print("   ✓ Database created successfully")

        gene_exon_lengths = defaultdict(list)
        gene_intron_lengths = defaultdict(list)
        gene_names = {}
        gene_biotypes = defaultdict(set)
        transcript_counts = defaultdict(int)

        print("   ⚙️ Processing transcripts to compute exon and intron lengths...")
        for transcript in tqdm(db.features_of_type("transcript"), desc="Processing Transcripts"):
            gene_id = transcript.attributes["gene_id"][0]
            gene_name = transcript.attributes.get("gene_name", ["unknown"])[0]
            transcript_biotype = transcript.attributes.get("transcript_type", ["unknown"])[0]

            exons = list(db.children(transcript, featuretype="exon", order_by="start"))
            if len(exons) == 0:
                continue  # skip transcripts with no exons

            exon_length = sum(exon.end - exon.start + 1 for exon in exons)
            transcript_span = exons[-1].end - exons[0].start + 1
            intron_length = transcript_span - exon_length  # includes gaps between exons
            
            # Skip if the transcript has zero length or negative intron length
            if exon_length <= 0 or intron_length <= 0:
                continue
                
            gene_exon_lengths[gene_id].append(exon_length)
            gene_intron_lengths[gene_id].append(intron_length)
            gene_names[gene_id] = gene_name
            gene_biotypes[gene_id].add(transcript_biotype)
            transcript_counts[gene_id] += 1

        print("   ✓ Finished processing transcripts")

        # Create DataFrame
        gene_ids = list(gene_exon_lengths.keys())
        gene_info_df = pd.DataFrame({
            "gene_id": gene_ids,
            "gene_name": [gene_names.get(g, "unknown") for g in gene_ids],
            "mean_transcript_length": [sum(gene_exon_lengths[g]) / len(gene_exon_lengths[g]) for g in gene_ids],
            "mean_intron_length": [sum(gene_intron_lengths[g]) / len(gene_intron_lengths[g]) for g in gene_ids],
            "num_transcripts": [transcript_counts[g] for g in gene_ids],
            "transcript_biotypes": [", ".join(sorted(gene_biotypes[g])) for g in gene_ids]
        })
        
        # Clean up gene info
        gene_info_df = gene_info_df.drop_duplicates(subset="gene_id")
        gene_info_df = gene_info_df.drop_duplicates(subset="gene_name")
        
        # Verify no zero-length genes remain
        zero_length_genes = (gene_info_df["mean_transcript_length"] <= 0) | (gene_info_df["mean_intron_length"] <= 0)
        if zero_length_genes.any():
            print(f"   ⚠️ Warning: Found {zero_length_genes.sum()} genes with zero length - removing them")
            gene_info_df = gene_info_df[~zero_length_genes].copy()
            
        print(f"   ✓ Final gene info contains {len(gene_info_df)} genes with valid length information")
        return gene_info_df
        
    except Exception as e:
        print(f"   ❌ Error processing gene information: {str(e)}")
        raise

def normalize_by_gene_length(adata, input_layer="raw_counts", output_layer="length_norm"):
    """
    Normalize gene expression counts by gene length.
    
    Args:
        adata (AnnData): AnnData object
        input_layer (str): Layer containing raw counts
        output_layer (str): Layer to store normalized values
        
    Returns:
        AnnData: Object with normalized counts
    """

    # Make sure input data is in CSR format
    counts = adata.layers[input_layer]
    if not isinstance(counts, csr_matrix):
        counts = csr_matrix(counts)
    
    # Get per-gene mean transcript lengths and enforce column order
    lengths = adata.var["mean_transcript_length"].reindex(adata.var.index)  # enforce order
    if lengths.isnull().any():
        raise ValueError("Some genes are missing length annotations.")
    lengths = lengths.values

    overall_median_length = np.median(lengths)

     # Do sparse division
    inv_lengths = 1.0 / lengths
    row, col = counts.nonzero()
    counts_norm = counts.copy()
    print("First normalizing by individual gene length")
    counts_norm.data = counts_norm.data * inv_lengths[col]
    print("Then scaling by overall median length")
    counts_norm.data = counts_norm.data * overall_median_length

    # Round down to integer values so we maintain count based data
    counts_norm.data = np.floor(counts_norm.data)

    # Save back
    adata.layers[output_layer] = counts_norm
    return adata
        
def safe_stringify_obs(adata):
    """
    Convert object columns in adata.obs to strings to avoid compatibility issues.
    
    Args:
        adata (AnnData): AnnData object to process
        
    Returns:
        AnnData: Processed AnnData object
    """
    for col in adata.obs.columns:
        if adata.obs[col].dtype == 'object' or adata.obs[col].apply(type).nunique() > 1:
            adata.obs[col] = adata.obs[col].astype(str)
    return adata

def preprocess_anndata(adata, metadata=None, dataset_label=None, metadata_key=None):
    """
    Standardize AnnData object with metadata integration.
    
    Args:
        adata (AnnData): The AnnData object to process
        metadata (pd.DataFrame, optional): Metadata to integrate
        dataset_label (str, optional): Label for the dataset
        metadata_key (str, optional): Column to use for matching metadata
        
    Returns:
        AnnData: Processed AnnData object
    """
    adata = adata.copy()
    
    # Add dataset label if provided
    if dataset_label:
        adata.obs["dataset"] = dataset_label
    
    # Store raw counts if not already present
    if "raw_counts" not in adata.layers:
        adata.layers["raw_counts"] = adata.X.copy()
    
    # If metadata is provided, integrate it
    if metadata is not None and metadata_key is not None:
        # Add metadata_key column to adata.obs if it doesn't exist
        if metadata_key not in adata.obs.columns:
            adata.obs[metadata_key] = adata.obs_names
        
        # Check that metadata contains the needed key
        if metadata_key not in metadata.columns:
            raise ValueError(f"Metadata key '{metadata_key}' not found in metadata columns")
        
        # Subset metadata to matching cells
        metadata_sub = metadata[metadata[metadata_key].isin(adata.obs_names)]
        if len(metadata_sub) == 0:
            raise ValueError(f"No matching cells found in metadata using key '{metadata_key}'")
            
        print(f"   ✓ Found {len(metadata_sub)} matching cells in metadata (out of {adata.shape[0]} total)")
        
        # Set index for joining
        metadata_sub = metadata_sub.set_index(metadata_key)
        
        # Align metadata to adata
        adata = adata[adata.obs_names.isin(metadata_sub.index)].copy()
        adata.obs = metadata_sub.loc[adata.obs_names]
    
    return adata

def normalize_and_log_transform(adata, norm_layer="length_norm", output_layer="log_norm"):
    """
    Perform library size normalization and log transformation.
    
    Args:
        adata (AnnData): AnnData object
        norm_layer (str): Layer with length-normalized data
        output_layer (str): Layer to store log-normalized values
        
    Returns:
        AnnData: Processed AnnData object with new layer
    """
    try:
        # Get length-normalized counts
        norm_counts = adata.layers[norm_layer]
        
        # Compute library size factors (sum per cell)
        lib_size = norm_counts.sum(axis=1).A1
        
        # Check for cells with zero counts
        zero_cells = (lib_size == 0)
        if zero_cells.any():
            print(f"Warning: Found {zero_cells.sum()} cells with zero total counts - setting size factor to 1")
            lib_size[zero_cells] = 1
        
        # Normalize by library size to CPM (counts per million)
        size_norm = norm_counts.multiply(1e4 / lib_size[:, None])
        
        # Log transform (log1p)
        log_norm = np.log1p(size_norm)
        
        # Save as new layer
        adata.layers[output_layer] = log_norm
        
        return adata
        
    except Exception as e:
        print(f"Error in normalization and log transformation: {str(e)}")
        raise