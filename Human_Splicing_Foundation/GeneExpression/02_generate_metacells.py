#!/usr/bin/env python
"""
Gene Expression Metacell Generation for Mouse Splicing Foundation

This script:
1. Loads processed gene expression and intron data
2. Maps cell types to standardized broader categories
3. Generates pseudobulk counts for common cell types
4. Saves the resulting pseudobulk data for downstream analysis
"""

import os
import sys
import datetime
import pandas as pd
import numpy as np
import anndata as ad
from tqdm import tqdm
import scanpy as sc
from scipy.sparse import csr_matrix, issparse as sp_issparse
import logging

# Setup logging
log_file = f"/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/processed_data/prep_ge_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

# Custom print function that also logs
def log_print(message):
    print(message)
    logging.info(message)

# Configuration
WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/processed_data"
OUTPUT_DIR = WD  # Save in the same directory
today = datetime.datetime.now().strftime("%Y-%m-%d")
date = "2025-09-30"
# Cell types to focus on for pseudobulk analysis
COMMON_BROAD_TYPES = ['Other_Neuron', 'Pericyte', 'General_Fibroblast', 'Microglia', 'CNS_Glia']

# Load data
print("Loading data...")
ab_exons = sc.read_h5ad(f"{WD}/ab_adata_exons_{date}.h5ad")
ab_introns = sc.read_h5ad(f"{WD}/ab_adata_introns_{date}.h5ad")
ts_adata = sc.read_h5ad(f"{WD}/tabsap_adata_{date}.h5ad")
gene_info_df = pd.read_csv(f"{WD}/gene_info_df_{date}.csv")

# === Clean up gene symbols and merge gene info ===
print("Cleaning up gene symbols and merging gene information...")

# First, let's standardize gene symbols across all datasets
for adata in [ab_exons, ab_introns, ts_adata]:
    adata.var["gene_symbol"] = adata.var["gene_symbol"].astype(str)

# Remove duplicated gene symbols from each dataset
ab_exons = ab_exons[:, ~ab_exons.var["gene_symbol"].duplicated(keep=False)].copy()
ab_exons.var_names = ab_exons.var["gene_symbol"]

ab_introns = ab_introns[:, ~ab_introns.var["gene_symbol"].duplicated(keep=False)].copy()
ab_introns.var_names = ab_introns.var["gene_symbol"]

ts_adata = ts_adata[:, ~ts_adata.var["gene_symbol"].duplicated(keep=False)].copy()
ts_adata.var_names = ts_adata.var["gene_symbol"]

# Prepare gene_info_df for merging
# Make sure we have the right column names
if 'gene_name' in gene_info_df.columns:
    gene_info_df = gene_info_df.rename(columns={'gene_name': 'gene_symbol'})
elif 'gene_symbol' not in gene_info_df.columns:
    print("Warning: No 'gene_name' or 'gene_symbol' column found in gene_info_df")
    print("Available columns:", gene_info_df.columns.tolist())

# Create a comprehensive gene info mapping
gene_info_dict = {}
if 'gene_symbol' in gene_info_df.columns:
    for _, row in gene_info_df.iterrows():
        gene_symbol = str(row['gene_symbol'])
        gene_info_dict[gene_symbol] = {
            'mean_transcript_length': row.get('mean_transcript_length', 0),
            'mean_intron_length': row.get('mean_intron_length', 0),
            'gene_id': row.get('gene_id', ''),
            'transcript_biotype': row.get('transcript_biotypes', 'unknown')
        }

print(f"Gene info available for {len(gene_info_dict)} genes")

# === Subset to shared genes and add gene metadata ===
print("Subsetting to shared genes...")
common_genes = set(ab_exons.var_names).intersection(ab_introns.var_names).intersection(ts_adata.var_names)
print(f"Found {len(common_genes)} common genes")

# Subset each dataset to common genes
gene_mask_ab_exons = [g in common_genes for g in ab_exons.var_names]
gene_mask_ab_introns = [g in common_genes for g in ab_introns.var_names]
gene_mask_ts = [g in common_genes for g in ts_adata.var_names]

ab_exons = ab_exons[:, gene_mask_ab_exons].copy()
ab_introns = ab_introns[:, gene_mask_ab_introns].copy()
ts_adata = ts_adata[:, gene_mask_ts].copy()

# Add gene metadata to all datasets
print("Adding gene metadata...")
for adata in [ab_exons, ab_introns, ts_adata]:
    # Initialize metadata columns
    adata.var['mean_transcript_length'] = 0.0
    adata.var['mean_intron_length'] = 0.0
    adata.var['gene_id'] = ''
    adata.var['transcript_biotype'] = 'unknown'
    
    # Fill in metadata from gene_info_dict
    for gene in adata.var_names:
        if gene in gene_info_dict:
            adata.var.loc[gene, 'mean_transcript_length'] = gene_info_dict[gene]['mean_transcript_length']
            adata.var.loc[gene, 'mean_intron_length'] = gene_info_dict[gene]['mean_intron_length']
            adata.var.loc[gene, 'gene_id'] = gene_info_dict[gene]['gene_id']
            adata.var.loc[gene, 'transcript_biotype'] = gene_info_dict[gene]['transcript_biotype']

print(f"Gene metadata added for {len([g for g in ab_exons.var_names if g in gene_info_dict])} out of {len(ab_exons.var_names)} genes")

# === Refined cell type mapping with intermediate granularity ===
grouped_refined_map = {
    # === NEURONS - Split by major functional classes ===
    'Excitatory_Neuron': [
        'IT', 'L4 IT', 'L5 ET', 'L6 CT', 'L6b', 'L5/6 IT Car3', 'L5/6 NP'
    ],
    'Inhibitory_Neuron': [
        'VIP', 'PVALB', 'SST', 'LAMP5'
    ],
    'Other_Neuron': [
        'PAX6', 'retinal bipolar neuron'
    ],
    
    # === T CELLS - Organized by major functional subsets ===
    'CD4_T_cell': [
        'cd4-positive, alpha-beta t cell', 'cd4-positive helper t cell', 
        'cd4-positive, alpha-beta memory t cell', 'naive thymus-derived cd4-positive, alpha-beta t cell',
        'activated cd4-positive, alpha-beta t cell', 'cd4-positive, alpha-beta thymocyte',
        'cd4-positive memory t cell', 't follicular helper cell'
    ],
    'CD8_T_cell': [
        'cd8-positive, alpha-beta t cell', 'cd8-positive, alpha-beta memory t cell',
        'cd8-positive, alpha-beta thymocyte', 'naive cd8-positive t cell',
        'activated cd8-positive, alpha-beta t cell', 'cd8-positive cytotoxic t cell',
        'cd8+, alpha-beta cytokine secreting effector t cell'
    ],
    'Regulatory_T_cell': [
        'regulatory t cell', 'naive regulatory t cell'
    ],
    'Other_T_cell': [
        't cell', 'gamma-delta t cell', 'thymocyte'
    ],
    
    # === B CELLS ===
    'B_cell': [
        'b cell', 'memory b cell', 'naive b cell'
    ],
    'Plasma_cell': [
        'plasma cell', 'antibody secreting cell'
    ],
    
    # === MYELOID CELLS - Split by major lineages ===
    'Microglia': [
        'Microglia', 'microglial cell', 'retina - microglia'
    ],
    'Macrophage': [
        'macrophage', 'Monocyte_Macrophage', 'tissue-resident macrophage', 
        'muscle macrophage'
    ],
    'Monocyte': [
        'monocyte', 'classical monocyte', 'non-classical monocyte', 'intermediate monocyte'
    ],
    'Dendritic_cell': [
        'dendritic cell', 'myeloid dendritic cell', 'plasmacytoid dendritic cell',
        'cd1c-positive myeloid dendritic cell', 'cd141-positive myeloid dendritic cell',
        'cdc1', 'cdc2', 'conventional dendritic cell'
    ],
    'Granulocyte': [
        'neutrophil', 'cd24 neutrophil', 'nampt neutrophil', 'granulocyte',
        'basophil', 'mast cell'
    ],
    
    # === NK/ILC ===
    'NK_ILC': [
        'nk cell', 'natural killer cell', 'nk t cell', 'innate lymphoid cell', 
        'mature nk t cell', 'type i nk t cell', 'uterine nk cell', 
        'proliferating nk cell', 'immature natural killer cell'
    ],
    
    # === EPITHELIAL - Split by organ system ===
    'Respiratory_Epithelial': [
        'club cell', 'ionocyte', 'goblet cell', 'ciliated epithelial cell',
        'pulmonary ionocyte', 'serous cell of epithelium of bronchus',
        'respiratory goblet cell', 'ciliated columnar cell of tracheobronchial tree',
        'tracheal goblet cell'
    ],
    'GI_Epithelial': [
        'enterocyte of epithelium of large intestine', 
        'enterocyte of epithelium proper of small intestine',
        'large intestine goblet cell', 'best4+ intestinal epithelial cell',
        'small intestine goblet cell', 'enterocyte of epithelium proper of ileum',
        'enterocyte of epithelium proper of duodenum', 
        'paneth cell of epithelium of small intestine', 'paneth cell of colon',
        'mature enterocyte', 'intestinal tuft cell', 'tuft cell of colon'
    ],
    'Urogenital_Epithelial': [
        'bladder urothelial cell', 'basal bladder urothelial cell', 
        'intermediate bladder urothelial cell', 'epithelial cell of uterus'
    ],
    'Mammary_Epithelial': [
        'HR positive luminal epithelial cell of mammary gland',
        'secretory luminal epithelial cell of mammary gland',
        'luminal epithelial cell'
    ],
    'Other_Epithelial': [
        'epithelial cell', 'duct epithelial cell', 'ltf+ epithelial cell',
        'basal epithelial cell', 'salivary gland cell', 'medullary thymic epithelial cell',
        'conjunctival epithelial cell', 'corneal epithelial cell', 'glandular epithelial cell',
        'cycling epithelial cell', 'mucus secreting cell', 'biliary epithelial cell',
        'pancreatic ductal cell', 'stratified squamous epithelial cell', 'sebum secreting cell'
    ],
    
    # === ENDOTHELIAL - Split by vessel type ===
    'Arterial_Endothelial': [
        'arterial endothelial cell', 'endothelial cell of arteriole', 'endothelial cell of artery'
    ],
    'Venous_Endothelial': [
        'vein endothelial cell', 'venous capillary endothelial cell', 'endothelial cell of venule'
    ],
    'Capillary_Endothelial': [
        'capillary endothelial cell', 'blood vessel endothelial cell'
    ],
    'Lymphatic_Endothelial': [
        'endothelial cell of lymphatic vessel'
    ],
    'Specialized_Endothelial': [
        'endothelial cell', 'endothelial cell of vascular tree', 'cardiac endothelial cell',
        'colon endothelial cell', 'retinal blood vessel endothelial cell', 'vascular endothelial cell'
    ],
    
    # === GLIA - Split by CNS vs PNS ===
    'CNS_Glia': [
        'Astrocyte', 'OPC', 'Oligodendrocyte', 'retina - muller glia', 'mueller cell'
    ],
    'PNS_Glia': [
        'enteroglial cell', 'schwann cell'
    ],
    'Glia_Other': [
        'glial cell'
    ],
    
    # === MUSCLE - Split by muscle type ===
    'Smooth_Muscle': [
        'smooth muscle cell', 'airway smooth muscle cell', 'vascular associated smooth muscle cell'
    ],
    'Cardiac_Muscle': [
        'atrial cardiac muscle cell', 'ventricular cardiac muscle cell'
    ],
    'Skeletal_Muscle': [
        'skeletal muscle satellite stem cell', 'fast muscle cell', 'slow muscle cell',
        'tongue muscle cell'
    ],
    'Muscle_Other': [
        'muscle cell', 'tendon cell'
    ],
    
    # === STROMAL/FIBROBLAST - More specific organ groupings ===
    'General_Fibroblast': [
        'fibroblast', 'stromal cell', 'myofibroblast cell', 'adventitial fibroblast',
        'cd34+ fibroblasts', 'VLMC', 'adventitial cell', 'connective tissue cell'
    ],
    'Organ_Specific_Fibroblast': [
        'alveolar fibroblast', 'fibroblast of breast', 'fibroblast of cardiac tissue',
        'uterine fibroblast', 'stellate_fibroblast', 'endometrial stromal fibroblast'
    ],
    'Specialized_Stromal': [
        'fat cell', 'cornea - mesenchymal cell - stromal keratinocytes',
        'limbal stromal cell', 'follicle', 'granulosa cell', 'mesothelial cell', 'theca cell'
    ],
    
    # === LIVER - Split by major cell types ===
    'Hepatocyte': [
        'hepatocyte'
    ],
    'Liver_Non_Parenchymal': [
        'hepatic stellate cell', 'intrahepatic cholangiocyte'
    ],
    
    # === SPECIALIZED CELLS ===
    'Pericyte': [
        'pericyte', 'Pericyte', 'myofibroblast cell and pericyte', 'mural cell'
    ],
    
    'Photoreceptor': [
        'retinal pigment epithelial cell', 'retina - photoreceptor cell', 'eye photoreceptor cell'
    ],
    
    'Alveolar_cell': [
        'type ii pneumocyte', 'type i pneumocyte', 'capillary aerocyte'
    ],
    
    'Enteroendocrine': [
        'enteroendocrine cell of small intestine', 'type l enteroendocrine cell',
        'enterochromaffin-like cell'
    ],
    
    'Hematopoietic_Mature': [
        'platelet', 'erythrocyte'
    ],
    
    'Hematopoietic_Progenitor': [
        'erythroid progenitor cell', 'hematopoietic stem cell', 'myeloid progenitor',
        'common myeloid progenitor'
    ],
    
    'Mesenchymal_Stem': [
        'mesenchymal stem cell', 'mesenchymal stem cell of adipose tissue'
    ],
    
    'Stem_Progenitor_Other': [
        'oocyte', 'radial glia progenitor cell', 'intestinal crypt stem cell of small intestine',
        'intestinal crypt stem cell of large intestine'
    ],
    
    'Secretory_Gland': [
        'acinar cell of salivary gland', 'lacrimal gland functional unit cell', 'myoepithelial cell'
    ],
    
    'Pigment_cell': [
        'melanocyte', 'melanocyte or limbal stem cell'
    ],
    
    'Sensory_cell': [
        'taste receptor cell'
    ],
    
    'Skin_cell': [
        'keratocyte'
    ],
    
    'Myeloid_Other': [
        'myeloid cell', 'mononuclear phagocyte'
    ],
    
    'Immune_Other': [
        'leukocyte', 'langerhans cell', 'immune cell'
    ],
    
    'Unknown': [
        'unknown'
    ]
}

# Define flatten function
def flatten_grouped_map(grouped_map):
    return {label: broad_type for broad_type, labels in grouped_map.items() for label in labels}

# Create the flat map and apply cell type mapping
print("Mapping cell types...")
grouped_broad_map_flat = flatten_grouped_map(grouped_refined_map)

# Apply mapping with proper column names (check if these columns exist)
if "free_annotation" in ts_adata.obs.columns:
    ts_adata.obs["broad_cell_type"] = ts_adata.obs["free_annotation"].map(grouped_broad_map_flat).fillna('Other')
else:
    print("Warning: 'free_annotation' column not found in ts_adata. Available columns:", ts_adata.obs.columns.tolist())
    # Use an alternative column if available
    if "cell_type" in ts_adata.obs.columns:
        ts_adata.obs["broad_cell_type"] = ts_adata.obs["cell_type"].map(grouped_broad_map_flat).fillna('Other')
    else:
        ts_adata.obs["broad_cell_type"] = 'Other'

if "subclass_label" in ab_exons.obs.columns:
    ab_exons.obs["broad_cell_type"] = ab_exons.obs["subclass_label"].map(grouped_broad_map_flat).fillna('Other')
else:
    print("Warning: 'subclass_label' column not found in ab_exons. Available columns:", ab_exons.obs.columns.tolist())
    if "cell_type" in ab_exons.obs.columns:
        ab_exons.obs["broad_cell_type"] = ab_exons.obs["cell_type"].map(grouped_broad_map_flat).fillna('Other')
    else:
        ab_exons.obs["broad_cell_type"] = 'Other'

if "subclass_label" in ab_introns.obs.columns:
    ab_introns.obs["broad_cell_type"] = ab_introns.obs["subclass_label"].map(grouped_broad_map_flat).fillna('Other')
else:
    print("Warning: 'subclass_label' column not found in ab_introns. Available columns:", ab_introns.obs.columns.tolist())
    if "cell_type" in ab_introns.obs.columns:  
        ab_introns.obs["broad_cell_type"] = ab_introns.obs["cell_type"].map(grouped_broad_map_flat).fillna('Other')
    else:
        ab_introns.obs["broad_cell_type"] = 'Other'

# Find common broad cell types across ts_adata and ab_exons 
common_broad_types = set(ts_adata.obs["broad_cell_type"]).intersection(set(ab_exons.obs["broad_cell_type"]))
print(f"Common broad cell types: {common_broad_types}")

# Remove 'Other' and 'Neuron' from common_broad_types
common_broad_types = [broad_type for broad_type in common_broad_types if broad_type not in ['Other', 'Neuron']]
print(f"Filtered common broad cell types: {common_broad_types}")

# === Identify indices ===
print("Identifying indices for shared cell types across TS and AB...")
idx_ts = ts_adata.obs["broad_cell_type"].isin(common_broad_types)
idx_ab_exons = ab_exons.obs["broad_cell_type"].isin(common_broad_types)
idx_ab_introns = ab_introns.obs["broad_cell_type"].isin(common_broad_types)

print(f"Number of cells in Tabula Sapiens: {sum(idx_ts)}")
print(f"Number of cells in Allen Brain Exon: {sum(idx_ab_exons)}")
print(f"Number of cells in Allen Brain Intron: {sum(idx_ab_introns)}")

# Print breakdown of each broad cell type in each dataset 
print("\nTabula Sapiens breakdown:")
print(ts_adata.obs[idx_ts]["broad_cell_type"].value_counts())
print("\nAllen Brain Exon breakdown:")
print(ab_exons.obs[idx_ab_exons]["broad_cell_type"].value_counts())

# === Make sparse if needed ===
print("Converting to sparse matrices if needed...")
for adata in [ab_exons, ab_introns, ts_adata]:
    if not sp_issparse(adata.X):
        adata.X = csr_matrix(adata.X)

# === Ensure gene order consistency ===
print("Ensuring gene order consistency...")
# Sort all datasets by gene names to ensure consistent ordering
gene_order = sorted(common_genes)

# Reorder datasets
ab_exons = ab_exons[:, [g in gene_order for g in ab_exons.var_names]].copy()
ab_exons = ab_exons[:, [ab_exons.var_names.tolist().index(g) for g in gene_order if g in ab_exons.var_names]].copy()

ab_introns = ab_introns[:, [g in gene_order for g in ab_introns.var_names]].copy() 
ab_introns = ab_introns[:, [ab_introns.var_names.tolist().index(g) for g in gene_order if g in ab_introns.var_names]].copy()

ts_adata = ts_adata[:, [g in gene_order for g in ts_adata.var_names]].copy()
ts_adata = ts_adata[:, [ts_adata.var_names.tolist().index(g) for g in gene_order if g in ts_adata.var_names]].copy()

# Verify gene order consistency
print("Verifying gene order...")
print(f"ab_exons genes: {ab_exons.shape[1]}")
print(f"ab_introns genes: {ab_introns.shape[1]}")  
print(f"ts_adata genes: {ts_adata.shape[1]}")

# === Pseudobulk generation function ===
def generate_pseudobulk(ge_adata, intron_adata, ts_data, cell_types_to_use):
    """
    Generate pseudobulk counts for specified cell types by summing RAW counts
    
    Args:
        ge_adata (AnnData): Gene expression data (Allen Brain exons)
        intron_adata (AnnData): Intron data (Allen Brain introns)
        ts_data (AnnData): Tabula Sapiens data
        cell_types_to_use (list): List of cell types to process
        
    Returns:
        None: Saves files to disk
    """
    print(f"\n>> Generating pseudobulk counts for {len(cell_types_to_use)} cell types...")
    
    # Check for raw counts layers
    if "raw_counts" in ge_adata.layers and "raw_counts" in intron_adata.layers and "raw_counts" in ts_data.layers:
        print("   ✓ Using 'raw_counts' layer for pseudobulk generation")
        ge_matrix = ge_adata.layers["raw_counts"]
        intron_matrix = intron_adata.layers["raw_counts"]
        ts_matrix = ts_data.layers["raw_counts"]
    else:
        print("   ⚠️ 'raw_counts' layer not found - using .X matrix instead")
        print("   ⚠️ Make sure .X contains raw counts, not normalized values!")
        ge_matrix = ge_adata.X
        intron_matrix = intron_adata.X
        ts_matrix = ts_data.X
    
    # Check gene consistency
    genes_match = (
        np.array_equal(ge_adata.var_names, intron_adata.var_names) and
        np.array_equal(ge_adata.var_names, ts_data.var_names)
    )
    
    if not genes_match:
        raise ValueError("Gene names and order don't match between datasets")
    else:
        print(f"   ✓ All {ge_adata.shape[1]} genes match between datasets")
    
    # Process each cell type
    for cell_type in tqdm(cell_types_to_use, desc="Processing cell types"):
        # Get cells for this cell type
        idx_ab = ge_adata.obs["broad_cell_type"] == cell_type
        idx_ts = ts_data.obs["broad_cell_type"] == cell_type
        idx_intron = intron_adata.obs["broad_cell_type"] == cell_type
        
        num_ab_cells = sum(idx_ab)
        num_ts_cells = sum(idx_ts)
        num_intron_cells = sum(idx_intron)
        
        print(f"   ⚙️ {cell_type}: {num_ab_cells} Allen Brain cells, {num_ts_cells} TMS cells, {num_intron_cells} intron cells")
        
        # Sum RAW counts for each gene across cells of the same type
        if num_ab_cells > 0:
            ab_exons_sum = np.array(ge_matrix[idx_ab].sum(axis=0)).flatten()
        else:
            ab_exons_sum = np.zeros(ge_adata.shape[1])
            
        if num_ts_cells > 0:
            ts_sum = np.array(ts_matrix[idx_ts].sum(axis=0)).flatten()
        else:
            ts_sum = np.zeros(ts_data.shape[1])
            
        if num_intron_cells > 0:
            ab_int_sum = np.array(intron_matrix[idx_intron].sum(axis=0)).flatten()
        else:
            ab_int_sum = np.zeros(intron_adata.shape[1])
        
        # Create pseudobulk dataframe
        safe_cell_type = cell_type.replace(' ', '_').replace('/', '_')
        filename = f"{OUTPUT_DIR}/pseudobulk_{safe_cell_type}.tsv"
        
        # Get gene metadata from the datasets (now guaranteed to have these columns)
        gene_names = ge_adata.var_names.values
        mean_transcript_length = ge_adata.var["mean_transcript_length"].values
        mean_intron_length = ge_adata.var["mean_intron_length"].values
        gene_ids = ge_adata.var["gene_id"].values
        transcript_biotypes = ge_adata.var["transcript_biotype"].values
        
        pseudobulk_df = pd.DataFrame({
            "gene": gene_names,
            "gene_id": gene_ids,
            "transcript_biotype": transcript_biotypes,
            "mean_transcript_length": mean_transcript_length,
            "mean_intron_length": mean_intron_length,
            "ab_exons_sum": ab_exons_sum,
            "ab_introns_sum": ab_int_sum,
            "ts_sum": ts_sum,
            "num_ab_cells": num_ab_cells,
            "num_ts_cells": num_ts_cells,
            "num_intron_cells": num_intron_cells
        })
        
        # Save to file
        pseudobulk_df.to_csv(filename, sep="\t", index=False)
        
    print(f"   ✓ Pseudobulk data generated for {len(cell_types_to_use)} cell types")
    print(f"   ✓ Files saved to: {OUTPUT_DIR}")

# === Save filtered AnnData objects ===
print("\nSaving filtered AnnData objects...")

# Create output filenames with today's date
ab_exons_filtered_file = f"{OUTPUT_DIR}/ab_adata_exons_filtered_{today}.h5ad"
ab_introns_filtered_file = f"{OUTPUT_DIR}/ab_adata_introns_filtered_{today}.h5ad"
ts_adata_filtered_file = f"{OUTPUT_DIR}/tabsap_adata_filtered_{today}.h5ad"

# Save the filtered datasets
print(f"   📁 Saving Allen Brain exons to: {ab_exons_filtered_file}")
ab_exons.write(ab_exons_filtered_file)

print(f"   📁 Saving Allen Brain introns to: {ab_introns_filtered_file}")
ab_introns.write(ab_introns_filtered_file)

print(f"   📁 Saving Tabula Sapiens to: {ts_adata_filtered_file}")
ts_adata.write(ts_adata_filtered_file)

# Save summary information
summary_info = {
    'processing_date': today,
    'total_common_genes': len(common_genes),
    'total_common_cell_types': len(common_broad_types),
    'common_cell_types': common_broad_types,
    'ab_exons_shape': ab_exons.shape,
    'ab_introns_shape': ab_introns.shape,
    'ts_adata_shape': ts_adata.shape,
    'ab_exons_cells_used': sum(ab_exons.obs["broad_cell_type"].isin(common_broad_types)),
    'ab_introns_cells_used': sum(ab_introns.obs["broad_cell_type"].isin(common_broad_types)),
    'ts_adata_cells_used': sum(ts_adata.obs["broad_cell_type"].isin(common_broad_types)),
    'gene_metadata_coverage': len([g for g in ab_exons.var_names if g in gene_info_dict])
}

summary_file = f"{OUTPUT_DIR}/filtering_summary_{today}.json"
print(f"   📋 Saving processing summary to: {summary_file}")

import json
with open(summary_file, 'w') as f:
    json.dump(summary_info, f, indent=2, default=str)

print(f"   ✓ Filtered datasets saved successfully!")
print(f"   ✓ Summary: {len(common_genes)} genes, {len(common_broad_types)} cell types")

# === Run pseudobulk generation ===
if common_broad_types:
    generate_pseudobulk(ab_exons, ab_introns, ts_adata, common_broad_types)
    print("\n=== Script completed successfully! ===")
    print(f"📊 Generated pseudobulk files for {len(common_broad_types)} cell types")
    print(f"📁 Saved filtered AnnData objects with {len(common_genes)} genes")
    print(f"📋 Processing summary saved to: {summary_file}")
else:
    print("No common broad cell types found between datasets!")

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/processed_data
# sbatch --job-name=prep_ge_data \
#       --mem=100G \
#       --partition dev,cpu,bigmem \
#       --output=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/processed_data/prep_ge_data_%j.out \
#       --error=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/processed_data/prep_ge_data_%j.err \
#       --wrap "python /gpfs/commons/home/kisaev/Leaflet-analysis/Human_Splicing_Foundation/GeneExpression/02_generate_metacells.py"