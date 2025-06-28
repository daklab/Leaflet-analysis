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

# Configuration
WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/processed_data"
OUTPUT_DIR = WD  # Save in the same directory
today = datetime.datetime.now().strftime("%Y-%m-%d")

# Cell types to focus on for pseudobulk analysis
COMMON_BROAD_TYPES = ['MICROGLIA', "ENDOTHELIAL CELL", "GLIAL CELL", "PERICYTE"]

def load_datasets():
    """Load processed gene expression and intron datasets"""
    print("\n>> Loading processed datasets...")
    
    try:
        ge_path = f"{WD}/tms_ab_exons_combo_ge_adata_2025-06-24.h5ad"
        intron_path = f"{WD}/ab_adata_introns_2025-06-24.h5ad"
        
        print(f"   ⚙️ Loading combined gene expression data from: {ge_path}")
        ge_adata = ad.read_h5ad(ge_path)
        
        print(f"   ⚙️ Loading intron data from: {intron_path}")
        intron_adata = ad.read_h5ad(intron_path)
        
        print(f"   ✓ Loaded gene expression data: {ge_adata.shape[0]} cells, {ge_adata.shape[1]} genes")
        print(f"   ✓ Loaded intron data: {intron_adata.shape[0]} cells, {intron_adata.shape[1]} genes")
        
        return ge_adata, intron_adata
        
    except Exception as e:
        print(f"   ❌ Error loading datasets: {str(e)}")
        sys.exit(1)

def create_cell_type_mappings():
    """Create dictionaries for cell type standardization"""
    
    # Cell type mappings organized by category for better readability
    cell_type_mappings = {
        # Neuronal cells
        "Sst": "Inhibitory Neurons",
        "Pvalb": "Inhibitory Neurons",
        "Vip": "Inhibitory Neurons",
        "Lamp5": "Inhibitory Neurons",
        "Sncg": "Inhibitory Neurons",
        "Meis2": "Inhibitory Neurons",
        "Ntng1": "Inhibitory Neurons",
        "Pax6": "Inhibitory Neurons",
        "CR": "Inhibitory Neurons",
        
        # Excitatory neuron layer markers
        "L2": "Excitatory Neurons",
        "L3": "Excitatory Neurons",
        "L4": "Excitatory Neurons",
        "L5": "Excitatory Neurons",
        "L6": "Excitatory Neurons",
        
        # Excitatory subregions (hippocampal/entorhinal)
        "CA1": "Excitatory Neurons",
        "CA2": "Excitatory Neurons",
        "CA3": "Excitatory Neurons",
        "DG": "Excitatory Neurons",
        "SUB": "Excitatory Neurons",
        "ProS": "Excitatory Neurons",
        "HATA": "Excitatory Neurons",
        "Mossy": "Excitatory Neurons",
        "PPP": "Excitatory Neurons",
        "RHP": "Excitatory Neurons",
        
        # Excitatory region markers
        "IT": "Excitatory Neurons",
        "CT": "Excitatory Neurons",
        "PT": "Excitatory Neurons",
        "NP": "Excitatory Neurons",
        "CTX": "Excitatory Neurons",
        "ENT": "Excitatory Neurons",
        "PAR": "Excitatory Neurons",
        "POST": "Excitatory Neurons",
        "RSP": "Excitatory Neurons",
        "HPF": "Excitatory Neurons",
        
        # Brain glial cells
        "Micro-PVM": "MICROGLIA",
        "Astro": "GLIAL CELL",
        "Oligo": "GLIAL CELL",
        "VLMC": "VLMCs",
        "Endo": "ENDOTHELIAL CELL",
        "SMC-Peri": "PERICYTE",
        
        # Car3+ special case
        "Car3": "Other non-neuronal (Car3+)",
        
        # Basal cells
        'basal cell of epidermis': 'BASAL CELL',
        'basal cell': 'BASAL CELL',

        # Endothelial cells
        'endothelial cell': 'ENDOTHELIAL CELL',
        'endothelial cell of coronary artery': 'ENDOTHELIAL CELL',
        'endothelial cell of hepatic sinusoid': 'ENDOTHELIAL CELL',
        'aortic endothelial cell': 'ENDOTHELIAL CELL',
        'vein endothelial cell': 'ENDOTHELIAL CELL',
        'endothelial cell of lymphatic vessel': 'ENDOTHELIAL CELL',

        # T cells
        'T cell': 'T CELL',
        'CD4-positive, alpha-beta T cell': 'T CELL',
        'CD8-positive, alpha-beta T cell': 'T CELL',
        'regulatory T cell': 'T CELL',
        'mature NK T cell': 'T CELL',
        'mature alpha-beta T cell': 'T CELL',
        
        # B cells
        'B cell': 'B CELL',
        'immature B cell': 'B CELL',
        'naive B cell': 'B CELL',
        'precursor B cell': 'B CELL',
        'early pro-B cell': 'B CELL',
        'late pro-B cell': 'B CELL',
        'plasma cell': 'B CELL',

        # Fibroblasts
        'fibroblast': 'FIBROBLAST',
        'fibroblast of cardiac tissue': 'FIBROBLAST',
        'fibroblast of lung': 'FIBROBLAST',
        'pulmonary interstitial fibroblast': 'FIBROBLAST',
        'kidney interstitial fibroblast': 'FIBROBLAST',
        'fibrocyte': 'FIBROBLAST',

        # Macrophages 
        'macrophage': 'MACROPHAGE',
        'Kupffer cell': 'MACROPHAGE',  # macrophages in the liver
        'lung macrophage': 'MACROPHAGE',

        # Monocytes
        'monocyte': 'MONOCYTE',
        'classical monocyte': 'MONOCYTE',
        'non-classical monocyte': 'MONOCYTE',
        'intermediate monocyte': 'MONOCYTE',

        # General Immune Cells
        'granulocyte': 'GRANULOCYTE',
        'basophil': 'GRANULOCYTE', 
        'granulocyte monocyte progenitor cell': 'GRANULOCYTE',
        'leukocyte': 'GENERAL IMMUNE CELL',
        'professional antigen presenting cell': 'ANTIGEN PRESENTING CELL',
        'lymphocyte': 'LYMPHOID IMMUNE CELL',
        'NK cell': 'LYMPHOID IMMUNE CELL',
        'myeloid cell': 'MYELOID IMMUNE CELL',
        'myeloid leukocyte': 'MYELOID IMMUNE CELL',
        'granulocytopoietic cell': 'MYELOID IMMUNE CELL',
        'promonocyte': 'MYELOID IMMUNE CELL',
        'thymocyte': 'THYMOCYTE',
        'DN4 thymocyte': 'THYMOCYTE',

        # Neutrophils
        'neutrophil': 'NEUTROPHIL',

        # Dendritic Cells
        'dendritic cell': 'DENDRITIC CELL',
        'plasmacytoid dendritic cell': 'DENDRITIC CELL',
        'myeloid dendritic cell': 'DENDRITIC CELL',

        # Microglia (Brain Immune Cells)
        'microglial cell': 'MICROGLIA',
        
        # Pancreatic cells
        'pancreatic A cell': 'PANCREATIC CELL',
        'pancreatic B cell': 'PANCREATIC CELL',
        'pancreatic D cell': 'PANCREATIC CELL',
        'pancreatic acinar cell': 'PANCREATIC CELL',
        'pancreatic PP cell': 'PANCREATIC CELL',
        'pancreatic ductal cell': 'PANCREATIC CELL',
        'pancreatic stellate cell': 'PANCREATIC CELL',
        
        # Smooth muscle cells
        'smooth muscle cell': 'SMOOTH MUSCLE CELL',
        'bronchial smooth muscle cell': 'SMOOTH MUSCLE CELL',
        'smooth muscle cell of the pulmonary artery': 'SMOOTH MUSCLE CELL',
        'smooth muscle cell of trachea': 'SMOOTH MUSCLE CELL',
        
        # Epithelial cells
        'epithelial cell': 'EPITHELIAL CELL',
        'epidermal cell': 'EPITHELIAL CELL',
        'epithelial cell of large intestine': 'EPITHELIAL CELL',
        'enterocyte of epithelium of large intestine': 'EPITHELIAL CELL',
        'epithelial cell of proximal tubule': 'EPITHELIAL CELL',
        'epithelial cell of thymus': 'EPITHELIAL CELL',
        'bladder urothelial cell': 'EPITHELIAL CELL',
        'basal epithelial cell of tracheobronchial tree': 'EPITHELIAL CELL',
        'luminal epithelial cell of mammary gland': 'EPITHELIAL CELL',

        # Neurons
        'neuron': 'NEURON',
        'medium spiny neuron': 'NEURON',
        'interneuron': 'NEURON',
        'neuronal stem cell': 'NEURON',

        # Glial Cells (excluding microglia)
        'oligodendrocyte': 'GLIAL CELL',
        'oligodendrocyte precursor cell': 'GLIAL CELL',
        'astrocyte': 'GLIAL CELL',
        'Bergmann glial cell': 'GLIAL CELL',
        'ependymal cell': 'GLIAL CELL',

        # Stem cells
        'mesenchymal stem cell': 'STEM CELL',
        'mesenchymal stem cell of adipose': 'STEM CELL',
        'hematopoietic stem cell': 'STEM CELL',
        'neuronal stem cell': 'STEM CELL',
        'intestinal crypt stem cell': 'STEM CELL',
        'keratinocyte stem cell': 'STEM CELL',
        'lymphoid progenitor cell': 'STEM CELL',
        'proerythroblast': 'STEM CELL',
        'megakaryocyte-erythroid progenitor cell': 'STEM CELL',

        # Other specialized cells
        'ventricular myocyte': 'CARDIAC MUSCLE CELL',
        'atrial myocyte': 'CARDIAC MUSCLE CELL',
        'skeletal muscle satellite cell': 'SKELETAL MUSCLE CELL',
        'kidney collecting duct principal cell': 'KIDNEY CELL',
        'kidney collecting duct epithelial cell': 'KIDNEY CELL',
        'kidney interstitial fibroblast': 'FIBROBLAST',
        'mesangial cell': 'KIDNEY CELL',
        'type I pneumocyte': 'LUNG CELL',
        'type II pneumocyte': 'LUNG CELL',
        'club cell of bronchiole': 'LUNG CELL',
        'lung neuroendocrine cell': 'LUNG CELL',
        'ciliated columnar cell of tracheobronchial tree': 'LUNG CELL',
        'respiratory basal cell': 'LUNG CELL',
        'Brush cell of epithelium proper of large intestine': 'INTESTINAL CELL',
        'large intestine goblet cell': 'INTESTINAL CELL',
        'enteroendocrine cell': 'INTESTINAL CELL',
        'stromal cell': 'STROMAL CELL',
        'pericyte cell': 'PERICYTE',
        'brain pericyte': 'PERICYTE',
        'adventitial cell': 'STROMAL CELL',
        'keratinocyte': 'KERATINOCYTE',
        'bulge keratinocyte': 'KERATINOCYTE',
        'hepatocyte': 'HEPATOCYTE',
        'bladder cell': 'BLADDER CELL',
        'secretory cell': 'SECRETORY CELL',
        'endocardial cell': 'ENDOCARDIAL CELL',
        'valve cell': 'VALVE CELL',
        'chondrocyte': 'STROMAL CELL',
        'fenestrated cell': 'FENESTRATED CELL',
        'neuroepithelial cell': 'NEUROEPITHELIAL CELL',
        'kidney loop of Henle ascending limb epithelial cell': 'KIDNEY CELL',
        'mucus secreting cell': 'SECRETORY CELL'
    }

    # Subtissue corrections
    subtissue_corrections = {
        'T cells': 'T-cells',
        'ENDOMUCIN': 'Endomucin',
        'forelimb and hindlimb': 'ForelimbandHindlimb',
        'Liver non-hepato/SCs_st': 'Liver non-hepato/SCs',
        'Skin Anagen': 'Anagen'
    }
    
    return cell_type_mappings, subtissue_corrections

def map_cell_type(cell_label, mappings):
    """
    Map cell types based on pattern matching for more flexible classification
    
    Args:
        cell_label (str): Original cell type label
        mappings (dict): Dictionary of cell type mappings
        
    Returns:
        str: Standardized cell type
    """
    # Handle missing or NaN values
    if not isinstance(cell_label, str) or pd.isna(cell_label):
        return "Unknown"
        
    # Handle special cases with Car3+
    if "Car3" in cell_label:
        return "Other non-neuronal (Car3+)"
    
    # Check for inhibitory neuron markers
    inhibitory_markers = ["Sst", "Pvalb", "Vip", "Lamp5", "Sncg", "Meis2", "Ntng1", "Pax6", "CR"]
    if any(marker in cell_label for marker in inhibitory_markers):
        return "Inhibitory Neurons"
    
    # Check for excitatory neuron markers
    layer_markers = ["L2", "L3", "L4", "L5", "L6"]
    excitatory_regions = ["CA1", "CA2", "CA3", "DG", "SUB", "ProS", "HATA", "Mossy", "PPP", "RHP"]
    other_excitatory = ["IT", "CT", "PT", "NP", "CTX", "ENT", "PAR", "POST", "RSP", "HPF"]
    
    if (any(marker in cell_label for marker in layer_markers) or
        any(marker in cell_label for marker in excitatory_regions) or
        any(marker in cell_label for marker in other_excitatory)):
        return "Excitatory Neurons"
    
    # Check for broad cell type markers
    broad_markers = {
        "Micro-PVM": "MICROGLIA",
        "Astro": "GLIAL CELL",
        "Oligo": "GLIAL CELL",
        "VLMC": "VLMCs",
        "Endo": "ENDOTHELIAL CELL",
        "SMC-Peri": "PERICYTE"
    }
    
    for marker, cell_type in broad_markers.items():
        if marker in cell_label:
            return cell_type
    
    # Check for direct matches in the main dictionary
    if cell_label in mappings:
        return mappings[cell_label]
    
    # Default: return the original label
    return cell_label

def prepare_annotations(ge_adata, intron_adata):
    """
    Standardize cell type annotations and prepare metadata
    
    Args:
        ge_adata (AnnData): Gene expression data
        intron_adata (AnnData): Intron data
        
    Returns:
        tuple: Updated gene expression and intron data
    """
    print("\n>> Standardizing cell type annotations...")
    
    # Create mapping dictionaries
    cell_type_mappings, subtissue_corrections = create_cell_type_mappings()
    
    # Convert cell ontology class to string to avoid errors
    ge_adata.obs['cell_ontology_class'] = ge_adata.obs['cell_ontology_class'].astype(str)
    intron_adata.obs['cell_ontology_class'] = intron_adata.obs['cell_ontology_class'].astype(str)
    
    # Standardize subtissue
    if 'subtissue' in ge_adata.obs.columns:
        ge_adata.obs['subtissue'] = ge_adata.obs['subtissue'].str.strip()
        ge_adata.obs['subtissue_clean'] = ge_adata.obs['subtissue'].replace(subtissue_corrections)
        # Drop the old subtissue 
        ge_adata.obs.drop(columns=['subtissue'], inplace=True)
    
    # Apply cell type mapping to both datasets
    # Use vectorized operation with a lambda function for efficiency
    print("   ⚙️ Mapping cell types to standardized categories...")
    ge_adata.obs['broad_cell_type'] = ge_adata.obs['cell_ontology_class'].apply(
        lambda x: map_cell_type(x, cell_type_mappings)
    )
    intron_adata.obs['broad_cell_type'] = intron_adata.obs['cell_ontology_class'].apply(
        lambda x: map_cell_type(x, cell_type_mappings)
    )
    
    # Fix tissue labels for Allen Brain data
    # Use the dataset name consistently - 'allen_brain_exons' instead of 'AB'
    print("   ⚙️ Standardizing tissue labels...")
    ab_mask = ge_adata.obs["dataset"] == "allen_brain_exons"
    ge_adata.obs.loc[ab_mask, "tissue"] = "Brain_Non-Myeloid"
    
    # Set brain myeloid tissue for microglia
    ab_microglia_mask = ab_mask & (ge_adata.obs["broad_cell_type"] == "MICROGLIA")
    ge_adata.obs.loc[ab_microglia_mask, "tissue"] = "Brain_Myeloid"
    
    # Get counts of each cell type 
    cell_type_counts = ge_adata.obs['broad_cell_type'].value_counts()
    print(f"   ✓ Found {len(cell_type_counts)} distinct broad cell types")
    print(f"   ✓ Top 5 most common cell types:")
    for cell_type, count in cell_type_counts.head(5).items():
        print(f"      - {cell_type}: {count} cells")
    
    return ge_adata, intron_adata

def generate_pseudobulk(ge_adata, intron_adata, cell_types_to_use):
    """
    Generate pseudobulk counts for specified cell types by summing RAW counts
    
    Args:
        ge_adata (AnnData): Gene expression data
        intron_adata (AnnData): Intron data
        cell_types_to_use (list): List of cell types to process
        
    Returns:
        None: Saves files to disk
    """
    print(f"\n>> Generating pseudobulk counts for {len(cell_types_to_use)} cell types...")
    
    # IMPORTANT: We need to use raw counts for pseudobulk generation
    # since downstream analysis will perform normalization
    if "raw_counts" in ge_adata.layers and "raw_counts" in intron_adata.layers:
        print("   ✓ Using 'raw_counts' layer for pseudobulk generation")
        ge_matrix = ge_adata.layers["raw_counts"]
        intron_matrix = intron_adata.layers["raw_counts"]
    else:
        print("   ⚠️ 'raw_counts' layer not found - using .X matrix instead")
        print("   ⚠️ Make sure .X contains raw counts, not normalized values!")
        ge_matrix = ge_adata.X
        intron_matrix = intron_adata.X
    
    # Check that genes match between datasets
    gene_match = all(ge_adata.var["gene_name"].values == intron_adata.var["gene_name"].values)
    if not gene_match:
        print("   ⚠️ Warning: Gene names don't match exactly between datasets")
        # Find common genes
        common_genes = set(ge_adata.var["gene_name"]) & set(intron_adata.var["gene_name"])
        print(f"   ⚙️ Using {len(common_genes)} common genes for pseudobulk analysis")
    else:
        print(f"   ✓ All {ge_adata.shape[1]} genes match between datasets")
    
    # Process each cell type
    for cell_type in tqdm(cell_types_to_use, desc="Processing cell types"):
        # Get cells for this cell type
        idx_all = ge_adata.obs["broad_cell_type"] == cell_type
        
        # Get counts by dataset
        idx_ab = idx_all & (ge_adata.obs["dataset"] == "allen_brain_exons")
        num_ab_cells = sum(idx_ab)
        
        idx_ts = idx_all & (ge_adata.obs["dataset"] == "tabula_muris_senis")
        num_ts_cells = sum(idx_ts)
        
        # Get counts for introns
        idx_intron = intron_adata.obs["broad_cell_type"] == cell_type
        num_intron_cells = sum(idx_intron)
        
        print(f"   ⚙️ {cell_type}: {num_ab_cells} Allen Brain cells, {num_ts_cells} TMS cells, {num_intron_cells} intron cells")
        
        # Sum RAW counts for each gene across cells of the same type and dataset
        if num_ab_cells > 0:
            ab_exons_sum = np.array(ge_matrix[idx_ab].sum(axis=0)).flatten()
        else:
            ab_exons_sum = np.zeros(ge_adata.shape[1])
            
        if num_ts_cells > 0:
            ts_sum = np.array(ge_matrix[idx_ts].sum(axis=0)).flatten()
        else:
            ts_sum = np.zeros(ge_adata.shape[1])
            
        if num_intron_cells > 0:
            ab_int_sum = np.array(intron_matrix[idx_intron].sum(axis=0)).flatten()
        else:
            ab_int_sum = np.zeros(intron_adata.shape[1])
        
        # Create and save pseudobulk data - use underscore in cell type names for consistency
        safe_cell_type = cell_type.replace(' ', '_').replace('/', '_')
        filename = f"{OUTPUT_DIR}/pseudobulk_{safe_cell_type}.tsv"
        
        # Create pseudobulk dataframe
        pseudobulk_df = pd.DataFrame({
            "gene": ge_adata.var["gene_name"].values,
            "mean_transcript_length": ge_adata.var["mean_transcript_length"].values,
            "mean_intron_length": ge_adata.var["mean_intron_length"].values,
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

# Execute processing pipeline
print("="*80)
print(f"GENE EXPRESSION METACELL GENERATION - {today}")
print("="*80)

# Load data
ge_adata, intron_adata = load_datasets()

# Prepare annotations
ge_adata, intron_adata = prepare_annotations(ge_adata, intron_adata)

# Generate pseudobulk counts
generate_pseudobulk(ge_adata, intron_adata, COMMON_BROAD_TYPES)

print("\n" + "="*80)
print("PIPELINE COMPLETE")
print(f"Pseudobulk data generated for {len(COMMON_BROAD_TYPES)} cell types")
print("="*80)

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/processed_data
# sbatch --job-name=prep_ge_data --mem=100G --partition dev,cpu,bigmem --wrap "python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/GeneExpression/02_generate_metacells.py"
