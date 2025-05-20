#!/usr/bin/env python
"""
Apply Regression Model to Allen Brain Dataset - Mouse Splicing Foundation

This script:
1. Loads preprocessed gene expression and intron data
2. Applies a pre-trained linear model to predict total spliced counts
3. Harmonizes cell type annotations across datasets
4. Outputs a combined dataset with adjusted gene expression values

This script bridges single-nucleus data (Allen Brain) to estimated single-cell data
(Tabula Muris Senis) by applying a regression model to account for differences in
spliced products between protocols.
"""

import os
import sys
import pickle
import datetime
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix
from tqdm import tqdm

# Set up logging and configuration
today = datetime.datetime.now().strftime("%Y-%m-%d")
print("="*80)
print(f"APPLY REGRESSION MODEL TO GENE EXPRESSION DATA - {today}")
print("="*80)

# Configuration
WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/processed_data"
OUTDIR = os.path.join(WD, "adjusted_gene_expression")
os.makedirs(OUTDIR, exist_ok=True)

# Input files
LINEAR_MODEL_PATH = "/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/GeneExpression/linear_log_norm_ts_model.pkl"
GE_ADATA_PATH = f"{WD}/tms_ab_exons_combo_ge_adata_{today}.h5ad"
INTRON_ADATA_PATH = f"{WD}/ab_adata_introns_{today}.h5ad"

def load_datasets():
    """Load processed gene expression and intron datasets"""
    print("\n>> Loading processed datasets...")
    
    try:
        # Try with today's date first
        ge_adata_path = GE_ADATA_PATH
        intron_adata_path = INTRON_ADATA_PATH
        
        # Fall back to hardcoded date if today's files don't exist
        if not os.path.exists(ge_adata_path):
            print(f"   ⚠️ Could not find file with current date: {ge_adata_path}")
            ge_adata_path = f"{WD}/tms_ab_exons_combo_ge_adata_2025-05-12.h5ad"
            print(f"   ⚙️ Trying alternative path: {ge_adata_path}")
            
        if not os.path.exists(intron_adata_path):
            print(f"   ⚠️ Could not find file with current date: {intron_adata_path}")
            intron_adata_path = f"{WD}/ab_adata_introns_2025-05-12.h5ad"
            print(f"   ⚙️ Trying alternative path: {intron_adata_path}")
        
        print(f"   ⚙️ Loading combined gene expression data from: {ge_adata_path}")
        ge_adata = ad.read_h5ad(ge_adata_path)
        
        print(f"   ⚙️ Loading intron data from: {intron_adata_path}")
        intron_adata = ad.read_h5ad(intron_adata_path)
        
        print(f"   ✓ Loaded gene expression data: {ge_adata.shape[0]} cells, {ge_adata.shape[1]} genes")
        print(f"   ✓ Loaded intron data: {intron_adata.shape[0]} cells, {intron_adata.shape[1]} genes")
        
        return ge_adata, intron_adata
        
    except Exception as e:
        print(f"   ❌ Error loading datasets: {str(e)}")
        sys.exit(1)

def load_linear_model(model_path):
    """Load the pre-trained linear regression model"""
    print("\n>> Loading linear regression model...")
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        print("   ✓ Model loaded successfully")
        print("   ✓ Model summary:")
        print(model.summary())
        
        return model
    except Exception as e:
        print(f"   ❌ Error loading model: {str(e)}")
        sys.exit(1)

def create_cell_type_mappings():
    """Create dictionaries for mapping cell types to standardized categories"""
    
    # Cell type mappings organized by category
    cell_type_mappings = {
        # Broad cell type map (previously separate)
        "Micro-PVM": "MICROGLIA",
        "Astro": "GLIAL CELL",
        "Oligo": "GLIAL CELL",
        "VLMC": "VLMCs",
        "Endo": "ENDOTHELIAL CELL",
        "SMC-Peri": "PERICYTE",
        
        # Car3+ special case
        "Car3": "Other non-neuronal (Car3+)",
        
        # Inhibitory neuron markers
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
    Map cell type labels to standardized categories
    
    Args:
        cell_label (str): Original cell type label
        mappings (dict): Dictionary of standardized mappings
        
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
    Standardize cell annotations and prepare metadata
    
    Args:
        ge_adata (AnnData): Gene expression data
        intron_adata (AnnData): Intron data
        
    Returns:
        tuple: Updated gene expression and intron AnnData objects
    """
    print("\n>> Standardizing cell annotations...")
    
    # Create mapping dictionaries
    cell_type_mappings, subtissue_corrections = create_cell_type_mappings()
    
    # Ensure cell ontology class is string type
    ge_adata.obs['cell_ontology_class'] = ge_adata.obs['cell_ontology_class'].astype(str)
    intron_adata.obs['cell_ontology_class'] = intron_adata.obs['cell_ontology_class'].astype(str)
    
    # Clean subtissue information if it exists
    if 'subtissue' in ge_adata.obs.columns:
        ge_adata.obs['subtissue'] = ge_adata.obs['subtissue'].str.strip()
        ge_adata.obs['subtissue_clean'] = ge_adata.obs['subtissue'].replace(subtissue_corrections)
        # Drop the old subtissue 
        ge_adata.obs.drop(columns=['subtissue'], inplace=True)
    
    # Apply cell type mapping to both datasets
    print("   ⚙️ Mapping cell types to standardized categories...")
    ge_adata.obs['broad_cell_type'] = ge_adata.obs['cell_ontology_class'].apply(
        lambda x: map_cell_type(x, cell_type_mappings)
    )
    intron_adata.obs['broad_cell_type'] = intron_adata.obs['cell_ontology_class'].apply(
        lambda x: map_cell_type(x, cell_type_mappings)
    )
    
    # Set tissue labels for Allen Brain data
    print("   ⚙️ Setting tissue labels for Allen Brain data...")
    ab_mask = ge_adata.obs["dataset"] == "allen_brain_exons"
    ge_adata.obs.loc[ab_mask, "tissue"] = "Brain_Non-Myeloid"
    
    # Set myeloid-specific tissue labels
    ab_microglia_mask = ab_mask & (ge_adata.obs["broad_cell_type"] == "MICROGLIA")
    ge_adata.obs.loc[ab_microglia_mask, "tissue"] = "Brain_Myeloid"
    
    print(f"   ✓ Standardized annotations for {ge_adata.shape[0]} cells")
    
    return ge_adata, intron_adata

def apply_linear_model(ge_adata, intron_adata, model):
    """
    Apply linear model to predict total spliced counts for Allen Brain data
    
    Args:
        ge_adata (AnnData): Gene expression data
        intron_adata (AnnData): Intron data
        model: Statsmodels linear model
        
    Returns:
        AnnData: Updated combined dataset with predicted values
    """
    print("\n>> Applying linear model to estimate total spliced counts...")
    
    # Extract model coefficients
    intercept = model.params["const"]
    coef_exons = model.params["exons"]
    coef_introns = model.params["introns"]
    
    print(f"   ✓ Model coefficients: intercept={intercept:.4f}, exons={coef_exons:.4f}, introns={coef_introns:.4f}")
    
    # Create a copy of the gene expression data
    combined_adata = ge_adata.copy()
    
    # Identify Allen Brain cells
    ab_mask = combined_adata.obs["dataset"] == "allen_brain_exons"
    ab_cells = combined_adata[ab_mask]
    
    print(f"   ⚙️ Applying model to {sum(ab_mask)} Allen Brain cells...")
    
    # Extract normalized expression matrices
    log_exons = ab_cells.layers["log_norm"]
    log_introns = intron_adata.layers["log_norm"]
    
    # Apply the linear regression equation: intercept + coef_exons * exons + coef_introns * introns
    log_pred_tot = log_exons.multiply(coef_exons) + log_introns.multiply(coef_introns)
    log_pred_tot_with_intercept = log_pred_tot.copy()
    log_pred_tot_with_intercept.data += intercept
    
    # Create the new layer for all cells (initialized with log_norm values)
    combined_adata.layers["predicted_log_norm_tms"] = combined_adata.layers["log_norm"].copy()
    
    # Update Allen Brain cells with predicted values
    combined_adata.layers["predicted_log_norm_tms"][ab_mask] = log_pred_tot_with_intercept
    
    print(f"   ✓ Applied model successfully")
    print(f"   ✓ Final dataset contains {combined_adata.n_obs} cells:")
    print(f"      - Allen Brain cells: {sum(ab_mask)}")
    print(f"      - Tabula Muris Senis cells: {sum(~ab_mask)}")
    
    return combined_adata

def save_dataset(adata, filename, compression_method="lzf"):
    """
    Save AnnData object with optimized compression settings
    
    Args:
        adata (AnnData): Object to save
        filename (str): Output filename
        compression_method (str): Compression method to use:
            - 'lzf': Fast but less compression (default)
            - 'gzip': Slower but better compression
            - None: No compression
        
    Returns:
        bool: Success status
    """
    try:
        print(f"   ⚙️ Saving dataset with '{compression_method}' compression...")
        start_time = datetime.datetime.now()
        
        # Save the AnnData object with specified compression
        # LZF doesn't accept compression options
        if compression_method == "lzf":
            adata.write_h5ad(filename, compression=compression_method)
        elif compression_method in ["gzip", "zlib"]:
            adata.write_h5ad(filename, compression=compression_method, compression_opts=4)
        else:
            adata.write_h5ad(filename, compression=compression_method)
        
        # Calculate file size
        file_size_bytes = os.path.getsize(filename)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # Calculate elapsed time
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        
        print(f"   ✓ Successfully saved to {filename}")
        print(f"   ✓ File size: {file_size_mb:.2f} MB")
        print(f"   ✓ Save time: {elapsed_time:.2f} seconds")
        
        return True
    except Exception as e:
        print(f"   ❌ Error saving {filename}: {str(e)}")
        return False
    
# Main execution flow
try:
    # Load datasets
    ge_adata, intron_adata = load_datasets()
    
    # Load the pre-trained linear model
    model = load_linear_model(LINEAR_MODEL_PATH)
    
    # Standardize cell type annotations
    ge_adata, intron_adata = prepare_annotations(ge_adata, intron_adata)
    
    # Apply the linear model to predict total spliced counts
    combined_adata = apply_linear_model(ge_adata, intron_adata, model)
    
    # Save the combined dataset
    print("\n>> Saving adjusted gene expression data...")
    output_path = os.path.join(OUTDIR, f"Combined_adjusted_GeneExpression_{today}.h5ad")
    save_dataset(combined_adata, output_path)
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print(f"Adjusted gene expression data saved to: {OUTDIR}")
    print("="*80)
    
except Exception as e:
    print("\n" + "="*80)
    print("PROCESSING FAILED")
    print(f"Error: {str(e)}")
    print("="*80)
    sys.exit(1)

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION
# sbatch --mem=250G -p cpu,dev --wrap "python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/GeneExpression/04_apply_regression_model.py"