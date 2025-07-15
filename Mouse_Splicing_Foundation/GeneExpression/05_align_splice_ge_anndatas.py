#!/usr/bin/env python
"""
Align Splicing and Gene Expression Data - Mouse Splicing Foundation

This script:
1. Loads splicing data from ATSEmapper and gene expression data
2. Aligns cells between the two datasets
3. Standardizes cell type annotations
4. Ensures both datasets have cells in the same order
5. Adds derived information like library size and centered PSI values (via raw data)
6. Saves aligned datasets for downstream analysis
"""

import os
import sys
import pandas as pd 
import numpy as np
import anndata as ad
import datetime
import traceback
from scipy.sparse import coo_matrix, csr_matrix
from tqdm import tqdm

# Define module paths for custom modules
src_path = "/gpfs/commons/home/kisaev/Leaflet-private/src/"
if src_path not in sys.path:
    sys.path.append(src_path)

# Import custom modules
import BetaDirichletFactor.waypoints as wayp  # for getting centered sparse junction usage ratios 

# Configuration
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/062025"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}", flush=True)

# Input file paths
SPLICE_INPUT = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper/junction_processing_20250622/anndatas/merged_anndata.h5ad"
GE_INPUT = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/processed_data/adjusted_gene_expression/Combined_adjusted_GeneExpression_2025-06-24.h5ad"
ATSE_FILE = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper/ATSE_files/MOUSE_FOUNDATION_ATSE_FILE_unanno_also_2025-07-01_00-02-00.txt.gz"

METADATA_FILE = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/metadata_mouse_metadata_combined.csv"

# Cell type mapping functions
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

def load_datasets():
    """Load splicing, gene expression and metadata datasets"""
    print("\n>> Loading datasets...")
    
    try:
        # Load splicing data
        print("   ⚙️ Loading splicing AnnData...")
        splice_adata = ad.read_h5ad(SPLICE_INPUT)
        splice_adata.obs.reset_index(drop=True, inplace=True)
        splice_adata.obs["cell_id_index"] = splice_adata.obs.index 
        print(f"   ✓ Loaded splicing data: {splice_adata.shape[0]} cells")
        
        # Load ATSE file
        print("   ⚙️ Loading ATSE information...")
        atses = pd.read_csv(ATSE_FILE, sep="\t")
        print(f"   ✓ Loaded ATSE info: {len(atses['event_id'].unique())} unique events")
        
        # Load gene expression data
        print("   ⚙️ Loading gene expression AnnData...")
        ge_adata = ad.read_h5ad(GE_INPUT)
        ge_adata.obs.reset_index(drop=True, inplace=True)
        ge_adata.obs["cell_id_index"] = ge_adata.obs.index 
        print(f"   ✓ Loaded gene expression data: {ge_adata.shape[0]} cells")
        
        # Load metadata
        print("   ⚙️ Loading metadata...")
        metadata = pd.read_csv(METADATA_FILE)
        print(f"   ✓ Loaded metadata for {len(metadata)} cells")
        
        return splice_adata, ge_adata, atses, metadata
        
    except Exception as e:
        print(f"   ❌ Error loading datasets: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def process_splicing_data(splice_adata, metadata):
    """Process and clean splicing data with metadata"""
    print("\n>> Processing splicing data...")
    
    # Drop existing metadata columns if they exist
    drop_cols = ["age", "cell_ontology_class", "mouse.id", "sex", "subtissue", "tissue"]
    existing_cols = [col for col in drop_cols if col in splice_adata.obs.columns]
    if existing_cols:
        print(f"   ⚙️ Dropping existing metadata columns: {existing_cols}")
        splice_adata.obs = splice_adata.obs.drop(columns=existing_cols)
    else:
        print("   ⚙️ No metadata columns to reset")
    
    # Filter to cells in metadata
    splice_adata = splice_adata[splice_adata.obs["cell_id"].isin(metadata["cell_id"])]
    print(f"   ✓ Filtered to {splice_adata.shape[0]} cells with metadata")
    
    # Validate cell IDs
    assert splice_adata.obs['cell_id'].is_unique, "splice_adata.obs['cell_id'] is not unique"
    assert metadata['cell_id'].is_unique, "metadata['cell_id'] is not unique"
    
    # Set index and merge metadata
    splice_adata.obs = splice_adata.obs.set_index('cell_id')
    metadata = metadata.set_index('cell_id')
    assert metadata.index.is_unique, "Non-unique cell IDs in metadata!"
    
    # Find common cells and subset
    common_cells = splice_adata.obs.index.intersection(metadata.index)
    print(f"   ✓ Found {len(common_cells)} common cells between splicing data and metadata")
    
    splice_adata = splice_adata[common_cells].copy()
    assert splice_adata.n_obs == len(common_cells)
    
    # Merge metadata with splicing observations
    merged_obs = splice_adata.obs.merge(metadata, left_index=True, right_index=True, how="left", validate="one_to_one")
    assert merged_obs.shape[0] == splice_adata.n_obs
    splice_adata.obs = merged_obs
    
    # Add additional annotations
    splice_adata.obs["cell_id_index"] = range(len(splice_adata.obs))
    splice_adata.obs["dataset"] = "TMS"
    splice_adata.obs.loc[splice_adata.obs["age"] == "2m", "dataset"] = "AB"
    print(f"   ✓ Dataset distribution: {dict(splice_adata.obs.dataset.value_counts())}")
    
    # Update cell IDs and names
    splice_adata.obs["cell_name"] = splice_adata.obs["tissue"]
    splice_adata.obs["cell_id"] = splice_adata.obs.index
    splice_adata.obs.loc[splice_adata.obs["dataset"] == "TMS", "cell_name"] = splice_adata.obs.loc[splice_adata.obs["dataset"] == "TMS", "cell_id"]
    splice_adata.obs.reset_index(drop=True, inplace=True)
    
    # Clean cell IDs for TMS cells
    splice_adata.obs["cell_clean"] = splice_adata.obs["cell_id"]
    is_tms = splice_adata.obs["dataset"] == "TMS"
    splice_adata.obs.loc[is_tms, "cell_clean"] = (
        splice_adata.obs.loc[is_tms, "cell_id"]
        .str.replace(r'-(?=.*_)', '_', regex=True)
        .str.split('_')
        .str[:2]
        .str.join('_')
    )
    
    splice_adata.obs["cell_id"] = splice_adata.obs["cell_name"]
    print(f"   ✓ Splicing data processing complete")
    
    return splice_adata

def standardize_cell_types(splice_adata, ge_adata):
    """Apply standardized cell type annotations to both datasets"""
    print("\n>> Standardizing cell type annotations...")
    
    # Get cell type mappings
    cell_type_mappings, _ = create_cell_type_mappings()

    # Print some sample data before standardization
    print("Before standardization:")
    print("Splicing data sample:", flush=True)
    print(splice_adata.obs[["cell_id", "cell_ontology_class"]].head(), flush=True)
    print("Gene expression data sample:", flush=True)
    print(ge_adata.obs[["cell_id", "cell_ontology_class"]].head(), flush=True)
        
    # Apply cell type mappings to both datasets
    splice_adata.obs["broad_cell_type"] = splice_adata.obs["cell_ontology_class"].apply(
        lambda x: map_cell_type(x, cell_type_mappings)
    )
    ge_adata.obs["broad_cell_type"] = ge_adata.obs["cell_ontology_class"].apply(
        lambda x: map_cell_type(x, cell_type_mappings)
    )
    
    # Print samples after standardization to verify
    print("\nAfter standardization:")
    print("Splicing data with standardized cell types:", flush=True)
    print(splice_adata.obs[["cell_id", "cell_ontology_class", "broad_cell_type"]].head(), flush=True)
    print("Gene expression data with standardized cell types:", flush=True)
    print(ge_adata.obs[["cell_id", "cell_ontology_class", "broad_cell_type"]].head(), flush=True)

    print(f"   ✓ Top splicing cell types: {dict(splice_adata.obs['broad_cell_type'].value_counts().head(5))}")
    print(f"   ✓ Top gene expression cell types: {dict(ge_adata.obs['broad_cell_type'].value_counts().head(5))}")
    
    # Update tissue labels based on cell types (moved from process_splicing_data)
    splice_adata.obs.loc[splice_adata.obs["dataset"] == "AB", "tissue"] = "Brain_Non-Myeloid"
    is_ab_microglia = (splice_adata.obs["dataset"] == "AB") & (splice_adata.obs["broad_cell_type"] == "microglial cell")
    splice_adata.obs.loc[is_ab_microglia, "tissue"] = "Brain_Myeloid"
    
    # Add sequencing technology information
    splice_adata.obs["seqtech"] = "single_nuclei"
    splice_adata.obs.loc[splice_adata.obs["dataset"] == "TMS", "seqtech"] = "single_cell"
    print(f"   ✓ Technology distribution: {dict(splice_adata.obs.seqtech.value_counts())}")
    
    return splice_adata, ge_adata

def extract_cell_name(cell_id):
    """Extract standardized cell name from cell ID"""
    # First, replace dots with underscores to standardize
    if not isinstance(cell_id, str):
        return str(cell_id)  # Handle non-string input
        
    cell_id = cell_id.replace(".", "_")
    parts = cell_id.split("_")
    if len(parts) >= 2:
        return parts[0] + "_" + parts[1]
    else:
        return cell_id  # fallback if somehow weird

def align_datasets(splice_adata, ge_adata):
    """Align splicing and gene expression datasets"""
    print("\n>> Aligning datasets...")
    
    # Apply cell name standardization to gene expression data
    print("   ⚙️ Standardizing cell names...")
    ge_adata.obs["cell_name"] = ge_adata.obs["cell_id"]
    
    # Only apply cleaning to Tabula Muris Senis cells
    mask = ge_adata.obs["dataset"] == "tabula_muris_senis"
    ge_adata.obs.loc[mask, "cell_name"] = ge_adata.obs.loc[mask, "cell_id"].apply(extract_cell_name)
    
    # Fix cell_name column in splice_adata to match
    mask = splice_adata.obs["dataset"] == "TMS"
    splice_adata.obs.loc[mask, "cell_name"] = splice_adata.obs.loc[mask, "cell_clean"]
    
    # Replace cell_id with cell_name in both datasets
    splice_adata.obs["cell_id"] = splice_adata.obs["cell_name"]
    ge_adata.obs["cell_id"] = ge_adata.obs["cell_name"]
    
    # Find common cells between datasets
    common_cells = pd.Index(ge_adata.obs['cell_name']).intersection(pd.Index(splice_adata.obs['cell_name']))
    print(f"   ✓ Found {len(common_cells)} cells common to both datasets")
    
    # Create masks for each dataset
    ge_mask = pd.Series(ge_adata.obs['cell_name'].isin(common_cells).values)
    splice_mask = pd.Series(splice_adata.obs['cell_name'].isin(common_cells).values)
    
    # Get indices that match common cells
    ge_idx = np.where(ge_mask)[0]
    splice_idx = np.where(splice_mask)[0]
    
    # Create mapping for consistent ordering
    print("   ⚙️ Creating consistent cell ordering...")
    common_cells_sorted = np.sort(common_cells)
    order_map = {cell: i for i, cell in enumerate(common_cells_sorted)}
    
    # Get sorted indices
    ge_ordered = np.array([order_map[ge_adata.obs['cell_name'].iloc[i]] for i in ge_idx])
    splice_ordered = np.array([order_map[splice_adata.obs['cell_name'].iloc[i]] for i in splice_idx])
    
    # Sort indices
    ge_sorted_idx = ge_idx[np.argsort(ge_ordered)]
    splice_sorted_idx = splice_idx[np.argsort(splice_ordered)]
    
    # Apply indices to slice data
    print("   ⚙️ Applying consistent ordering to both datasets...")
    ge_adata_view = ge_adata[ge_sorted_idx].copy()  # Create explicit copies
    splice_adata_view = splice_adata[splice_sorted_idx].copy()
    
    # Update cell identifiers
    ge_adata_view.obs['cell_id'] = ge_adata_view.obs['cell_name'].values
    splice_adata_view.obs['cell_id'] = splice_adata_view.obs['cell_name'].values
    
    ge_adata_view.obs['cell_id_index'] = np.arange(len(ge_adata_view))
    splice_adata_view.obs['cell_id_index'] = np.arange(len(splice_adata_view))
    
    # Add library size information (via length normalized counts) to gene expression data
    print("   ⚙️ Computing library size information...")
    library_size = np.asarray(ge_adata_view.layers["length_norm"].sum(axis=1)).flatten()
    ge_adata_view.obs["library_size"] = library_size  # 1D column 
    ge_adata_view.obsm["X_library_size"] = library_size[:, np.newaxis]  # 2D array
    
    # Verify alignment of cells
    print("   ⚙️ Verifying cell alignment...")
    verify_cells = np.all(ge_adata_view.obs['cell_id'].values == splice_adata_view.obs['cell_id'].values)
    if verify_cells:
        print("   ✓ Cell IDs verified to be in identical order")
    else:
        print("   ❌ ERROR: Cell IDs are not aligned properly")
        sys.exit(1)
    
    return splice_adata_view, ge_adata_view

def compute_centered_psi(splice_adata):
    """Compute centered PSI values from junction-level splicing data"""
    print("\n>> Computing centered PSI values...")
    
    try:
        # Extract the junction and cluster count matrices
        print("   ⚙️ Extracting count matrices...")
        junction_counts = splice_adata.layers["cell_by_junction_matrix"].tocoo()
        cluster_counts = splice_adata.layers["cell_by_cluster_matrix"].tocoo()
        
        # Compute centered PSI values using the existing function
        print("   ⚙️ Computing centered PSI values...")
        psi = wayp.calculate_centered_psi(junction_counts, cluster_counts)
        
        # Convert to CSR format and store in the AnnData object
        print("   ⚙️ Storing centered PSI values...")
        splice_adata.layers["junc_ratio"] = csr_matrix(psi)
        
        print(f"   ✓ Successfully computed centered PSI values")
        
        return splice_adata
        
    except Exception as e:
        print(f"   ❌ Error computing centered PSI values: {str(e)}")
        traceback.print_exc()
        return splice_adata

def save_aligned_data(splice_adata, ge_adata):
    """Save aligned datasets to disk"""
    print("\n>> Saving aligned datasets...")
    
    try:
        # Generate output filenames with timestamp
        splicing_output = os.path.join(OUTPUT_DIR, f"aligned_splicing_data_{timestamp}.h5ad")
        ge_output = os.path.join(OUTPUT_DIR, f"aligned_gene_expression_data_{timestamp}.h5ad")
        
        # Save the AnnData objects
        print(f"   ⚙️ Saving splicing data to {splicing_output}...")
        splice_adata.write(splicing_output)
        
        print(f"   ⚙️ Saving gene expression data to {ge_output}...")
        ge_adata.write(ge_output)
        
        print(f"   ✓ Successfully saved aligned datasets")
        
        # Save dataset summary
        summary_file = os.path.join(OUTPUT_DIR, f"dataset_summary_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write(f"# Mouse Splicing Foundation - Aligned Dataset Summary\n")
            f.write(f"# Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Splicing Dataset\n")
            f.write(f"Number of cells: {splice_adata.n_obs}\n")
            f.write(f"Number of junctions: {splice_adata.n_vars}\n")
            f.write(f"Cell type distribution: {dict(splice_adata.obs['broad_cell_type'].value_counts())}\n")
            f.write(f"Dataset distribution: {dict(splice_adata.obs['dataset'].value_counts())}\n\n")
            
            f.write(f"## Gene Expression Dataset\n")
            f.write(f"Number of cells: {ge_adata.n_obs}\n")
            f.write(f"Number of genes: {ge_adata.n_vars}\n")
            f.write(f"Cell type distribution: {dict(ge_adata.obs['broad_cell_type'].value_counts())}\n")
            f.write(f"Dataset distribution: {dict(ge_adata.obs['dataset'].value_counts())}\n")
        
        print(f"   ✓ Saved dataset summary to {summary_file}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error saving datasets: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Main execution function"""
    print("\n========================================")
    print("Align Splicing and Gene Expression Data")
    print("Mouse Splicing Foundation")
    print("========================================\n")
    
    # Load source datasets
    splice_adata, ge_adata, atses, metadata = load_datasets()
    
    # Process splicing data with metadata
    splice_adata = process_splicing_data(splice_adata, metadata)
    
    # Standardize cell type annotations
    splice_adata, ge_adata = standardize_cell_types(splice_adata, ge_adata)
    
    # Align datasets
    splice_adata, ge_adata = align_datasets(splice_adata, ge_adata)
    
    # Compute centered PSI values
    splice_adata = compute_centered_psi(splice_adata)
    
    # Save aligned datasets
    save_aligned_data(splice_adata, ge_adata)
    
    print("\n========================================")
    print("Data alignment and processing complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("========================================\n")

if __name__ == "__main__":
    main()

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION
# sbatch --mem=500G -p cpu,bigmem --wrap "python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/GeneExpression/05_align_splice_ge_anndatas.py"