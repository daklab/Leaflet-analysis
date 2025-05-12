import os

# Get the current working directory
current_dir = os.getcwd()
print("Current working directory:", current_dir)

import pandas as pd
import scanpy as sc
import anndata as ad
from tqdm import tqdm
import matplotlib.pyplot as plt # import matplotlib to visualize our qc metrics
import subprocess
import sys
import seaborn as sns
import numpy as np
from scipy.sparse import csr_matrix
import scanpy.external as sce
from sklearn.metrics import silhouette_score
import datetime
from collections import defaultdict
import scipy.sparse as sp
import gffutils 
import gffutils 
import pickle 

# Load gene gtf file for gene length 
gtf_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/genome_files/gencode.vM19/genes/genes.gtf"
db_file="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/genome_files/GENCODE_vM19"

today = datetime.datetime.now()
today = today.strftime("%Y-%m-%d")

# Load in the gene expression dataset 
WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/processed_data" 
ge_adata = ad.read_h5ad(f"{WD}/tms_ab_exons_combo_ge_adata_2025-04-29.h5ad")
intron_adata = ad.read_h5ad(f"{WD}/ab_adata_introns_2025-04-29.h5ad")

# Add better cell type and tissue labels
# Dictionary to consolidate duplicates

# Complete cell type classification dictionary
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

# Subtissue corrections remain separate as they serve a different purpose
subtissue_corrections = {
    'T cells': 'T-cells',
    'ENDOMUCIN': 'Endomucin',
    'forelimb and hindlimb': 'ForelimbandHindlimb',
    'Liver non-hepato/SCs_st': 'Liver non-hepato/SCs',
    'Skin Anagen': 'Anagen'
}

# Remove any leading/trailing whitespace from subtissue values
ge_adata.obs['subtissue'] = ge_adata.obs['subtissue'].str.strip()
ge_adata.obs['subtissue_clean'] = ge_adata.obs['subtissue'].replace(subtissue_corrections)

# Drop the old subtissue 
ge_adata.obs.drop(columns=['subtissue'], inplace=True)

# Add new cell type groupings 
ge_adata.obs['cell_ontology_class'] = ge_adata.obs['cell_ontology_class'].astype(str)

# Create a function to map cell types based on pattern matching instead of exact matching
def map_cell_type(cell_label):
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
    if cell_label in cell_type_mappings:
        return cell_type_mappings[cell_label]
    
    # Default: return the original label
    return cell_label

# Apply the mapping function to each cell
ge_adata.obs['broad_cell_type'] = ge_adata.obs['cell_ontology_class'].apply(map_cell_type)
intron_adata.obs['broad_cell_type'] = intron_adata.obs['cell_ontology_class'].apply(map_cell_type)

print(f"The number of cells in the splicing dataset is {ge_adata.shape[0]}")

# If AB and Microglia cells
ge_adata.obs.loc[ge_adata.obs["dataset"] == "allen_brain_exons", "tissue"] = "Brain_Non-Myeloid"
is_ab_microglia = (ge_adata.obs["dataset"] == "allen_brain_exons") & (ge_adata.obs["broad_cell_type"] == "MICROGLIA")
ge_adata.obs.loc[is_ab_microglia, "tissue"] = "Brain_Myeloid"

# Load model for mapping exon + intron log1p length normalized counts to total spliced counts (single cell from single nuclei estimatoin)
linear_model_file = "/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/GeneExpression/linear_log_norm_ts_model.pkl"

# Load the model
with open(linear_model_file, 'rb') as f:
    model_linear = pickle.load(f)

print("Linear Model Summary:")
print(model_linear.summary())

# Estimate total spliced counts from exons and introns
# This converts single-nucleus data to estimated single-cell data to account for differences in spliced products
print("Estimating total spliced counts using the linear model...")

# Extract coefficients from our linear model
intercept = model_linear.params["const"]
coef_exons = model_linear.params["exons"]
coef_introns = model_linear.params["introns"]

# Create a copy of the original dataset that we'll modify
combined_adata = ge_adata.copy()

# First apply the model to all Allen Brain cells
ab_mask = combined_adata.obs["dataset"] == "allen_brain_exons"
ab_cells = combined_adata[ab_mask]

# Get the corresponding intron data
log_exons = ab_cells.layers["log_norm"]
log_introns = intron_adata.layers["log_norm"]

# Apply the linear model
log_pred_tot = log_exons.multiply(coef_exons) + log_introns.multiply(coef_introns)
log_pred_tot_with_intercept = log_pred_tot.copy()
log_pred_tot_with_intercept.data += intercept

# Create the new layer for all cells (initialize with None/zeros)
combined_adata.layers["predicted_log_norm_tms"] = combined_adata.layers["log_norm"].copy()

# Update AB cells with the predicted values
combined_adata.layers["predicted_log_norm_tms"][ab_mask] = log_pred_tot_with_intercept

# For TMS cells, just keep the original log_norm values (already copied above)
print(f"Combined object contains {combined_adata.n_obs} cells total")
print(f"- Allen Brain cells: {sum(ab_mask)}")
print(f"- Tabula Muris Senis cells: {sum(~ab_mask)}")

# Create output directory
from datetime import date
outdir = os.path.join(WD, "adjusted_gene_expression")
os.makedirs(outdir, exist_ok=True)

# Save the combined dataset
today = date.today().strftime("%Y-%m-%d")
combined_outfile = os.path.join(outdir, f"Combined_adjusted_GeneExpression_{today}.h5ad")
combined_adata.write_h5ad(combined_outfile, compression="lzf")
print(f"Saved the combined object to {combined_outfile}")

print("Processing complete!")

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION
# sbatch --mem=250G -p cpu,dev --wrap "python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/GeneExpression/TabulaSenis_vs_Allen_4.py"