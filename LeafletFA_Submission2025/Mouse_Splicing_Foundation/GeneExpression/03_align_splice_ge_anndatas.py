#!/usr/bin/env python
"""
Align Splicing and Gene Expression Data - Mouse Splicing Foundation

This script:
1. Loads splicing data from ATSEmapper and gene expression data
2. Aligns cells between the two datasets
3. Standardizes cell type annotations using two-level mapping system
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
OUTPUT_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}", flush=True)

# Input file paths
SPLICE_INPUT = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper/junction_processing_20250929/anndatas/merged_anndata.h5ad"
GE_INPUT = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/processed_data/combined_gene_expression/Combined_GeneExpression_2025-10-03.h5ad"
ATSE_FILE = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper/ATSE_files/MOUSE_FOUNDATION_ATSE_FILE_unanno_also_2025-10-01_21-36-40.txt.gz"

METADATA_FILE = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/metadata_mouse_metadata_combined.csv"

# Cell type mapping functions with two levels of granularity
def create_cell_type_mappings():
    """Create dictionaries for mapping cell types to standardized categories"""
    
    # LEVEL 1: SPECIFIC MAPPINGS (for detailed analysis)
    specific_cell_type_mappings = {
        # Brain-specific cell types (keep detailed for neuroscience)
        "Micro-PVM": "Microglia",
        "Astro": "Astrocyte",
        "Oligo": "Oligodendrocyte", 
        "VLMC": "Vascular stromal cell",
        "Endo": "Endothelial cell",
        "SMC-Peri": "Pericyte",
        
        # Car3+ special case
        "Car3": "Car3+ cell",
        
        # Inhibitory neuron subtypes (detailed)
        "Sst": "Sst+ inhibitory neuron",
        "Pvalb": "Pvalb+ inhibitory neuron", 
        "Vip": "Vip+ inhibitory neuron",
        "Lamp5": "Lamp5+ inhibitory neuron",
        "Sncg": "Sncg+ inhibitory neuron",
        "Meis2": "Meis2+ inhibitory neuron",
        "Ntng1": "Ntng1+ inhibitory neuron",
        "Pax6": "Pax6+ inhibitory neuron",
        "CR": "Cr+ inhibitory neuron",
        
        # Excitatory neuron layer subtypes (detailed)
        "L2": "Layer 2 excitatory neuron",
        "L3": "Layer 3 excitatory neuron", 
        "L4": "Layer 4 excitatory neuron",
        "L5": "Layer 5 excitatory neuron",
        "L6": "Layer 6 excitatory neuron",
        
        # Excitatory subregions (hippocampal/entorhinal) - detailed
        "CA1": "Ca1 pyramidal neuron",
        "CA2": "Ca2 pyramidal neuron",
        "CA3": "Ca3 pyramidal neuron", 
        "DG": "Dentate gyrus neuron",
        "SUB": "Subicular neuron",
        "ProS": "Prosubicular neuron",
        "HATA": "Hata neuron",
        "Mossy": "Mossy cell",
        "PPP": "Ppp neuron",
        "RHP": "Rhp neuron",
        
        # Excitatory projection types (detailed)
        "IT": "Intratelencephalic neuron",
        "CT": "Corticothalamic neuron", 
        "PT": "Pyramidal tract neuron",
        "NP": "Near-projecting neuron",
        "CTX": "Cortical excitatory neuron",
        "ENT": "Entorhinal neuron",
        "PAR": "Parietal neuron", 
        "POST": "Posterior neuron",
        "RSP": "Retrosplenial neuron",
        "HPF": "Hippocampal formation neuron",

        # Immune cells (specific subtypes)
        'T cell': 'T cell',
        'CD4-positive, alpha-beta T cell': 'Cd4+ T cell',
        'CD8-positive, alpha-beta T cell': 'Cd8+ T cell', 
        'regulatory T cell': 'Regulatory T cell',
        'mature NK T cell': 'Nk T cell',
        'mature alpha-beta T cell': 'Mature T cell',
        
        'B cell': 'B cell',
        'immature B cell': 'Immature B cell',
        'naive B cell': 'Naive B cell', 
        'precursor B cell': 'Precursor B cell',
        'early pro-B cell': 'Early pro-B cell',
        'late pro-B cell': 'Late pro-B cell',
        'plasma cell': 'Plasma cell',

        'macrophage': 'Macrophage',
        'Kupffer cell': 'Kupffer cell',
        'lung macrophage': 'Lung macrophage', 

        'monocyte': 'Monocyte',
        'classical monocyte': 'Classical monocyte',
        'non-classical monocyte': 'Non-classical monocyte',
        'intermediate monocyte': 'Intermediate monocyte',

        'dendritic cell': 'Dendritic cell',
        'plasmacytoid dendritic cell': 'Plasmacytoid dendritic cell',
        'myeloid dendritic cell': 'Myeloid dendritic cell',

        'neutrophil': 'Neutrophil',
        'granulocyte': 'Granulocyte',
        'basophil': 'Basophil',
        'granulocyte monocyte progenitor cell': 'Granulocyte progenitor',

        'NK cell': 'Nk cell',
        'thymocyte': 'Thymocyte',
        'DN4 thymocyte': 'Dn4 thymocyte',

        'microglial cell': 'Microglia',

        # Muscle cell subtypes
        'smooth muscle cell': 'Smooth muscle cell',
        'bronchial smooth muscle cell': 'Bronchial smooth muscle cell',
        'smooth muscle cell of the pulmonary artery': 'Pulmonary artery smooth muscle',
        'smooth muscle cell of trachea': 'Tracheal smooth muscle',
        'ventricular myocyte': 'Ventricular cardiomyocyte',
        'atrial myocyte': 'Atrial cardiomyocyte', 
        'skeletal muscle satellite cell': 'Skeletal muscle satellite cell',

        # Epithelial cell subtypes
        'basal cell of epidermis': 'Epidermal basal cell',
        'basal cell': 'Basal cell',
        'keratinocyte': 'Keratinocyte',
        'bulge keratinocyte': 'Bulge keratinocyte',
        
        'epithelial cell': 'Epithelial cell',
        'epidermal cell': 'Epidermal cell',
        'epithelial cell of large intestine': 'Large intestine epithelial cell',
        'enterocyte of epithelium of large intestine': 'Large intestine enterocyte',
        'epithelial cell of proximal tubule': 'Proximal tubule epithelial cell',
        'epithelial cell of thymus': 'Thymic epithelial cell',
        'bladder urothelial cell': 'Bladder urothelial cell',
        'basal epithelial cell of tracheobronchial tree': 'Tracheobronchial basal epithelial cell',
        'luminal epithelial cell of mammary gland': 'Mammary luminal epithelial cell',

        # Vascular cells
        'endothelial cell': 'Endothelial cell',
        'endothelial cell of coronary artery': 'Coronary endothelial cell',
        'endothelial cell of hepatic sinusoid': 'Hepatic sinusoid endothelial cell',
        'aortic endothelial cell': 'Aortic endothelial cell',
        'vein endothelial cell': 'Venous endothelial cell',
        'endothelial cell of lymphatic vessel': 'Lymphatic endothelial cell',
        'pericyte cell': 'Pericyte',
        'brain pericyte': 'Brain pericyte',

        # Fibroblasts and stromal cells
        'fibroblast': 'Fibroblast',
        'fibroblast of cardiac tissue': 'Cardiac fibroblast',
        'fibroblast of lung': 'Lung fibroblast',
        'pulmonary interstitial fibroblast': 'Pulmonary interstitial fibroblast',
        'kidney interstitial fibroblast': 'Kidney interstitial fibroblast',
        'fibrocyte': 'Fibrocyte',
        'stromal cell': 'Stromal cell',
        'adventitial cell': 'Adventitial cell',

        # Specialized organ cells
        'pancreatic A cell': 'Pancreatic alpha cell',
        'pancreatic B cell': 'Pancreatic beta cell', 
        'pancreatic D cell': 'Pancreatic delta cell',
        'pancreatic acinar cell': 'Pancreatic acinar cell',
        'pancreatic PP cell': 'Pancreatic pp cell',
        'pancreatic ductal cell': 'Pancreatic ductal cell',
        'pancreatic stellate cell': 'Pancreatic stellate cell',

        'hepatocyte': 'Hepatocyte',
        
        'kidney collecting duct principal cell': 'Kidney collecting duct principal cell',
        'kidney collecting duct epithelial cell': 'Kidney collecting duct epithelial cell',
        'mesangial cell': 'Mesangial cell',
        'kidney loop of Henle ascending limb epithelial cell': 'Kidney loop of henle epithelial cell',

        'type I pneumocyte': 'Type I pneumocyte',
        'type II pneumocyte': 'Type II pneumocyte',
        'club cell of bronchiole': 'Club cell',
        'lung neuroendocrine cell': 'Lung neuroendocrine cell',
        'ciliated columnar cell of tracheobronchial tree': 'Ciliated tracheobronchial cell',
        'respiratory basal cell': 'Respiratory basal cell',

        'Brush cell of epithelium proper of large intestine': 'Large intestine brush cell',
        'large intestine goblet cell': 'Large intestine goblet cell',
        'enteroendocrine cell': 'Enteroendocrine cell',

        # Stem cells
        'mesenchymal stem cell': 'Mesenchymal stem cell',
        'mesenchymal stem cell of adipose': 'Adipose mesenchymal stem cell',
        'hematopoietic stem cell': 'Hematopoietic stem cell',
        'neuronal stem cell': 'Neuronal stem cell',
        'intestinal crypt stem cell': 'Intestinal crypt stem cell',
        'keratinocyte stem cell': 'Keratinocyte stem cell',
        'lymphoid progenitor cell': 'Lymphoid progenitor cell',
        'proerythroblast': 'Proerythroblast',
        'megakaryocyte-erythroid progenitor cell': 'Megakaryocyte-erythroid progenitor',

        # Other glial cells
        'oligodendrocyte': 'Oligodendrocyte',
        'oligodendrocyte precursor cell': 'Oligodendrocyte precursor cell',
        'astrocyte': 'Astrocyte',
        'Bergmann glial cell': 'Bergmann glial cell',
        'ependymal cell': 'Ependymal cell',

        # General neurons
        'neuron': 'Neuron',
        'medium spiny neuron': 'Medium spiny neuron',
        'interneuron': 'Interneuron',

        # Other specialized cells  
        'bladder cell': 'Bladder cell',
        'secretory cell': 'Secretory cell',
        'mucus secreting cell': 'Mucus secreting cell',
        'endocardial cell': 'Endocardial cell',
        'valve cell': 'Valve cell',
        'chondrocyte': 'Chondrocyte',
        'fenestrated cell': 'Fenestrated cell',
        'neuroepithelial cell': 'Neuroepithelial cell',

        # Catch-all categories
        'leukocyte': 'Leukocyte',
        'professional antigen presenting cell': 'Antigen presenting cell',
        'lymphocyte': 'Lymphocyte',
        'myeloid cell': 'Myeloid cell',
        'myeloid leukocyte': 'Myeloid leukocyte',
        'granulocytopoietic cell': 'Granulocytopoietic cell',
        'promonocyte': 'Promonocyte',
    }
    
    # LEVEL 2: BROAD MAPPINGS (for high-level analysis)
    broad_cell_type_mappings = {
        # All neurons -> Neuron
        "Sst+ inhibitory neuron": "Neuron",
        "Pvalb+ inhibitory neuron": "Neuron", 
        "Vip+ inhibitory neuron": "Neuron",
        "Lamp5+ inhibitory neuron": "Neuron",
        "Sncg+ inhibitory neuron": "Neuron",
        "Meis2+ inhibitory neuron": "Neuron",
        "Ntng1+ inhibitory neuron": "Neuron",
        "Pax6+ inhibitory neuron": "Neuron",
        "Cr+ inhibitory neuron": "Neuron",
        
        "Layer 2 excitatory neuron": "Neuron",
        "Layer 3 excitatory neuron": "Neuron",
        "Layer 4 excitatory neuron": "Neuron", 
        "Layer 5 excitatory neuron": "Neuron",
        "Layer 6 excitatory neuron": "Neuron",
        
        "Ca1 pyramidal neuron": "Neuron",
        "Ca2 pyramidal neuron": "Neuron",
        "Ca3 pyramidal neuron": "Neuron",
        "Dentate gyrus neuron": "Neuron",
        "Subicular neuron": "Neuron",
        "Prosubicular neuron": "Neuron",
        "Hata neuron": "Neuron",
        "Mossy cell": "Neuron",
        "Ppp neuron": "Neuron",
        "Rhp neuron": "Neuron",
        
        "Intratelencephalic neuron": "Neuron",
        "Corticothalamic neuron": "Neuron",
        "Pyramidal tract neuron": "Neuron", 
        "Near-projecting neuron": "Neuron",
        "Cortical excitatory neuron": "Neuron",
        "Entorhinal neuron": "Neuron",
        "Parietal neuron": "Neuron",
        "Posterior neuron": "Neuron",
        "Retrosplenial neuron": "Neuron",
        "Hippocampal formation neuron": "Neuron",
        
        "Neuron": "Neuron",
        "Medium spiny neuron": "Neuron",
        "Interneuron": "Neuron",

        # All glial cells -> Glial cell  
        "Microglia": "Glial cell",
        "Astrocyte": "Glial cell",
        "Oligodendrocyte": "Glial cell",
        "Oligodendrocyte precursor cell": "Glial cell",
        "Bergmann glial cell": "Glial cell",
        "Ependymal cell": "Glial cell",

        # All immune cells -> Immune cell
        "T cell": "Immune cell",
        "Cd4+ T cell": "Immune cell", 
        "Cd8+ T cell": "Immune cell",
        "Regulatory T cell": "Immune cell",
        "Nk T cell": "Immune cell",
        "Mature T cell": "Immune cell",
        
        "B cell": "Immune cell",
        "Immature B cell": "Immune cell",
        "Naive B cell": "Immune cell",
        "Precursor B cell": "Immune cell", 
        "Early pro-B cell": "Immune cell",
        "Late pro-B cell": "Immune cell",
        "Plasma cell": "Immune cell",

        "Macrophage": "Immune cell",
        "Kupffer cell": "Immune cell",
        "Lung macrophage": "Immune cell",

        "Monocyte": "Immune cell",
        "Classical monocyte": "Immune cell", 
        "Non-classical monocyte": "Immune cell",
        "Intermediate monocyte": "Immune cell",

        "Dendritic cell": "Immune cell",
        "Plasmacytoid dendritic cell": "Immune cell",
        "Myeloid dendritic cell": "Immune cell",

        "Neutrophil": "Immune cell",
        "Granulocyte": "Immune cell",
        "Basophil": "Immune cell",
        "Granulocyte progenitor": "Immune cell",

        "Nk cell": "Immune cell",
        "Thymocyte": "Immune cell",
        "Dn4 thymocyte": "Immune cell",
        
        "Leukocyte": "Immune cell",
        "Antigen presenting cell": "Immune cell",
        "Lymphocyte": "Immune cell",
        "Myeloid cell": "Immune cell",
        "Myeloid leukocyte": "Immune cell", 
        "Granulocytopoietic cell": "Immune cell",
        "Promonocyte": "Immune cell",

        # All muscle cells -> Muscle cell
        "Smooth muscle cell": "Muscle cell",
        "Bronchial smooth muscle cell": "Muscle cell",
        "Pulmonary artery smooth muscle": "Muscle cell",
        "Tracheal smooth muscle": "Muscle cell", 
        "Ventricular cardiomyocyte": "Muscle cell",
        "Atrial cardiomyocyte": "Muscle cell",
        "Skeletal muscle satellite cell": "Muscle cell",

        # All epithelial -> Epithelial cell
        "Epidermal basal cell": "Epithelial cell",
        "Basal cell": "Epithelial cell",
        "Keratinocyte": "Epithelial cell",
        "Bulge keratinocyte": "Epithelial cell",
        
        "Epithelial cell": "Epithelial cell",
        "Epidermal cell": "Epithelial cell",
        "Large intestine epithelial cell": "Epithelial cell",
        "Large intestine enterocyte": "Epithelial cell",
        "Proximal tubule epithelial cell": "Epithelial cell",
        "Thymic epithelial cell": "Epithelial cell",
        "Bladder urothelial cell": "Epithelial cell",
        "Tracheobronchial basal epithelial cell": "Epithelial cell",
        "Mammary luminal epithelial cell": "Epithelial cell",
        "Kidney collecting duct epithelial cell": "Epithelial cell",
        "Kidney loop of henle epithelial cell": "Epithelial cell",

        # All vascular -> Vascular cell
        "Endothelial cell": "Vascular cell",
        "Coronary endothelial cell": "Vascular cell",
        "Hepatic sinusoid endothelial cell": "Vascular cell",
        "Aortic endothelial cell": "Vascular cell",
        "Venous endothelial cell": "Vascular cell", 
        "Lymphatic endothelial cell": "Vascular cell",
        "Pericyte": "Vascular cell",
        "Brain pericyte": "Vascular cell",

        # All stromal -> Stromal cell
        "Vascular stromal cell": "Stromal cell",
        "Fibroblast": "Stromal cell",
        "Cardiac fibroblast": "Stromal cell", 
        "Lung fibroblast": "Stromal cell",
        "Pulmonary interstitial fibroblast": "Stromal cell",
        "Kidney interstitial fibroblast": "Stromal cell",
        "Fibrocyte": "Stromal cell",
        "Stromal cell": "Stromal cell",
        "Adventitial cell": "Stromal cell",
        "Chondrocyte": "Stromal cell",

        # All stem cells -> Stem cell
        "Mesenchymal stem cell": "Stem cell",
        "Adipose mesenchymal stem cell": "Stem cell",
        "Hematopoietic stem cell": "Stem cell",
        "Neuronal stem cell": "Stem cell",
        "Intestinal crypt stem cell": "Stem cell",
        "Keratinocyte stem cell": "Stem cell",
        "Lymphoid progenitor cell": "Stem cell",
        "Proerythroblast": "Stem cell",
        "Megakaryocyte-erythroid progenitor": "Stem cell",

        # Organ-specific cells (keep some specialization)
        "Pancreatic alpha cell": "Pancreatic cell",
        "Pancreatic beta cell": "Pancreatic cell",
        "Pancreatic delta cell": "Pancreatic cell",
        "Pancreatic acinar cell": "Pancreatic cell",
        "Pancreatic pp cell": "Pancreatic cell",
        "Pancreatic ductal cell": "Pancreatic cell",
        "Pancreatic stellate cell": "Pancreatic cell",

        "Hepatocyte": "Liver cell",
        
        "Kidney collecting duct principal cell": "Kidney cell",
        "Mesangial cell": "Kidney cell",

        "Type I pneumocyte": "Lung cell",
        "Type II pneumocyte": "Lung cell",
        "Club cell": "Lung cell",
        "Lung neuroendocrine cell": "Lung cell", 
        "Ciliated tracheobronchial cell": "Lung cell",
        "Respiratory basal cell": "Lung cell",

        "Large intestine brush cell": "Intestinal cell",
        "Large intestine goblet cell": "Intestinal cell",
        "Enteroendocrine cell": "Intestinal cell",

        # Special/other cells
        "Car3+ cell": "Other cell",
        "Bladder cell": "Bladder cell",
        "Secretory cell": "Secretory cell",
        "Mucus secreting cell": "Secretory cell",
        "Endocardial cell": "Cardiac cell",
        "Valve cell": "Cardiac cell", 
        "Fenestrated cell": "Fenestrated cell",
        "Neuroepithelial cell": "Neuroepithelial cell",
    }

    # LEVEL 2.5: MEDIUM MAPPINGS (intermediate granularity)
    medium_cell_type_mappings = {
        # Neurons - keep major functional distinctions
        "Sst+ inhibitory neuron": "Inhibitory neuron",
        "Pvalb+ inhibitory neuron": "Inhibitory neuron", 
        "Vip+ inhibitory neuron": "Inhibitory neuron",
        "Lamp5+ inhibitory neuron": "Inhibitory neuron",
        "Sncg+ inhibitory neuron": "Inhibitory neuron",
        "Meis2+ inhibitory neuron": "Inhibitory neuron",
        "Ntng1+ inhibitory neuron": "Inhibitory neuron",
        "Pax6+ inhibitory neuron": "Inhibitory neuron",
        "Cr+ inhibitory neuron": "Inhibitory neuron",

        "Layer 2 excitatory neuron": "Cortical excitatory neuron",
        "Layer 3 excitatory neuron": "Cortical excitatory neuron",
        "Layer 4 excitatory neuron": "Cortical excitatory neuron", 
        "Layer 5 excitatory neuron": "Cortical excitatory neuron",
        "Layer 6 excitatory neuron": "Cortical excitatory neuron",

        "Ca1 pyramidal neuron": "Hippocampal neuron",
        "Ca2 pyramidal neuron": "Hippocampal neuron",
        "Ca3 pyramidal neuron": "Hippocampal neuron",
        "Dentate gyrus neuron": "Hippocampal neuron",
        "Subicular neuron": "Hippocampal neuron",
        "Prosubicular neuron": "Hippocampal neuron",
        "Hata neuron": "Hippocampal neuron",
        "Mossy cell": "Hippocampal neuron",
        "Ppp neuron": "Hippocampal neuron",
        "Rhp neuron": "Hippocampal neuron",

        "Intratelencephalic neuron": "Projection neuron",
        "Corticothalamic neuron": "Projection neuron",
        "Pyramidal tract neuron": "Projection neuron", 
        "Near-projecting neuron": "Projection neuron",
        "Cortical excitatory neuron": "Cortical excitatory neuron",
        "Entorhinal neuron": "Cortical excitatory neuron",
        "Parietal neuron": "Cortical excitatory neuron",
        "Posterior neuron": "Cortical excitatory neuron",
        "Retrosplenial neuron": "Cortical excitatory neuron",
        "Hippocampal formation neuron": "Hippocampal neuron",

        "Neuron": "Other neuron",
        "Medium spiny neuron": "Other neuron",
        "Interneuron": "Other neuron",

        # Glial cells - keep major subtypes
        "Microglia": "Microglia",
        "Astrocyte": "Astrocyte",
        "Oligodendrocyte": "Oligodendrocyte",
        "Oligodendrocyte precursor cell": "Oligodendrocyte precursor",
        "Bergmann glial cell": "Other glial cell",
        "Ependymal cell": "Other glial cell",

        # Immune cells - group by major lineages
        "T cell": "T cell",
        "Cd4+ T cell": "T cell", 
        "Cd8+ T cell": "T cell",
        "Regulatory T cell": "T cell",
        "Nk T cell": "T cell",
        "Mature T cell": "T cell",

        "B cell": "B cell",
        "Immature B cell": "B cell",
        "Naive B cell": "B cell",
        "Precursor B cell": "B cell", 
        "Early pro-B cell": "B cell",
        "Late pro-B cell": "B cell",
        "Plasma cell": "B cell",

        "Macrophage": "Macrophage",
        "Kupffer cell": "Macrophage",
        "Lung macrophage": "Macrophage",

        "Monocyte": "Monocyte",
        "Classical monocyte": "Monocyte", 
        "Non-classical monocyte": "Monocyte",
        "Intermediate monocyte": "Monocyte",

        "Dendritic cell": "Dendritic cell",
        "Plasmacytoid dendritic cell": "Dendritic cell",
        "Myeloid dendritic cell": "Dendritic cell",

        "Neutrophil": "Granulocyte",
        "Granulocyte": "Granulocyte",
        "Basophil": "Granulocyte",
        "Granulocyte progenitor": "Immune progenitor",

        "Nk cell": "Nk cell",
        "Thymocyte": "T cell precursor",
        "Dn4 thymocyte": "T cell precursor",

        "Leukocyte": "Other immune cell",
        "Antigen presenting cell": "Other immune cell",
        "Lymphocyte": "Other immune cell",
        "Myeloid cell": "Other immune cell",
        "Myeloid leukocyte": "Other immune cell", 
        "Granulocytopoietic cell": "Immune progenitor",
        "Promonocyte": "Immune progenitor",

        # Muscle cells - distinguish cardiac, smooth, skeletal
        "Smooth muscle cell": "Smooth muscle cell",
        "Bronchial smooth muscle cell": "Smooth muscle cell",
        "Pulmonary artery smooth muscle": "Smooth muscle cell",
        "Tracheal smooth muscle": "Smooth muscle cell", 
        "Ventricular cardiomyocyte": "Cardiomyocyte",
        "Atrial cardiomyocyte": "Cardiomyocyte",
        "Skeletal muscle satellite cell": "Skeletal muscle cell",

        # Epithelial cells - group by location/function
        "Epidermal basal cell": "Skin epithelial cell",
        "Basal cell": "Basal epithelial cell",
        "Keratinocyte": "Skin epithelial cell",
        "Bulge keratinocyte": "Skin epithelial cell",

        "Epithelial cell": "Other epithelial cell",
        "Epidermal cell": "Skin epithelial cell",
        "Large intestine epithelial cell": "Intestinal epithelial cell",
        "Large intestine enterocyte": "Intestinal epithelial cell",
        "Proximal tubule epithelial cell": "Kidney epithelial cell",
        "Thymic epithelial cell": "Other epithelial cell",
        "Bladder urothelial cell": "Urogenital epithelial cell",
        "Tracheobronchial basal epithelial cell": "Respiratory epithelial cell",
        "Mammary luminal epithelial cell": "Mammary epithelial cell",
        "Kidney collecting duct epithelial cell": "Kidney epithelial cell",
        "Kidney loop of henle epithelial cell": "Kidney epithelial cell",

        # Vascular cells - distinguish endothelial vs pericytes
        "Endothelial cell": "Endothelial cell",
        "Coronary endothelial cell": "Endothelial cell",
        "Hepatic sinusoid endothelial cell": "Endothelial cell",
        "Aortic endothelial cell": "Endothelial cell",
        "Venous endothelial cell": "Endothelial cell", 
        "Lymphatic endothelial cell": "Lymphatic endothelial cell",
        "Pericyte": "Pericyte",
        "Brain pericyte": "Pericyte",

        # Stromal cells - distinguish fibroblasts vs other stromal
        "Vascular stromal cell": "Stromal cell",
        "Fibroblast": "Fibroblast",
        "Cardiac fibroblast": "Fibroblast", 
        "Lung fibroblast": "Fibroblast",
        "Pulmonary interstitial fibroblast": "Fibroblast",
        "Kidney interstitial fibroblast": "Fibroblast",
        "Fibrocyte": "Fibroblast",
        "Stromal cell": "Stromal cell",
        "Adventitial cell": "Stromal cell",
        "Chondrocyte": "Chondrocyte",

        # Stem cells - distinguish by lineage
        "Mesenchymal stem cell": "Mesenchymal stem cell",
        "Adipose mesenchymal stem cell": "Mesenchymal stem cell",
        "Hematopoietic stem cell": "Hematopoietic stem cell",
        "Neuronal stem cell": "Neural stem cell",
        "Intestinal crypt stem cell": "Epithelial stem cell",
        "Keratinocyte stem cell": "Epithelial stem cell",
        "Lymphoid progenitor cell": "Hematopoietic progenitor",
        "Proerythroblast": "Hematopoietic progenitor",
        "Megakaryocyte-erythroid progenitor": "Hematopoietic progenitor",

        # Organ-specific cells - keep organ distinction
        "Pancreatic alpha cell": "Pancreatic endocrine cell",
        "Pancreatic beta cell": "Pancreatic endocrine cell",
        "Pancreatic delta cell": "Pancreatic endocrine cell",
        "Pancreatic acinar cell": "Pancreatic exocrine cell",
        "Pancreatic pp cell": "Pancreatic endocrine cell",
        "Pancreatic ductal cell": "Pancreatic ductal cell",
        "Pancreatic stellate cell": "Pancreatic stellate cell",

        "Hepatocyte": "Hepatocyte",

        "Kidney collecting duct principal cell": "Kidney tubular cell",
        "Mesangial cell": "Kidney glomerular cell",

        "Type I pneumocyte": "Alveolar epithelial cell",
        "Type II pneumocyte": "Alveolar epithelial cell",
        "Club cell": "Airway epithelial cell",
        "Lung neuroendocrine cell": "Lung neuroendocrine cell", 
        "Ciliated tracheobronchial cell": "Airway epithelial cell",
        "Respiratory basal cell": "Airway epithelial cell",

        "Large intestine brush cell": "Specialized intestinal cell",
        "Large intestine goblet cell": "Specialized intestinal cell",
        "Enteroendocrine cell": "Enteroendocrine cell",

        # Special/other cells
        "Car3+ cell": "Car3+ cell",
        "Bladder cell": "Bladder cell",
        "Secretory cell": "Secretory cell",
        "Mucus secreting cell": "Secretory cell",
        "Endocardial cell": "Cardiac stromal cell",
        "Valve cell": "Cardiac stromal cell", 
        "Fenestrated cell": "Specialized endothelial cell",
        "Neuroepithelial cell": "Neuroepithelial cell",

    }

    # Subtissue corrections
    subtissue_corrections = {
            'T cells': 'T-cells',
            'ENDOMUCIN': 'Endomucin',
            'forelimb and hindlimb': 'ForelimbandHindlimb',
            'Liver non-hepato/SCs_st': 'Liver non-hepato/SCs',
            'Skin Anagen': 'Anagen'
        }

    return specific_cell_type_mappings, medium_cell_type_mappings, broad_cell_type_mappings, subtissue_corrections

def apply_cell_type_mapping(cell_types, mapping_level="specific"):
    """
    Apply cell type mapping to a list of cell types
    
    Parameters:
    cell_types: list of cell type strings
    mapping_level: "specific" or "broad"
    
    Returns:
    list of mapped cell types
    """
    specific_mappings, medium_mappings, broad_mappings, _ = create_cell_type_mappings()
    
    if mapping_level == "specific":
        return [specific_mappings.get(cell_type, cell_type) for cell_type in cell_types]
    elif mapping_level == "medium":
        # First apply specific mapping, then medium mapping
        specific_mapped = [specific_mappings.get(cell_type, cell_type) for cell_type in cell_types]
        return [medium_mappings.get(cell_type, cell_type) for cell_type in specific_mapped]
    elif mapping_level == "broad":
        # First apply specific, then medium, then broad mapping
        specific_mapped = [specific_mappings.get(cell_type, cell_type) for cell_type in cell_types]
        medium_mapped = [medium_mappings.get(cell_type, cell_type) for cell_type in specific_mapped]
        return [broad_mappings.get(cell_type, cell_type) for cell_type in medium_mapped]
    else:
        raise ValueError("mapping_level must be 'specific', 'medium', or 'broad'")

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
        return "Car3+ cell"
    
    # Check for inhibitory neuron markers
    inhibitory_markers = ["Sst", "Pvalb", "Vip", "Lamp5", "Sncg", "Meis2", "Ntng1", "Pax6", "CR"]
    if any(marker in cell_label for marker in inhibitory_markers):
        # Return specific subtype if found in mappings
        for marker in inhibitory_markers:
            if marker in cell_label and marker in mappings:
                return mappings[marker]
        return "Inhibitory neuron"  # fallback
    
    # Check for excitatory neuron markers
    layer_markers = ["L2", "L3", "L4", "L5", "L6"]
    excitatory_regions = ["CA1", "CA2", "CA3", "DG", "SUB", "ProS", "HATA", "Mossy", "PPP", "RHP"]
    other_excitatory = ["IT", "CT", "PT", "NP", "CTX", "ENT", "PAR", "POST", "RSP", "HPF"]
    
    all_excitatory_markers = layer_markers + excitatory_regions + other_excitatory
    for marker in all_excitatory_markers:
        if marker in cell_label and marker in mappings:
            return mappings[marker]
    
    # Check for broad cell type markers
    broad_markers = {
        "Micro-PVM": "Microglia",
        "Astro": "Astrocyte",
        "Oligo": "Oligodendrocyte",
        "VLMC": "Vascular stromal cell",
        "Endo": "Endothelial cell",
        "SMC-Peri": "Pericyte"
    }
    
    for marker, cell_type in broad_markers.items():
        if marker in cell_label:
            return cell_type
    
    # Check for direct matches in the main dictionary
    if cell_label in mappings:
        return mappings[cell_label]
    
    # Default: return the original label with proper capitalization
    return cell_label.capitalize() if isinstance(cell_label, str) else "Unknown"

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
        print(f"   Error loading datasets: {str(e)}")
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
    specific_mappings, medium_mappings, broad_mappings, _ = create_cell_type_mappings()

    # Print some sample data before standardization
    print("Before standardization:")
    print("Splicing data sample:", flush=True)
    print(splice_adata.obs[["cell_id", "cell_ontology_class"]].head(), flush=True)
    print("Gene expression data sample:", flush=True)
    print(ge_adata.obs[["cell_id", "cell_ontology_class"]].head(), flush=True)
        
    # Apply cell type mappings to both datasets - SPECIFIC LEVEL
    splice_adata.obs["specific_cell_type"] = splice_adata.obs["cell_ontology_class"].apply(
        lambda x: map_cell_type(x, specific_mappings)
    )
    ge_adata.obs["specific_cell_type"] = ge_adata.obs["cell_ontology_class"].apply(
        lambda x: map_cell_type(x, specific_mappings)
    )
    
    # Apply BROAD LEVEL mappings
    splice_adata.obs["broad_cell_type"] = splice_adata.obs["specific_cell_type"].apply(
        lambda x: broad_mappings.get(x, x)
    )
    ge_adata.obs["broad_cell_type"] = ge_adata.obs["specific_cell_type"].apply(
        lambda x: broad_mappings.get(x, x)
    )
    
    # Apply MEDIUM LEVEL mappings
    splice_adata.obs["medium_cell_type"] = splice_adata.obs["specific_cell_type"].apply(
        lambda x: medium_mappings.get(x, x)
    )
    ge_adata.obs["medium_cell_type"] = ge_adata.obs["specific_cell_type"].apply(
        lambda x: medium_mappings.get(x, x)
    )

    # Print samples after standardization to verify
    print("\nAfter standardization:")
    print("Splicing data with standardized cell types:", flush=True)
    print(splice_adata.obs[["cell_id", "cell_ontology_class", "specific_cell_type", "broad_cell_type"]].head(), flush=True)
    print("Gene expression data with standardized cell types:", flush=True)
    print(ge_adata.obs[["cell_id", "cell_ontology_class", "specific_cell_type", "broad_cell_type"]].head(), flush=True)

    print(f"   ✓ Top specific splicing cell types: {dict(splice_adata.obs['specific_cell_type'].value_counts().head(5))}")
    print(f"   ✓ Top broad splicing cell types: {dict(splice_adata.obs['broad_cell_type'].value_counts().head(5))}")
    print(f"   ✓ Top specific gene expression cell types: {dict(ge_adata.obs['specific_cell_type'].value_counts().head(5))}")
    print(f"   ✓ Top broad gene expression cell types: {dict(ge_adata.obs['broad_cell_type'].value_counts().head(5))}")
    
    # Update tissue labels based on cell types (moved from process_splicing_data)
    splice_adata.obs.loc[splice_adata.obs["dataset"] == "AB", "tissue"] = "Brain_Non-Myeloid"
    is_ab_microglia = (splice_adata.obs["dataset"] == "AB") & (splice_adata.obs["broad_cell_type"] == "Glial cell")
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
    
    # Add library size information for GENE EXPRESSION data
    print("   ⚙️ Computing gene expression library size information...")
    ge_library_size = np.asarray(ge_adata_view.layers["length_norm"].sum(axis=1)).flatten()
    # Convert to integer (library size should be count data)
    ge_library_size = ge_library_size.astype(int)
    ge_adata_view.obs["library_size"] = ge_library_size  # 1D column 
    ge_adata_view.obsm["X_library_size"] = ge_library_size[:, np.newaxis]  # 2D array
    
    # Add library size information for SPLICING data (sum of junction reads)
    print("   ⚙️ Computing splicing library size information...")
    splice_library_size = np.asarray(splice_adata_view.layers["cell_by_junction_matrix"].sum(axis=1)).flatten()
    # Convert to integer (library size should be count data)
    splice_library_size = splice_library_size.astype(int)
    splice_adata_view.obs["library_size"] = splice_library_size  # 1D column
    splice_adata_view.obsm["X_library_size"] = splice_library_size[:, np.newaxis]  # 2D array
    
    # Verify alignment of cells
    print("   ⚙️ Verifying cell alignment...")
    verify_cells = np.all(ge_adata_view.obs['cell_id'].values == splice_adata_view.obs['cell_id'].values)
    if verify_cells:
        print("   ✓ Cell IDs verified to be in identical order")
    else:
        print("   ❌ ERROR: Cell IDs are not aligned properly")
        sys.exit(1)
    
    return splice_adata_view, ge_adata_view

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
            f.write(f"Specific cell type distribution: {dict(splice_adata.obs['specific_cell_type'].value_counts())}\n")
            f.write(f"Broad cell type distribution: {dict(splice_adata.obs['broad_cell_type'].value_counts())}\n")
            f.write(f"Dataset distribution: {dict(splice_adata.obs['dataset'].value_counts())}\n\n")
            
            f.write(f"## Gene Expression Dataset\n")
            f.write(f"Number of cells: {ge_adata.n_obs}\n")
            f.write(f"Number of genes: {ge_adata.n_vars}\n")
            f.write(f"Specific cell type distribution: {dict(ge_adata.obs['specific_cell_type'].value_counts())}\n")
            f.write(f"Broad cell type distribution: {dict(ge_adata.obs['broad_cell_type'].value_counts())}\n")
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

    # Save breakdown of broad, specific, and medium cell types to text in the same output directory
    with open(os.path.join(OUTPUT_DIR, f"cell_type_breakdown_{timestamp}.txt"), "w") as f:
        f.write(f"# Mouse Splicing Foundation - Cell Type Breakdown\n")
        f.write(f"# Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Splicing Dataset\n")
        f.write(f"Specific cell type distribution: {dict(splice_adata.obs['specific_cell_type'].value_counts())}\n")
        f.write(f"Broad cell type distribution: {dict(splice_adata.obs['broad_cell_type'].value_counts())}\n")
        f.write(f"Medium cell type distribution: {dict(splice_adata.obs['medium_cell_type'].value_counts())}\n")

    # Save aligned datasets
    save_aligned_data(splice_adata, ge_adata)
    
    print("\n========================================")
    print("Data alignment and processing complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("========================================\n")

if __name__ == "__main__":
    main()

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION
# script=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/GeneExpression/03_align_splice_ge_anndatas.py
# sbatch --mem=550G -p cpu,bigmem --wrap "python $script"