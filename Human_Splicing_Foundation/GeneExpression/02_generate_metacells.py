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
WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/processed_data"
OUTPUT_DIR = WD  # Save in the same directory
today = datetime.datetime.now().strftime("%Y-%m-%d")

# Cell types to focus on for pseudobulk analysis
COMMON_BROAD_TYPES = ['Other_Neuron', 'Pericyte', 'General_Fibroblast', 'Microglia', 'CNS_Glia']

ab_exons = sc.read_h5ad(f"{WD}/ab_adata_exons_2025-04-15.h5ad")
ab_introns = sc.read_h5ad(f"{WD}/ab_adata_introns_2025-04-15.h5ad")
ts_adata = sc.read_h5ad(f"{WD}/tabsap_adata_2025-04-15.h5ad")

# === Clean up gene symbols ===
print(f"Cleaning up gene symbols...")
for adata in [ab_exons, ab_introns, ts_adata]:
    adata.var["gene_symbol"] = adata.var["gene_symbol"].astype(str)
    adata = adata[:, ~adata.var["gene_symbol"].duplicated(keep=False)].copy()
    adata.var_names = adata.var["gene_symbol"]

# === Subset to shared genes ===
print(f"Subsetting to shared genes...")
common_genes = set(ab_exons.var_names).intersection(ab_introns.var_names).intersection(ts_adata.var_names)
for adata in [ab_exons, ab_introns, ts_adata]:
    adata._inplace_subset_var([g in common_genes for g in adata.var_names])

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

# 1. Define flatten function once
def flatten_grouped_map(grouped_map):
    return {label: broad_type for broad_type, labels in grouped_map.items() for label in labels}

# 2. Create the flat map once (outside the loop)
grouped_broad_map_flat = flatten_grouped_map(grouped_refined_map)
ts_adata.obs["broad_cell_type"] = ts_adata.obs["free_annotation"].map(grouped_broad_map_flat).fillna('Other')
ab_exons.obs["broad_cell_type"] = ab_exons.obs["subclass_label"].map(grouped_broad_map_flat).fillna('Other')
ab_introns.obs["broad_cell_type"] = ab_introns.obs["subclass_label"].map(grouped_broad_map_flat).fillna('Other')

# Find common broad cell types across ts_adata and ab_exons 
common_broad_types = set(ts_adata.obs["broad_cell_type"]).intersection(set(ab_exons.obs["broad_cell_type"]))
print(f"Common broad cell types: {common_broad_types}")

# Removed Other from common_broad_types
common_broad_types = [broad_type for broad_type in common_broad_types if broad_type != 'Other']
# remove neuron 
common_broad_types = [broad_type for broad_type in common_broad_types if broad_type != 'Neuron']

# === Identify indices ===
print(f"Identifying indices for shared cell types across TS and AB...")
idx_ts = ts_adata.obs["broad_cell_type"].isin(common_broad_types)
idx_ab_exons = ab_exons.obs["broad_cell_type"].isin(common_broad_types)
idx_ab_introns = ab_introns.obs["broad_cell_type"].isin(common_broad_types)

print(f"Number of cells in Tabula Sapiens: {len(ts_adata.obs[idx_ts])}")
print(f"Number of cells in Allen Brain Exon: {len(ab_exons.obs[idx_ab_exons])}")
print(f"Number of cells in Allen Brain Intron: {len(ab_introns.obs[idx_ab_introns])}")

# Print breakdown of each broad cell type in each dataset 
print("Tabula Sapiens breakdown:")
print(ts_adata.obs[idx_ts]["broad_cell_type"].value_counts())
print("Allen Brain Exon breakdown:")
print(ab_exons.obs[idx_ab_exons]["broad_cell_type"].value_counts())

# === Make sparse if needed ===
for adata in [ab_exons, ab_introns, ts_adata]:
    if not sp.issparse(adata.X):
        adata.X = csr_matrix(adata.X)

# === Pseudobulk counts ===
print(f"Getting pseudobulk counts for each broad cell type...")

# Assert that genes are ordered the same in all datasets in the same exact order
assert np.array_equal(ab_exons.var_names, ab_introns.var_names), "Gene names in ab_exons and ab_introns are not the same."
assert np.array_equal(ab_exons.var_names, ts_adata.var_names), "Gene names in ab_exons and ts_adata are not the same."
assert np.array_equal(ab_introns.var_names, ts_adata.var_names), "Gene names in ab_introns and ts_adata are not the same."

for cell_type in common_broad_types:
    print(f"Processing {cell_type}...")
    # Get indices for each cell type
    idx_ts = ts_adata.obs["broad_cell_type"] == cell_type
    idx_ab_exons = ab_exons.obs["broad_cell_type"] == cell_type
    idx_ab_introns = ab_introns.obs["broad_cell_type"] == cell_type

    # Sum counts for each gene across cells of the same type
    ts_sum = np.array(ts_adata.X[idx_ts].sum(axis=0)).flatten()
    ab_exons_sum = np.array(ab_exons.X[idx_ab_exons].sum(axis=0)).flatten()
    ab_introns_sum = np.array(ab_introns.X[idx_ab_introns].sum(axis=0)).flatten()
    genes = ab_exons.var_names.to_list()

    # Save pseudobulk data for that cell type
    pd.DataFrame({
        "gene": genes,
        "ab_exons_sum": ab_exons_sum,
        "ab_introns_sum": ab_introns_sum,
        "ts_sum": ts_sum
    }).to_csv(f"{outdir}/pseudobulk_{cell_type}.tsv", sep="\t", index=False)
    print(f"Pseudobulk data saved for {cell_type}.")

print("All pseudobulk files saved.")

