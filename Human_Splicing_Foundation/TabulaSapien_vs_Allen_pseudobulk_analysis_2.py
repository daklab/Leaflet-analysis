# %%
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
import numpy as np
from collections import defaultdict
import scipy.sparse as sp
from collections import defaultdict
import gffutils 
import scipy.sparse as sp
import gffutils 

# %% [markdown]
# ### Normalize and combine each version of the data: exons, introns, total

# %%
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

# === Paths and output ===
print(f"Reading in the anndata objects...")
outdir = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/processed_data/"
ab_exons = sc.read_h5ad(f"{outdir}/ab_adata_exons_2025-04-15.h5ad")
ab_introns = sc.read_h5ad(f"{outdir}/ab_adata_introns_2025-04-15.h5ad")
ts_adata = sc.read_h5ad(f"{outdir}/tabsap_adata_2025-04-15.h5ad")

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

# === Broad cell type mapping ===
# Grouped mappings by broad cell type
grouped_broad_map = {
    'Neuron': [
        'IT', 'L4 IT', 'VIP', 'PVALB', 'SST', 'L6 CT', 'L6b', 'L5 ET', 'L5/6 IT Car3',
        'L5/6 NP', 'PAX6', 'LAMP5', 'retinal bipolar neuron'
    ],
    'T cell': [
        'cd4-positive, alpha-beta t cell', 'cd8-positive, alpha-beta t cell', 't cell',
        'regulatory t cell', 'cd4-positive helper t cell', 'cd4-positive, alpha-beta memory t cell',
        'cd8-positive, alpha-beta memory t cell', 'naive thymus-derived cd4-positive, alpha-beta t cell',
        'activated cd4-positive, alpha-beta t cell', 'cd8-positive, alpha-beta thymocyte',
        'naive cd8-positive t cell', 'cd4-positive, alpha-beta thymocyte', 'gamma-delta t cell',
        'cd4-positive memory t cell', 'activated cd8-positive, alpha-beta t cell',
        't follicular helper cell', 'naive regulatory t cell', 'thymocyte',
        'cd8-positive cytotoxic t cell', 'cd8+, alpha-beta cytokine secreting effector t cell'
    ],
    'B cell': ['b cell', 'memory b cell', 'plasma cell', 'naive b cell'],
    'Microglia_Macrophage': [
        'macrophage', 'Microglia', 'microglial cell', 'Monocyte_Macrophage',
        'tissue-resident macrophage', 'muscle macrophage', 'retina - microglia'
    ],
    'Monocyte': [
        'monocyte', 'classical monocyte', 'non-classical monocyte', 'intermediate monocyte'
    ],
    'NK/ILC': [
        'nk cell', 'natural killer cell', 'nk t cell', 'innate lymphoid cell', 'mature nk t cell',
        'type i nk t cell', 'uterine nk cell', 'proliferating nk cell', 'immature natural killer cell'
    ],
    'Dendritic': [
        'dendritic cell', 'myeloid dendritic cell', 'plasmacytoid dendritic cell',
        'cd1c-positive myeloid dendritic cell', 'cd141-positive myeloid dendritic cell',
        'cdc1', 'cdc2', 'conventional dendritic cell'
    ],
    'Granulocyte': [
        'neutrophil', 'cd24 neutrophil', 'nampt neutrophil', 'granulocyte',
        'basophil', 'mast cell'
    ],
    'Epithelial': [
        'epithelial cell', 'club cell', 'ionocyte', 'duct epithelial cell', 'goblet cell',
        'ltf+ epithelial cell', 'ciliated epithelial cell', 'basal epithelial cell',
        'bladder urothelial cell', 'basal bladder urothelial cell', 'intermediate bladder urothelial cell',
        'enterocyte of epithelium of large intestine', 'enterocyte of epithelium proper of small intestine',
        'salivary gland cell', 'HR positive luminal epithelial cell of mammary gland',
        'epithelial cell of uterus', 'pulmonary ionocyte', 'large intestine goblet cell',
        'medullary thymic epithelial cell', 'antibody secreting cell', 'conjunctival epithelial cell',
        'corneal epithelial cell', 'secretory luminal epithelial cell of mammary gland',
        'glandular epithelial cell', 'luminal epithelial cell', 'cycling epithelial cell',
        'mucus secreting cell', 'serous cell of epithelium of bronchus', 'best4+ intestinal epithelial cell',
        'biliary epithelial cell', 'pancreatic ductal cell', 'small intestine goblet cell',
        'stratified squamous epithelial cell', 'sebum secreting cell', 'enterocyte of epithelium proper of ileum',
        'enterocyte of epithelium proper of duodenum', 'paneth cell of epithelium of small intestine',
        'paneth cell of colon', 'mature enterocyte', 'intestinal tuft cell',
        'tuft cell of colon'
    ],
    'Endothelial': [
        'endothelial cell', 'capillary endothelial cell', 'arterial endothelial cell',
        'vein endothelial cell', 'endothelial cell of vascular tree', 'endothelial cell of lymphatic vessel',
        'cardiac endothelial cell', 'colon endothelial cell', 'retinal blood vessel endothelial cell',
        'endothelial cell of arteriole', 'venous capillary endothelial cell', 'blood vessel endothelial cell',
        'endothelial cell of venule', 'endothelial cell of artery', 'vascular endothelial cell'
    ],
    'Glia': [
        'Astrocyte', 'OPC', 'Oligodendrocyte', 'retina - muller glia', 'mueller cell',
        'enteroglial cell', 'schwann cell', 'glial cell'
    ],
    'Muscle': [
        'smooth muscle cell', 'skeletal muscle satellite stem cell', 'fast muscle cell',
        'slow muscle cell', 'atrial cardiac muscle cell', 'muscle cell', 'tongue muscle cell',
        'ventricular cardiac muscle cell', 'airway smooth muscle cell', 'tendon cell',
        'vascular associated smooth muscle cell'
    ],
    'Stromal': [
        'fibroblast', 'stromal cell', 'myofibroblast cell', 'adventitial fibroblast', 'cd34+ fibroblasts',
        'VLMC', 'fat cell', 'adventitial cell', 'cornea - mesenchymal cell - stromal keratinocytes',
        'limbal stromal cell', 'follicle', 'granulosa cell', 'mesothelial cell', 'theca cell',
        'connective tissue cell', 'endometrial stromal fibroblast'
    ],
    'Fibroblast': [
        'alveolar fibroblast', 'fibroblast of breast', 'fibroblast of cardiac tissue', 'uterine fibroblast',
        'stellate_fibroblast'
    ],
    'Pericyte': [
        'pericyte', 'Pericyte', 'myofibroblast cell and pericyte', 'mural cell'
    ],
    'Liver': [
        'hepatocyte', 'hepatic stellate cell', 'intrahepatic cholangiocyte'
    ],
    'Photoreceptor': [
        'retinal pigment epithelial cell', 'retina - photoreceptor cell', 'eye photoreceptor cell'
    ],
    'Alveolar cell': [
        'type ii pneumocyte', 'type i pneumocyte', 'capillary aerocyte', 'respiratory goblet cell',
        'ciliated columnar cell of tracheobronchial tree', 'tracheal goblet cell'
    ],
    'Endocrine': [
        'enteroendocrine cell of small intestine', 'type l enteroendocrine cell', 'enterochromaffin-like cell'
    ],
    'Hematopoietic': [
        'platelet', 'erythroid progenitor cell', 'erythrocyte'
    ],
    'Stem/Progenitor': [
        'hematopoietic stem cell', 'mesenchymal stem cell', 'myeloid progenitor', 'common myeloid progenitor',
        'mesenchymal stem cell of adipose tissue'
    ],
    'Progenitor': [
        'oocyte', 'radial glia progenitor cell', 'intestinal crypt stem cell of small intestine',
        'intestinal crypt stem cell of large intestine'
    ],
    'Secretory': [
        'acinar cell of salivary gland', 'lacrimal gland functional unit cell', 'myoepithelial cell'
    ],
    'Pigment cell': [
        'melanocyte', 'melanocyte or limbal stem cell'
    ],
    'Sensory': [
        'taste receptor cell'
    ],
    'Skin': [
        'keratocyte'
    ],
    'Myeloid': [
        'myeloid cell', 'mononuclear phagocyte'
    ],
    'Immune other': [
        'leukocyte', 'langerhans cell', 'immune cell'
    ],
    'Unknown': [
        'unknown'
    ],
}

# 1. Define flatten function once
def flatten_grouped_map(grouped_map):
    return {label: broad_type for broad_type, labels in grouped_map.items() for label in labels}

# 2. Create the flat map once (outside the loop)
grouped_broad_map_flat = flatten_grouped_map(grouped_broad_map)

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

# === Negative controls: 2 random samples from each dataset ===
np.random.seed(42)
ts_other_idx = ts_adata.obs[~ ts_adata.obs["broad_cell_type"].isin(common_broad_types)].index
ab_exons_other_idx = ab_exons.obs[ab_exons.obs["broad_cell_type"].isin(common_broad_types)].index
ab_introns_other_idx = ab_introns.obs[ab_introns.obs["broad_cell_type"].isin(common_broad_types)].index
num_cells_sample = 2000 

ts_rand_cells = np.random.choice(ts_other_idx, size=num_cells_sample, replace=False)
ab_exons_rand_cells = np.random.choice(ab_exons_other_idx, size=num_cells_sample, replace=False)
ab_introns_rand_cells = np.random.choice(ab_introns_other_idx, size=num_cells_sample, replace=False)
print(f"Got {len(ts_rand_cells)} random cells from Tabula Sapiens.")
print(f"Got {len(ab_exons_rand_cells)} random cells from Allen Brain Exon.")
print(f"Got {len(ab_introns_rand_cells)} random cells from Allen Brain Intron.")
ts_ctrl_sum = np.array(ts_adata[ts_rand_cells].X.sum(axis=0)).flatten()
ab_ctrl_exons_sum = np.array(ab_exons[ab_exons_rand_cells].X.sum(axis=0)).flatten()
ab_ctrl_introns_sum = np.array(ab_introns[ab_introns_rand_cells].X.sum(axis=0)).flatten()

# === Save control pseudobulk ===
pd.DataFrame({
    "gene": genes,
    "ab_exons_ctrl_sum": ab_ctrl_exons_sum,
    "ab_introns_ctrl_sum": ab_ctrl_introns_sum,
    "ts_ctrl_sum": ts_ctrl_sum
}).to_csv(f"{outdir}/pseudobulk_microglia_macrophage_controls.tsv", sep="\t", index=False)

print("✅ All pseudobulk files saved.")


# SCRIPT=/gpfs/commons/home/kisaev/Leaflet-analysis/Human_Splicing_Foundation/TabulaSapien_vs_Allen_pseudobulk_analysis_2.py
# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/processed_data
#sbatch --mem=500G \
#  --output=merge_microglia.out \
#  --wrap "python -u $SCRIPT"
# %%

