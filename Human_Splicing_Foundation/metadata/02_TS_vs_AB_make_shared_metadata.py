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
ts_adata = sc.read_h5ad(f"{outdir}/tabsap_adata_2025-04-15.h5ad")

# Columns that we have in the shared metadata file in mouse foundation 
# cell_id, age, cell_ontology_class, mouse.id, sex, subtissue, tissue

# === Clean up gene symbols ===
print(f"Cleaning up gene symbols...")
for adata in [ab_exons, ts_adata]:
    adata.var["gene_symbol"] = adata.var["gene_symbol"].astype(str)
    adata = adata[:, ~adata.var["gene_symbol"].duplicated(keep=False)].copy()
    adata.var_names = adata.var["gene_symbol"]

# === Subset to shared genes ===
print(f"Subsetting to shared genes...")
common_genes = set(ab_exons.var_names).intersection(ts_adata.var_names)
for adata in [ab_exons, ts_adata]:
    adata._inplace_subset_var([g in common_genes for g in adata.var_names])

ts_adata.obs["cell_type_grouped"] = ts_adata.obs["free_annotation"]
ab_exons.obs["cell_type_grouped"] = ab_exons.obs["subclass_label"]

# F2S4_190227_086_D01_junctions_with_barcodes.bed
# TSP1_TSP1_smartseq2_NA_B107921_Muscle_NA.multi_star.output_raw.output_raw_per_TSP1_smartseq2_NA_NA_B107921_M23_Muscle_NA_NA_junctions_with_barcodes.bed

ab_metadata = ab_exons.obs[["sample_name", "cell_type_designation_label", "cell_type_alias_label",  "specimen_type", "subclass_label", "donor_sex_label", "external_donor_name_label", "cell_type_grouped", "class_label", "region_label"]].reset_index(drop=True)
ab_metadata["dataset"] = "allen_brain"
# adding age manuall for donors using https://pmc.ncbi.nlm.nih.gov/articles/PMC6919571/table/T1/
# H200.1030 --> 54 (Caucasian)
# H200.1023 --> 43 (iranian descent)
# H200.1025 --> 50 (Caucasian)
ab_metadata["age"] = 0 
ab_metadata.loc[ab_metadata["external_donor_name_label"] == "H200.1030", "age"] = 54
ab_metadata.loc[ab_metadata["external_donor_name_label"] == "H200.1023", "age"] = 43
ab_metadata.loc[ab_metadata["external_donor_name_label"] == "H200.1025", "age"] = 50

# Collect metadata from TabulaSapien
ts_metadata = ts_adata.obs[["old_index", "sample_id", "donor", "tissue", "cell_ontology_class", "compartment", "broad_cell_class", "cell_type_grouped", "age", "sex", "dataset"]].reset_index(drop=True)
# Extract project prefix (e.g., TSP1) from sample_id
ts_metadata["project_id"] = ts_metadata["sample_id"].str.extract(r'^(TSP\d+)')
parts = ts_metadata["old_index"].str.split("_")

# Correct prefix from specific fields
ts_metadata["junc_prefix_core"] = (
    parts.str[0] + "_" +  # TSP1
    parts.str[1] + "_" +  # smartseq2
    parts.str[2] + "_" +  # NA
    parts.str[4] + "_" +  # B107921
    parts.str[6] + "_" +  # Muscle
    parts.str[7]          # NA
)

# Add project prefix and fixed literal string
ts_metadata["cell_id_prefix"] = (
    ts_metadata["project_id"] + "_" +
    ts_metadata["junc_prefix_core"] +
    ".multi_star.output_raw.output_raw_per_"
)

# Full cell_id
ts_metadata["cell_id"] = ts_metadata["cell_id_prefix"] + ts_metadata["old_index"]

# Make a copy
ts_meta = ts_metadata.copy()
ab_meta = ab_metadata.copy()

# Rename columns to match target structure
ts_meta = ts_meta.rename(columns={
    "donor": "donor",
    "sex": "sex",
    "cell_ontology_class": "cell_type",
})

ab_meta = ab_meta.rename(columns={
    "sample_name": "cell_id",
    "external_donor_name_label": "donor",
    "donor_sex_label": "sex",
    "region_label": "tissue", 
    "cell_type_alias_label": "cell_type"
})

# Ensure same columns exist
final_columns = ["cell_id", "donor", "sex", "age", "dataset", "tissue", "cell_type_grouped", "cell_type"]

# Ensure all columns are in the right order and exist
ts_meta = ts_meta[final_columns].reset_index(drop=True)
ab_meta = ab_meta[final_columns].reset_index(drop=True)

# --- Coerce any categoricals to strings ---

from pandas.api.types import CategoricalDtype

for df in [ts_meta, ab_meta]:
    for col in df.columns:
        if isinstance(df[col].dtype, CategoricalDtype):
            df[col] = df[col].astype(str)

# --- Combine ---

combined_metadata = pd.concat([ts_meta, ab_meta], ignore_index=True)

# Finally clean up cell types!

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
combined_metadata['broad_cell_type'] = (combined_metadata['cell_type_grouped'].map(grouped_broad_map_flat).fillna('Other'))

# Determine shared cell types with at least 50 cells per dataset
broad_counts = (
       combined_metadata.groupby(["dataset", "broad_cell_type"])
        .size().reset_index(name="count")
    )

pivot_counts = broad_counts.pivot(index="broad_cell_type", columns="dataset", values="count").fillna(0)
shared_broad_cell_types = pivot_counts[(pivot_counts >= 50).all(axis=1)].index.tolist()
    
print(f"Shared broad cell types (≥50 per dataset): {shared_broad_cell_types}")

combined_metadata.to_csv(
    "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/human_metadata_combined.tsv",
    sep="\t", index=False
)

print("Done saving metadata!")

# Ensure only samples with metadata go through Leaflet and ATSEmapper pipeline
from pathlib import Path

# Paths generated via /gpfs/commons/home/kisaev/Leaflet-analysis/Human_Splicing_Foundation/ATSE_mapper/ATSEmap_SLURM/01_split_junctions.sh
ab_junc_filelist = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/ATSE_mapper/junction_files_AB.txt"
ts_junc_filelist = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/ATSE_mapper/junction_files_TS.txt"

# Output
clean_ts = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/ATSE_mapper/junction_files_TS_subset.txt"
clean_ab = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/ATSE_mapper/junction_files_AB_subset.txt"

# Load metadata
valid_cell_ids = set(combined_metadata["cell_id"])

# Helper function
def filter_junction_paths(filelist_path, output_path, valid_ids):
    with open(filelist_path) as f:
        lines = f.read().splitlines()

    def extract_cell_id_from_path(path):
        fname = Path(path).name
        return fname.replace("_junctions_with_barcodes.bed", "")

    filtered_lines = [
        path for path in lines
        if extract_cell_id_from_path(path) in valid_ids
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(filtered_lines))

    print(f"Retained {len(filtered_lines)} of {len(lines)} in {output_path}")

# Filter both AB and TS junction lists
filter_junction_paths(ab_junc_filelist, clean_ab, valid_cell_ids)
filter_junction_paths(ts_junc_filelist, clean_ts, valid_cell_ids)