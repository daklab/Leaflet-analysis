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
from pathlib import Path

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
# F2S4_190227_086_D01_junctions_with_barcodes.bed
# TSP1_TSP1_smartseq2_NA_B107921_Muscle_NA.multi_star.output_raw.output_raw_per_TSP1_smartseq2_NA_NA_B107921_M23_Muscle_NA_NA_junctions_with_barcodes.bed

ab_metadata = ab_exons.obs[["sample_name", "cell_type_designation_label", "cell_type_alias_label", "specimen_type", "subclass_label", "donor_sex_label", "external_donor_name_label", "class_label", "region_label"]].reset_index(drop=True)
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
ts_metadata = ts_adata.obs[["old_index", "sample_id", "donor", "tissue", "cell_ontology_class", "compartment", "broad_cell_class", "age", "sex", "dataset", "free_annotation"]].reset_index(drop=True)
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
    "free_annotation": "cell_type",
})

ab_meta = ab_meta.rename(columns={
    "sample_name": "cell_id",
    "external_donor_name_label": "donor",
    "donor_sex_label": "sex",
    "region_label": "tissue", 
    "subclass_label": "cell_type"
})

# Ensure same columns exist
final_columns = ["cell_id", "donor", "sex", "age", "dataset", "tissue", "cell_type"]

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
# sex column is currently cateogrical need to reset make just M and F groups 
combined_metadata["sex"] = combined_metadata["sex"].astype(str)
combined_metadata.loc[combined_metadata["sex"] == "male", "sex"] = "M"
combined_metadata.loc[combined_metadata["sex"] == "female", "sex"] = "F"

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

# === Map cell types to broad cell types ===
def flatten_grouped_map(grouped_map):
    return {label: broad_type for broad_type, labels in grouped_map.items() for label in labels}
grouped_broad_map_flat = flatten_grouped_map(grouped_refined_map)
combined_metadata['broad_cell_type'] = (combined_metadata['cell_type'].map(grouped_broad_map_flat).fillna('Other'))
# For cell type that remain Other just use "cell_type" value for them
combined_metadata.loc[combined_metadata["broad_cell_type"] == "Other", "broad_cell_type"] = combined_metadata["cell_type"]

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