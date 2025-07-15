# %% [markdown]
# ### Take in the raw splice_adata produced by ATSEmapper, clean up its metadata and read in corresponding GE based anndata file 
# 
# - ensure both anndatas contain the same cells and metadata for them in the SAME order 

# %%
import os
import pandas as pd 
from sklearn.decomposition import TruncatedSVD
import anndata as ad
from scipy.sparse import coo_matrix, csr_matrix
from datetime import datetime
import anndata as ad
import numpy as np
import torch 
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from scipy.spatial.distance import cdist
import numpy as np
import sys
import os
import json
import numpy as np
import torch
import anndata as ad
from importlib import reload
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import pyro 
import umap.umap_ as umap
import matplotlib.patches as mpatches
import scipy.sparse
import datetime
import sys
import random

# Output DIR 
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/062025"
print(f"Output directory: {output_dir}", flush=True)

# --- Splice data ---
input_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/ATSE_mapper/junction_processing_20250421/anndatas/merged_anndata.h5ad"
assert os.path.exists(input_file), f"Splice input file does not exist: {input_file}"

splice_adata = ad.read_h5ad(input_file)
assert splice_adata.shape[0] > 0, "Splice AnnData has zero cells"
assert splice_adata.shape[1] > 0, "Splice AnnData has zero features"

splice_adata.obs.reset_index(drop=True, inplace=True)
splice_adata.obs["cell_id_index"] = splice_adata.obs.index 
print(f"The number of cells in the splice dataset is {splice_adata.shape[0]}", flush=True)

# --- ATSEs ---
ATSE_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/ATSE_mapper/ATSE_files/stella_gtf/TMS_atse_file_unanno_also_2025-07-01_03-02-03.txt.gz"
assert os.path.exists(ATSE_file), f"ATSE file does not exist: {ATSE_file}"

atses = pd.read_csv(ATSE_file, sep="\t")
assert "event_id" in atses.columns, "'event_id' column missing from ATSE file"
assert len(atses) > 0, "ATSE file is empty"

print(f"The number of ATSEs in this dataset is {len(atses['event_id'].unique())}", flush=True)

# --- Gene expression data ---
ge_input = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/processed_data/ts_ab_exons_combo_ge_adata_2025-06-23.h5ad"
assert os.path.exists(ge_input), f"Gene expression file does not exist: {ge_input}"

ge_adata = ad.read_h5ad(ge_input)
assert ge_adata.shape[0] > 0, "Gene expression AnnData has zero cells"
assert ge_adata.shape[1] > 0, "Gene expression AnnData has zero features"

ge_adata.obs.reset_index(drop=True, inplace=True)
ge_adata.obs["cell_id_index"] = ge_adata.obs.index 
print(f"The number of cells in the gene expression dataset is {ge_adata.shape[0]}", flush=True)

#----------------------------------------
# Clean up cell_ids in both datasets 
#----------------------------------------

# Gene expression 
# 1. Only Tabula Sapiens cells
ts_like_cells = ge_adata.obs["cell_id_backup"].str.startswith("TSP")

# 2. Split on "_"
ge_parts = ge_adata.obs.loc[ts_like_cells, "cell_id_backup"].str.split("_")

# 3. Parse all fields
sample_id = ge_parts.str[0]
platform = ge_parts.str[1]
donor1 = ge_parts.str[2]        # L144742
plate1 = ge_parts.str[3]        # B1
donor2 = ge_parts.str[4]        # B133819
plate2 = ge_parts.str[5]        # D2
tissue = ge_parts.str[6]        # Liver
subtissue = ge_parts.str[7]     # NA
celltype = ge_parts.str[8]      # Hepatocyte

# 4. Build full cleaned ID
ge_adata.obs.loc[ts_like_cells, "cell_id_clean"] = (
    sample_id + "_donor_" + donor1 + "_" + plate1 + "_" + donor2 + "_" + plate2 + "_" + tissue + "_" + celltype
)

# 5. Fallback for non-TS cells
ge_adata.obs.loc[~ts_like_cells, "cell_id_clean"] = ge_adata.obs.loc[~ts_like_cells, "cell_id_backup"]

# Splicing 
# 1. Only Tabula Sapiens cells
ts_cells = splice_adata.obs["dataset"] == "tabula_sapiens"

# 2. Split after "per_"
per_parts = splice_adata.obs.loc[ts_cells, "cell_id"].str.split("per_").str[-1]
split_parts = per_parts.str.split("_")

# 3. Parse all fields
sample_id = split_parts.str[0]
platform = split_parts.str[1]
donor1 = split_parts.str[2]     # L144736
plate1 = split_parts.str[3]     # G2
donor2 = split_parts.str[4]     # B133819
plate2 = split_parts.str[5]     # M4
tissue = split_parts.str[6]     # Liver
subtissue = split_parts.str[7]  # NA
celltype = split_parts.str[8]   # Hepatocyte

# 4. Build full cleaned ID
splice_adata.obs.loc[ts_cells, "cell_id_clean"] = (
    sample_id + "_donor_" + donor1 + "_" + plate1 + "_" + donor2 + "_" + plate2 + "_" + tissue + "_" + celltype
)

# 5. Fallback for non-TS cells
splice_adata.obs.loc[~ts_cells, "cell_id_clean"] = splice_adata.obs.loc[~ts_cells, "cell_id"]

# Common cells 
common_cells = np.intersect1d(splice_adata.obs["cell_id_clean"], ge_adata.obs["cell_id_clean"])
print(len(common_cells))  # Should be the number of matched cells

# 1. Set obs_names to be the cleaned cell IDs
splice_adata.obs_names = splice_adata.obs["cell_id_clean"]
ge_adata.obs_names = ge_adata.obs["cell_id_clean"]

splice_adata = splice_adata[splice_adata.obs_names.isin(common_cells)].copy()
ge_adata = ge_adata[ge_adata.obs_names.isin(common_cells)].copy()
splice_adata = splice_adata[common_cells].copy()
ge_adata = ge_adata[common_cells].copy()
assert all(splice_adata.obs_names == ge_adata.obs_names), "Mismatch: cell order is wrong!"
print("Cells are now perfectly aligned.")

splice_adata.obs_names = pd.Index(map(str, range(splice_adata.n_obs)))
ge_adata.obs_names = pd.Index(map(str, range(ge_adata.n_obs)))

# Save in output_dir
import datetime
today = datetime.datetime.now().strftime("%Y-%m-%d")

grouped_refined_map = {
    # === NEURONS - Split by major functional classes ===
    'Excitatory_Neuron': [
        'IT', 'L4 IT', 'L5 ET', 'L6 CT', 'L6b', 'L5/6 IT Car3', 'L5/6 NP',
        # Add all the cortical excitatory neurons from your data
        'Exc L2-3 LINC00507 RPL9P17', 'Exc L2-4 RORB GRIK1', 'Exc L3 LINC00507 CTXN3',
        'Exc L3 LINC00507 PSRC1', 'Exc L3 RORB CARTPT', 'Exc L3 THEMIS PLA2G7',
        'Exc L3-4 RORB FOLH1B', 'Exc L3-4 RORB PRSS12', 'Exc L3-4 RORB RPS3P6',
        'Exc L3-4 RORB SEMA6D', 'Exc L3-5 FEZF2 DCN', 'Exc L3-5 FEZF2 ONECUT1',
        'Exc L3-5 LINC00507 SLN', 'Exc L3-5 RORB CD24', 'Exc L3-5 RORB CMAHP',
        'Exc L3-5 RORB HSPB3', 'Exc L3-5 THEMIS ELOF1', 'Exc L3-5 THEMIS UBE2F',
        'Exc L4 RORB BHLHE22', 'Exc L4 RORB CACNG5', 'Exc L4 RORB CCDC168',
        'Exc L4-5 RORB AIM2', 'Exc L4-5 RORB ASCL1', 'Exc L4-5 RORB HNRNPA1P46',
        'Exc L4-5 RORB LCN15', 'Exc L4-5 RORB LINC01474', 'Exc L4-5 RORB RPL31P31',
        'Exc L4-6 RORB HPCA', 'Exc L5 FEZF2 DYRK2', 'Exc L5 FEZF2 MORN2',
        'Exc L5 FEZF2 SCN7A', 'Exc L5 RORB LINC01202', 'Exc L5 RORB SNHG7',
        'Exc L5-6 FEZF2 ANKRD20A1', 'Exc L5-6 FEZF2 CABP7', 'Exc L5-6 FEZF2 CYP26B1',
        'Exc L5-6 FEZF2 MYBPHL', 'Exc L5-6 FEZF2 RSAD2', 'Exc L5-6 RORB LINC00320',
        'Exc L5-6 THEMIS GPR21', 'Exc L5-6 THEMIS IL7R', 'Exc L5-6 THEMIS OR1J1',
        'Exc L5-6 THEMIS THTPA', 'Exc L5-6 THEMIS TMEM233', 'Exc L6 FEZF2 CPZ',
        'Exc L6 FEZF2 ETV4', 'Exc L6 FEZF2 FAM95C', 'Exc L6 FEZF2 KRT17',
        'Exc L6 FEZF2 P4HA3', 'Exc L6 FEZF2 SLITRK6', 'Exc L6 FEZF2 TBC1D26',
        'Exc L6 FEZF2 TBCC', 'Exc L6 FEZF2 VWA2', 'Exc L6 THEMIS C6orf48',
        'Exc L6 THEMIS EGR3', 'Exc L6 THEMIS LINC00343'
    ],
    'Inhibitory_Neuron': [
        'VIP', 'PVALB', 'SST', 'LAMP5',
        # Add all the cortical inhibitory neurons from your data
        'Inh L1 ADARB2 ADAM33', 'Inh L1 ADARB2 DISP2', 'Inh L1 LAMP5 GGT8P',
        'Inh L1 LAMP5 NDNF', 'Inh L1 PAX6 CA4', 'Inh L1 PAX6 GRIP2',
        'Inh L1 SST CXCL14', 'Inh L1 VIP PCDH20', 'Inh L1 VIP PRSS8',
        'Inh L1 VIP SOX11', 'Inh L1 VIP TNFAIP8L3', 'Inh L1-2 PAX6 SCGN',
        'Inh L1-2 PVALB TAC1', 'Inh L1-2 VIP PPAPDC1A', 'Inh L1-2 VIP RPL41P3',
        'Inh L1-3 PAX6 NABP1', 'Inh L1-3 PVALB WFDC2', 'Inh L1-3 VIP ACHE',
        'Inh L1-3 VIP CCDC184', 'Inh L1-3 VIP GGH', 'Inh L1-3 VIP SSTR1',
        'Inh L1-3 VIP ZNF322P1', 'Inh L1-4 LAMP5 DUSP4', 'Inh L1-4 VIP CHRNA2',
        'Inh L1-5 VIP KCNJ2', 'Inh L1-6 LAMP5 CA13', 'Inh L1-6 PVALB SCUBE3',
        'Inh L1-6 VIP PENK', 'Inh L1-6 VIP RCN1', 'Inh L1-6 VIP RGS16',
        'Inh L2-4 PVALB C8orf4', 'Inh L2-4 SST AHR', 'Inh L2-4 VIP DSEL',
        'Inh L2-4 VIP LGI2', 'Inh L2-5 VIP TOX2', 'Inh L2-6 VIP VIP',
        'Inh L3 VIP CBLN1', 'Inh L3-4 PVALB HOMER3', 'Inh L3-5 SST MAFB',
        'Inh L3-6 PVALB MFI2', 'Inh L3-6 VIP KCTD13', 'Inh L4-5 PVALB TRIM67',
        'Inh L4-6 SST MTHFD2P6', 'Inh L5 PVALB CNTNAP3P2', 'Inh L5-6 LAMP5 SFTA3',
        'Inh L5-6 PVALB FAM150B', 'Inh L5-6 PVALB STON2', 'Inh L5-6 SST ISOC1',
        'Inh L5-6 SST KLHL14', 'Inh L5-6 SST TH', 'Inh L6 LAMP5 ANKRD20A11P',
        'Inh L6 LAMP5 C1QL2', 'Inh L6 LHX6 GLP1R', 'Inh L6 SST NPY'
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
        't cell', 'gamma-delta t cell', 'thymocyte', 'mature nk t cell'
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
        'Microglia', 'microglial cell', 'retina - microglia',
        # Add brain microglia from your data
        'Micro L1-6 C1QC'
    ],
    'Macrophage': [
        'macrophage', 'Monocyte_Macrophage', 'tissue-resident macrophage', 
        'muscle macrophage', 'colon macrophage'
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
        'tracheal goblet cell', 'lung ciliated cell'
    ],
    'GI_Epithelial': [
        'enterocyte of epithelium of large intestine', 
        'enterocyte of epithelium proper of small intestine',
        'large intestine goblet cell', 'best4+ intestinal epithelial cell',
        'small intestine goblet cell', 'enterocyte of epithelium proper of ileum',
        'enterocyte of epithelium proper of duodenum', 
        'paneth cell of epithelium of small intestine', 'paneth cell of colon',
        'mature enterocyte', 'intestinal tuft cell', 'tuft cell of colon',
        'best4+ intestinal epithelial cell, human', 'transit amplifying cell of small intestine'
    ],
    'Urogenital_Epithelial': [
        'bladder urothelial cell', 'basal bladder urothelial cell', 
        'intermediate bladder urothelial cell', 'epithelial cell of uterus',
        'kidney epithelial cell', 'basal cell of prostate epithelium',
        'luminal cell of prostate epithelium'
    ],
    'Mammary_Epithelial': [
        'HR positive luminal epithelial cell of mammary gland',
        'secretory luminal epithelial cell of mammary gland',
        'luminal epithelial cell', 'luminal epithelial cell of mammary gland'
    ],
    'Other_Epithelial': [
        'epithelial cell', 'duct epithelial cell', 'ltf+ epithelial cell',
        'basal epithelial cell', 'salivary gland cell', 'medullary thymic epithelial cell',
        'conjunctival epithelial cell', 'corneal epithelial cell', 'glandular epithelial cell',
        'cycling epithelial cell', 'mucus secreting cell', 'biliary epithelial cell',
        'pancreatic ductal cell', 'stratified squamous epithelial cell', 'sebum secreting cell',
        'basal cell', 'epithelial fate stem cell'
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
        'colon endothelial cell', 'retinal blood vessel endothelial cell', 'vascular endothelial cell',
        # Add brain endothelial from your data
        'Endo L2-5 CLDN5'
    ],
    
    # === GLIA - Split by CNS vs PNS ===
    'CNS_Glia': [
        'Astrocyte', 'OPC', 'Oligodendrocyte', 'retina - muller glia', 'mueller cell',
        # Add brain glia from your data
        'Astro L1 FGFR3 FOS', 'Astro L1 FGFR3 MT1G', 'Astro L1-6 FGFR3 ETNPPL',
        'OPC L1-6 MYT1', 'Oligo L4-6 MOBP COL18A1', 'Oligo L4-6 OPALIN',
        'radial glial cell'
    ],
    'PNS_Glia': [
        'enteroglial cell', 'schwann cell'
    ],
    'Glia_Other': [
        'glial cell'
    ],
    
    # === MUSCLE - Split by muscle type ===
    'Smooth_Muscle': [
        'smooth muscle cell', 'airway smooth muscle cell', 'vascular associated smooth muscle cell',
        'bronchial smooth muscle cell'
    ],
    'Cardiac_Muscle': [
        'atrial cardiac muscle cell', 'ventricular cardiac muscle cell',
        'regular atrial cardiac myocyte'
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
        'cd34+ fibroblasts', 'VLMC', 'adventitial cell', 'connective tissue cell',
        'mesenchymal cell', 'alveolar type 2 fibroblast cell',
        # Add brain VLMC from your data
        'VLMC L1-3 CYP1B1'
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
        'pericyte', 'Pericyte', 'myofibroblast cell and pericyte', 'mural cell',
        # Add brain pericyte from your data
        'Peri L1-6 MUSTN1'
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
        'platelet', 'erythrocyte', 'erythroid lineage cell'
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
        'intestinal crypt stem cell of large intestine', 'intestinal crypt stem cell of colon'
    ],
    
    'Secretory_Gland': [
        'acinar cell of salivary gland', 'lacrimal gland functional unit cell', 'myoepithelial cell',
        'acinar cell'
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
        'unknown', ''  # Add empty string for the blank entry in your data
    ]
}

# 1. Define flatten function once
def flatten_grouped_map(grouped_map):
    return {label: broad_type for broad_type, labels in grouped_map.items() for label in labels}

# 2. Create the flat map once (outside the loop)
grouped_broad_map_flat = flatten_grouped_map(grouped_refined_map)

# Update cell type in splice_adata (gene expression was already updated)
splice_adata.obs["broad_cell_type"] = splice_adata.obs["cell_type"].map(grouped_broad_map_flat)

# Handle any remaining NaN values by setting them to "Unknown"
splice_adata.obs["broad_cell_type"] = splice_adata.obs["broad_cell_type"].fillna("Unknown")

print("Mapping complete!")
print(f"Total cells: {len(splice_adata.obs)}")
print(f"Mapped cells: {(splice_adata.obs['broad_cell_type'] != 'Unknown').sum()}")
print(f"Unknown cells: {(splice_adata.obs['broad_cell_type'] == 'Unknown').sum()}")
print("\nBroad cell type counts:")
print(splice_adata.obs["broad_cell_type"].value_counts())

# Save splice_adata
splice_adata_file = os.path.join(output_dir, f"splice_adata_matched_{today}.h5ad")
splice_adata.write_h5ad(splice_adata_file, compression="lzf")
print(f"Saved splice_adata to {splice_adata_file}")

# Save ge_adata
ge_adata_file = os.path.join(output_dir, f"ge_adata_matched_{today}.h5ad")
ge_adata.write_h5ad(ge_adata_file, compression="lzf")
print(f"Saved ge_adata to {ge_adata_file}")

# Submit script like this:
# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/062025
# sbatch --mem=100G --partition=cpu,dev,bigmem --wrap="python /gpfs/commons/home/kisaev/Leaflet-analysis/Human_Splicing_Foundation/GeneExpression/06_align_splice_ge_anndatas.py"