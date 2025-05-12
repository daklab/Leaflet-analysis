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

# Define module paths
src_path = "/gpfs/commons/home/kisaev/Leaflet-private/src/"

# Add to sys.path if not already present
if src_path not in sys.path:
    sys.path.append(src_path)

# Import custom modules
import BetaDirichletFactor.waypoints as wayp # for getting centered sparse junction usage ratios 

# %%
# Output DIR 
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/042025"
print(f"Output directory: {output_dir}", flush=True)

# %%
print("Loading splicing anndata object!")
input_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper/junction_processing_20250415/anndatas/merged_anndata.h5ad"
splice_adata = ad.read_h5ad(input_file)
splice_adata.obs.reset_index(drop=True, inplace=True)
splice_adata.obs["cell_id_index"] = splice_adata.obs.index 
print(f"The number of cells in the dataset is {splice_adata.shape[0]}", flush=True)

ATSE_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper/ATSE_files/MOUSE_FOUNDATION_ATSE_FILE_unanno_also_2025-04-26_19-55-26.txt.gz"
atses = pd.read_csv(ATSE_file, sep="\t")
print(f"The number of ATSEs in this dataset is {len(atses['event_id'].unique())}", flush=True)

# combine gene expression data 
print("Loading gene expression object!")
ge_input = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/processed_data/adjusted_gene_expression/Combined_adjusted_GeneExpression_2025-04-30.h5ad"
ge_adata = ad.read_h5ad(ge_input)
ge_adata.obs.reset_index(drop=True, inplace=True)
ge_adata.obs["cell_id_index"] = ge_adata.obs.index 
print(f"The number of cells in the dataset is {ge_adata.shape[0]}", flush=True)

# %%
# Additional processing that needs to be done on the splicing data 
metadata = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/metadata_mouse_metadata_combined.csv"
metadata = pd.read_csv(metadata)

# Need to reset the columns because the AB junction files sample_names don't correspond to cell_ids the same way
splice_adata.obs = splice_adata.obs.drop(columns=["age", "cell_ontology_class", "mouse.id", "sex", "subtissue", "tissue"])
splice_adata = splice_adata[splice_adata.obs["cell_id"].isin(metadata["cell_id"])]
print(f"The number of cells in the dataset is {splice_adata.shape[0]}", flush=True)

# BEFORE ANYTHING: check cell_id uniqueness
assert splice_adata.obs['cell_id'].is_unique, "splice_adata.obs['cell_id'] is not unique"
assert metadata['cell_id'].is_unique, "metadata['cell_id'] is not unique"

# Set index in metadata to 'cell_id'
splice_adata.obs = splice_adata.obs.set_index('cell_id')
metadata = metadata.set_index('cell_id')
assert metadata.index.is_unique, "Non-unique cell IDs in metadata!"
common_cells = splice_adata.obs.index.intersection(metadata.index)
print(f"Number of common cells: {len(common_cells)}")

splice_adata = splice_adata[common_cells].copy()
assert splice_adata.n_obs == len(common_cells)
merged_obs = splice_adata.obs.merge(metadata, left_index=True, right_index=True, how="left", validate="one_to_one")
assert merged_obs.shape[0] == splice_adata.n_obs
splice_adata.obs = merged_obs

splice_adata.obs["cell_id_index"] = range(len(splice_adata.obs))
splice_adata.obs["dataset"] = "TMS"
splice_adata.obs.loc[splice_adata.obs["age"] == "2m", "dataset"] = "AB"
print(splice_adata.obs.dataset.value_counts())

# Update cell_ids 
splice_adata.obs["cell_name"] = splice_adata.obs["tissue"]
splice_adata.obs["cell_id"] = splice_adata.obs.index
splice_adata.obs.loc[splice_adata.obs["dataset"] == "TMS", "cell_name"] = splice_adata.obs.loc[splice_adata.obs["dataset"] == "TMS", "cell_id"]
splice_adata.obs.reset_index(drop=True, inplace=True)

# Update major cell type labels
broad_cell_type_map = {
    "Micro-PVM": "microglial cell", # to match how the cells are in the TMS dataset
    "Astro": "GLIAL CELL",
    "Oligo": "GLIAL CELL",
    "VLMC": "VLMCs",
    "Endo": "ENDOTHELIAL CELL",
    "SMC-Peri": "PERICYTE",
}

# Supplemental known excitatory subregions (often hippocampal / entorhinal)
known_exc_subregions = [
    "CA1", "CA3", "CA2", "DG", "SUB", "ProS", "HATA", "Mossy", "PPP", "RHP"
]

def map_to_broad_category(label):
    for keyword, category in broad_cell_type_map.items():
        if keyword in label:
            return category

    if "Car3" in label:
        return "Other non-neuronal (Car3+)"

    if any(x in label for x in ["Sst", "Pvalb", "Vip", "Lamp5", "Sncg", "Meis2", "Ntng1", "Pax6", "CR"]):
        return "Inhibitory Neurons"

    if (
        any(x in label for x in ["L2", "L3", "L4", "L5", "L6"])
        or any(x in label for x in known_exc_subregions)
        or any(x in label for x in ["IT", "CT", "PT", "NP", "CTX", "ENT", "PAR", "POST", "RSP", "HPF"])
    ):
        return "Excitatory Neurons"

    return label  # fallback to original label

# Apply 
splice_adata.obs["broad_cell_type"] = splice_adata.obs["cell_ontology_class"].apply(map_to_broad_category)

# If AB and Microglia cells
splice_adata.obs.loc[splice_adata.obs["dataset"] == "AB", "tissue"] = "Brain_Non-Myeloid"
is_ab_microglia = (splice_adata.obs["dataset"] == "AB") & (splice_adata.obs["broad_cell_type"] == "microglial cell")
splice_adata.obs.loc[is_ab_microglia, "tissue"] = "Brain_Myeloid"

# Add technology used 
splice_adata.obs["seqtech"] = "single_nuclei"
splice_adata.obs.loc[splice_adata.obs["dataset"] == "TMS", "seqtech"] = "single_cell"
splice_adata.obs.seqtech.value_counts()

# Clean cell_id only for TMS cells
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
print(f"Done processing splicing AnnData object!")

def extract_cell_name(cell_id):
    # First, replace dots with underscores to standardize
    cell_id = cell_id.replace(".", "_")
    parts = cell_id.split("_")
    if len(parts) >= 2:
        return parts[0] + "_" + parts[1]
    else:
        return cell_id  # fallback if somehow weird

# Apply to ge_adata
ge_adata.obs["cell_name"] = ge_adata.obs["cell_id"]

# Now, only apply the cleaning to 'tabula_muris_senis' cells
mask = ge_adata.obs["dataset"] == "tabula_muris_senis"
ge_adata.obs.loc[mask, "cell_name"] = ge_adata.obs.loc[mask, "cell_id"].apply(extract_cell_name)

# Fix cell_name column in splice_adata so they match
mask = splice_adata.obs["dataset"] == "TMS"
splice_adata.obs.loc[mask, "cell_name"] = splice_adata.obs.loc[mask, "cell_clean"]

# Replace_cell_id with cell_name now in both
splice_adata.obs["cell_id"] = splice_adata.obs["cell_name"]
ge_adata.obs["cell_id"] = ge_adata.obs["cell_name"]

print(ge_adata[ge_adata.obs["cell_name"].isin(splice_adata.obs["cell_name"])])

# Step 1: Get common cells
common_cells = pd.Index(ge_adata.obs['cell_name']).intersection(pd.Index(splice_adata.obs['cell_name']))

# Step 2: Create boolean masks for each dataset
ge_mask = pd.Series(ge_adata.obs['cell_name'].isin(common_cells).values)
splice_mask = pd.Series(splice_adata.obs['cell_name'].isin(common_cells).values)

# Step 3: Get indices that would sort the common cells
ge_idx = np.where(ge_mask)[0]
splice_idx = np.where(splice_mask)[0]

# Step 4: Create a mapping from cell_name to the desired order
common_cells_sorted = np.sort(common_cells)
order_map = {cell: i for i, cell in enumerate(common_cells_sorted)}

# Step 5: Get new indices in the common cells order
ge_ordered = np.array([order_map[ge_adata.obs['cell_name'].iloc[i]] for i in ge_idx])
splice_ordered = np.array([order_map[splice_adata.obs['cell_name'].iloc[i]] for i in splice_idx])

# Step 6: Sort indices by the ordering
ge_sorted_idx = ge_idx[np.argsort(ge_ordered)]
splice_sorted_idx = splice_idx[np.argsort(splice_ordered)]

# Step 7: Apply the indices to slice anndata without copy
ge_adata_view = ge_adata[ge_sorted_idx]
splice_adata_view = splice_adata[splice_sorted_idx]

# Step 8: Update the indices without copy
ge_adata_view.obs['cell_id'] = ge_adata_view.obs['cell_name'].values
splice_adata_view.obs['cell_id'] = splice_adata_view.obs['cell_name'].values

ge_adata_view.obs['cell_id_index'] = np.arange(len(ge_adata_view))
splice_adata_view.obs['cell_id_index'] = np.arange(len(splice_adata_view))

# Add column to gene expressed anndata for total library_size using the "length_norm" layer values (sparse matrix)
library_size = np.asarray(ge_adata_view.layers["length_norm"].sum(axis=1)).flatten()
ge_adata_view.obs["library_size"] = library_size # 1D column 
# Store in .obsm (2D array, with shape [n_cells, 1])
ge_adata_view.obsm["X_library_size"] = library_size[:, np.newaxis]

# Add to obsm part 
print(f"Done getting library sizes for all cells using length adjusted counts!")

# Verify indices match
assert np.all(ge_adata_view.obs['cell_name'].values == splice_adata_view.obs['cell_name'].values)
assert np.all(ge_adata_view.obs['cell_id'].values == splice_adata_view.obs['cell_id'].values)

# Save both files with related names
base_filename = f"mouse_foundation_data_{timestamp}"
splice_file = os.path.join(output_dir, f"{base_filename}_splice.h5ad")
ge_file = os.path.join(output_dir, f"{base_filename}_ge.h5ad")
ge_adata_view.obs.reset_index(inplace=True, drop=True)
splice_adata_view.obs.reset_index(inplace=True, drop=True)

# After defining splice_adata_view
junction_counts = splice_adata_view.layers["cell_by_junction_matrix"].tocoo()
cluster_counts = splice_adata_view.layers["cell_by_cluster_matrix"].tocoo()

# Add centered values to splice_adata 
from scipy.sparse import csr_matrix
psi = wayp.calculate_centered_psi(junction_counts, cluster_counts)
# Convert to CSR before saving
splice_adata_view.layers["junc_ratio"] = csr_matrix(psi)
print(f"Done getting sparse centered PSI values!")

# Final ATSE clean up
print(f"Saving splice AnnData to: {splice_file}", flush=True)
splice_adata_view.write(splice_file, compression="lzf")
print(splice_adata_view)

print(f"Saving gene expression AnnData to: {ge_file}", flush=True)
ge_adata_view.write(ge_file, compression="lzf")
print(ge_adata_view)

print("All saves complete!", flush=True)

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025
# sbatch --mem=300G --partition cpu,dev,bigmem --wrap="python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/align_splice_ge_anndatas.py"
