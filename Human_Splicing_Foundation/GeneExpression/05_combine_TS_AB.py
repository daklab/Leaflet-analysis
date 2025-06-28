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
from datetime import datetime
from collections import defaultdict
import scipy.sparse as sp
from collections import defaultdict
import gffutils 
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import pickle
from datetime import date

# === Paths and input files ===
WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/processed_data"
ab_adata_file = "AB_adjusted_GeneExpression_via_exon_intron_regression_2025-06-23.h5ad"
ts_adata_file = "TS_GeneExpression_with_length_norm_2025-06-23.h5ad"

# === Load AnnData objects ===
print("Loading AnnData objects...")
ab_adata = ad.read_h5ad(f"{WD}/{ab_adata_file}")
ts_adata = ad.read_h5ad(f"{WD}/{ts_adata_file}")

# Get ab_adata metadata 
# Read in metadata
metadata = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/human_metadata_combined.tsv"
metadata = pd.read_csv(metadata, sep="\t")

# Align metadata to adata
ab_adata.obs["external_name"] = ab_adata.obs_names
ab_adata.obs.index = ab_adata.obs["sample_name"]
ab_adata.obs_names = ab_adata.obs.index
# Add metadata to ab_adata
metadata_key = "cell_id"
metadata_sub = metadata[metadata[metadata_key].isin(ab_adata.obs_names)].copy()
metadata_sub = metadata_sub.set_index(metadata_key)
ab_adata = ab_adata[ab_adata.obs_names.isin(metadata_sub.index)].copy()
ab_adata.obs = metadata_sub.loc[ab_adata.obs_names]
print(all(ts_adata.var_names == ab_adata.var_names))

ts_adata.obs["cell_type_grouped"] = ts_adata.obs["free_annotation"]
ts_adata.obs["cell_id"] = ts_adata.obs["old_index"]
ts_adata.obs["cell_type"] = ts_adata.obs["broad_cell_class"]
ab_adata.obs["cell_id"] = ab_adata.obs.index 
ab_adata.obs.reset_index(drop=True, inplace=True)

# Columns to keep for ab_adata.obs
final_columns = ["cell_id", "donor", "sex", "age", "dataset", "tissue", "cell_type", "broad_cell_type"]

# Ensure all columns are in the right order and exist
ts_adata.obs = ts_adata.obs[final_columns].reset_index(drop=True)
ab_adata.obs = ab_adata.obs[final_columns].reset_index(drop=True)
ab_adata.layers["log_norm"] = ab_adata.layers.pop("predicted_log_norm_ts")

# === Check 1: Gene order ===
assert np.array_equal(ab_adata.var_names, ts_adata.var_names), \
    "ERROR: Gene names not aligned across ab_adata and ts_adata!"
assert np.array_equal(ab_adata.var.index, ts_adata.var.index), \
    "ERROR: Gene indices not in the same order!"
print("Gene order is consistent between ab_adata and ts_adata.")

# === Verify gene metadata ===
shared_gene_cols = ["gene_symbol", "gene_name", "mean_transcript_length"]
for col in shared_gene_cols:
    assert np.all(ab_adata.var[col].values == ts_adata.var[col].values), \
        f"ERROR: Mismatch in gene metadata column: {col}"
print("Gene metadata columns are consistent.")

# === Verify non-zero counts ===
library_size_ab = np.asarray(ab_adata.layers["log_norm"].sum(axis=1)).flatten()
zero_count_cells = np.sum(library_size_ab == 0)
assert zero_count_cells == 0, f"ERROR: {zero_count_cells} cells in ab_adata have zero predicted log counts."
print("All ab_adata cells have non-zero predicted log counts.")

# === Compute library size based on length-normalized raw counts ===
for adata, name in zip([ab_adata, ts_adata], ["ab_adata", "ts_adata"]):
    if "length_norm" not in adata.layers:
        raise ValueError(f"ERROR: {name} is missing 'length_norm' layer.")

    # Calculate total normalized counts per cell
    library_size = np.asarray(adata.layers["length_norm"].sum(axis=1)).flatten()
    # Make sure to do np.floor of library_size
    library_size = np.floor(library_size)

    # Store in .obs and .obsm
    adata.obs["library_size"] = library_size
    adata.obsm["library_size"] = library_size[:, np.newaxis]

    # Sanity checks
    num_nan = np.sum(np.isnan(library_size))
    num_zero = np.sum(library_size == 0)
    assert num_nan == 0, f"ERROR: {num_nan} NaNs found in library size for {name}"
    assert num_zero == 0, f"ERROR: {num_zero} zero-library cells found in {name}"
    print(f"{name}: Library size stored with no NaNs or zeros.")

# === Combine AnnData objects ===
print("Combining AnnData objects...")
combined_adata = ad.concat([ab_adata, ts_adata], join="outer", 
                        label="batch", keys=["allen_brain_exons", "tabula_sapien"])

# Set cell_id as obs_names and drop redundant column
combined_adata.obs_names = combined_adata.obs["cell_id"]
combined_adata.obs["cell_id_backup"] = combined_adata.obs["cell_id"]
combined_adata.obs = combined_adata.obs.drop(columns=["cell_id"])

# sex column is currently cateogrical need to reset make just M and F groups 
combined_adata.obs["sex"] = combined_adata.obs["sex"].astype(str)
combined_adata.obs.loc[combined_adata.obs["sex"] == "male", "sex"] = "M"
combined_adata.obs.loc[combined_adata.obs["sex"] == "female", "sex"] = "F"

# === Save combined object ===
today = datetime.now().strftime("%Y-%m-%d")
output_file = os.path.join(WD, f"ts_ab_exons_combo_ge_adata_{today}.h5ad")
combined_adata.write_h5ad(output_file, compression="lzf")
print(f"Saved combined AnnData object to {output_file}")

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/processed_data
# sbatch --mem=150G --partition dev,cpu --wrap "python /gpfs/commons/home/kisaev/Leaflet-analysis/Human_Splicing_Foundation/GeneExpression/05_combine_TS_AB.py"