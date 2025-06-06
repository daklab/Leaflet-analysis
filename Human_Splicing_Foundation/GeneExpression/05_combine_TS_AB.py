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
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import pickle
from datetime import date

# Combine the cleaned up and length normalized AB + TS gene expression adata objects into one!
WD="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/processed_data"
ab_adata_file = "AB_adjusted_GeneExpression_via_exon_intron_regression_2025-05-03.h5ad"
ts_adata_file = "TS_GeneExpression_with_length_norm_2025-05-03.h5ad"

ab_adata = ad.read_h5ad(f"{WD}/{ab_adata_file}")
ts_adata = ad.read_h5ad(f"{WD}/{ts_adata_file}")

# Get ab_adata metadata 
# Read in metadata
metadata = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/human_metadata_combined.tsv"
metadata = pd.read_csv(metadata, sep="\t")
metadata = metadata[metadata["dataset"] == "allen_brain"]

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
final_columns = ["cell_id", "donor", "sex", "age", "dataset", "tissue", "cell_type_grouped", "cell_type", "broad_cell_type"]

# Ensure all columns are in the right order and exist
ts_adata.obs = ts_adata.obs[final_columns].reset_index(drop=True)
ab_adata.obs = ab_adata.obs[final_columns].reset_index(drop=True)
ab_adata.layers["log_norm"] = ab_adata.layers.pop("predicted_log_norm_ts")

# === Check 1: Gene order ===
assert np.array_equal(ab_adata.var_names, ts_adata.var_names), \
    "🚨 ERROR: Gene names not aligned across ab_adata and ts_adata!"
assert np.array_equal(ab_adata.var.index, ts_adata.var.index), \
    "🚨 ERROR: Gene indices not in the same order!"
print("✅ Gene order is consistent between ab_adata and ts_adata.")

# === Check 2: Gene metadata equality ===
shared_gene_cols = ["gene_symbol", "gene_name", "mean_transcript_length"]
for col in shared_gene_cols:
    assert np.all(ab_adata.var[col].values == ts_adata.var[col].values), \
        f"🚨 ERROR: Mismatch in gene metadata column: {col}"
print("✅ Gene metadata columns are consistent.")

# === Check 3: Nonzero predicted log counts in ab_adata ===
library_size_ab = np.asarray(ab_adata.layers["log_norm"].sum(axis=1)).flatten()
zero_count_cells = np.sum(library_size_ab == 0)
assert zero_count_cells == 0, f"🚨 ERROR: {zero_count_cells} cells in ab_adata have zero predicted log counts."
print("✅ All ab_adata cells have non-zero predicted log counts.")

# === Check 4: All required columns exist in obs ===
for df, name in zip([ab_adata.obs, ts_adata.obs], ["ab_adata", "ts_adata"]):
    for col in final_columns:
        assert col in df.columns, f"🚨 ERROR: Column {col} missing in {name}.obs!"
print("✅ All required metadata columns are present.")

# === Compute library size based on length-normalized raw counts ===
for adata, name in zip([ab_adata, ts_adata], ["ab_adata", "ts_adata"]):
    if "length_norm" not in adata.layers:
        raise ValueError(f"🚨 ERROR: {name} is missing 'length_norm' layer.")

    # Calculate total normalized counts per cell
    library_size = np.asarray(adata.layers["length_norm"].sum(axis=1)).flatten()

    # Store in .obs and .obsm
    adata.obs["library_size"] = library_size
    adata.obsm["library_size"] = library_size[:, np.newaxis]

    # Sanity checks
    num_nan = np.sum(np.isnan(library_size))
    num_zero = np.sum(library_size == 0)
    assert num_nan == 0, f"🚨 ERROR: {num_nan} NaNs found in library size for {name}"
    assert num_zero == 0, f"🚨 ERROR: {num_zero} zero-library cells found in {name}"
    print(f"✅ {name}: Library size stored with no NaNs or zeros.")

combined_adata = ad.concat([ab_adata, ts_adata], join="outer", 
                        label="batch", keys=["allen_brain_exons", "tabula_sapien"])

# Set cell_id as obs_names and drop redundant column
combined_adata.obs_names = combined_adata.obs["cell_id"]
# Rename the column to something else (preserve info without conflict)
combined_adata.obs["cell_id_backup"] = combined_adata.obs["cell_id"]
combined_adata.obs = combined_adata.obs.drop(columns=["cell_id"])

# save in WD 
output_dir = WD
today = datetime.datetime.now()
today = today.strftime("%Y-%m-%d")
combined_adata.write_h5ad(os.path.join(output_dir, f"ts_ab_exons_combo_ge_adata_{today}.h5ad"), compression="lzf")
print(f"Saved combined anndata objects to {output_dir}")

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/processed_data
# sbatch --mem=100G --partition dev,cpu --wrap "python /gpfs/commons/home/kisaev/Leaflet-analysis/Human_Splicing_Foundation/TabulaSapien_vs_Allen_pseudobulk_analysis_5.py"