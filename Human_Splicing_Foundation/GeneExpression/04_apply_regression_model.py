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

# Add the directory containing the shared utils to the Python path
sys.path.append("/gpfs/commons/home/kisaev/Leaflet-analysis/Multi_Species_Splicing_Foundation/shared_utils")

# Import utility functions
from gene_processing import (
    extract_gene_transcript_info, 
    normalize_by_gene_length,
    safe_stringify_obs,
    preprocess_anndata,
    normalize_and_log_transform
)

# === Genome Paths ===
gtf_hg38 = "/gpfs/commons/datasets/controlled/BRAIN_NeMO/human-reference/gencode/gencode.v45.primary_assembly.annotation.gtf"
db_file = "/gpfs/commons/datasets/controlled/BRAIN_NeMO/human-reference/gencode/gencode_hg38.db"

WD="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/processed_data"
# Get lengths of genes 
gene_info_df = pd.read_csv(f"{WD}/gene_info_df_2025-06-22.csv")

# === Paths and output directories for anndatas ===
print(f"Reading in the anndata objects...")
outdir = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/processed_data/"
ab_exons = sc.read_h5ad(f"{outdir}/ab_adata_exons_filtered_2025-06-23.h5ad")
ab_introns = sc.read_h5ad(f"{outdir}/ab_adata_introns_filtered_2025-06-23.h5ad")
ts_adata = sc.read_h5ad(f"{outdir}/tabsap_adata_filtered_2025-06-23.h5ad")

# === Make sparse if needed ===
for adata in [ab_exons, ab_introns, ts_adata]:
    print(f"Checking {adata} for sparse matrix")
    if not sp.issparse(adata.layers["raw_counts"]):
        adata.layers["raw_counts"] = csr_matrix(adata.layers["raw_counts"]).copy()
        print(f"Converted layer to sparse matrix!")

# === Subset to shared genes ===
ts_adata.var["gene_name"] = ts_adata.var["gene_symbol"]
ab_introns.var["gene_name"] = ab_introns.var["gene_symbol"]
ab_exons.var["gene_name"] = ab_exons.var["gene_symbol"]

# assert that order of genes in ts_adata, ab_introns, and ab_adata_exons is the same 
assert np.array_equal(ts_adata.var["gene_name"], ab_introns.var["gene_name"]), "ERROR: Gene order mismatch between ts_adata and ab_introns!"
assert np.array_equal(ts_adata.var["gene_name"], ab_exons.var["gene_name"]), "ERROR: Gene order mismatch between ts_adata and ab_adata_exons!"

# === Sanity check: ensure gene order is the same across all AnnData objects ===
gene_order = ab_exons.var["gene_name"].values
assert np.array_equal(gene_order, ab_introns.var["gene_name"].values), \
    "Gene order mismatch between ab_adata_exons and ab_introns!"
assert np.array_equal(gene_order, ts_adata.var["gene_name"].values), \
    "Gene order mismatch between ab_adata_exons and ts_adata!"
print("Gene order is consistent across ab_adata_exons, ab_introns, and ts_adata.")

# Get gene lengths
gene_lengths = ts_adata.var.loc[ab_exons.var_names, ['mean_transcript_length', 'mean_intron_length']]

# === Extract exon and intron RAW counts ===
normalize_by_gene_length(ab_exons)
normalize_by_gene_length(ab_introns)
normalize_by_gene_length(ts_adata)

# Confirm no cells with zero counts across the board 
for name, adata in zip(["ab_exons", "ab_introns", "ts_adata"], [ab_exons, ab_introns, ts_adata]):
    length_norm = adata.layers["length_norm"]
    row_sums = length_norm.sum(axis=1).A1  # get sum per cell
    zero_cells = np.sum(row_sums == 0)
    print(f"{zero_cells} cells in {name} have zero length-normalized counts before library size adjustment.")
    assert zero_cells == 0, f"ERROR: {zero_cells} all-zero cells found in {name}!"

# Now normalize by library size 
normalize_and_log_transform(ab_exons)
normalize_and_log_transform(ab_introns)
normalize_and_log_transform(ts_adata)

# Confirm no cells with zero counts across the board after library size adjustment 
for name, adata in zip(["ab_exons", "ab_introns", "ts_adata"], [ab_exons, ab_introns, ts_adata]):
    log_norm = adata.layers["log_norm"]
    row_sums = log_norm.sum(axis=1).A1  # get sum per cell
    zero_cells = np.sum(row_sums == 0)
    print(f"{zero_cells} cells in {name} have zero length-normalized counts before library size adjustment.")
    assert zero_cells == 0, f" ERROR: {zero_cells} all-zero cells found in {name}!"

# === Load linear regression model ===
linear_model_file = "/gpfs/commons/home/kisaev/Leaflet-analysis/Human_Splicing_Foundation/GeneExpression/linear_log_norm_ts_model.pkl"
with open(linear_model_file, 'rb') as f:
    model_linear = pickle.load(f)
print("Linear Model Summary:")
print(model_linear.summary())

# Estimate total spliced counts from exons and introns (from single nuclei data to single cell data as a way to account for differences in spliced products)
# For each gene, we will estimate the total spliced counts using the linear model for each cell, if its counts were missing, we will not include it 
intercept = model_linear.params["const"]
coef_exons = model_linear.params["exons"]
coef_introns = model_linear.params["introns"]
print(f"   ✓ Model coefficients: intercept={intercept:.4f}, exons={coef_exons:.4f}, introns={coef_introns:.4f}")

log_exons = ab_exons.layers["log_norm"]  # CSR sparse matrix
log_introns = ab_introns.layers["log_norm"]  # CSR sparse matrix
log_pred_tot = log_exons.multiply(coef_exons) + log_introns.multiply(coef_introns)
log_pred_tot_with_intercept = log_pred_tot.copy()
log_pred_tot_with_intercept.data += intercept

# Create a new object using ab_exons as base
ab_adata = ab_exons.copy()
ab_adata.layers["predicted_log_norm_ts"] = ab_adata.layers["log_norm"].copy()
# Store adjusted log-normalized total spliced counts (predicted values)
ab_adata.layers["predicted_log_norm_ts"] = log_pred_tot_with_intercept

# Preserve raw counts explicitly and ensure CSR format
ab_adata.layers["raw_counts"] = csr_matrix(ab_adata.layers["raw_counts"])
ab_adata.layers["length_norm"] = csr_matrix(ab_adata.layers["length_norm"])
ab_adata.layers["predicted_log_norm_ts"] = csr_matrix(ab_adata.layers["predicted_log_norm_ts"])

# Clean up obs to only necessary metadata (optional but clean)
ab_adata.obs = ab_adata.obs[["sample_name"]]

# Remove old exon-only log_norm layer to avoid confusion
if "log_norm" in ab_adata.layers:
    del ab_adata.layers["log_norm"]

# === Sanity check 1: Non-zero total raw counts per cell ===
raw_sums = ab_adata.layers["raw_counts"].sum(axis=1).A1
zero_raw = np.sum(raw_sums == 0)
print(f"Cells with zero raw counts: {zero_raw}")
assert zero_raw == 0, "ERROR: Some cells have zero total raw counts!"

# === Sanity check 2: Non-zero total predicted log counts per cell ===
pred_sums = ab_adata.layers["predicted_log_norm_ts"].sum(axis=1).A1
zero_pred = np.sum(pred_sums == 0)
print(f"Cells with zero predicted log counts: {zero_pred}")
assert zero_pred == 0, "ERROR: Some cells have zero total predicted log counts!"

# === Sanity check: ensure gene order is the same in ab_adata and ts_adata ===
assert np.array_equal(ab_adata.var_names, ts_adata.var_names), \
    "ERROR: Gene order mismatch between ab_adata and ts_adata!"
print("Gene order is consistent between ab_adata and ts_adata.")

# Save the object
today = date.today().strftime("%Y-%m-%d")
outfile = os.path.join(outdir, f"AB_adjusted_GeneExpression_via_exon_intron_regression_{today}.h5ad")
ab_adata.write_h5ad(outfile, compression="lzf")
print(f"Saved the object to {outfile}")

# Ensure all sparse matrices in ts_adata are in CSR format
print("Converting ts_adata sparse matrices to CSR format...")
for layer in ts_adata.layers:
    if sp.issparse(ts_adata.layers[layer]):
        ts_adata.layers[layer] = csr_matrix(ts_adata.layers[layer])
        print(f"Converted {layer} to CSR format")

outfile = os.path.join(outdir, f"TS_GeneExpression_with_length_norm_{today}.h5ad")
ts_adata.write_h5ad(outfile, compression="lzf")
print(f"Saved the object to {outfile}")

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/processed_data
# sbatch --mem=140G --partition cpu,bigmem --wrap "python /gpfs/commons/home/kisaev/Leaflet-analysis/Human_Splicing_Foundation/GeneExpression/04_apply_regression_model.py"