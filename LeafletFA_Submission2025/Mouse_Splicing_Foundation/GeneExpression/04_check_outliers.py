# %%
import os
import sys
import pandas as pd
import numpy as np
import anndata as ad
import datetime
import traceback
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.sparse import csr_matrix
import re
import numpy as np
import pandas as pd
from scipy.stats import zscore

# Input file path (make sure using most recent aligned anndatas)
GE_INPUT = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/aligned_gene_expression_data_20251003_004515.h5ad"
SPLICE_INPUT = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/aligned_splicing_data_20251003_004515.h5ad"
OUTPUT_DIR_PLOTS = "/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/results/gene_expression/plots"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR_PLOTS, exist_ok=True)
print(f"=== PLOTS WILL BE SAVED TO: {OUTPUT_DIR_PLOTS} ===")

# Configure matplotlib to not use interactive backend
plt.switch_backend('Agg')

print("=== STARTING QC PIPELINE ===")
print(f"Start time: {datetime.datetime.now()}")

# Read the input file
print("=== LOADING DATA ===")

print(f"Loading splicing data from: {SPLICE_INPUT}")
splice_adata = ad.read_h5ad(SPLICE_INPUT)
print(splice_adata.var.head())

print(f"Loading gene expression data from: {GE_INPUT}")
ge_adata = ad.read_h5ad(GE_INPUT)

# Check the number of cells and genes
print(f"=== INITIAL DATA DIMENSIONS ===")
print(f"Gene expression - Number of cells: {ge_adata.n_obs}")
print(f"Gene expression - Number of genes: {ge_adata.n_vars}")
print(f"Splicing - Number of cells: {splice_adata.n_obs}")
print(f"Splicing - Number of junctions: {splice_adata.n_vars}")

# %%
print("=== ALIGNING CELL IDS ===")
splice_adata.obs_names = splice_adata.obs["cell_id"]
ge_adata.obs_names = ge_adata.obs["cell_id"]

# Assert that obs_names are the same
assert np.all(ge_adata.obs_names == splice_adata.obs_names)
print("✓ Cell IDs successfully aligned between datasets")

# %%
# reset the index
ge_adata.var = ge_adata.var.reset_index(drop=True)

print("=== CLEANING CELL IDS ===")

# Confirm that the cell ids are the same
assert np.all(ge_adata.obs_names == splice_adata.obs_names)
assert np.all(ge_adata.obs["cell_id"].values == splice_adata.obs["cell_id"].values)
print("✓ Cell order confirmed consistent between datasets")

# %%
print("=== FILTERING CELL TYPES WITH <50 CELLS ===")
# Count cells per broad_cell_type
cell_type_counts = ge_adata.obs['broad_cell_type'].value_counts()
print(f"Cell type counts before filtering:")
print(cell_type_counts.sort_values(ascending=False))

# Keep only cell types with at least 50 cells
valid_cell_types = cell_type_counts[cell_type_counts >= 50].index
print(f"\nCell types with ≥50 cells: {len(valid_cell_types)}")
print(f"Cell types being removed: {len(cell_type_counts) - len(valid_cell_types)}")

# Filter both datasets
cells_before = ge_adata.n_obs
cell_mask = ge_adata.obs['broad_cell_type'].isin(valid_cell_types)
ge_adata = ge_adata[cell_mask].copy()
splice_adata = splice_adata[cell_mask].copy()

print(f"Filtered from {cells_before} to {ge_adata.n_obs} cells")
print(f"Final cell type counts:")
print(ge_adata.obs['broad_cell_type'].value_counts().sort_values(ascending=False))

# Verify alignment
assert np.all(ge_adata.obs_names == splice_adata.obs_names)
print("✓ Datasets remain aligned after cell type filtering")

# %%
print("=== STARTING QC METRICS CALCULATION ===")

# ---------- 0   Tag rRNA genes once ---------------------------------
print("Flagging rRNA genes...")
# non-capturing groups (?: … ) silence the pandas warning
rrna_regex = r'(?:^|,\s*)rRNA(?:_pseudogene)?(?:$|,\s*)'
ge_adata.var['is_rRNA'] = ge_adata.var['gene_biotype'].str.contains(
    rrna_regex, regex=True, na=False
)
n_rrna = ge_adata.var['is_rRNA'].sum()
print(f"✓ Flagged {n_rrna} rRNA genes out of {ge_adata.n_vars} total genes")

# ---------- 1   All QC metrics in a single scanpy call --------------
print("Calculating QC metrics...")
sc.pp.calculate_qc_metrics(
    ge_adata,
    qc_vars=['is_rRNA'],               # adds total_counts_is_rRNA & pct_counts_is_rRNA
    layer='raw_counts',
    percent_top=None,
    inplace=True
)
print("✓ QC metrics calculated")

# ---------- 2   Batch-wise robust stats (median ± MAD) --------------
print("Computing batch-wise robust statistics...")
stats = (
    ge_adata.obs
      .groupby('batch')
      .agg(
          n_genes_median = ('n_genes_by_counts', 'median'),
          n_genes_mad    = ('n_genes_by_counts',
                            lambda x: np.median(np.abs(x - np.median(x)))),
          counts_median  = ('total_counts',       'median'),
          counts_mad     = ('total_counts',
                            lambda x: np.median(np.abs(x - np.median(x))))
      )
)
qc = ge_adata.obs.join(stats, on='batch')

sf = 1.4826     # converts MAD -> SD for a normal distribution
z_genes  = (qc['n_genes_by_counts'] - qc['n_genes_median']) / (sf * qc['n_genes_mad'])
z_counts = (qc['total_counts']       - qc['counts_median'])  / (sf * qc['counts_mad'])

print("=== APPLYING CELL FILTERS ===")
gene_filter   = (-3 <= z_genes)  & (z_genes  <= 3)        # keep within ±3 "robust SD"
count_filter  = (-3 <= z_counts) & (z_counts <= 3)
rrna_filter   =  qc['pct_counts_is_rRNA'] < 10
label_filter  =  qc['broad_cell_type'].notna()

print(f"Gene count filter: {gene_filter.sum()} / {len(gene_filter)} cells pass")
print(f"Total count filter: {count_filter.sum()} / {len(count_filter)} cells pass")
print(f"rRNA filter (<10%): {rrna_filter.sum()} / {len(rrna_filter)} cells pass")
print(f"Cell type label filter: {label_filter.sum()} / {len(label_filter)} cells pass")

keep = gene_filter & count_filter & rrna_filter & label_filter
initial_cells = ge_adata.n_obs
ge_adata = ge_adata[keep].copy()
final_cells = ge_adata.n_obs
print(f"=== CELL FILTERING COMPLETE ===")
print(f"Retained {final_cells} cells (filtered out {initial_cells - final_cells})")

# %%
print(f"Current data dimensions: {ge_adata.shape}")

print("=== STEP 1: Remove lncRNAs from ge_adata ===")

# Define lncRNA biotype keywords
lncrna_keywords = ["lncRNA", "lincRNA"]

# Identify lncRNA genes
lncrna_mask = ge_adata.var["gene_biotype"].str.contains(
    "|".join(lncrna_keywords), case=False, na=False
)
print(f"Removing {lncrna_mask.sum():,} lncRNA genes.")
ge_adata = ge_adata[:, ~lncrna_mask].copy()

# ----------------------------------------------------------------

print("=== STEP 2: Subset to genes shared with splice_adata ===")

# Compute gene overlap
splice_genes = set(splice_adata.var["gene_id"])
ge_genes = set(ge_adata.var["gene_id"])
common_genes = splice_genes & ge_genes

print(f"Keeping {len(common_genes):,} genes shared between GE and Splicing data.")

# Subset ge_adata to shared genes
ge_adata = ge_adata[:, ge_adata.var["gene_id"].isin(common_genes)].copy()

# Also subset splice_adata to shared genes 
splice_adata = splice_adata[:, splice_adata.var["gene_id"].isin(common_genes)].copy()

# ----------------------------------------------------------------

print("=== STEP 3: Align cells between ge_adata and splice_adata ===")
# Realign splice_adata to match filtered ge_adata cells
splice_adata = splice_adata[ge_adata.obs_names].copy()

# Now safe to compare
assert np.all(ge_adata.obs_names == splice_adata.obs_names)
print(f"✓ Gene expression data filtered to {ge_adata.n_vars} genes")

# ----------------------------------------------------------------

print("=== STEP 4: Summarize number of junction reads per cell ===")
print("=== CALCULATING JUNCTION QC METRICS ===")

# Confirm required column exists
assert "annotation_status" in splice_adata.var.columns, "Missing 'annotation_status' column in splice_adata.var"
assert "both" in splice_adata.var["annotation_status"].unique(), "'both' not found in annotation_status"

# Define annotation mask
is_annotated = splice_adata.var["annotation_status"] == "both"
is_unannotated = ~is_annotated

# Read count matrix
X_counts = splice_adata.layers["cell_by_junction_matrix"]

# Total read counts
splice_adata.obs["total_junction_reads"] = np.asarray(X_counts.sum(axis=1)).flatten()
splice_adata.obs["annotated_junction_reads"] = np.asarray(X_counts[:, is_annotated].sum(axis=1)).flatten()
splice_adata.obs["unannotated_junction_reads"] = np.asarray(X_counts[:, is_unannotated].sum(axis=1)).flatten()

# Detected junctions (binary matrix)
X_detected = X_counts > 0
splice_adata.obs["n_detected_annotated_junctions"] = np.asarray(X_detected[:, is_annotated].sum(axis=1)).flatten()
splice_adata.obs["n_detected_unannotated_junctions"] = np.asarray(X_detected[:, is_unannotated].sum(axis=1)).flatten()

# Apply filters based on 1st and 99th percentiles
def quantile_filter(series, lower=0.01, upper=0.99):
    q_low, q_high = series.quantile([lower, upper])
    return (series >= q_low) & (series <= q_high)

read_filter = quantile_filter(splice_adata.obs["total_junction_reads"])
junctions_filter = quantile_filter(splice_adata.obs["n_detected_annotated_junctions"])

# Additional filter: cell must have at least 1000 total junction reads
min_reads_filter = splice_adata.obs["total_junction_reads"] > 1000
min_junctions_filter = splice_adata.obs["n_detected_annotated_junctions"] >= 2

# Final per-cell QC: within percentile range AND above minimum count
splice_qc_filter = read_filter & junctions_filter & min_reads_filter & min_junctions_filter

# Filter both datasets
cells_before_filter = splice_adata.shape[0]
ge_adata = ge_adata[splice_qc_filter].copy()
splice_adata = splice_adata[splice_qc_filter].copy()
cells_after_filter = splice_adata.shape[0]  
print(f"Cells lost due to junction-based per-cell QC filters: {cells_before_filter - cells_after_filter}")

print("✓ Applied junction-based per-cell QC filters")

# ----------------------------------------------------------------

print("=== APPLYING JUNCTION-LEVEL QC FILTERS ===")

# Recompute total read counts per junction
junction_counts = np.asarray(splice_adata.layers["cell_by_junction_matrix"].sum(axis=0)).flatten()
junction_detected = np.asarray((splice_adata.layers["cell_by_junction_matrix"] > 0).sum(axis=0)).flatten()

# Annotate junctions based on detection across cells
detected_in_cells = np.asarray((X_counts > 0).sum(axis=0)).flatten()
splice_adata.var["n_cells_detected"] = detected_in_cells
splice_adata.var["confidence"] = np.where(
    splice_adata.var["n_cells_detected"] <= 100, "low", "high"
)
print(f"✓ Annotated {np.sum(splice_adata.var['confidence'] == 'low')} junctions as 'low confidence'")

# Remove junction-ATSEs that have more than 10 junctions suggesting overly complex splicing
junctions_before = splice_adata.shape[1]
atses_before = splice_adata.var["gene_name"].nunique()
splice_adata = splice_adata[:, splice_adata.var["num_junctions"] <= 10]
junctions_after = splice_adata.shape[1]
atses_after = splice_adata.var["gene_name"].nunique()

print(f"✓ Removed {junctions_before - junctions_after} junctions from {atses_before - atses_after} genes/ATSEs")
print(f"  (ATSEs with >10 junctions removed due to overly complex splicing patterns)")
print(f"  Remaining: {junctions_after} junctions across {atses_after} genes/ATSEs")

# %%
# Filter out singleton tissue/cell_type/medium_cell_type combinations
print("=== FILTERING SINGLETON CELL TYPE COMBINATIONS ===")

# Get value counts
cell_counts = splice_adata.obs[["tissue", "broad_cell_type", "medium_cell_type"]].value_counts()

# Filter out combinations that appear only once
cell_counts_filtered = cell_counts[cell_counts > 1]

print(f"Before filtering: {len(cell_counts)} unique combinations")
print(f"After filtering: {len(cell_counts_filtered)} unique combinations")
print(f"Removed {len(cell_counts) - len(cell_counts_filtered)} singleton combinations")

# Create a tuple column for the combination
splice_adata.obs['temp_combo'] = list(zip(
    splice_adata.obs['tissue'],
    splice_adata.obs['broad_cell_type'],
    splice_adata.obs['medium_cell_type']
))

# Get the combinations that have more than 1 cell
valid_combos = set(cell_counts_filtered.index)

# Filter to keep only cells in valid combinations
cells_before = splice_adata.shape[0]
splice_adata = splice_adata[splice_adata.obs['temp_combo'].isin(valid_combos)].copy()
cells_after = splice_adata.shape[0]

# Remove the temporary column
splice_adata.obs.drop(columns=['temp_combo'], inplace=True)

print(f"\nCells removed: {cells_before - cells_after}")
print(f"Cells remaining: {cells_after}")

# Visualize the filtered table
print("\n=== Full filtered cell type distribution ===")
cell_counts_filtered_display = splice_adata.obs[["tissue", "broad_cell_type", "medium_cell_type"]].value_counts()
print(cell_counts_filtered_display.to_string())

# ----------------------------------------------------------------
print("=== RE-SUBSETTING AND VALIDATING FINAL ALIGNMENT ===")

# Step 1: Realign cells — ensure identical cell set in both
common_cells = ge_adata.obs_names.intersection(splice_adata.obs_names)
print(f"Re-aligning to {len(common_cells)} common cells")
ge_adata = ge_adata[common_cells].copy()
splice_adata = splice_adata[common_cells].copy()
assert np.all(ge_adata.obs_names == splice_adata.obs_names)

# Step 2: Filter again for cell types with ≥50 cells
cell_type_counts = ge_adata.obs['broad_cell_type'].value_counts()
valid_cell_types = cell_type_counts[cell_type_counts >= 50].index
print(f"Cell types retained after final filtering: {len(valid_cell_types)}")

cell_mask = ge_adata.obs['broad_cell_type'].isin(valid_cell_types)
ge_adata = ge_adata[cell_mask].copy()
splice_adata = splice_adata[cell_mask].copy()
print(f"✓ Retained {ge_adata.n_obs} cells after final broad cell type filtering")

# Step 3: Recheck shared genes
shared_genes = set(ge_adata.var["gene_id"]).intersection(set(splice_adata.var["gene_id"]))
print(f"✓ {len(shared_genes)} genes shared between gene expression and splicing data")

ge_adata = ge_adata[:, ge_adata.var["gene_id"].isin(shared_genes)].copy()
splice_adata = splice_adata[:, splice_adata.var["gene_id"].isin(shared_genes)].copy()

# Final sanity checks
assert np.all(ge_adata.obs_names == splice_adata.obs_names)
print("✓ Final alignment confirmed between gene expression and splicing datasets")

# %%
print("=== FINAL DATA SUMMARY ===")
print(f"Final number of unique cells in splice_adata: {splice_adata.n_obs}")
print(f"Final number of unique junctions in splice_adata: {splice_adata.n_vars}")
print(f"Final number of unique ATSEs in splice_adata: {splice_adata.var['event_id'].nunique()}")
print(f"Final number of unique genes in splice_adata: {splice_adata.var['gene_id'].nunique()}")

print(f"Final number of unique cells in ge_adata: {ge_adata.n_obs}")
print(f"Final number of unique genes in ge_adata: {ge_adata.n_vars}")

# %%
print("=== ADDING COMPUTED SCORES TO SPLICE DATA ===")
# Copy broad_cell_type from ge_adata to splice_adata
splice_adata.obs["broad_cell_type"] = ge_adata.obs["broad_cell_type"]
print("✓ Added broad_cell_type to splice data")

# %%
import os
from datetime import datetime

print("=== SAVING FINAL DATASETS ===")

# Set new model date and shared output directory
new_model_date = "102025"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/{new_model_date}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Output directory: {OUTPUT_DIR}")

# ---- Gene Expression ----
ge_new_filename = f"aligned_gene_expression_data_{timestamp}.h5ad"
GE_OUTPUT_PATH = os.path.join(OUTPUT_DIR, ge_new_filename)

print("Saving filtered gene expression data...")
ge_adata.write(GE_OUTPUT_PATH, compression="lzf")
print(f"✓ Filtered gene expression AnnData written to:\n{GE_OUTPUT_PATH}")

# ---- Splicing Data ----
splice_new_filename = f"aligned_splicing_data_{timestamp}.h5ad"
SPLICE_OUTPUT_PATH = os.path.join(OUTPUT_DIR, splice_new_filename)

print("Saving filtered splicing data...")
splice_adata.write(SPLICE_OUTPUT_PATH, compression="lzf")
print(f"✓ Filtered splicing AnnData written to:\n{SPLICE_OUTPUT_PATH}")

# %%
print("=== SAVING CELL ID LIST ===")
# Save text file with final list of cell ids 
cell_ids_path = os.path.join(OUTPUT_DIR, f"filtered_cell_ids_{timestamp}.txt")
with open(cell_ids_path, "w") as f:
    for cell_id in ge_adata.obs_names:
        f.write(f"{cell_id}\n")

print(f"✓ Cell IDs saved to: {cell_ids_path}")

print("=== PIPELINE COMPLETE ===")
print(f"Plot directory: {OUTPUT_DIR_PLOTS}")
print("✓ All tasks completed successfully!")

# to submit
# conda activate LeafletSC 
# script=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/GeneExpression/04_check_outliers.py
# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/processed_data
# sbatch --job-name=qc_mouse --mem=600G --partition cpu,bigmem --wrap "python $script"