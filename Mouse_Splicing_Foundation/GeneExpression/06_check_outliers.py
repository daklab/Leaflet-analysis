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
GE_INPUT = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/aligned_gene_expression_data_20250722_235313.h5ad"
SPLICE_INPUT = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/aligned_splicing_data_20250722_235313.h5ad"
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
print(f"Loading gene expression data from: {GE_INPUT}")
ge_adata = ad.read_h5ad(GE_INPUT)
print(f"Loading splicing data from: {SPLICE_INPUT}")
splice_adata = ad.read_h5ad(SPLICE_INPUT)

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
# Add the directory containing the shared utils to the Python path
sys.path.append("/gpfs/commons/home/kisaev/Leaflet-analysis/Multi_Species_Splicing_Foundation/shared_utils")
# Input file paths
GTF_FILE = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/genome_files/gencode.vM19/genes/genes.gtf"
DB_FILE = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/genome_files/GENCODE_vM19"

# %%
# reset the index
ge_adata.var = ge_adata.var.reset_index(drop=True)

print("=== CLEANING CELL IDS ===")

# Confirm that the cell ids are the same
# Assert that obs_names are the same
assert np.all(ge_adata.obs_names == splice_adata.obs_names)
assert np.all(ge_adata.obs["cell_id"].values == splice_adata.obs["cell_id"].values)
print("✓ Cell order confirmed consistent between datasets")

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
ge_adata.var['is_rRNA'] = ge_adata.var['transcript_biotypes'].str.contains(
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

NUCLEAR_GENES = ['Malat1','Neat1','Xist','Meg3','Kcnq1ot1',
                 'Pvt1','Gas5','Snhg1','Snhg6']
CYTO_GENES    = ['Rpl3','Rpl4','Rpl5','Rpl6','Rpl7','Rpl8',
                 'Rps2','Rps3','Rps4x','Rps6','Rps8','Rps9',
                 'Eef1a1','Eef2','Actb','Gapdh']

def add_nuclear_score(
        adata,
        layer=('log_norm',),            # str or tuple/list
        verbose=True,
        weights=dict(ratio=1.0, rrna=0.0) # must sum to 1.0
):
    if isinstance(layer, str):
        layer = (layer,)

    gnames = adata.var.get('gene_name', adata.var_names).str.upper()
    nuc_idx  = np.flatnonzero(gnames.isin([g.upper() for g in NUCLEAR_GENES]))
    cyto_idx = np.flatnonzero(gnames.isin([g.upper() for g in CYTO_GENES]))
    rrna_idx = np.flatnonzero(adata.var.get('is_rRNA', False))

    if verbose:
        print(f'nuclear genes: {len(nuc_idx)}/{len(NUCLEAR_GENES)}, '
              f'cyto: {len(cyto_idx)}/{len(CYTO_GENES)}, '
              f'rRNA: {rrna_idx.size}')

    mean1d = lambda M, idx: np.asarray(M[:, idx].mean(axis=1)).ravel() if idx.size else np.zeros(adata.n_obs)
    sum1d  = lambda M, idx: np.asarray(M[:, idx].sum(axis=1)).ravel()  if idx.size else np.zeros(adata.n_obs)
    z      = lambda v: zscore(v, nan_policy='omit')

    for lyr in layer:
        X = adata.layers.get(lyr, adata.X)

        # component 1 – nuclear / cytoplasmic ratio
        nuc_expr  = mean1d(X, nuc_idx)
        cyto_expr = mean1d(X, cyto_idx)
        ratio_z   = z(np.log2((nuc_expr + 1)/(cyto_expr + 1)))

        # component 2 – rRNA depletion
        rrna_expr  = sum1d(X, rrna_idx)
        total_expr = np.asarray(X.sum(axis=1)).ravel()
        rrna_frac  = rrna_expr / (total_expr + 1e-9)
        rrna_z     = z(-np.log10(rrna_frac + 1e-9))

        composite = (weights['ratio'] * ratio_z +
                     weights['rrna'] * rrna_z)

        composite = (composite - composite.min()) / (composite.max() - composite.min() + 1e-9)

        suf = '' if len(layer) == 1 else f'_{lyr}'
        adata.obs[f'nuclear_score{suf}'] = composite
        adata.obs[f'nuclear_ratio{suf}'] = ratio_z
        adata.obs[f'nuclear_rrna{suf}']  = rrna_z
        adata.obs[f'nuclear_class{suf}'] = pd.cut(
            composite, bins=[0, .25, .75, 1],
            labels=['cellular','mixed','nuclear'],  include_lowest=True
        )
    return adata


# %%
print("=== CALCULATING NUCLEAR SCORES ===")
ge_adata = add_nuclear_score(
    ge_adata,
    layer=('log_norm', 'predicted_log_norm_tms'),
    verbose=True
)

print("=== GENERATING NUCLEAR SCORE PLOTS ===")

# Summarize with plot 
cls_col   = 'dataset'
score_col = 'nuclear_ratio_predicted_log_norm_tms'   # or nuclear_score_length_norm

palette   = {'cellular':'#d62728', 'mixed':'#ff7f0e', 'nuclear':'#2ca02c'}

fig, ax = plt.subplots(figsize=(5,4))

# 1️violin per batch
sns.violinplot(
    data = ge_adata.obs,
    hue = cls_col,
    x    = 'dataset',
    y    = score_col,
    ax   = ax,
    inner='quartile',      # draw the Q1–median–Q3 bars
    cut  = 0
)

# Move legend outside of plot
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# 2️dashed line at y = 0.5
ax.axhline(0.5, ls='--', lw=1, color='black')

# 3️annotate median for each batch
medians = ge_adata.obs.groupby('dataset')[score_col].median()
for tick, batch in enumerate(medians.index):
    y = medians[batch]
    ax.text(tick, y + 0.02,          # slight upward offset
            f"{y:.2f}",
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Composite nuclear score')
ax.set_xlabel('Batch / dataset')
plt.tight_layout()

# Save plot
plot_path = os.path.join(OUTPUT_DIR_PLOTS, "nuclear_score_predicted_log_norm_tms_by_dataset.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved nuclear score plot (predicted_log_norm_tms): {plot_path}")
plt.close()

# %%
# Second nuclear score plot
score_col = 'nuclear_ratio_log_norm'   # or nuclear_score_length_norm

fig, ax = plt.subplots(figsize=(5,4))

# 1️violin per batch
sns.violinplot(
    data = ge_adata.obs,
    hue = cls_col,
    x    = 'dataset',
    y    = score_col,
    ax   = ax,
    inner='quartile',      # draw the Q1–median–Q3 bars
    cut  = 0
)

# Move legend outside of plot
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# 2️dashed line at y = 0.5
ax.axhline(0.5, ls='--', lw=1, color='black')

# 3️annotate median for each batch
medians = ge_adata.obs.groupby('dataset')[score_col].median()
for tick, batch in enumerate(medians.index):
    y = medians[batch]
    ax.text(tick, y + 0.02,          # slight upward offset
            f"{y:.2f}",
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Composite nuclear score')
ax.set_xlabel('Batch / dataset')
plt.tight_layout()

# Save plot
plot_path = os.path.join(OUTPUT_DIR_PLOTS, "nuclear_score_log_norm_by_dataset.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved nuclear score plot (log_norm): {plot_path}")
plt.close()

# %%
print("=== LOADING AGING GENES ===")
# read the aging-gene table you pasted
df = pd.read_csv("/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/062025/global_aging_genes.tsv", sep="\t")
# positive genes = those whose coefficient > 0  (up-regulated with age)
age_pos = df.loc[df.gag_score_coef > 0, "global_aging_genes"].str.upper().tolist()
# negative genes = coefficient < 0 (down-regulated with age)
age_neg = df.loc[df.gag_score_coef < 0, "global_aging_genes"].str.upper().tolist()
print(f"Loaded {len(age_pos)} age-positive genes and {len(age_neg)} age-negative genes")

# %%
ge_adata.X = ge_adata.layers["log_norm"]

# %%
# be sure var_names are also upper-case to match
ge_adata.var_names = ge_adata.var["gene_name"].str.upper()
ge_adata.var['gene_name'] = ge_adata.var_names        # now column ≡ index

# Make sure to also make gene_name upper case in splicing anndata 
splice_adata.var["gene_name"] = splice_adata.var["gene_name"].str.upper()

print("=== CALCULATING AGING SCORES ===")

# --- Up-regulated gene score ---
sc.tl.score_genes(
    ge_adata,
    gene_list=age_pos,
    score_name="AgingScore_pos"
)
print("✓ Calculated aging score for up-regulated genes")

# --- Down-regulated gene score ---
sc.tl.score_genes(
    ge_adata,
    gene_list=age_neg,
    score_name="AgingScore_neg"
)
print("✓ Calculated aging score for down-regulated genes")

# Final composite (higher = "older")
ge_adata.obs["AgingScore_unweighted"] = (
    ge_adata.obs["AgingScore_pos"] - ge_adata.obs["AgingScore_neg"]
)
print("✓ Calculated composite aging score")

# %%
print("=== GENERATING AGING SCORE PLOTS ===")

# Make violin plot of aging score
plt.figure(figsize=(5, 5))
sns.violinplot(data=ge_adata.obs, x="age", y="AgingScore_unweighted", inner="quartile")
plt.title("Aging Score by Age")
plt.tight_layout()

# Save plot
plot_path = os.path.join(OUTPUT_DIR_PLOTS, "aging_score_by_age.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved aging score by age plot: {plot_path}")
plt.close()

# %%
# --- Compute ordering by median AgingScore ---
median_by_ct = (
    ge_adata.obs
      .groupby("broad_cell_type")["AgingScore_unweighted"]
      .median()
      .sort_values(ascending=False)
)

ordered_cts = median_by_ct.index.tolist()

# --- Make a horizontal violin plot, ordered by median score ---
fig_height = 12      # autoscale height

plt.figure(figsize=(5, fig_height))
sns.violinplot(
    data=ge_adata.obs,
    y="broad_cell_type",            # cell types on y-axis
    x="AgingScore_unweighted",      # score on x-axis
    order=ordered_cts,              # descending median order
    inner="quartile",               # show median & IQR
    hue="age",
    cut=0,                          # trim tails to data range
)
# Add a thin dashed line at x=0 
plt.axvline(0, color='red', linewidth=0.3, linestyle='--')
plt.xlabel("AgingScore (unweighted)")
plt.ylabel("")                      # optional: hide "broad_cell_type" label
plt.title("AgingScore distribution by cell type (ordered by median)")
plt.tight_layout()

# Save plot
plot_path = os.path.join(OUTPUT_DIR_PLOTS, "aging_score_by_celltype_ordered.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved aging score by cell type plot: {plot_path}")
plt.close()

print("=== STEP 1: Remove lncRNAs from ge_adata ===")

# Define lncRNA biotype keywords
lncrna_keywords = ["lncRNA", "lincRNA"]

# Identify lncRNA genes
lncrna_mask = ge_adata.var["transcript_biotypes"].str.contains(
    "|".join(lncrna_keywords), case=False, na=False
)
print(f"Removing {lncrna_mask.sum():,} lncRNA genes.")
ge_adata = ge_adata[:, ~lncrna_mask].copy()

# ----------------------------------------------------------------

print("=== STEP 2: Subset to genes shared with splice_adata ===")

# Compute gene overlap
splice_genes = set(splice_adata.var["gene_name"])
ge_genes = set(ge_adata.var["gene_name"])
common_genes = splice_genes & ge_genes

print(f"Keeping {len(common_genes):,} genes shared between GE and Splicing data.")

# Subset ge_adata to shared genes
ge_adata = ge_adata[:, ge_adata.var["gene_name"].isin(common_genes)].copy()

# Also subset splice_adata to shared genes 
splice_adata = splice_adata[:, splice_adata.var["gene_name"].isin(common_genes)].copy()

# ----------------------------------------------------------------

print("=== STEP 3: HVG selection on filtered ge_adata ===")

# Run HVG selection on length-normalized layer
sc.pp.highly_variable_genes(
    ge_adata,
    layer="length_norm",
    n_top_genes=15000,
    batch_key="dataset", 
    flavor="seurat_v3", 
    inplace=True
)

print(f"✓ Selected {ge_adata.var['highly_variable'].sum():,} HVGs.")

# ----------------------------------------------------------------

print("=== STEP 4: Align cells between ge_adata and splice_adata ===")
# Realign splice_adata to match filtered ge_adata cells
splice_adata = splice_adata[ge_adata.obs_names].copy()

# Now safe to compare
assert np.all(ge_adata.obs_names == splice_adata.obs_names)
print(f"✓ Gene expression data filtered to {ge_adata.n_vars} genes")


# ----------------------------------------------------------------

print("=== STEP 5: Summarize number of junction reads per cell ===")

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

# Additional filter: cell must have at least 100 total junction reads
min_reads_filter = splice_adata.obs["total_junction_reads"] > 10
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
shared_genes = set(ge_adata.var["gene_name"]).intersection(set(splice_adata.var["gene_name"]))
print(f"✓ {len(shared_genes)} genes shared between gene expression and splicing data")

ge_adata = ge_adata[:, ge_adata.var["gene_name"].isin(shared_genes)].copy()
splice_adata = splice_adata[:, splice_adata.var["gene_name"].isin(shared_genes)].copy()

# Final sanity checks
assert np.all(ge_adata.obs_names == splice_adata.obs_names)
print("✓ Final alignment confirmed between gene expression and splicing datasets")

# %%
print("=== FINAL DATA SUMMARY ===")
print(f"Final number of unique cells in splice_adata: {splice_adata.n_obs}")
print(f"Final number of unique junctions in splice_adata: {splice_adata.n_vars}")
print(f"Final number of unique ATSEs in splice_adata: {splice_adata.var['event_id'].nunique()}")
print(f"Final number of unique genes in splice_adata: {splice_adata.var['gene_name'].nunique()}")

print(f"Final number of unique cells in ge_adata: {ge_adata.n_obs}")
print(f"Final number of unique genes in ge_adata: {ge_adata.n_vars}")

# %%
print("=== ADDING COMPUTED SCORES TO SPLICE DATA ===")
# add the new .obs columns to the splice_adata
splice_adata.obs["broad_cell_type"] = ge_adata.obs["broad_cell_type"]
splice_adata.obs["AgingScore_unweighted"] = ge_adata.obs["AgingScore_unweighted"]
splice_adata.obs["AgingScore_pos"] = ge_adata.obs["AgingScore_pos"]
splice_adata.obs["AgingScore_neg"] = ge_adata.obs["AgingScore_neg"]
print("✓ Added aging scores and cell type annotations to splice data")

print("=== ADDING COMPUTED NUCLEAR RATIO SCORES TO SPLICE DATA ===")
splice_adata.obs["nuclear_ratio_log_norm"] = ge_adata.obs["nuclear_ratio_log_norm"]

# %%
import os
from datetime import datetime

print("=== SAVING FINAL DATASETS ===")

# Set new model date and shared output directory
new_model_date = "072025"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
GE_OUTPUT_DIR = f"/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/{new_model_date}"
os.makedirs(GE_OUTPUT_DIR, exist_ok=True)

print(f"Output directory: {GE_OUTPUT_DIR}")

# ---- Gene Expression ----
ge_new_filename = f"aligned_gene_expression_data_{timestamp}.h5ad"
GE_OUTPUT_PATH = os.path.join(GE_OUTPUT_DIR, ge_new_filename)

print("Saving filtered gene expression data...")
ge_adata.write(GE_OUTPUT_PATH, compression="lzf")
print(f"✓ Filtered gene expression AnnData written to:\n{GE_OUTPUT_PATH}")

# ---- Splicing Data ----
splice_new_filename = f"aligned_splicing_data_{timestamp}.h5ad"
SPLICE_OUTPUT_PATH = os.path.join(GE_OUTPUT_DIR, splice_new_filename)

print("Saving filtered splicing data...")
# Subset and save matched splicing data
splice_adata.write(SPLICE_OUTPUT_PATH, compression="lzf")
print(f"✓ Filtered splicing AnnData written to:\n{SPLICE_OUTPUT_PATH}")

# %%
print("=== SAVING CELL ID LIST ===")
# Save also text file with just final list of cell ids 
cell_ids_path = os.path.join(GE_OUTPUT_DIR, "filtered_cell_ids.txt")
with open(cell_ids_path, "w") as f:
    for cell_id in ge_adata.obs_names:
        f.write(f"{cell_id}\n")

print(f"✓ Cell IDs saved to: {cell_ids_path}")

print("=== PIPELINE COMPLETE ===")
print(f"Total plots saved: 3")
print(f"Plot directory: {OUTPUT_DIR_PLOTS}")
print("✓ All tasks completed successfully!")

# to submit
# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/processed_data
# sbatch --job-name=outlier_check --mem=600G --partition cpu,bigmem --wrap "python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/GeneExpression/06_check_outliers.py"