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


# %%
def extract_gene_transcript_info(gtf_file, db_file):
    """
    Parses a GENCODE GTF file to compute:
    - Mean transcript length per gene (sum of exons)
    - Mean intron length per gene (transcript span - exon length)
    - Number of transcripts per gene
    - Transcript biotypes
    - Gene name
    
    Returns a DataFrame with gene_id, gene_name, mean_transcript_length, mean_intron_length, 
    num_transcripts, and transcript_biotypes.
    """

    if os.path.exists(db_file):
        print("Using existing GTF database.")
        db = gffutils.FeatureDB(db_file, keep_order=True)
        print("Database loaded successfully!")
    else:
        print("Creating GTF database (this may take a few minutes)...")
        db = gffutils.create_db(
            gtf_file,
            db_file,
            force=True,
            keep_order=True,
            disable_infer_transcripts=False,
            disable_infer_genes=True
        )
        print("Database created successfully!")

    gene_exon_lengths = defaultdict(list)
    gene_intron_lengths = defaultdict(list)
    gene_names = {}
    gene_biotypes = defaultdict(set)
    transcript_counts = defaultdict(int)

    print("Processing transcripts to compute exon and intron lengths...")
    for transcript in tqdm(db.features_of_type("transcript"), desc="Processing Transcripts", unit=" transcript"):
        gene_id = transcript.attributes["gene_id"][0]
        gene_name = transcript.attributes.get("gene_name", ["unknown"])[0]
        transcript_biotype = transcript.attributes.get("transcript_type", ["unknown"])[0]

        exons = list(db.children(transcript, featuretype="exon", order_by="start"))
        if len(exons) == 0:
            continue  # skip transcripts with no exons

        exon_length = sum(exon.end - exon.start + 1 for exon in exons)
        transcript_start = exons[0].start
        transcript_end = exons[-1].end
        transcript_span = transcript_end - transcript_start + 1
        intron_length = transcript_span - exon_length  # includes gaps between exons

        gene_exon_lengths[gene_id].append(exon_length)
        gene_intron_lengths[gene_id].append(max(0, intron_length))  # avoid negative values
        gene_names[gene_id] = gene_name
        gene_biotypes[gene_id].add(transcript_biotype)
        transcript_counts[gene_id] += 1

    print("Finished processing transcripts.")

    gene_ids = list(gene_exon_lengths.keys())
    gene_info_df = pd.DataFrame({
        "gene_id": gene_ids,
        "gene_name": [gene_names[g] for g in gene_ids],
        "mean_transcript_length": [sum(gene_exon_lengths[g]) / len(gene_exon_lengths[g]) for g in gene_ids],
        "mean_intron_length": [sum(gene_intron_lengths[g]) / len(gene_intron_lengths[g]) for g in gene_ids],
        "num_transcripts": [transcript_counts[g] for g in gene_ids],
        "transcript_biotypes": [", ".join(sorted(gene_biotypes[g])) for g in gene_ids]
    })

    return gene_info_df

gtf_hg38 = "/gpfs/commons/datasets/controlled/BRAIN_NeMO/human-reference/gencode/gencode.v45.primary_assembly.annotation.gtf"
db_file = "/gpfs/commons/datasets/controlled/BRAIN_NeMO/human-reference/gencode/gencode_hg38.db"

# Get lengths of genes 
gene_info_df = extract_gene_transcript_info(gtf_hg38, db_file)
gene_info_df = gene_info_df.drop_duplicates(subset="gene_id")
gene_info_df = gene_info_df.drop_duplicates(subset="gene_name")

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

# 1. Define flatten function once
def flatten_grouped_map(grouped_map):
    return {label: broad_type for broad_type, labels in grouped_map.items() for label in labels}

# 2. Create the flat map once (outside the loop)
grouped_broad_map_flat = flatten_grouped_map(grouped_refined_map)

ts_adata.obs["broad_cell_type"] = ts_adata.obs["free_annotation"].map(grouped_broad_map_flat).fillna('Other')
ab_exons.obs["broad_cell_type"] = ab_exons.obs["subclass_label"].map(grouped_broad_map_flat).fillna('Other')
ab_introns.obs["broad_cell_type"] = ab_introns.obs["subclass_label"].map(grouped_broad_map_flat).fillna('Other')

# Find common broad cell types across ts_adata and ab_exons 
common_broad_types = set(ts_adata.obs["broad_cell_type"]).intersection(set(ab_exons.obs["broad_cell_type"]))

# Removed Other from common_broad_types
common_broad_types = [broad_type for broad_type in common_broad_types if broad_type != 'Other']
print(f"Common broad cell types: {common_broad_types}")

# === Make sparse if needed ===
for adata in [ab_exons, ab_introns, ts_adata]:
    if not sp.issparse(adata.X):
        adata.X = csr_matrix(adata.X)

ts_adata.var["gene_name"] = ts_adata.var["gene_symbol"]
ts_adata = ts_adata[:, ts_adata.var["gene_name"].isin(gene_info_df["gene_name"])].copy()

# Clean up allen brain exon and intron datasets 
ab_introns.var["gene_name"] = ab_introns.var["gene_symbol"]
ab_introns = ab_introns[:, ab_introns.var["gene_name"].isin(gene_info_df["gene_name"])]

ab_exons.var["gene_name"] = ab_exons.var["gene_symbol"]
ab_adata_exons = ab_exons[:, ab_exons.var["gene_name"].isin(gene_info_df["gene_name"])]

# merge tms_adata and ab_adata with gene_info_df 
ts_adata.var = ts_adata.var.reset_index().merge(gene_info_df, on="gene_name").set_index("index")
ab_introns.var = ab_introns.var.reset_index().merge(gene_info_df, on="gene_name").set_index("index")
ab_exons.var = ab_exons.var.reset_index().merge(gene_info_df, on="gene_name").set_index("index")

# Load model for mapping exon + intron log1p length normalized counts to total spliced counts (single cell from single nuclei estimatoin)
linear_model_file = "/gpfs/commons/home/kisaev/Leaflet-analysis/Human_Splicing_Foundation/linear_norm_ts_model.pkl"

# Load the model
with open(linear_model_file, 'rb') as f:
    model_linear = pickle.load(f)

print("Linear Model Summary:")
print(model_linear.summary())

# Estimate total spliced counts from exons and introns (from single nuclei data to single cell data as a way to account for differences in spliced products)
# For each gene, we will estimate the total spliced counts using the linear model for each cell, if its counts were missing, we will not include it 

# Step 1: Filter genes in gene_info_df
valid_gene_info = gene_info_df[
    (gene_info_df["mean_transcript_length"] > 0) &
    (gene_info_df["mean_intron_length"] > 0)
].copy()

print(f"Retaining {len(valid_gene_info)} genes with nonzero transcript/intron length.")

# Step 2: Subset all adata objects to valid genes
valid_genes = set(valid_gene_info["gene_name"])

for adata in [ts_adata, ab_exons, ab_introns]:
    adata._inplace_subset_var(adata.var["gene_name"].isin(valid_genes))

# Extract exon and intron raw counts
exon_counts = ab_exons.X  # shape: cells x genes
intron_counts = ab_introns.X  # same shape
tot_counts = ts_adata.X  # shape: cells x genes

# === Sanity check: ensure gene order is the same across all AnnData objects ===
gene_order = ab_exons.var["gene_name"].values

assert np.array_equal(gene_order, ab_introns.var["gene_name"].values), \
    "Gene order mismatch between ab_exons and ab_introns!"

assert np.array_equal(gene_order, ts_adata.var["gene_name"].values), \
    "Gene order mismatch between ab_exons and ts_adata!"

print("✅ Gene order is consistent across ab_exons, ab_introns, and ts_adata.")

# Get gene lengths
gene_lengths = ts_adata.var.loc[ab_exons.var_names, ['mean_transcript_length', 'mean_intron_length']]

# Broadcast normalization
length_norm_exons = exon_counts / gene_lengths['mean_transcript_length'].values
length_norm_introns = intron_counts / gene_lengths['mean_intron_length'].values
length_norm_tot = tot_counts / gene_lengths['mean_transcript_length'].values

ab_exons.layers["length_norm"] = csr_matrix(length_norm_exons)
ab_introns.layers["length_norm"] = csr_matrix(length_norm_introns)
ts_adata.layers["length_norm"] = csr_matrix(length_norm_tot)

# Confirm no cells with zero counts across the board 
# === Check for cells with zero total normalized expression before library size adjustment ===
for name, adata in zip(["ab_exons", "ab_introns", "ts_adata"], [ab_exons, ab_introns, ts_adata]):
    length_norm = adata.layers["length_norm"]
    row_sums = length_norm.sum(axis=1).A1  # get sum per cell
    zero_cells = np.sum(row_sums == 0)
    print(f"{zero_cells} cells in {name} have zero length-normalized counts before library size adjustment.")
    assert zero_cells == 0, f"🚨 ERROR: {zero_cells} all-zero cells found in {name}!"

# Now normalize by library size 
length_norm_exons = length_norm_exons.multiply(1e4 / length_norm_exons.sum(axis=1).A1[:, None])
length_norm_introns = length_norm_introns.multiply(1e4 / length_norm_introns.sum(axis=1).A1[:, None])
length_norm_tot = length_norm_tot.multiply(1e4 / length_norm_tot.sum(axis=1).A1[:, None])

# Apply log1p transformation
log_norm_exons = np.log1p(length_norm_exons)
log_norm_introns = np.log1p(length_norm_introns)
log_norm_tot = np.log1p(length_norm_tot)

# Add as new layers
ab_exons.layers["log_norm"] = csr_matrix(log_norm_exons)
ab_introns.layers["log_norm"] = csr_matrix(log_norm_introns)
ts_adata.layers["log_norm"] = csr_matrix(log_norm_tot)

for name, adata in zip(["ab_exons", "ab_introns", "ts_adata"], [ab_exons, ab_introns, ts_adata]):
    length_norm = adata.layers["log_norm"]
    row_sums = length_norm.sum(axis=1).A1  # get sum per cell
    zero_cells = np.sum(row_sums == 0)
    print(f"{zero_cells} cells in {name} have zero length-normalized counts before library size adjustment.")
    assert zero_cells == 0, f"🚨 ERROR: {zero_cells} all-zero cells found in {name}!"

intercept = model_linear.params["const"]
coef_exons = model_linear.params["exons"]
coef_introns = model_linear.params["introns"]

log_exons = ab_exons.layers["log_norm"]  # CSR sparse matrix
log_introns = ab_introns.layers["log_norm"]  # CSR sparse matrix

log_pred_tot = log_exons.multiply(coef_exons) + log_introns.multiply(coef_introns)

log_pred_tot_with_intercept = log_pred_tot.copy()
log_pred_tot_with_intercept.data += intercept

# Create a new object using ab_exons as base
ab_adata = ab_exons.copy()
# Store adjusted log-normalized total spliced counts
ab_adata.layers["predicted_log_norm_ts"] = log_pred_tot_with_intercept
# Preserve raw counts explicitly
ab_adata.layers["raw_counts"] = csr_matrix(ab_adata.layers["raw_counts"])
# Clean up obs to only necessary metadata (optional but clean)
ab_adata.obs = ab_adata.obs[["sample_name"]]

# Remove old exon-only log_norm layer to avoid confusion
if "log_norm" in ab_adata.layers:
    del ab_adata.layers["log_norm"]

# === Sanity check 1: Non-zero total raw counts per cell ===
raw_sums = ab_adata.layers["raw_counts"].sum(axis=1).A1
zero_raw = np.sum(raw_sums == 0)
print(f"🔍 Cells with zero raw counts: {zero_raw}")
assert zero_raw == 0, "🚨 ERROR: Some cells have zero total raw counts!"

# === Sanity check 2: Non-zero total predicted log counts per cell ===
pred_sums = ab_adata.layers["predicted_log_norm_ts"].sum(axis=1).A1
zero_pred = np.sum(pred_sums == 0)
print(f"🔍 Cells with zero predicted log counts: {zero_pred}")
assert zero_pred == 0, "🚨 ERROR: Some cells have zero total predicted log counts!"

# === Sanity check: ensure gene order is the same in ab_adata and ts_adata ===
assert np.array_equal(ab_adata.var_names, ts_adata.var_names), \
    "🚨 ERROR: Gene order mismatch between ab_adata and ts_adata!"
print("✅ Gene order is consistent between ab_adata and ts_adata.")

# Save the object
today = date.today().strftime("%Y-%m-%d")
outfile = os.path.join(outdir, f"AB_adjusted_GeneExpression_via_exon_intron_regression_{today}.h5ad")
ab_adata.write_h5ad(outfile, compression="lzf")
print(f"Saved the object to {outfile}")

outfile = os.path.join(outdir, f"TS_GeneExpression_with_length_norm_{today}.h5ad")
ts_adata.write_h5ad(outfile, compression="lzf")
print(f"Saved the object to {outfile}")

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/processed_data
# sbatch --mem=100G --wrap "python /gpfs/commons/home/kisaev/Leaflet-analysis/Human_Splicing_Foundation/TabulaSapien_vs_Allen_pseudobulk_analysis_4.py"