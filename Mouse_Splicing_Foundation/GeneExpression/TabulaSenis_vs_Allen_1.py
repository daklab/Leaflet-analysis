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
from collections import defaultdict
import scipy.sparse as sp
import gffutils 
import gffutils 

# Load gene gtf file for gene length 
gtf_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/genome_files/gencode.vM19/genes/genes.gtf"
db_file="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/genome_files/GENCODE_vM19"

today = datetime.datetime.now()
today = today.strftime("%Y-%m-%d")

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

# ### First, get gene info (average transcript length and average total intron length)
# Add gene length information to the adata object [done]
# follow recommended practices for smart-seq2 raw count normalization
# https://docs.scvi-tools.org/en/1.0.0/tutorials/notebooks/tabula_muris.html

# Load gene info
gene_info_df = extract_gene_transcript_info(gtf_file, db_file)
gene_info_df = gene_info_df.drop_duplicates(subset="gene_id")
gene_info_df = gene_info_df.drop_duplicates(subset="gene_name")

# ### Prep the three anndata objects (total counts, exons only and introns only)# 
def preprocess_ab_adata(adata, metadata, dataset_label="allen_brain", 
                        metadata_key="sample_name", rename_var=True):
    """
    Standardizes Allen Brain adata object:
    - Subsets metadata to only matching cells
    - Renames gene info
    - Stores raw counts layer
    """

    adata.var['gene_name'] = adata.var_names
    adata.obs["dataset"] = dataset_label

    # Add metadata_key column to adata.obs if it doesn't exist 
    if metadata_key not in adata.obs.columns:
        adata.obs[metadata_key] = adata.obs_names

    # Use correct column to index metadata
    if metadata_key not in metadata.columns:
        raise ValueError(f"Metadata key '{metadata_key}' not found in metadata columns.")
    
    metadata_sub = metadata[metadata[metadata_key].isin(adata.obs_names)]
    print(f"Remaining cells after metadata filtering: {metadata_sub.shape[0]}")
    metadata_sub = metadata_sub.set_index(metadata_key)

    # Align metadata to adata
    adata = adata[adata.obs_names.isin(metadata_sub.index)].copy()
    adata.obs = metadata_sub.loc[adata.obs_names]

    # Rename gene_name -> gene_symbol
    if rename_var:
        adata.var.rename(columns={"gene_name": "gene_symbol"}, inplace=True)
    else:
        adata.var["gene_symbol"] = adata.var["gene_name"]
    print(f"Saving raw counts!")
    adata.layers["raw_counts"] = adata.X.copy()
    print(f"Number of cells/nuclei: {adata.shape[0]}")
    return adata

# Read in metadata
metadata = "/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/METADATA/metadata.csv"
metadata = pd.read_csv(metadata, low_memory=False)

# Set up paths for files 
introns_only = "/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/GeneExpression/expression_matrix_introns.csv"
exons_only = "/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/GeneExpression/expression_matrix_exons.csv"
    
ab_adata_introns = sc.read_csv(introns_only)
print(f"Done reading {introns_only}")

ab_adata_exons = sc.read_csv(exons_only)
print(f"Done reading {exons_only}")

# Add sample_name to obs in exons and introns 
ab_adata_exons.obs["sample_name"] = ab_adata_exons.obs.index
ab_adata_introns.obs["sample_name"] = ab_adata_introns.obs.index

# Process and merge with metadata
ab_adata_introns = preprocess_ab_adata(ab_adata_introns, metadata, metadata_key="sample_name")
ab_adata_exons = preprocess_ab_adata(ab_adata_exons, metadata, metadata_key="sample_name")
print("All AB datasets preprocessed with correct metadata mapping.")

# ### Read in the Tabula Muris Senis gene expression data 
# # Load the expression data (.h5ad file)
exp_file = "/gpfs/commons/projects/knowles_singlecell_splicing/TabulaSenis/data/AWS/processed_for_scanpy/tabulamurissenisfacsofficialrawobj.h5ad"
tms_adata = sc.read_h5ad(exp_file)
tms_adata.obs["dataset"] = "tabula_muris_senis"
tms_adata.obs["cell_id"] = tms_adata.obs.index.values

# Define a function to clean a single cell id
def clean_cell_id(cell_id):
    return cell_id.split(".mm10-plus-0-0")[0].split(".mus-2-1")[0]

# Apply it to all cells
tms_adata.obs_names = tms_adata.obs_names.map(clean_cell_id)
tms_adata.obs["cell_id"] = tms_adata.obs_names.values
tms_adata.layers["raw_counts"] = tms_adata.X.copy()

# Reset the index of adata.obs to integers and drop the old index
tms_adata.obs.reset_index(drop=True, inplace=True)
tms_adata.var["gene_symbol"] = tms_adata.var.index

# Clean up Cell IDs in the gene expression object
# If age column is 3m, adjust cell_clean to replace the first "." with "_" and the second "." also with "_" (annoying fix but that's how the data came...)
tms_adata.obs['cell_clean'] = tms_adata.obs['cell']  # Start by copying the 'cell' column to 'cell_clean'
tms_adata.obs.loc[tms_adata.obs['age'] == '3m', 'cell_clean'] = tms_adata.obs.loc[tms_adata.obs['age'] == '3m', 'cell'].str.replace('.', '_', 2)
tms_adata.obs["cell_clean"] = tms_adata.obs["cell_clean"].str.split('_').str[:2].str.join('_') 

# First, subset adata to just genes in gene_info_df
gene_info_df = gene_info_df.drop_duplicates(subset="gene_name")
# Remove genes with zero-length transcripts or introns
gene_info_df = gene_info_df[
    (gene_info_df["mean_transcript_length"] > 0) & 
    (gene_info_df["mean_intron_length"] > 0)
].copy()

print(f"Gene info after removing zero-length genes: {gene_info_df.shape[0]} genes")

# Merge safely and preserve original index
for adata in [tms_adata, ab_adata_introns, ab_adata_exons]:
    adata.var["gene_name"] = adata.var["gene_symbol"]
    adata.var["orig_index"] = adata.var.index  # store original index
    merged = adata.var.reset_index().merge(gene_info_df, on="gene_name", how="inner")
    if "orig_index" in merged:
        merged = merged.set_index("orig_index")
    adata._inplace_subset_var(merged.index)  # ensure dimensions match
    adata.var = merged

# Step 1: Find common genes
common_genes = set(tms_adata.var["gene_name"]).intersection(
    ab_adata_introns.var["gene_name"]
).intersection(
    ab_adata_exons.var["gene_name"]
)

print(f"Number of common genes: {len(common_genes)}")

# Step 2: Subset each AnnData to common genes
tms_adata = tms_adata[:, tms_adata.var["gene_name"].isin(common_genes)]
ab_adata_introns = ab_adata_introns[:, ab_adata_introns.var["gene_name"].isin(common_genes)]
ab_adata_exons = ab_adata_exons[:, ab_adata_exons.var["gene_name"].isin(common_genes)]

# Step 3: Sort each by gene_name
tms_adata = tms_adata[:, tms_adata.var.sort_values("gene_name").index]
ab_adata_introns = ab_adata_introns[:, ab_adata_introns.var.sort_values("gene_name").index]
ab_adata_exons = ab_adata_exons[:, ab_adata_exons.var.sort_values("gene_name").index]

# Step 4: Sanity check
assert all(tms_adata.var["gene_name"].values == ab_adata_introns.var["gene_name"].values)
assert all(tms_adata.var["gene_name"].values == ab_adata_exons.var["gene_name"].values)

# convert .X to sparse array in ab_adata, ab_adata_exons, ab_adata_introns
ab_adata_introns.X = csr_matrix(ab_adata_introns.X)
ab_adata_exons.X = csr_matrix(ab_adata_exons.X)

# Before saving clean up metadata in adata objects 
# 1. introns
ab_adata_introns = ab_adata_introns.copy()
ab_adata_introns.obs["cell_id"] = ab_adata_introns.obs["exp_component_name"]
ab_adata_introns.obs["cell_ontology_class"] = ab_adata_introns.obs["cell_type_alias_label"]
ab_adata_introns.obs["tissue"] = ab_adata_introns.obs["exp_component_name"]
ab_adata_introns.obs["age"] = "2m" 
ab_adata_introns.obs["mouse.id"] = ab_adata_introns.obs["external_donor_name_label"] # not sure if this is actually a mouse ID but most likely? 
ab_adata_introns.obs["subtissue"] = ab_adata_introns.obs["region_label"] 
ab_adata_introns.obs["sex"] = ab_adata_introns.obs["donor_sex_label"]
ab_adata_introns.obs = ab_adata_introns.obs[["cell_id", "age", "cell_ontology_class", "mouse.id", "sex", "subtissue", "tissue"]]

# 2. Exons 
ab_adata_exons = ab_adata_exons.copy()
ab_adata_exons.obs["cell_id"] = ab_adata_exons.obs["exp_component_name"]
ab_adata_exons.obs["cell_ontology_class"] = ab_adata_exons.obs["cell_type_alias_label"]
ab_adata_exons.obs["tissue"] = ab_adata_exons.obs["exp_component_name"]
ab_adata_exons.obs["age"] = "2m" 
ab_adata_exons.obs["mouse.id"] = ab_adata_exons.obs["external_donor_name_label"] # not sure if this is actually a mouse ID but most likely? 
ab_adata_exons.obs["subtissue"] = ab_adata_exons.obs["region_label"] 
ab_adata_exons.obs["sex"] = ab_adata_exons.obs["donor_sex_label"]
ab_adata_exons.obs = ab_adata_exons.obs[["cell_id", "age", "cell_ontology_class", "mouse.id", "sex", "subtissue", "tissue"]]

# 3. TMS single cell 
tms_adata.obs = tms_adata.obs.copy()
tms_adata.obs = tms_adata.obs[['cell_id', 'age', 'cell_ontology_class', 
                                 'mouse.id', 'sex', 'subtissue', 'tissue']]
ab_adata_introns.var["mean_transcript_length"] = ab_adata_introns.var["mean_intron_length"]

def normalize_by_gene_length(adata, input_layer="raw_counts", output_layer="length_norm"):
    # Make sure it's CSR
    counts = adata.layers[input_layer]
    if not isinstance(counts, csr_matrix):
        counts = csr_matrix(counts)
    
    # Get per-gene mean transcript lengths
    lengths = adata.var["mean_transcript_length"].values.copy()

    # Prevent division by zero
    lengths[lengths == 0] = np.nan

    # Do sparse division
    inv_lengths = 1.0 / lengths
    row, col = counts.nonzero()
    counts_norm = counts.copy()  # don't modify in-place unless you want to
    counts_norm.data = counts_norm.data * inv_lengths[col]

    # Save back
    adata.layers[output_layer] = counts_norm

# Apply to your datasets
normalize_by_gene_length(ab_adata_introns)
normalize_by_gene_length(ab_adata_exons)
normalize_by_gene_length(tms_adata)

# Normalize by library size and apply log1p transformation
# Now normalize by library size 
length_norm_exons = ab_adata_exons.layers["length_norm"].multiply(1e4 / ab_adata_exons.layers["length_norm"].sum(axis=1).A1[:, None])
length_norm_introns = ab_adata_introns.layers["length_norm"].multiply(1e4 / ab_adata_introns.layers["length_norm"].sum(axis=1).A1[:, None])
length_norm_tot = tms_adata.layers["length_norm"].multiply(1e4 / tms_adata.layers["length_norm"].sum(axis=1).A1[:, None])

# Apply log1p transformation
log_norm_exons = np.log1p(length_norm_exons)
log_norm_introns = np.log1p(length_norm_introns)
log_norm_tot = np.log1p(length_norm_tot)

# Add as new layers
ab_adata_exons.layers["log_norm"] = log_norm_exons
ab_adata_introns.layers["log_norm"] = log_norm_introns
tms_adata.layers["log_norm"] = log_norm_tot

### Save adata objects 
assert all(ab_adata_exons.var["gene_name"].values == tms_adata.var["gene_name"].values)
assert ab_adata_exons.shape[1] == tms_adata.shape[1]

# Drop tms_adata.var column that's not found in ab_adata
tms_adata.var = tms_adata.var.reset_index(drop=True)
tms_adata.var.drop(columns=["index", "n_cells"], inplace=True)
ab_adata_exons.var.drop(columns=["index"], inplace=True)
tms_adata.obs.reset_index(drop=True, inplace=True)

tms_adata.obs["dataset"] = "tabula_muris_senis"
ab_adata_exons.obs["dataset"] = "allen_brain_exons"

# Safer version: convert only object or mixed-type columns to string
def safe_stringify_obs(adata):
    for col in adata.obs.columns:
        if adata.obs[col].dtype == 'object' or adata.obs[col].apply(type).nunique() > 1:
            adata.obs[col] = adata.obs[col].astype(str)
    return adata

# write compressed file tabsap_adata to file 
ab_adata_introns = safe_stringify_obs(ab_adata_introns)
ab_adata_exons = safe_stringify_obs(ab_adata_exons)
tms_adata = safe_stringify_obs(tms_adata)
ab_adata_exons.var.reset_index(drop=True, inplace=True)

# First, ensure that both datasets have the same gene index type and format
print(f"ab_adata_exons var index type: {type(ab_adata_exons.var.index)}")
print(f"tms_adata var index type: {type(tms_adata.var.index)}")

# Convert both indices to strings if they aren't already
ab_adata_exons.var.index = ab_adata_exons.var.index.astype(str)
tms_adata.var.index = tms_adata.var.index.astype(str)

# Check if indices match
common_genes = set(ab_adata_exons.var.index).intersection(set(tms_adata.var.index))
print(f"Number of common genes: {len(common_genes)} out of {ab_adata_exons.var.shape[0]} and {tms_adata.var.shape[0]}")

# Before combining keep just .var columns that are the same in both
combined_adata = ad.concat(
    [ab_adata_exons, tms_adata],
    axis=0,  # concatenate cells (not genes)
    join="inner",  # require exact matching genes
    label="batch",  # new column in .obs to record original source
    keys=["allen_brain_exons", "tabula_muris_senis"],  # batch labels
    index_unique=None  # don't modify cell ids
)

# Manually copy over the var dataframe (since you confirmed genes match exactly)
combined_adata.var = ab_adata_exons.var.copy()
print(combined_adata.var)
print(combined_adata.obs["batch"].value_counts())

output_dir="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/processed_data" 
# make a dir called processed_data
os.makedirs(output_dir, exist_ok=True)
combined_adata.write_h5ad(os.path.join(output_dir, f"tms_ab_exons_combo_ge_adata_{today}.h5ad"), compression="lzf")
print(f"Saved combined anndata objects to {output_dir}")

ab_adata_introns.write_h5ad(os.path.join(output_dir, f"ab_adata_introns_{today}.h5ad"), compression="lzf")
print(f"Done saving ab_adata_introns to {output_dir}")
print(f"Done processing all datasets.")

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/processed_data
# sbatch --mem=100G --wrap "python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/TabulaSenis_vs_Allen_1.py"