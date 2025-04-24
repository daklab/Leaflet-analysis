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

# sys.path.append('../utils')
# from functions import * 
# Flag for whether to also read intron and exons files 
read_introns_exons = True

# %%
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
    adata = adata[adata.obs_names.isin(metadata_sub.index)]
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

if read_introns_exons:
    ab_adata_introns = sc.read_csv(introns_only)
    ab_adata_introns = ab_adata_introns.transpose()
    print(f"Done reading {introns_only}")

    ab_adata_exons = sc.read_csv(exons_only)
    ab_adata_exons = ab_adata_exons.transpose()
    print(f"Done reading {exons_only}")
    
## Process total counts normally (uses 'sample_name')
if read_introns_exons:
    ab_adata_introns = preprocess_ab_adata(ab_adata_introns, metadata, metadata_key="sample_name")
    ab_adata_exons = preprocess_ab_adata(ab_adata_exons, metadata, metadata_key="sample_name")

print("All AB datasets preprocessed with correct metadata mapping.")

# ### Read in the Tabula Muris Senis gene expression data 
# # Load the expression data (.h5ad file)
exp_file = "/gpfs/commons/projects/knowles_singlecell_splicing/TabulaSenis/data/AWS/processed_for_scanpy/tabulamurissenisfacsofficialrawobj.h5ad"
tms_adata = sc.read_h5ad(exp_file)
tms_adata.obs["dataset"] = "tabula_sapiens"
tms_adata.layers["raw_counts"] = tms_adata.X.copy()

# Reset the index of adata.obs to integers and drop the old index
tms_adata.obs.reset_index(drop=True, inplace=True)
tms_adata.var["gene_symbol"] = tms_adata.var.index

# Clean up Cell IDs in the gene expression object
# If age column is 3m, adjust cell_clean to replace the first "." with "_" and the second "." also with "_" (annoying fix but that's how the data came...)
tms_adata.obs['cell_clean'] = tms_adata.obs['cell']  # Start by copying the 'cell' column to 'cell_clean'
tms_adata.obs.loc[tms_adata.obs['age'] == '3m', 'cell_clean'] = tms_adata.obs.loc[tms_adata.obs['age'] == '3m', 'cell'].str.replace('.', '_', 2)
tms_adata.obs["cell_clean"] = tms_adata.obs["cell_clean"].str.split('_').str[:2].str.join('_') 
print(f"The number of cells in the expression dataset is {tms_adata.shape[0]}")

print("Number of genes in common between the two datasets:", len(set(tms_adata.var["gene_symbol"]).intersection(set(ab_adata_introns.var["gene_symbol"]))))
# Find the rows in tms_adata.var that are duplicated
duplicated_genes = tms_adata.var[tms_adata.var["gene_symbol"].duplicated(keep=False)]
tms_adata = tms_adata[:, ~tms_adata.var.index.isin(duplicated_genes.index)].copy()
tms_adata.var["gene_name"] = tms_adata.var["gene_symbol"]
tms_adata = tms_adata[:, tms_adata.var["gene_name"].isin(gene_info_df["gene_name"])].copy()

# Clean up allen brain exon and intron datasets 
ab_adata_introns.var["gene_name"] = ab_adata_introns.var["gene_symbol"]
ab_adata_introns = ab_adata_introns[:, ab_adata_introns.var["gene_name"].isin(gene_info_df["gene_name"])]

ab_adata_exons.var["gene_name"] = ab_adata_exons.var["gene_symbol"]
ab_adata_exons = ab_adata_exons[:, ab_adata_exons.var["gene_name"].isin(gene_info_df["gene_name"])]

# merge tms_adata and ab_adata with gene_info_df 
tms_adata.var = tms_adata.var.reset_index().merge(gene_info_df, on="gene_name").set_index("index")
ab_adata_introns.var = ab_adata_introns.var.reset_index().merge(gene_info_df, on="gene_name").set_index("index")
ab_adata_exons.var = ab_adata_exons.var.reset_index().merge(gene_info_df, on="gene_name").set_index("index")

# Sort by gene_name to ensure identical gene order
tms_adata = tms_adata[:, tms_adata.var["gene_name"].sort_values().index]
ab_adata_introns = ab_adata_introns[:, ab_adata_introns.var["gene_name"].sort_values().index]
ab_adata_exons = ab_adata_exons[:, ab_adata_exons.var["gene_name"].sort_values().index]

# convert .X to sparse array in ab_adata, ab_adata_exons, ab_adata_introns
ab_adata_introns.X = csr_matrix(ab_adata_introns.X)
ab_adata_exons.X = csr_matrix(ab_adata_exons.X)

# %%
output_dir="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/processed_data" 
# make a dir called processed_data
os.makedirs(output_dir, exist_ok=True)

# write compressed file tabsap_adata to file 
tms_adata.write_h5ad(os.path.join(output_dir, f"tms_adata_{today}.h5ad"), compression="gzip", compression_opts=9)
print(f"Done saving tms_adata to {output_dir}")

# Safer version: convert only object or mixed-type columns to string
def safe_stringify_obs(adata):
    for col in adata.obs.columns:
        if adata.obs[col].dtype == 'object' or adata.obs[col].apply(type).nunique() > 1:
            adata.obs[col] = adata.obs[col].astype(str)
    return adata

ab_adata_introns = safe_stringify_obs(ab_adata_introns)
ab_adata_exons = safe_stringify_obs(ab_adata_exons)

ab_adata_exons.write_h5ad(os.path.join(output_dir, f"ab_adata_exons_{today}.h5ad"), compression="gzip", compression_opts=9)
print(f"Done saving ab_adata_exons to {output_dir}")

ab_adata_introns.write_h5ad(os.path.join(output_dir, f"ab_adata_introns_{today}.h5ad"), compression="gzip", compression_opts=9)
print(f"Done saving ab_adata_introns to {output_dir}")

print(f"Saved all anndata objects to {output_dir}")
print(f"Done processing all datasets.")

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/processed_data
# sbatch --mem=100G --wrap "python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/TabulaSenis_vs_Allen_1.py"