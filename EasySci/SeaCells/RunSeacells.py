# Load dependencies
import numpy as np
import pandas as pd
import scanpy as sc
import SEACells
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import fast_matrix_market as fmm
import sys
import os

# Some plotting aesthetics
sns.set_style('ticks')

# Working directory 
WD="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/Seacells/"

# Write new log file in the working directory to which all print statements will be written
sys.stdout = open(f"{WD}log.txt", "w")

print("Starting Seacells analysis on EasySci2024 dataset...", flush=True)

# Generate Anndata object from gene count matrix and annotation file 
# GE matrix 
ge_matrix="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/EasySci-RNA-mouse-brain/GSM6538356_RNA_gene_count.txt.gz"
# Metadata 
metadata="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/EasySci-RNA-mouse-brain/GSM6538356_RNA_cell_annotation.csv.gz"

print("Reading in gene expression matrix and metadata...", flush=True)
# Replace 'data.mtx' with the path to your MatrixMarket file
matrix = fmm.mmread(ge_matrix)

# transpose matrix 
matrix = matrix.T

# convert COO to CSR or CSC sparse matrix
matrix = matrix.tocsr()

print("Making anndata object...", flush=True)
adata = sc.AnnData(X=matrix)

# Gene annotations 
gene_annots = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/EasySci-RNA-mouse-brain/GSM6538356_RNA_gene_annotation.csv.gz"
print("Adding in gene annotations...", flush=True)

# Read in gene_annots and clean up dataframe 
gene_annots = pd.read_csv(gene_annots)

# now add gene_annots as gene annotations in the anndata object
adata.var = gene_annots

# create anndata object from gene_matrix and metadata
metadata = pd.read_csv(metadata)
adata.obs = metadata

print("Done creating anndata object", flush=True)

# Pre-processing
# The following section describes basic pre-processing steps for scRNA-seq.

# Copy the counts to ".raw" attribute of the anndata since it is necessary for downstream analysis
# This step should be performed after filtering 
ad = adata
raw_ad = sc.AnnData(ad.X)
raw_ad.obs_names, raw_ad.var_names = ad.obs_names, ad.var_names
ad.raw = raw_ad

# Normalize to 10,000 reads per cell
sc.pp.normalize_total(ad, target_sum=1e4)  
# Log transform
sc.pp.log1p(ad)
# Compute highly variable genes
sc.pp.highly_variable_genes(ad, n_top_genes=1500, flavor="seurat_v3") # "seurat_v3" is a method similar to Seurat v3, which can handle large datasets efficiently.

# Compute principal components - 
# Here we use 50 components. This number may also be selected by examining variance explaint
sc.tl.pca(ad, n_comps=50, use_highly_variable=True)

# Subset adata object to contain just two major cell types 
# Check how many cells are in each cell type
print(ad.obs['Main_cluster_name'].value_counts(), flush=True)

# As a test subset ad to just the "Main_cluster_name" which equals "Habenula neurons" or "Pituitary cells"
# ad = ad[ad.obs['Main_cluster_name'].isin(["Habenula neurons", "Pituitary cells"])] 

# Running SEACells
# As a rule of thumb, we recommended choosing one metacell for every 75 single-cells. Since this dataset contains ~7k cells, we choose 90 metacells.
num_cells = ad.shape[0]
print(f"Number of cells: {num_cells}", flush=True)

num_metacells = num_cells / 75
print(f"Number of metacells: {num_metacells}", flush=True) 
# round num_meta_cells to the nearest integer 
num_metacells = int(round(num_metacells))
print(f"Number of rounded metacells to learn: {num_metacells}", flush=True)

# renumber the index in adata_subset 
ad.obs_names = [f"cell-{i}" for i in range(ad.shape[0])]
print("Renamed the index in adata object", flush=True)

# Let's rerun PCA and UMAP on this adata_subset object and then plot it
sc.tl.pca(ad, n_comps=50, use_highly_variable=True)
sc.pp.neighbors(ad, n_neighbors=15, n_pcs=50)
sc.tl.umap(ad)

print("Done running PCA, UMAP on ad object", flush=True)

os.makedirs(WD, exist_ok=True)

# Plot UMAP, colour by Main-cluster_name and save in the working directory
sc.pl.umap(ad, color='Main_cluster_name', save=".pdf")

#User defined parameters

## Core parameters 
n_SEACells = num_metacells
build_kernel_on = 'X_pca' # key in ad.obsm to use for computing metacells
                          # This would be replaced by 'X_svd' for ATAC data

## Additional parameters
n_waypoint_eigs = 10 # Number of eigenvalues to consider when initializing metacells

model = SEACells.core.SEACells(ad, 
                  build_kernel_on=build_kernel_on, 
                  n_SEACells=n_SEACells, 
                  n_waypoint_eigs=n_waypoint_eigs,
                  convergence_epsilon = 1e-3)

model.construct_kernel_matrix()
M = model.kernel_matrix

# Initialize archetypes
model.initialize_archetypes()

# Plot the initilization to ensure they are spread across phenotypic space and save plot in working directory
SEACells.plot.plot_initialization(ad, model)
plt.savefig(f"{WD}initialization.pdf")

model.fit(min_iter=10, max_iter=500)

# Accessing results
# Check for convergence 
model.plot_convergence()
plt.savefig(f"{WD}convergence.pdf")

print("Done fitting SEACells model", flush=True)

# how many cells in each SEACell?
print("The number of cells in each SEACell:", flush=True)
print(ad.obs['SEACell'].value_counts(), flush=True)

# write code to save adata to file
print("Saving adata to file", flush=True)
ad.write('/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/Seacells/EasySci.h5ad')

print("Done running Seacells on EasySci2024 dataset.", flush=True)

#cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/Seacells/slurm/03212024
#script=/gpfs/commons/home/kisaev/Leaflet-analysis/EasySci/SeaCells/RunSeacells.py
#sbatch --mem=128G --job-name=SeaCells --wrap="python $script"
