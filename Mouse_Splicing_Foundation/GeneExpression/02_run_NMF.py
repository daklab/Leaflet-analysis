# %%
import os

# Get the current working directory
current_dir = os.getcwd()
print("Current working directory:", current_dir)

from sklearn.decomposition import NMF
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

# %%
# Set up paths for files 
gene_exp_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/processed_data/adjusted_gene_expression/Combined_adjusted_GeneExpression_2025-04-30.h5ad"  
ge_adata = ad.read_h5ad(gene_exp_file)
print("Loaded gene expression file containing both TMS + Allen Brain data!")

# %%
print(ge_adata.layers)

# %%
# Extract the non-negative matrix
X = ge_adata.layers["log_norm"]

# If it's a sparse matrix, densify it first
if not isinstance(X, np.ndarray):
    X = X.toarray()

print(X.shape)  # (n_cells, n_genes)

# %%
# For now X = np.nan_to_num(X, nan=0.0)
X = np.nan_to_num(X, nan=0.0)
print("Replacing any NaNs with zeroes... (check on this...)")

# %%
from sklearn.decomposition import MiniBatchNMF

n_components = 30
print(f"Starting MiniBatch NMF with {n_components}!")

nmf_model = MiniBatchNMF(
    n_components=n_components,
    init="nndsvda",         
    batch_size=512,       
    max_iter=100,         
    random_state=0,
    tol=1e-3,              
    verbose=1,             
)

W = nmf_model.fit_transform(X)  # Cells × Factors (n_obs × K)
H = nmf_model.components_       # Factors × Genes (K × n_vars)
print(f"Done training NMF and saving to anndata object!")
# Save
ge_adata.obsm["X_NMF"] = W
ge_adata.uns["NMF_loadings"] = H

# %%
print("Calculating neighbors, umap and leiden clusters!")
# Step 1: Compute neighbors using NMF latent space
sc.pp.neighbors(ge_adata, use_rep="X_NMF", key_added="neighbors_NMF", n_neighbors=20)
# Step 2: Compute UMAP from NMF latent space
sc.tl.umap(ge_adata, neighbors_key="neighbors_NMF", min_dist=0.3)
# Step 3: Cluster cells based on NMF space
sc.tl.leiden(ge_adata, neighbors_key="neighbors_NMF", resolution=0.8)

# %%
# Save updated gene expression anndata object 
# Define save path
today = datetime.datetime.now().strftime("%Y-%m-%d")

import gc
print("Saving file!")

# Only keep necessary layers and annotations before saving
layers_to_keep = ["length_norm", "raw_counts"]  # adjust as needed
for key in list(ge_adata.layers.keys()):
    if key not in layers_to_keep:
        del ge_adata.layers[key]

# Force garbage collection to free memory
gc.collect()

# Save using gzip (higher compression, slower, but uses less disk/mem than lzf)
output_file = f"/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/processed_data/adjusted_gene_expression/Combined_adjusted_GeneExpression_with_NMF_{today}.h5ad"
ge_adata.write_h5ad(output_file, compression="gzip", compression_opts=9)

print(f"Saved updated AnnData with NMF results to {output_file}")

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION
# sbatch --mem=500G -p cpu,dev,bigmem --wrap "python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/GeneExpression/02_run_NMF.py"


