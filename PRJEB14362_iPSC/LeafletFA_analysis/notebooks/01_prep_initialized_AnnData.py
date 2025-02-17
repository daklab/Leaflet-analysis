# %%
import os
import pandas as pd 
from sklearn.decomposition import TruncatedSVD
import anndata as ad
from scipy.sparse import coo_matrix
from datetime import datetime

# turn this into AnnData object 
import anndata as ad
from scipy.sparse import csr_matrix
import numpy as np
import torch 
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

from scipy.spatial.distance import cdist

import numpy as np

import sys
import os
import json
import numpy as np
import torch
import anndata as ad
from importlib import reload
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import pyro 
import umap.umap_ as umap
import matplotlib.patches as mpatches
import scipy.sparse
import datetime
import sys
import random

# Import custom modules
sys.path.append('/gpfs/commons/home/kisaev/Leaflet-private/src/BetaDirichletFactor')

import waypoints as wayp
reload(wayp)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

float_type = {"device": device, "dtype": torch.float}
if device == torch.device('cuda'):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# %%
gtf_annot=True
filter_junctions=True 
gtf_file = "/gpfs/commons/datasets/controlled/BRAIN_NeMO/human-reference/gencode/gencode.v45.primary_assembly.annotation.gtf"

# %% [markdown]
# ### Load merged anndata file containing all cells

# %%
WD="/gpfs/commons/projects/knowles_singlecell_splicing/PRJEB14362/LeafletFA/ATSEs/022025/anndatas/"
anndata="merged_anndata.h5ad"
input_file=f"{WD}/{anndata}"

print(f"Loading anndata object from {input_file}")
adata_full = ad.read_h5ad(input_file)
splice_adata = adata_full.copy()
splice_adata.obs.reset_index(drop=True, inplace=True)
splice_adata.obs["cell_id_index"] = splice_adata.obs.index 

# %%
# Note: this ATSE file was also generated through the script mentioned above
ATSE_file="/gpfs/commons/projects/knowles_singlecell_splicing/PRJEB14362/LeafletFA/ATSEs/022025/output/TMS_atse_file_unanno_also_2025-02-11_21-50-55.txt.gz"
print(f"Loading ATSE file from {ATSE_file}")
atses = pd.read_csv(ATSE_file, sep="\t")

# %%
splice_adata.obs.reset_index(drop=True, inplace=True)
splice_adata.obs["cell_id_index"] = splice_adata.obs.index 
print(splice_adata.obs.shape, splice_adata.obs.cell_id_index.max())

# %% [markdown]
# ### Merge ATSE gene information if it's there to splice_adata object!

# %%
if gtf_annot:
    # find common columns between splice_adata.var and atses
    splice_adata.var = splice_adata.var.merge(atses[["gene_id", "gene_name", "junction_id", "annotation_status", "position_off_5_prime", "position_off_3_prime"]], on=["junction_id", "gene_id", "annotation_status", "gene_name", "position_off_5_prime", "position_off_3_prime"])
else:
    splice_adata.var = splice_adata.var.merge(atses[["junction_id"]], on=["junction_id"])    

# %% [markdown]
# ### Fix "day" variable, add "none" where it is not specified 

# %%
# Assign "unknown" to "day" which is a categorical column that's missing a value 
# add "unknown" to the list of categories
splice_adata.obs["day"] = splice_adata.obs["day"].astype("category")
splice_adata.obs["day"] = splice_adata.obs["day"].cat.add_categories("unknown")
splice_adata.obs["day"].fillna("unknown", inplace=True)

# Remove cells with no day information 
splice_adata = splice_adata[splice_adata.obs["day"] != "unknown", :]

# %% [markdown]
# #### Figure out which junctions to include for model training object

# %%
splice_adata

# %%
print(f"Calculating number of cells with non-zero splice junctions for each junction")
splice_adata.var["non_zero_count_cells"] = np.array((splice_adata.X > 0).sum(axis=0)).flatten()
splice_adata.var["non_zero_cell_prop"] = splice_adata.var["non_zero_count_cells"] / splice_adata.shape[0]

# %%
# let's make an ATSE event id score to decide which ATSEs to keep at the end, since we are limited to under 100,000 splice junctions 
# what do we care about? 
# - number of fully annotated splice junctions in the ATSE vs partially annotated vs unannotated (annotation_status)
# - number of cells that have splice junctions in this ATSE expressed above zero (non_zero_cell_prop)
# - which splice_motif is used in the ATSE (splice_motif)

# let's make a score for each ATSE based on these three criteria 
# - annotation_status: 1 for fully annotated, 0.5 for partially annotated, 0 for unannotated
# - non_zero_cell_prop: 1 if non_zero_cell_prop > 0.01, 0 otherwise
# - splice_motif: 1 if splice_motif is "GT-AG", 0 otherwise

# let's calculate these for every junction and combine and normalize score per ATSE event_id 
splice_adata.var["annotation_status_score"] = splice_adata.var["annotation_status"].map({"both": 1, "five_prime": 0.5, "three_prime": 0.5, "unannotated": 0})
splice_adata.var["non_zero_cell_prop_score"] = (splice_adata.var["non_zero_cell_prop"] > 0.01).astype(int)
splice_adata.var["splice_motif_score"] = (splice_adata.var["splice_motif"] == "GT-AG").astype(int)

# weigh the annotation_status_score more heavily than the other two scores 
splice_adata.var["annotation_status_score"] *= 2
splice_adata.var["non_zero_cell_prop_score"] *= 1.5

# Combine and normalize score per ATSE event_id 
# First group by event_id and sum the scores before normalizing by the number of junctions in the event_id
atse_scores = splice_adata.var.groupby("event_id")[
    ["annotation_status_score", "non_zero_cell_prop_score"]
].sum()

# Normalize by the number of junctions in each event_id
junction_counts = splice_adata.var["event_id"].value_counts().rename("junction_count")

# Add the three columns to make ATSE_score 
atse_scores["atse_score"] = atse_scores.sum(axis=1)
atse_scores["number_of_junctions"] = junction_counts
atse_scores["normalized_atse_score"] = atse_scores["atse_score"] / junction_counts
atse_scores.sort_values("number_of_junctions", ascending=False).head()

# %%
# summarize histogram of normalized_atse_score show percentilues 10% 50% 90% 
atse_scores["normalized_atse_score"].describe(percentiles=[0.1, 0.5, 0.6, 0.9])
# make a histogram of the normalized_atse_score 
atse_scores["normalized_atse_score"].hist(bins=30)

# draw dashed lines for the percentiles 
plt.axvline(atse_scores["normalized_atse_score"].quantile(0.1), color="red", linestyle="--")
plt.axvline(atse_scores["normalized_atse_score"].quantile(0.5), color="green", linestyle="--")
plt.axvline(atse_scores["normalized_atse_score"].quantile(0.6), color="orange", linestyle="--")
plt.axvline(atse_scores["normalized_atse_score"].quantile(0.9), color="blue", linestyle="--")

# add xlaxis and yaxis labels 
plt.xlabel("Normalized ATSE score")
plt.ylabel("# of ATSEs")
plt.show()

# Print how many ATSEs remain after filtering at each of the percentiles
print(f"Number of ATSEs remaining at 10th percentile: {atse_scores[atse_scores['normalized_atse_score'] > atse_scores['normalized_atse_score'].quantile(0.1)].shape[0]}")
print(f"Number of ATSEs remaining at 50th percentile: {atse_scores[atse_scores['normalized_atse_score'] > atse_scores['normalized_atse_score'].quantile(0.5)].shape[0]}")
print(f"Number of ATSEs remaining at 60th percentile: {atse_scores[atse_scores['normalized_atse_score'] > atse_scores['normalized_atse_score'].quantile(0.6)].shape[0]}")
print(f"Number of ATSEs remaining at 90th percentile: {atse_scores[atse_scores['normalized_atse_score'] > atse_scores['normalized_atse_score'].quantile(0.9)].shape[0]}")


# %%
# For splice_adata object, let's filter out the ATSEs that have a normalized_atse_score below the 10th percentile
atse_scores_filt = atse_scores[atse_scores["normalized_atse_score"] > atse_scores["normalized_atse_score"].quantile(0.6)]
splice_adata = splice_adata[:, splice_adata.var["event_id"].isin(atse_scores_filt.index)]

# %%
splice_adata.var.reset_index(drop=True, inplace=True)
# Rename the old junction_id_index to old_junction_id_index 
splice_adata.var.rename(columns={'junction_id_index': 'old_junction_id_index'}, inplace=True)
# Redo the junction_id_index column now that we have removed some junctions
splice_adata.var['junction_id_index'] = splice_adata.var.index
splice_adata.var.head()

# %%
# Print final number of splice junctions, ATSEs, and genes in splice_adata
print(f"Number of splice junctions: {splice_adata.shape[1]}")
print(f"Number of ATSEs: {splice_adata.var['event_id'].nunique()}")

# %%
print(splice_adata.var.splice_motif.value_counts())
print(splice_adata.var.annotation_status.value_counts())

# %%
# Update junction_counts and cluster_counts
junction_counts = splice_adata.layers["cell_by_junction_matrix"].tocoo()
cluster_counts = splice_adata.layers["cell_by_cluster_matrix"].tocoo()

# %%
# Get sparse centered PSI values 
splice_adata.layers["junc_ratio"] = wayp.calculate_centered_psi(junction_counts, cluster_counts)
print(f"Done calculating centered PSI values")

# Step 1: Perform PCA using sparse data
n_components = 30  
n_iter = 3  # Increasing the number of iterations for better convergence
svd = TruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=42)
print(f"Initalizing TruncatedSVD with n_components={n_components} and n_iter={n_iter}")

# Fit and transform the junction ratio data (this gives U)
U = svd.fit_transform(splice_adata.layers["junc_ratio"])

# Get the singular values (S)
S = svd.singular_values_

# Multiply U by S to get U * S
U_by_S = U * S  # This scales each component in U by the corresponding singular value in S
print(f"Done with PCA")

# Store the PCA results (U * S) in the 'X_pca' field of .obsm (multi-dimensional)
splice_adata.obsm['X_pca'] = U_by_S

# store explained variance ratio for future reference
splice_adata.uns['pca_explained_variance_ratio'] = svd.explained_variance_ratio_

# Step 2: Compute UMAP on the PCA-reduced data
sc.pp.neighbors(splice_adata, use_rep='X_pca')
print(f"Done with computing neighbors")

# Step 3. Calculate UMAP 
sc.tl.umap(splice_adata)
print(f"Done with UMAP")

# %%
# plot UMAP with cell types and age groups
sc.pl.umap(splice_adata, color="day", title="UMAP")

# %% [markdown]
# ### Identify cell waypoints using splicing based PCs!

# %%
# Define possible number of waypoints to learn
n_waypoints_learn = [30, 100]

# Placeholder to store waypoints and metacell dictionaries for each n_waypoints
waypoints_dict = {}
metacell_dicts = {}

# Parameters
num_components = 30  # Number of components to consider
metacell_size = 50    # Number of nearest cells to assign to each waypoint
pca_components = splice_adata.obsm["X_pca"]

# Loop over different n_waypoints to generate waypoints and metacell assignments
for n_waypoints in n_waypoints_learn:

    print(f"Finding {n_waypoints} waypoints from the PCA components!")
    random_seed = np.random.randint(0, 10000 + 1)  # Generate random seed
    
    # Max-min sampling to identify waypoints
    waypoints = wayp.max_min_sampling(pca_components, n_waypoints, num_components=num_components, seed=random_seed)
    
    # Store waypoints for this particular number of waypoints
    waypoints_dict[n_waypoints] = waypoints

    # Assign nearest cells to each waypoint (metacells)
    metacell_dict = wayp.assign_nearest_cells(waypoints, pca_components, num_nearest=metacell_size)
    
    # Store the metacell dictionary for this number of waypoints
    metacell_dicts[n_waypoints] = metacell_dict

# %% [markdown]
# #### Try looking at waypoints just using PC space...

# %%
# variable to plot 
var="day"
wayp.plot_PCA_with_waypoints(splice_adata, waypoints_dict, n_waypoints=30, color_by=var, waypoint_color='red', first_waypoint_color='blue', size=10)

# %% [markdown]
# ### Generate_initializations matrices for Phi and Psi

# %%
rho_hat = splice_adata.layers["junc_ratio"]

# Generate multiple initializations
psi_initializations, phi_initializations = wayp.generate_initializations(rho_hat, waypoints_dict, metacell_dicts, epsilon=0.001)

# %%
# Loop through the waypoints_dict and corresponding initializations
for i, n_waypoints in enumerate(waypoints_dict.keys()):

    print(f"Adding waypoint based initializations to anndata for {n_waypoints} waypoints!")

    # Extract the corresponding psi and phi initializations
    psi = psi_initializations[i]
    phi = phi_initializations[i]

    # Convert psi and phi to torch tensors if needed
    psi = torch.tensor(psi)
    phi = torch.tensor(phi)

    # Convert to NumPy arrays if psi and phi are torch tensors (just in case)
    if isinstance(psi, torch.Tensor):
        psi = psi.cpu().numpy()  # Convert to NumPy array
    if isinstance(phi, torch.Tensor):
        phi = phi.cpu().numpy()  # Convert to NumPy array

    # Store psi in `adata.varm` and phi in `adata.obsm` with keys based on the number of waypoints
    splice_adata.varm[f'psi_init_{n_waypoints}_waypoints'] = psi  # Store psi with name 'psi_init_{n_waypoints}_waypoints'
    splice_adata.obsm[f'phi_init_{n_waypoints}_waypoints'] = phi  # Store phi with name 'phi_init_{n_waypoints}_waypoints'


# %%
# remove junc_ratio layer from splice_adata to save memory prior to saving object 
splice_adata.layers.pop("junc_ratio")

# %%
# remove first old_junction_id_index column from splice_adata.var
splice_adata.var.drop(columns=['old_junction_id_index'], inplace=True)

# %% [markdown]
# ### Save splice_adata Anndata object

# %%
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# in file name include timestamp 
new_filename = f"iPSC_human_Anndata_ATSE_counts_with_waypoints_{timestamp}.h5ad"

# Create the new full file path using {WD}
new_file_path = f"{WD}/{new_filename}"

# Save the AnnData object with the new filename
splice_adata.write_h5ad(new_file_path, compression='gzip')

print(f"AnnData saved as {new_file_path}")

# sbatch --wrap "python /gpfs/commons/home/kisaev/Leaflet-analysis/PRJEB14362_iPSC/LeafletFA_analysis/notebooks/01_prep_initialized_AnnData.py" -p bigmem --mem 400G -J prep_initialized_AnnData --time 24:00:00