#!/usr/bin/env python
"""
LeafletFA Model Evaluation in Mouse 

This script:
1. Loads trained LeafletFA model outputs and associated data (mouse foundation smart-seq data)
2. Loads EasySci data (just brain) meta-cells anndata with junctions and ATSEs mapped using mouse foundation atse file
3. Applies the model to the EasySci data
4. Saves the predicted factor activities and factor usage
"""

# -------------------------------------------------------------------------------------
# IMPORT LIBRARIES
# -------------------------------------------------------------------------------------

import os
import sys
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

import importlib
import datetime
import numpy as np
import pandas as pd
import anndata as ad
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import gffutils
import scanpy as sc
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mord import OrdinalRidge
from scipy.stats import spearmanr, pearsonr
from sklearn.utils import resample
from tqdm import tqdm
import torch
import gzip
import pickle
import glob
from scipy.sparse import csr_matrix
torch.manual_seed(0)

# LeafletFA model related imports 
# Ensure CUDA is available and if not use CPU
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device found")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

float_type = {"device": device, "dtype": torch.float}
torch.set_default_tensor_type("torch.FloatTensor" if device.type == "cpu" else "torch.cuda.FloatTensor")

# Import utility functions - simple direct import
sys.path.append('/gpfs/commons/home/kisaev/Leaflet-analysis/Multi_Species_Splicing_Foundation/shared_utils/')
from utils import load_model

# Define module paths
src_path = "/gpfs/commons/home/kisaev/Leaflet-private/src/"

# Add to sys.path if not already present
if src_path not in sys.path:
    sys.path.append(src_path)

# Import custom modules
import BetaDirichletFactor.LeafletFA as LeafletFA
import BetaDirichletFactor.utils as utilsFA

# Configure plotting styles
sns.set_theme()
sc.set_figure_params(figsize=(7, 7), frameon=True, dpi=80, facecolor='white')

# Check if using CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------------------------------
# SET PATHS TO DATA 
# -------------------------------------------------------------------------------------

# Mouse foundation data 
BASE_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
ATSE_ANNDATA_PATH = f"{BASE_DIR}/MODEL_INPUT/052025/MOUSE_SPLICING_FOUNDATION_Anndata_ATSE_counts_with_waypoints_20250513_073829.h5ad"
ATSE_MOUSE_FOUNDATION_FILE_PATH = f"{BASE_DIR}/ATSE_mapper/ATSE_files/MOUSE_FOUNDATION_ATSE_FILE_unanno_also_2025-04-26_19-55-26.txt.gz"
ATSE_MOUSE_FOUNDATION_FILE_LIFTOVER = f"{BASE_DIR}/ATSE_mapper/ATSE_files/MOUSE_FOUNDATION_ATSE_FILE_unanno_also_2025-04-26_19-55-26_lifted_mm39.txt.gz"

# -------------------------------------------------------------------------------------
# READ IN ATSE FILES
# -------------------------------------------------------------------------------------

# Read in original atse file
atse_mouse_foundation_df = pd.read_csv(ATSE_MOUSE_FOUNDATION_FILE_PATH, sep="\t")
atse_easysci_df = pd.read_csv(ATSE_MOUSE_FOUNDATION_FILE_LIFTOVER, sep="\t")

# -------------------------------------------------------------------------------------
# READ IN ANNDATA MOUSE FOUNDATION AND EASYSCI
# -------------------------------------------------------------------------------------

# Read in mouse foundation anndata
mouse_foundation_adata = ad.read_h5ad(ATSE_ANNDATA_PATH)
print(f"The number of cells in the dataset is {mouse_foundation_adata.shape[0]}")
print(f"The number of junctions in the dataset is {mouse_foundation_adata.shape[1]}")

# EasySci mouse data (just brain)
easysci_input_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/ATSEmap/RH/output/junction_processing_20250223/anndatas/merged_anndata.h5ad"
easysci_adata = ad.read_h5ad(easysci_input_file)
print(f"The number of cells in the dataset is {easysci_adata.shape[0]}")
print(f"The number of junctions in the dataset is {easysci_adata.shape[1]}")

# Mapping dictionary for cleaned cluster names
cluster_mapping = {
    'Cerebellum_granule_neurons': 'Neurons',
    'Interbrain_and_midbrain_neurons_1': 'Neurons',
    'Interbrain_and_midbrain_neurons_2': 'Neurons',
    'Oligodendrocytes': 'Oligodendrocytes',
    'Cortical_projection_neurons_1': 'Neurons',
    'Astrocytes': 'Astrocytes',
    'Striatal_neurons_1': 'Neurons',
    'Oligodendrocyte_progenitor_cells': 'OPCs',
    'OB_neurons_1': 'Neurons',
    'Interneurons_1': 'Neurons',
    'Cortical_projection_neurons_2': 'Neurons',
    'Cerebellum_interneurons': 'Neurons',
    'Interneurons_2': 'Neurons',
    'Endothelial_cells': 'Endothelial Cells',
    'Microglia': 'Microglia',
    'Dentate_gyrus_neurons': 'Neurons',
    'Bergmann_glia': 'Astrocytes',
    'Purkinje_neurons': 'Neurons',
    'Vascular_leptomeningeal_cells': 'Endothelial Cells',
    'OB_neurons_2': 'Neurons',
    'Choroid_plexus_epithelial_cells': 'Epithelial Cells',
    'Cortical_projection_neurons_3': 'Neurons',
    'Striatal_neurons_2': 'Neurons',
    'Ependymal_cells': 'Ependymal Cells',
    'OB_neurons_3': 'Neurons',
    'Hindbrain_neurons_2': 'Neurons',
    'Hindbrain_neurons_1': 'Neurons',
    'Habenula_neurons': 'Neurons'
}

# Apply mapping to create a new column
easysci_adata.obs['broad_cell_type'] = easysci_adata.obs['Main_cluster_name'].map(cluster_mapping)

# Subset atse_easysci_df to just the junctions that are in the mouse foundation anndata
atse_easysci_df = atse_easysci_df[atse_easysci_df['mouse_foundation_junction_id'].isin(mouse_foundation_adata.var['junction_id'])]

# Need to replace junction_id column now with values in atse_easysci_df['mouse_foundation_junction_id']
atse_easysci_df_sub = atse_easysci_df[["junction_id", "mouse_foundation_junction_id"]].drop_duplicates()
# check if any junction_id or mouse_foundation_junction_id are duplicated 
if atse_easysci_df_sub['junction_id'].duplicated().any() or atse_easysci_df_sub['mouse_foundation_junction_id'].duplicated().any():
    print("Warning: junction_id or mouse_foundation_junction_id are duplicated")
    print(f"Number of duplicated junction_id: {atse_easysci_df_sub['junction_id'].duplicated().sum()}")
    print(f"Number of duplicated mouse_foundation_junction_id: {atse_easysci_df_sub['mouse_foundation_junction_id'].duplicated().sum()}")

# Remove the duplicated junction_id values, remove  atse_easysci_df_sub['junction_id'].duplicated()
atse_easysci_df_sub = atse_easysci_df_sub[~atse_easysci_df_sub['junction_id'].duplicated()]

# Subset easysci_adata to only include junctions that are in atse_easysci_df_sub
easysci_adata = easysci_adata[:, easysci_adata.var['junction_id'].isin(atse_easysci_df_sub['junction_id'])]

# Merge easysci_adata.var with atse_easysci_df_sub on the junction_id column
easysci_adata.var = pd.merge(easysci_adata.var, atse_easysci_df_sub, on='junction_id')
# Now rename the junction_id column to mouse_foundation_junction_id (swap the names)
easysci_adata.var.rename(columns={'junction_id': 'mouse_foundation_junction_id', 'mouse_foundation_junction_id': 'junction_id'}, inplace=True)
print(f"The number of junctions in the dataset is {easysci_adata.shape[1]}")

# Confirm number of junctions in easysci_adata.var is the same as in mouse_foundation_adata.var
print(f"The number of junctions EasySci that are the same as in Mouse Foundation is {len(easysci_adata.var['junction_id'].isin(mouse_foundation_adata.var['junction_id']))}")

# -------------------------------------------------------------------------------------
# LOAD TRAINED LEAFLET FA MODEL
# -------------------------------------------------------------------------------------

# Load the best model that was trained on the Mouse Foundation data 
print("\n>> Extracting model parameters...")
model_home = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel/2025-05-13/"
param_id = 0
model_path = f"{model_home}/run_{param_id}/leafletfa_model.pkl.xz"
leaflet_model = load_model(model_path)

# Extract hyperparameters
alpha_pi = leaflet_model["alpha_pi"]
bb_conc = leaflet_model["bb_conc"]
dir_conc = leaflet_model["dir_conc"]

# Extract factor activities and usage
PHI = leaflet_model["assign_post"]
PI = leaflet_model["pi"]
PSI = leaflet_model["psi_learned"].T
K = leaflet_model["K"]

# -------------------------------------------------------------------------------------
# APPLY MODEL TO EASYSCI DATA
# -------------------------------------------------------------------------------------

# how do we do this? using junctions that are mapped in EasySci, get their factor usage values  
# we also have estimated PSI values in EasySci using raw junction and ATSE counts 
# can we estimate factor activities from the PSI values? 
# Ensure the PSI matrix is aligned with easysci junctions
# First align index to mouse_foundation_adata.var so we can match order correctly
psi_df = pd.DataFrame(PSI, columns=[f"factor_{i}" for i in range(K)])
psi_df["junction_id"] = mouse_foundation_adata.var["junction_id"].values
psi_df = psi_df.set_index("junction_id")

# Subset PSI matrix to junctions in EasySci
shared_junctions = easysci_adata.var["junction_id"]
psi_subset = psi_df.loc[shared_junctions]

# Ensure PSI matrix order of junctions follows the order of easysci_adata.var["junction_id"]
assert (psi_subset.index == easysci_adata.var["junction_id"]).all()

# -------------------------------------------------------------------------------------
# ESTIMATE FACTOR ACTIVITIES FROM PSI VALUES
# -------------------------------------------------------------------------------------

# We will run LeafletFA here but using a fixed PSI matrix 
# Check what shape and type PSI matrix need to be in 
# leaflet_model["psi_learned"].shape
# Out[14]: (30, 16530)
# so need to make sure psi_subset is a numpy array of shape (J, K)
psi_input = psi_subset.T.values
output_dir = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel/EasySci_results"
# if doesn't exist, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#Initialize model (maybe should also use fixed PI here...)
print("Initializing LeafletFA model...")
easysci_leaflet_model = LeafletFA.LeafletFA(
    adata=easysci_adata, 
    K=K, 
    fixed_psi=torch.tensor(psi_input),
    junc_specific_prior=leaflet_model["junc_specific_prior"], 
    waypoints_use=False, # should i use them here? feels like too much... 
    input_conc_prior=None, 
    delta_fixed=leaflet_model["dir_conc"],
    num_epochs=300, 
    print_epochs=5, 
    ELBO_num_particles=10, 
    lr=0.3, 
    gamma=0.001, 
    min_delta=10,
    num_samples=100, 
    patience=5,
    output_dir=output_dir,
    log_wandb=False  # Log to wandb
)

# Print confirm that model has dir_conc 
print(f"Model initialized with dir_conc: {easysci_leaflet_model.dir_conc}")

# Train model
print(f"Extracting sparse tensors from anndata object")
easysci_leaflet_model.from_anndata()

print(f"Obtaining mask for sparse operations")
easysci_leaflet_model.initialize_triton_mask()

print("Training LeafletFA model...")
easysci_leaflet_model.train(num_initializations=1)

print("Training complete, extracting results...")
easysci_leaflet_model.get_all_variables()

# Save latent variables
easysci_adata.obsm[f"X_leafletFA_K{K}"] = easysci_leaflet_model.assign_post

# Make a quick barplot of PI and add to wandb log 
alpha_pi=easysci_leaflet_model.alpha_pi
PI = easysci_leaflet_model.pi
PI_df = pd.DataFrame(PI, columns=["PI"])

temp_save="/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/model_evaluation/"
# Run neighbor analysis and UMAP on easysci_adata X_leafletFA_K{K}
sc.pp.neighbors(easysci_adata, use_rep="X_leafletFA_K30")
sc.tl.umap(easysci_adata, min_dist=0.5)
# Make plot using broad_cell_type as color 
sc.pl.umap(easysci_adata, color="broad_cell_type", legend_loc="on data", title="UMAP of EasySci data colored by broad cell type")
# save plot 
plt.savefig(f"{temp_save}/easysci_adata_leafletFA_K{K}_umap.png")














