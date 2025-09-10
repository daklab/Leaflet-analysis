#!/usr/bin/env python
"""
LeafletFA Model Training Script
Trains a factor analysis model on mouse splicing foundation data.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import mudata as mu
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import TruncatedSVD
import pickle
import numpy as np
from scipy.sparse import csr_matrix

# Configure environment
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.set_default_tensor_type("torch.FloatTensor" if device.type == "cpu" else "torch.cuda.FloatTensor")
torch.manual_seed(0)

# Configure plotting
sns.set_theme()
sc.set_figure_params(figsize=(7, 7), frameon=True, dpi=80, facecolor='white')

# Add custom module paths
src_path = "/gpfs/commons/home/kisaev/Leaflet-private/src/"
if src_path not in sys.path:
    sys.path.append(src_path)

import BetaDirichletFactor.LeafletFA as LeafletFA
import BetaDirichletFactor.utils as utils
import BetaDirichletFactor.waypoints as wayp

def load_model(model_file):
    """Load LeafletFA model from file with device handling"""
    model = {}

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model to device: {device}")

    # Patch torch.load to respect map_location
    def device_load(*args, **kwargs):
        kwargs.setdefault("map_location", device)
        return original_torch_load(*args, **kwargs)

    # Save original and patch
    original_torch_load = torch.load
    torch.load = device_load

    try:
        with gzip.open(model_file, "rb") as f:
            while True:
                try:
                    attr_dict = pickle.load(f)
                    model.update(attr_dict)
                except EOFError:
                    break
    finally:
        torch.load = original_torch_load  # Restore original

    return model

# =============================================================================
# Configuration
# =============================================================================

# File paths
TEST_ADATA_PATH = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/MASKED_0.2_test_30_70_ge_splice_combined_20250730_164104.h5mu"
OUTPUT_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_OUTPUT/082025"

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

print(f"Loading Training MuData from {TEST_ADATA_PATH}...")
mdata = mu.read_h5mu(TEST_ADATA_PATH)
ad = mdata["splicing"]

print(f"Found layers in training AnnData: {list(ad.layers.keys())}")

# Reset and prepare cell indices
ad.obs.reset_index(drop=True, inplace=True)
ad.obs["cell_id_index"] = ad.obs.index

# Reduce ad to only junctions in ATSEs that have <= 3 junctions in them 
ad = ad[:, ad.var["num_junctions"] <= 3].copy()

# Reset the junction_id_index to be 0 to num_junctions - 1
ad.var["junction_id_index"] = np.arange(ad.shape[1])
print(f"Done reducing adata to junctions in ATSEs that have <= 3 junctions in them")
print(f"New adata shape: {ad.shape}")

# =============================================================================
# Make sure that the sparsity pattern of the cell_by_junction_matrix and cell_by_cluster_matrix match 
# =============================================================================

ad.layers["cell_by_junction_matrix"], ad.layers["cell_by_cluster_matrix"]
print("Fixing sparsity pattern mismatch...")

# Get the matrices
junction_matrix = ad.layers["cell_by_junction_matrix"] 
cluster_matrix = ad.layers["cell_by_cluster_matrix"]

print(f"Before fix:")
print(f"  Junction matrix: {junction_matrix.nnz} non-zeros")
print(f"  Cluster matrix: {cluster_matrix.nnz} non-zeros")

# Convert both to COO format to work with indices
junction_coo = junction_matrix.tocoo()
cluster_coo = cluster_matrix.tocoo()

# Get the sparsity pattern (locations of non-zeros) from cluster matrix
cluster_indices = set(zip(cluster_coo.row, cluster_coo.col))
junction_indices = set(zip(junction_coo.row, junction_coo.col))

# Find missing positions in junction matrix
missing_positions = cluster_indices - junction_indices
print(f"Missing positions in junction matrix: {len(missing_positions)}")

if len(missing_positions) > 0:
    # Create arrays for the missing positions
    missing_rows = [pos[0] for pos in missing_positions]
    missing_cols = [pos[1] for pos in missing_positions]
    missing_values = np.zeros(len(missing_positions), dtype=junction_matrix.dtype)
    
    # Combine existing data with missing zeros
    all_rows = np.concatenate([junction_coo.row, missing_rows])
    all_cols = np.concatenate([junction_coo.col, missing_cols])
    all_values = np.concatenate([junction_coo.data, missing_values])
    
    # Create new junction matrix with explicit zeros
    new_junction_matrix = csr_matrix(
        (all_values, (all_rows, all_cols)), 
        shape=junction_matrix.shape,
        dtype=junction_matrix.dtype
    )
    
    # Update the layer
    ad.layers["cell_by_junction_matrix"] = new_junction_matrix
    
    print(f"After fix:")
    print(f"  Junction matrix: {new_junction_matrix.nnz} non-zeros")
    print(f"  Cluster matrix: {cluster_matrix.nnz} non-zeros")
    print(f"  ✓ Sparsity patterns now match!")
    
else:
    print("✓ No missing positions found - sparsity patterns already match")

# Verify they now have the same number of non-zeros
final_junction = ad.layers["cell_by_junction_matrix"]
final_cluster = ad.layers["cell_by_cluster_matrix"]

if final_junction.nnz == final_cluster.nnz:
    print(f"✓ Success! Both matrices now have {final_junction.nnz} non-zeros")
else:
    print(f"❌ Still mismatched: {final_junction.nnz} vs {final_cluster.nnz}")
    
print("✓ Ready for LeafletFA model!")

# =============================================================================
# Load in model trained using the 70% training data 
# =============================================================================

leaflet_model = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_OUTPUT/082025/leafletfa_model_20250821_131921.pkl.gz"
leaflet_model = load_model(leaflet_model)

psi_learned = 1 / (1 + np.exp(-(leaflet_model["psis_loc"] + leaflet_model["psis_scale"] * np.random.standard_normal(leaflet_model["psis_loc"].shape))))

# =============================================================================
# ESTIMATE FACTOR ACTIVITIES FROM PSI VALUES
# =============================================================================

# We will run LeafletFA here but using a fixed PSI matrix 
# Check what shape and type PSI matrix need to be in 
# leaflet_model["psi_learned"].shape K by J
psi_input = psi_learned
K = 20
print(f"The shape of the PSI input is: {psi_input.shape}")

#Initialize model (maybe should also use fixed PI here...)
print("Initializing LeafletFA model...")
masked_test_leaflet_model = LeafletFA.LeafletFA(
    adata=ad, # test data with 20% masked values 
    K=K, 
    fixed_psi=torch.tensor(psi_input),
    pi_init=torch.tensor(leaflet_model["pi"]),
    alpha_pi_init = torch.tensor(leaflet_model["alpha_pi"]),
    junc_specific_prior=leaflet_model["junc_specific_prior"], 
    waypoints_use=False, 
    input_conc_prior=np.inf, # ideally should use the one the original model learned
    delta_fixed=torch.tensor(leaflet_model["dir_conc"]),
    num_epochs=20, 
    print_epochs=5, 
    ELBO_num_particles=10, 
    lr=0.6, 
    gamma=0.005, 
    min_delta=10,
    num_samples=100, 
    patience=10,
    output_dir=OUTPUT_DIR,
    log_wandb=False  # Log to wandb
)

# Print confirm that model has dir_conc 
print(f"Model initialized with dir_conc: {masked_test_leaflet_model.dir_conc}")
print(f"Model initialized with pi_init: {masked_test_leaflet_model.pi_init} and alpha_pi_init: {masked_test_leaflet_model.alpha_pi_init}")

# Train model
print(f"Extracting sparse tensors from anndata object")
masked_test_leaflet_model.from_anndata()

print(f"Obtaining mask for sparse operations")
masked_test_leaflet_model.initialize_triton_mask()

print("Training LeafletFA model...")
masked_test_leaflet_model.train(num_initializations=1)

print("Done training model!")

# =============================================================================
# Extract learned PHI
# =============================================================================

print("Training complete, extracting results...")
masked_test_leaflet_model.get_all_variables()

# Save latent variables
ad.obsm[f"X_leafletFA_K{K}"] = masked_test_leaflet_model.assign_post

# Make a quick barplot of PI and add to wandb log 
alpha_pi=masked_test_leaflet_model.alpha_pi
PI = masked_test_leaflet_model.pi
PI_df = pd.DataFrame(PI, columns=["PI"])

# Calculate imputed PSI by multiplying PHI by PI 
imputed_psi = masked_test_leaflet_model.assign_post @ psi_learned

# Add imputed PSI to adata 
ad.layers["imputed_psi"] = imputed_psi

from scipy import sparse

# ensure CSR for fast row/col lookups
masked_orig = ad.layers["junc_ratio_masked_original"]

if not sparse.isspmatrix_csr(masked_orig):
    masked_orig = sparse.csr_matrix(masked_orig)

bin_mask = ad.layers["junc_ratio_masked_bin_mask"]
if not sparse.isspmatrix_csr(bin_mask):
    bin_mask = sparse.csr_matrix(bin_mask)

# get masked locations (row, col indices)
rows, cols = bin_mask.nonzero()

# ground-truth original PSI values (may be zero)
orig_vals = masked_orig[rows, cols].A1  # .A1 = flatten to 1D

# model predictions (dense output)
pred_vals = imputed_psi[rows, cols]

import numpy as np
from scipy.stats import spearmanr
pearson_m  = np.corrcoef(orig_vals, pred_vals)[0, 1]
spearman_m = spearmanr(orig_vals, pred_vals, nan_policy="omit")[0]
print(f"[impute-test] masked‐ATSE PSI corr — Pearson: {pearson_m:.4f}, Spearman: {spearman_m:.4f}")


"""
conda activate LeafletSC
cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel/scVI_compare

To submit this script:
script=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/LeafletFA_analysis/05_impute_on_test.py

# CPU run (fallback option with high memory)
sbatch --job-name=leaflet_cpu \
       --partition=bigmem \
       --mem=900G \
       --time=5-00:00:00 \
       --output=leaflet_cpu_%j.out \
       --error=leaflet_cpu_%j.err \
       --wrap="python $script"
"""