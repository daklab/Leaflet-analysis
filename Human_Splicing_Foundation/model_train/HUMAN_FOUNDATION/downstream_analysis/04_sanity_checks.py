#!/usr/bin/env python
"""
LeafletFA Advanced Analysis - Mouse Splicing Foundation

Sanity checks for LeafletFA model
1. Extract PHI and PSI to calculate imputed PSI (Rho values)
2. Compare imputed PSI with original PSI values for cells that had ATSE counts > 0
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as stats
import scanpy as sc
from sklearn.decomposition import PCA, NMF
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from mord import OrdinalRidge
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tqdm import tqdm

# Import utility functions - simple direct import
sys.path.append('/gpfs/commons/home/kisaev/Leaflet-analysis/Multi_Species_Splicing_Foundation/shared_utils/')
from utils import *

#############################
### Configuration Section ###
#############################

# Input/Output paths
BASE_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
ATSE_ANNDATA_PATH = f"{BASE_DIR}/MODEL_INPUT/052025/MOUSE_SPLICING_FOUNDATION_Anndata_ATSE_counts_with_waypoints_20250513_073829.h5ad"

######################
### Analysis Functions 
######################

def correlate_imputed_psi_with_observed_psi(splice_adata, PLOTS_DIR, DATA_DIR, num_cells=5000, num_samples=10):
    # Splice_adata already contains PSI_CELLS layer
    PSI_CELLS = splice_adata.layers["PSI_CELLS"] # This is imputed PSI (dense: N_cells x N_junctions)
    
    # Retrieve sparse matrices for junction and cluster counts
    junction_counts_sparse = splice_adata.layers["cell_by_junction_matrix"] 
    cluster_counts_sparse = splice_adata.layers["cell_by_cluster_matrix"]

    # Find (cell_idx, junction_idx) coordinates where cluster_counts_sparse is non-zero.
    # These are the locations where observed PSI is well-defined.
    valid_cell_indices, valid_junction_indices = cluster_counts_sparse.nonzero()
    num_all_valid_pairs = len(valid_cell_indices)

    # Determine the number of pairs to sample in each iteration
    num_pairs_to_sample_this_iteration = num_cells    
    all_correlations = []
    
    # Use tqdm for a progress bar over sampling iterations
    iterator = tqdm(range(num_samples), desc="Correlating PSI samples")

    for _ in iterator:

        # Randomly choose indices from the list of all valid (cell, junction) pairs
        if num_pairs_to_sample_this_iteration < num_all_valid_pairs :
             # Sample without replacement if we're taking a subset of available pairs
             chosen_indices_in_valid_list = np.random.choice(num_all_valid_pairs, size=num_pairs_to_sample_this_iteration, replace=False)
        else: # num_pairs_to_sample_this_iteration == num_all_valid_pairs
             # If sampling size is equal to available (after clamping), take all unique pairs
             chosen_indices_in_valid_list = np.arange(num_all_valid_pairs)

        # Get the actual cell and junction indices for the current sample
        current_sampled_cell_idxs = valid_cell_indices[chosen_indices_in_valid_list]
        current_sampled_junction_idxs = valid_junction_indices[chosen_indices_in_valid_list]

        # Extract corresponding numerators and denominators for observed PSI
        # .A1 converts the sparse matrix slice to a 1D NumPy array
        obs_numerators = junction_counts_sparse[current_sampled_cell_idxs, current_sampled_junction_idxs].A1
        obs_denominators = cluster_counts_sparse[current_sampled_cell_idxs, current_sampled_junction_idxs].A1
        
        # Calculate observed PSI for the sample (denominators are guaranteed non-zero here)
        psi_observed_sample = obs_numerators / obs_denominators
        
        # Extract imputed PSI values for the *same* (cell, junction) pairs
        # PSI_CELLS is a dense NumPy array, so direct advanced indexing is efficient
        psi_imputed_sample = PSI_CELLS[current_sampled_cell_idxs, current_sampled_junction_idxs]
        
        # Calculate Pearson correlation for the current sample
        correlation = np.nan # Default to NaN if correlation cannot be computed
        if len(psi_observed_sample) >= 2 and np.std(psi_observed_sample) > 1e-9 and np.std(psi_imputed_sample) > 1e-9:
            correlation, _ = pearsonr(psi_observed_sample, psi_imputed_sample)
        
        all_correlations.append(correlation)

    # Summarize and print results
    mean_corr = np.nanmean(all_correlations)
    median_corr = np.nanmedian(all_correlations)
    std_corr = np.nanstd(all_correlations)
    num_valid_corrs = np.sum(~np.isnan(all_correlations)) # Count non-NaN correlations

    # Summarize in boxplot to show distribution of correlations
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=all_correlations, showfliers=False)
    plt.title("Distribution of Pearson R values")
    plt.xlabel("Pearson R")
    plt.show()

    # Save boxplot to file
    plt.savefig(os.path.join(PLOTS_DIR, "boxplot_imputed_vs_observed_pearson_r_values.png"))
    
    print(f"Finished {num_samples} sampling iterations ({num_valid_corrs} valid correlations computed).")
    print(f"Mean Pearson R: {mean_corr:.4f}" if not np.isnan(mean_corr) else "Mean Pearson R: N/A")
    print(f"Median Pearson R: {median_corr:.4f}" if not np.isnan(median_corr) else "Median Pearson R: N/A")
    print(f"Std Dev Pearson R: {std_corr:.4f}" if not np.isnan(std_corr) else "Std Dev Pearson R: N/A")
    
    # Save all_correlations to file
    np.save(os.path.join(DATA_DIR, "all_PSI_imputed_vs_observed_correlations.npy"), all_correlations)
    
    # Save num_samples, mean_corr, median_corr, std_corr to file
    with open(os.path.join(DATA_DIR, "PSI_imputed_vs_observed_correlations_summary.txt"), "w") as f:
        f.write(f"num_samples: {num_samples}\n")
        f.write(f"mean_corr: {mean_corr:.4f}\n")
        f.write(f"median_corr: {median_corr:.4f}\n")
        f.write(f"std_corr: {std_corr:.4f}\n")
    # num_samples, mean_corr, median_corr, std_corr
    return all_correlations

###########################
### Main Analysis Script ##
###########################

# Get param_id and MODEL_OUTPUTS_DIR from command line if provided
if len(sys.argv) > 1:
    param_id = sys.argv[1]
    MODEL_OUTPUTS_DIR = sys.argv[2]
    print(f"Using specified param_id: {param_id}")
    print(f"Using specified MODEL_OUTPUTS_DIR: {MODEL_OUTPUTS_DIR}")

def main():
    print("\n========================================")
    print("LeafletFA Model Analysis - Mouse Splicing Foundation")
    print("========================================\n")
    
    ############################
    # 1. Load Data and Model
    ############################
    print("\n>> Loading data and model...")
    
    splice_adata = ad.read_h5ad(ATSE_ANNDATA_PATH)
     
    # Choose model based on param_id or best performance
    model_path = os.path.join(MODEL_OUTPUTS_DIR, f"run_{param_id}", "leafletfa_model.pkl.xz")
    print(f"Using specified model: run_{param_id} with model path {model_path}")
    
    # Create output directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d")
    train_date = MODEL_OUTPUTS_DIR.split("/")[-1]
    OUTPUT_DIR = f"/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/results/{train_date}/param_id_{param_id}"
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
    DATA_DIR = os.path.join(OUTPUT_DIR, "data")
    
    os.makedirs(PLOTS_DIR, exist_ok=True); os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Load the model
    leaflet_model = load_model(model_path)
    print(f"   ✓ Data loaded successfully")
    
    ############################
    # 2. Extract Model Parameters
    ############################

    print("\n>> Extracting model parameters...")
    PHI = leaflet_model["assign_post"]
    K_factors_model = leaflet_model["K"]
    
    # Get imputed PSI values by the model
    splice_adata.obsm["X_PHI"] = PHI
    PSI_CELLS = np.dot(PHI, leaflet_model["psi_learned"])
    splice_adata.layers["PSI_CELLS"] = PSI_CELLS

    print(f"   ✓ Extracted {K_factors_model} factors from the model")

    # Call PSI correlation check
    print("\n>> Checking PSI correlation...")
    correlate_imputed_psi_with_observed_psi(splice_adata, PLOTS_DIR, DATA_DIR, num_cells=20000, num_samples=100)

    print("\n========================================")
    print("LeafletFA Model Analysis Completed.")
    print("========================================\n")

if __name__ == "__main__":
    main()

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025
# sbatch --mem=250G -p dev,cpu --wrap="python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/downstream_analysis/04_sanity_checks.py"