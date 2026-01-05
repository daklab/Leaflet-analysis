#!/usr/bin/env python
"""
LeafletFA Advanced Analysis - Mouse Splicing Foundation

This script performs extended analysis on LeafletFA model results:
1. Regression analysis of factors vs aging, cell type, and tissue
2. Correlation structure analysis across factors
3. Integration with gene expression data (NMF/scVI features)
4. Variance explained by different variables (ANOVA-based analysis)
5. RBP gene expression correlation with splicing factors
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
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# import LeafletFA differential splicing code
# Define module paths
src_path = "/gpfs/commons/home/kisaev/Leaflet-private/src/"

# Add to sys.path if not already present
if src_path not in sys.path:
    sys.path.append(src_path)

# Import custom modules
import BetaDirichletFactor.differential_splicing as ds

# Import utility functions - simple direct import
sys.path.append('/gpfs/commons/home/kisaev/Leaflet-analysis/Multi_Species_Splicing_Foundation/shared_utils/')
from utils import *

# Check if using CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###########################
### Main Analysis Script ##
###########################

# Get param_id and MODEL_OUTPUTS_DIR from command line if provided
if len(sys.argv) > 1:

    # ---------- command-line arguments (7 total) -------------------
    # 0: script name
    # 1: param_id
    # 2: MODEL_OUTPUTS_DIR
    # 3: ATSE_ANNDATA_PATH
    # 4: AGING_GENES_PATH
    # 5: RBP_FILE_PATH
    # 6: OUTPUT_DIR
    # 7: cell_type_index  ← optional; None if omitted
    # ---------------------------------------------------------------
    
    param_id          = sys.argv[1]
    MODEL_OUTPUTS_DIR = sys.argv[2]
    ATSE_ANNDATA_PATH = sys.argv[3]
    AGING_GENES_PATH  = sys.argv[4]
    RBP_FILE_PATH     = sys.argv[5]
    OUTPUT_DIR        = sys.argv[6]
    cell_type_index   = int(sys.argv[7]) if len(sys.argv) > 7 else None
    
    print(f"param_id        : {param_id}")
    print(f"MODEL_OUTPUTS_DIR: {MODEL_OUTPUTS_DIR}")
    print(f"ATSE_ANNDATA_PATH: {ATSE_ANNDATA_PATH}")
    print(f"AGING_GENES_PATH : {AGING_GENES_PATH}")
    print(f"RBP_FILE_PATH    : {RBP_FILE_PATH}")
    print(f"OUTPUT_DIR       : {OUTPUT_DIR}")
    print(f"cell_type_index  : {cell_type_index}")


def main():
    print("\n========================================")
    print("LeafletFA Model Analysis 03...")
    print("========================================\n")
    
    ############################
    # 1. Load Data and Model
    ############################
    print("\n>> Loading data and model...")
    
    final_cells = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/062025/filtered_cell_ids.txt"
    with open(final_cells, "r") as f:
        final_cells = f.read().splitlines()
    
    splice_adata = ad.read_h5ad(ATSE_ANNDATA_PATH)

    # Only subset by final_cells if in mouse so check if "MOUSE_" is in ATSE_ANNDATA_PATH
    if "MOUSE_" in ATSE_ANNDATA_PATH:
        print("   :gear: Subsetting to final cells (outlier removal)...")
        # Subset both anndatas to only include cells in final_cells
        splice_adata = splice_adata[splice_adata.obs["cell_id"].isin(final_cells)].copy()

    # Fix the sex column in the anndatas
    sex_str = splice_adata.obs["sex"].astype(str)

    # Step 2: Replace "M" → "male", "F" → "female"
    sex_fixed = sex_str.replace({"M": "male", "F": "female"})

    # Step 3: Convert back to categorical (optional)
    splice_adata.obs["sex"] = pd.Categorical(sex_fixed)
    print(splice_adata.obs["sex"].value_counts())

    # Load aging gene lists
    aging_genes_mouse, aging_genes_human = load_aging_genes(AGING_GENES_PATH)
    
    # Load RBP genes
    rbps = load_rbp_genes(RBP_FILE_PATH)
    splice_adata.var["gene_id"] = splice_adata.var["gene_id"].str.split(".").str[0]

    # if "mouse.id" is in splice_adata.obs rename it to donor_id 
    if "mouse.id" in splice_adata.obs.columns:
        print(f"Renaming mouse.id to donor_id in splice_adata.obs")
        splice_adata.obs.rename(columns={"mouse.id": "donor_id"}, inplace=True)
        # Update gene annotations
        splice_adata.var["RBP_gene"] = splice_adata.var["gene_name"].isin(rbps["mouse_gene_name"])
        splice_adata.var["Aging_gene"] = splice_adata.var["gene_name"].isin(aging_genes_mouse)
        aging_genes = aging_genes_mouse
        rbps = rbps["mouse_gene_name"]
    
    else:
        splice_adata.var = add_gene_symbols_to_var(splice_adata.var)
        splice_adata.var["RBP_gene"] = splice_adata.var["gene_name"].isin(rbps["gene_name"]) # when running with Human data... 
        splice_adata.var["Aging_gene"] = splice_adata.var["gene_name"].isin(aging_genes_human)
        
        aging_genes = aging_genes_human
        rbps = rbps["gene_name"]
    
    model_path = os.path.join(MODEL_OUTPUTS_DIR, f"run_{param_id}", "leafletfa_model.pkl.xz")
    print(f"Using specified model: run_{param_id} with model path {model_path}")
    
    # Create output directory
    from datetime import datetime
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
    DATA_DIR = os.path.join(OUTPUT_DIR, "data")
    
    os.makedirs(PLOTS_DIR, exist_ok=True); os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load the model
    leaflet_model = load_model(model_path)

    # Update gene annotations
    splice_adata.var["gene_id"] = splice_adata.var["gene_id"].str.split(".").str[0]
    print(f"   ✓ Data loaded successfully")
    
    ############################
    # 2. Extract Model Parameters
    ############################

    print("\n>> Extracting model parameters...")
    # Extract factor activities and usage
    PHI = leaflet_model["assign_post"]
    # Subset PHI based on cell_id_index in splice_adata.obs
    PHI = PHI[splice_adata.obs.cell_id_index, :]
    # assert shape of PHI matches shape of splice_adata.obs
    assert PHI.shape == (len(splice_adata.obs), leaflet_model["K"]), "PHI shape does not match the number of cells and factors."
    K_factors_model = leaflet_model["K"]
    print(f"   ✓ Extracted {K_factors_model} factors from the model")
    PSI_learned = leaflet_model["psi_learned"]
    print(f" The shape of PSI_learned is {PSI_learned.shape}")
    splice_adata.obsm["X_PHI"] = PHI
    PSI_CELLS = np.dot(PHI, leaflet_model["psi_learned"])
    splice_adata.layers["PSI_CELLS"] = PSI_CELLS
    psi_samples = leaflet_model["psi_samples"]
    phi_samples = leaflet_model["phi_samples"]

    ############################
    # 3. Run analysis functions 
    ############################
    print("\n>> Running differential-splicing analysis...")
    from joblib import Parallel, delayed
    import multiprocessing

    cell_types = np.unique(splice_adata.obs["broad_cell_type"])
    groupby_column  = "broad_cell_type"
    min_effect_size = 0.2
    junction_indices = splice_adata.var["junction_id_index"] #.values[0:100]   # full set

    print(f"Discovered {len(cell_types)} cell types:")
    for i, ct in enumerate(cell_types):
        print(f"  [{i}] {ct}")

    # Declare once at module scope so every worker can see them
    SPLICE_ADATA   = splice_adata
    PSI_SAMPLES    = psi_samples
    PHI_SAMPLES    = phi_samples
    GROUPBY_COLUMN = groupby_column
    MIN_ES         = min_effect_size

    def safe_compute(junction_idx, celltype):
        """Run DS for one (junction, cell-type) pair."""
        try:
            res = ds.compute_differential_splicing_groups(
                SPLICE_ADATA,
                PSI_SAMPLES,
                PHI_SAMPLES,
                junction_idx,
                group_1      = celltype,
                group_2      = None,
                groupby_column = GROUPBY_COLUMN,
                min_effect_size = MIN_ES,
            )
            res["junction_idx"] = junction_idx
            return res
        except Exception as e:
            print(f"Error {junction_idx} for {celltype}: {e}")
            return None

    # -------- loop over cell types (optionally restricted by index) ----
    for idx, ct in enumerate(cell_types):
        if cell_type_index is not None and idx != cell_type_index:
            continue                # skip everything except the requested index

        print(f"\n>> [{idx}] Computing DS for: {ct}")
        
        results = Parallel(
            n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 16)),
            backend = "loky"
        )(
            delayed(safe_compute)(jidx, ct) for jidx in tqdm(
                junction_indices,
                desc = f"Junctions → {ct}",
                leave = False
            )
        )
        
        results = [r for r in results if r is not None]
        if not results:
            print(f"  ⚠ No significant junctions for {ct}")
            continue

        df = pd.DataFrame(results)
        df_sig = ds.compute_junctions_significance_groups(df, min_effect_size)
        df_sig["junction_id_index"] = df_sig["junction_idx"]
    
        out_name = f"differential_splicing_{ct.replace(' ','_').replace('/','_')}.csv"
        df_sig.to_csv(os.path.join(DATA_DIR, out_name), index=False)
        print(f"  ✓ Saved: {out_name}")


    print("\n========================================")
    print("LeafletFA Model Analysis Completed.")
    print("========================================\n")

if __name__ == "__main__":
    main()

