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
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy.ma as ma
from matplotlib.colors import ListedColormap
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
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
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, classification_report, confusion_matrix
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
from statsmodels.stats.multitest import multipletests

# Import utility functions - simple direct import
sys.path.append('/gpfs/commons/home/kisaev/Leaflet-analysis/Multi_Species_Splicing_Foundation/shared_utils/')
from utils import *

# Check if using CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

######################
### Analysis Functions 
######################

def predict_age_with_factors_scvi(splice_adata, ge_scvi_matrix, PLOTS_DIR, DATA_DIR, tissue_col_name="tissue", n_bootstraps=1000, alpha_val=1.0):
    """
    Predicts age using splicing factors, scVI latent gene expression, and tissue metadata.
    Coefficients and 95% CIs are computed by bootstrapping Ridge regression.

    Args:
        splice_adata: AnnData with obs['age_numeric'] and obsm['X_PHI'] (splicing factors).
        ge_scvi_matrix: np.ndarray of shape (n_cells, n_scvi_latents), e.g., ge_adata.obsm["X_scVI_linear"].
        PLOTS_DIR: Output path for plots.
        DATA_DIR: Output path for data.
        tissue_col_name: Column name in .obs to use as tissue covariate (optional).
        n_bootstraps: Number of bootstrap iterations.
        alpha_val: Regularization strength for Ridge.

    Returns:
        dict: model results keyed by model name.
    """
    print("🔧 Predicting age using splicing, scVI gene expression, and tissue metadata...")

    all_results = {}

    # --- 1. Load inputs ---
    X_splicing = splice_adata.obsm["X_PHI"]
    y_age = splice_adata.obs["age_numeric"].values
    feature_names = {}
    feature_sets = {}

    # Splicing features
    if X_splicing.shape[0] != ge_scvi_matrix.shape[0]:
        raise ValueError("Mismatch in number of cells between splicing and scVI matrices.")
    feature_sets["splicing"] = X_splicing
    feature_names["splicing"] = [f"splice_factor_{i}" for i in range(X_splicing.shape[1])]

    # scVI latent gene expression
    feature_sets["scvi"] = ge_scvi_matrix
    feature_names["scvi"] = [f"scvi_Z_{i}" for i in range(ge_scvi_matrix.shape[1])]
    print(f"  ✅ Using {ge_scvi_matrix.shape[1]} scVI gene expression latent dimensions.")

    # Tissue covariates
    if tissue_col_name and tissue_col_name in splice_adata.obs.columns:
        tissue_data = splice_adata.obs[tissue_col_name]
        if pd.api.types.is_categorical_dtype(tissue_data) or tissue_data.nunique() < 20:
            print(f"  ✅ Treating '{tissue_col_name}' as categorical.")
            tissue_dummies = pd.get_dummies(tissue_data, prefix=tissue_col_name, drop_first=True)
            if tissue_dummies.shape[1] > 0:
                feature_sets["tissue"] = tissue_dummies.values.astype(float)
                feature_names["tissue"] = tissue_dummies.columns.tolist()
                print(f"     Created {len(feature_names['tissue'])} one-hot tissue covariates.")
        else:
            warnings.warn(f"Tissue column '{tissue_col_name}' has too many unique values or is numeric. Skipping.")

    # --- 2. Model configurations ---
    model_configs = [
        {"name": "tissue_only", "keys": ["tissue"]} if "tissue" in feature_sets else None,
        {"name": "tissue_and_splicing", "keys": ["tissue", "splicing"]} if "tissue" in feature_sets else None,
        {"name": "tissue_and_scvi", "keys": ["tissue", "scvi"]} if "tissue" in feature_sets else None,
        {"name": "tissue_splicing_scvi", "keys": ["tissue", "splicing", "scvi"]} if "tissue" in feature_sets else None
    ]

    model_configs = [cfg for cfg in model_configs if cfg is not None]

    # --- 3. Train and evaluate each model ---
    for cfg in model_configs:
        model_name = cfg["name"]
        print(f"\n🚀 Training model: {model_name}")

        X_parts = [feature_sets[k] for k in cfg["keys"]]
        name_parts = [feature_names[k] for k in cfg["keys"]]
        X_all = np.hstack(X_parts)
        feature_labels = sum(name_parts, [])

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_age, test_size=0.2, random_state=42)

        # Ridge model
        model = Ridge(alpha=alpha_val)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print(f"   📈 R² = {r2:.3f} | MSE = {mse:.2f}")

        # --- 4. Bootstrap for CIs ---
        print(f"   🌀 Bootstrapping {n_bootstraps} iterations...")
        boot_coefs = np.zeros((n_bootstraps, X_train.shape[1]))
        for i in tqdm(range(n_bootstraps), desc=f"Boot ({model_name})", leave=False):
            Xb, yb = resample(X_train, y_train, random_state=i)
            boot_model = Ridge(alpha=alpha_val).fit(Xb, yb)
            boot_coefs[i] = boot_model.coef_

        ci_lower = np.percentile(boot_coefs, 2.5, axis=0)
        ci_upper = np.percentile(boot_coefs, 97.5, axis=0)

        coef_df = pd.DataFrame({
            "Feature": feature_labels,
            "Coefficient": model.coef_,
            "CI_Lower": ci_lower,
            "CI_Upper": ci_upper
        }).sort_values("Coefficient")

        # --- 5. Save data ---
        coef_path = os.path.join(DATA_DIR, f"age_coeffs_{model_name}_alpha{alpha_val}.csv")
        coef_df.to_csv(coef_path, index=False)

        metrics_df = pd.DataFrame({
            "model_name": [model_name],
            "r2": [r2],
            "mse": [mse]
        })
        metrics_path = os.path.join(DATA_DIR, f"age_metrics_{model_name}_alpha{alpha_val}.csv")
        metrics_df.to_csv(metrics_path, index=False)

        # --- 6. Plot ---
        if PLOTS_DIR:
            plt.figure(figsize=(6, max(6, len(coef_df) * 0.2)))
            errors = [coef_df["Coefficient"] - coef_df["CI_Lower"],
                      coef_df["CI_Upper"] - coef_df["Coefficient"]]
            plt.errorbar(coef_df["Coefficient"], coef_df["Feature"], xerr=errors, fmt='o', ecolor='black', capsize=3)
            plt.axvline(0, color='red', linestyle='--', lw=0.8)
            plt.xlabel("Coefficient Value")
            plt.ylabel("Feature")
            plt.tight_layout()
            plot_path = os.path.join(PLOTS_DIR, f"age_coeffs_{model_name}_alpha{alpha_val}_CI.pdf")
            plt.savefig(plot_path, format="pdf")
            plt.close()

        all_results[model_name] = {
            "model": model,
            "r2": r2,
            "mse": mse,
            "coef_df": coef_df
        }
    return all_results
                
def run_variance_explained_analysis(
    splice_adata, 
    sample_id,
    PLOTS_DIR=None, 
    DATA_DIR=None
):
    """
    Run variance explained analysis for splicing factors using OLS regression and ANOVA.
    Uses pre-computed RBP NMF features and derives NMF features for aging genes.
    """
    print("Running variance explained analysis...")
        
    # 1. Prepare analysis DataFrame
    # --- Factor activities ---
    if "X_PHI" not in splice_adata.obsm:
        print("  Error: 'X_PHI' not found in splice_adata.obsm. Aborting variance analysis.")
        return
    X_phi = splice_adata.obsm["X_PHI"]
    factor_names = [f"factor_{i}" for i in range(X_phi.shape[1])]
    analysis_df = pd.DataFrame(X_phi, index=splice_adata.obs_names, columns=factor_names)

    # --- Basic Covariates from splice_adata.obs ---
    obs_cols_to_copy = {
        "broad_cell_type": "cell_type", 
        "tissue": "tissue",
        "sex": "sex",
        sample_id: "dataset",
        "age_numeric": "age"
    }
    for obs_col, df_col in obs_cols_to_copy.items():
        analysis_df[df_col] = splice_adata.obs[obs_col].values

    # 2. Define Covariate List for Formula (using the approach that worked before)
    base_formula_covariates = [
        'C(cell_type)', 
        'C(tissue)', 
        'C(sex)', 
        'C(dataset)', 
        'age'
    ]
    
    # Add NMF components as regular column names (no backticks)
    all_covariates = base_formula_covariates
    
    # Define interaction term
    interaction_terms = ['C(cell_type):age', 'C(tissue):age', 'C(sex):age']

    # 3. Run ANOVA for each factor (simplified based on working approach)
    r2_scores_list = []
    anova_results_list = []
    
    for factor_col in tqdm(factor_names, desc="Analyzing factors for variance explained"):
        # Build the formula string - without backticks for any variables
        formula = f"{factor_col} ~ " + " + ".join(all_covariates)
        if interaction_terms:
            formula += " + " + " + ".join(interaction_terms)
            
        # Print formula
        print(f"  Formula: {formula}")

        # Fit the model
        model = smf.ols(formula, data=analysis_df, missing='drop').fit()
        
        if model.nobs < (model.df_model + 2) or model.df_resid <= 0:
            print(f"    Skipping factor {factor_col} due to insufficient observations")
            continue
            
        r2_scores_list.append({'factor': factor_col, 'r2_overall': model.rsquared, 'n_obs': model.nobs})
        
        anova_res = anova_lm(model, typ=2)
        anova_res['factor'] = factor_col
        anova_results_list.append(anova_res.reset_index())

    r2_df = pd.DataFrame(r2_scores_list)
    r2_df.to_csv(os.path.join(DATA_DIR, "variance_explained_r_squared.csv"), index=False)
    print(f"  Saved R-squared values to {DATA_DIR}")
        
    anova_df_combined = pd.concat(anova_results_list)
    anova_df_combined.to_csv(os.path.join(DATA_DIR, "variance_explained_anova_details.csv"), index=False)
    print(f"  Saved detailed ANOVA results to {DATA_DIR}")

    # 4. Process ANOVA results for plotting (proportion of variance)
    anova_df_combined = anova_df_combined.rename(columns={'index': 'covariate_term'}) # 'index' is the default name from reset_index
    anova_df_filtered = anova_df_combined[anova_df_combined['covariate_term'] != 'Residual'].copy()
    anova_df_filtered['sum_sq'] = pd.to_numeric(anova_df_filtered['sum_sq'], errors='coerce')
    anova_df_filtered.dropna(subset=['sum_sq'], inplace=True)

    anova_pivot = anova_df_filtered.pivot_table(
            index='factor', columns='covariate_term', values='sum_sq', aggfunc='sum' )
    anova_pivot = anova_pivot.loc[:, (anova_pivot.sum(axis=0).abs() > 1e-9)] 
        
    total_explained_ss_per_factor = anova_pivot.sum(axis=1)
    anova_prop = anova_pivot.div(total_explained_ss_per_factor.replace(0, np.nan), axis=0) # Replace 0 with NaN to make resulting divisions NaN

    # Save results from variance explained analysis
    anova_prop.to_csv(os.path.join(DATA_DIR, "variance_explained_anova_proportions.csv"))
    print(f"  Saved ANOVA proportions to {DATA_DIR}")
        
    # Map: factor name → label with R²
    factor_r2_map = {
        row['factor']: f"{row['factor']} (R²={row['r2_overall']:.2f})"
        for _, row in r2_df.set_index("factor").loc[anova_prop.index].reset_index().iterrows()
    }
    
    # Ensure plot_data has finite values for masking operations
    plot_data = anova_prop
        
    height = 5
    width = 4

    # Ensure all data is float for clustermap
    plot_data = plot_data.astype(float)
    # Rename index of plot_data (rows of the clustermap)
    plot_data_annotated = plot_data.rename(index=factor_r2_map)
    
    # Round the values for annotation and convert to string
    annot_data = plot_data_annotated.round(2).astype(str)

    # Compute clustermap
    cg = sns.clustermap(
        plot_data_annotated, 
        cmap="PRGn", 
        center=0,             # ← center diverging colormap at zero (white)
        linewidths=0.2,
        linecolor='black',
        annot=annot_data,
        fmt='',  # Values already formatted as strings
        annot_kws={"size": 4, "color": "grey"},  # Smaller font size with grey color for annotations
        figsize=(width, height),
        xticklabels=True,
        yticklabels=True,
        vmin=0,
        vmax=max(1.0, plot_data.max().max()) if plot_data.size > 0 else 1.0
    )

    # Adjust tick labels
    plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(cg.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
    cg.ax_heatmap.tick_params(axis='x', which='major', labelsize=8, length=3)
    cg.ax_heatmap.tick_params(axis='y', which='major', labelsize=8, length=3)
    
    # Axis labels
    cg.ax_heatmap.set_xlabel('Covariate Terms', fontsize=11, fontweight='bold')
    cg.ax_heatmap.set_ylabel('LeafletFA Factors', fontsize=11, fontweight='bold')
    
    # Colorbar formatting
    cbar = cg.ax_heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('')
    
    # Save
    plot_path = os.path.join(PLOTS_DIR, "variance_explained_heatmap.pdf")
    cg.savefig(plot_path, format="pdf", bbox_inches="tight")
    print(f"  Saved variance explained heatmap to {plot_path}")
    plt.close(cg.fig)

    # Annotate each factor 
    # Convert string annotations to float
    annot_data_float = annot_data.astype(float)

    # Step 2: Get top covariates (1 or 2 depending on max contribution)
    factor_labels = []

    for factor, row in annot_data_float.iterrows():
        sorted_covs = row.sort_values(ascending=False)
        top1_val = sorted_covs.iloc[0]

        if top1_val < 0.75:
            top_covs = sorted_covs.head(2)
        else:
            top_covs = sorted_covs.head(1)

        label = ", ".join([f"{term} ({val:.2f})" for term, val in top_covs.items()])

        factor_labels.append({
            "factor": factor,
            "top_covariates": label,
            "top_covariate_name": top_covs.index[0],
            "top_covariate_value": float(top_covs.iloc[0])
        })

    # Step 3: Create DataFrame and extract R²
    factor_label_df = pd.DataFrame(factor_labels)
    factor_label_df[["factor_id", "r2_str"]] = factor_label_df["factor"].str.extract(r"(factor_\d+)\s+\(R²=(.*)\)")
    factor_label_df["r2"] = factor_label_df["r2_str"].astype(float)

    # Step 4: Compute explained variance
    factor_label_df["explained_variance_score"] = (
        factor_label_df["r2"] * factor_label_df["top_covariate_value"]
    )

    # Step 5: Sort and assign ranked labels directly by top_covariate_name
    factor_label_df = factor_label_df.sort_values(
        by=["top_covariate_name", "explained_variance_score"], ascending=[True, False]
    )

    factor_label_df["category_ranked_label"] = (
        factor_label_df["top_covariate_name"] + " #" + 
        (factor_label_df.groupby("top_covariate_name").cumcount() + 1).astype(str)
    )

    # Save factor_label_df to DATA_DIR
    factor_label_df.to_csv(os.path.join(DATA_DIR, "factor_label_df.csv"), index=False)
    print("Variance explained analysis complete.")

###########################
### Main Analysis Script ##
###########################

# Get param_id and MODEL_OUTPUTS_DIR from command line if provided
if len(sys.argv) > 1:
    param_id = sys.argv[1]
    MODEL_OUTPUTS_DIR = sys.argv[2]
    ATSE_ANNDATA_PATH = sys.argv[3]
    GE_ANNDATA_scVI_PATH = sys.argv[4]
    GE_ANNDATA_NMF_PATH = sys.argv[5]
    AGING_GENES_PATH = sys.argv[6]
    RBP_FILE_PATH = sys.argv[7]
    OUTPUT_DIR = sys.argv[8]
    print(f"Using specified param_id: {param_id}")
    print(f"Using specified MODEL_OUTPUTS_DIR: {MODEL_OUTPUTS_DIR}")
    print(f"Using specified ATSE_ANNDATA_PATH: {ATSE_ANNDATA_PATH}")
    print(f"Using specified GE_ANNDATA_scVI_PATH: {GE_ANNDATA_scVI_PATH}")
    print(f"Using specified GE_ANNDATA_NMF_PATH: {GE_ANNDATA_NMF_PATH}")
    print(f"Using specified AGING_GENES_PATH: {AGING_GENES_PATH}")
    print(f"Using specified RBP_FILE_PATH: {RBP_FILE_PATH}")
    print(f"Using output directory: {OUTPUT_DIR}")

def main():
    print("\n========================================")
    print("LeafletFA Model Regression Analysis 02...")
    print("========================================\n")
    
    ############################
    # 1. Load Data and Model
    ############################
    print("\n>> Loading data and model...")
    
    # Load splicing data
    splice_adata = ad.read_h5ad(ATSE_ANNDATA_PATH)
    ge_adata = ad.read_h5ad(GE_ANNDATA_scVI_PATH)
    ge_adata_nmf = ad.read_h5ad(GE_ANNDATA_NMF_PATH)

    # If ge_adata.obs doesn't have cell_id make it from cell_id_clean
    if "cell_id" not in ge_adata.obs.columns:
        ge_adata.obs["cell_id"] = ge_adata.obs["cell_id_clean"]
        ge_adata_nmf.obs["cell_id"] = ge_adata_nmf.obs["cell_id_clean"]
        splice_adata.obs["cell_id"] = splice_adata.obs["cell_id_clean"] 

    assert np.all(ge_adata.obs["cell_id"].values == splice_adata.obs["cell_id"].values), "Cell IDs in ge_adata and splice_adata do not match or are not in the same order."

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
    
    # If ge_adata.var["gene_name"] is not in ge_adata.var_names, then add it
    if "gene_name" not in ge_adata.var.columns:
        ge_adata.var["gene_name"] = ge_adata.var_names

    rbps["mouse_gene_name"] = rbps["mouse_gene_name"].str.upper()
    aging_genes_mouse = [g.upper() for g in aging_genes_mouse]

    # if "mouse.id" is in splice_adata.obs rename it to donor_id 
    if "mouse.id" in splice_adata.obs.columns:
        print(f"Renaming mouse.id to donor_id in splice_adata.obs")
        splice_adata.obs.rename(columns={"mouse.id": "donor_id"}, inplace=True)
        # Update gene annotations
        splice_adata.var["RBP_gene"] = splice_adata.var["gene_name"].isin(rbps["mouse_gene_name"])
        splice_adata.var["Aging_gene"] = splice_adata.var["gene_name"].isin(aging_genes_mouse)
        ge_adata.var["RBP_gene"] = ge_adata.var["gene_name"].isin(rbps["mouse_gene_name"])
        ge_adata.var["Aging_gene"] = ge_adata.var["gene_name"].isin(aging_genes_mouse)
        rbps = rbps["mouse_gene_name"]
    
    else:
        splice_adata.var = add_gene_symbols_to_var(splice_adata.var)
        splice_adata.var["RBP_gene"] = splice_adata.var["gene_name"].isin(rbps["gene_name"]) # when running with Human data... 
        splice_adata.var["Aging_gene"] = splice_adata.var["gene_name"].isin(aging_genes_human)
        
        ge_adata.var["RBP_gene"] = ge_adata.var["gene_name"].isin(rbps["gene_name"])
        ge_adata.var["Aging_gene"] = ge_adata.var["gene_name"].isin(aging_genes_human)
        rbps = rbps["gene_name"]
    
    print(ge_adata.var.RBP_gene.value_counts())

    assert np.all(ge_adata.obs_names == ge_adata_nmf.obs_names), "Cell IDs in ge_adata and ge_adata_nmf do not match or are not in the same order."
    assert np.all(ge_adata.var_names == ge_adata_nmf.var_names), "Gene names in ge_adata and ge_adata_nmf do not match or are not in the same order."
    ge_adata.obsm["X_nmf_standard_mb"] = ge_adata_nmf.obsm["X_nmf_standard_mb"]
    ge_adata.varm["nmf_standard_mb_components"] = ge_adata_nmf.varm["nmf_standard_mb_components"]
    
    # Check if predicted_log_norm_tms not in ge_adata.layers then ge_layer_name="log_norm"
    ge_layer_name = "log_norm"
    
    model_path = os.path.join(MODEL_OUTPUTS_DIR, f"run_{param_id}", "leafletfa_model.pkl.xz")
    print(f"Using specified model: run_{param_id} with model path {model_path}")
    
    # Create output directory
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
    DATA_DIR = os.path.join(OUTPUT_DIR, "data")
    
    os.makedirs(PLOTS_DIR, exist_ok=True); os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load the model
    leaflet_model = load_model(model_path)
    print(f"  ✓ Data loaded successfully")
    
    ############################
    # 2. Extract Model Parameters
    ############################

    print("\n>> Extracting model parameters...")

    # Extract factor activities
    PHI = leaflet_model["assign_post"]
    # Subset PHI based on cell_id_index in splice_adata.obs
    PHI = PHI[splice_adata.obs.cell_id_index, :]
    # assert shape of PHI matches shape of splice_adata.obs
    assert PHI.shape == (len(splice_adata.obs), leaflet_model["K"]), "PHI shape does not match the number of cells and factors."

    K_factors_model = leaflet_model["K"]
    print(f"  ✓ Extracted {K_factors_model} factors from the model")

    splice_adata.obsm["X_PHI"] = PHI
    if "psi_learned" in leaflet_model and 'PSI_CELLS' not in splice_adata.layers:
        PSI_CELLS = np.dot(PHI, leaflet_model["psi_learned"])
        splice_adata.layers["PSI_CELLS"] = PSI_CELLS

    # Add leaflet_model["psi_learned"] to splice_adata.varm 
    psi_learned = leaflet_model["psi_learned"].T
    splice_adata.varm["psi_learned"] = psi_learned

    # Handle age data which is categorical
    print("   :gear: Processing age data...")
    # Create numeric age values (extract the numbers from strings like "18m" for mouse age)
    if splice_adata.obs["age"].astype(str).str.contains("m").any():
        age_numeric = pd.to_numeric(splice_adata.obs["age"].astype(str).str.replace("m", "", regex=False))
        splice_adata.obs["age_numeric"] = age_numeric
        print(f"   ✓ Age range: {age_numeric.min()} - {age_numeric.max()} months")
    else:
        splice_adata.obs["age_numeric"] = pd.to_numeric(splice_adata.obs["age"])
        print(f"   ✓ Age range: {splice_adata.obs['age_numeric'].min()} - {splice_adata.obs['age_numeric'].max()} years")  # human age is in years

    # Split age into two groups, young and old using the median
    median_age = np.median(splice_adata.obs["age_numeric"].unique())
    splice_adata.obs["age_group"] = np.where(
        splice_adata.obs["age_numeric"] < median_age, "young", "old"
    )

    ############################
    # 3. Run analysis functions 
    ############################

    print("\n>> Running analysis functions...")

    # Plot correlation between splicing factors
    print("   :gear: Plotting correlation matrix of PHI...")
    plot_correlation_matrix(PHI, PLOTS_DIR)

    # ----------------------------------------------------------
    # Variance Explained Analysis
    print("   :gear: Running variance explained analysis...")
    run_variance_explained_analysis(
        splice_adata=splice_adata, 
        sample_id = "dataset",
        PLOTS_DIR=PLOTS_DIR, 
        DATA_DIR=DATA_DIR
    )

    # Predict age from factor activities with/without covariate
    print("   :gear: Predicting age from factor activities...")
    
    predict_age_with_factors_scvi(
        splice_adata=splice_adata,
        ge_scvi_matrix=ge_adata.obsm["X_scVI_linear"],
        PLOTS_DIR=PLOTS_DIR,
        DATA_DIR=DATA_DIR,
        tissue_col_name="tissue",  # or another obs column if needed
        n_bootstraps=100,
        alpha_val=1.0
    )

    # Combine splicing and gene expression latent space 
    print(f"Running UMAP on combined latent space...")
    
    # Save smaller file with just X_PHI, cell_id, broad_cell_type, sex, tissue, age_numeric, age_group
    # Copy AnnData
    splice_adata_small = splice_adata.copy()

    # Keep desired .obsm and .varm
    splice_adata_small.obsm = {"X_PHI": splice_adata.obsm["X_PHI"]}
    splice_adata_small.varm = {"psi_learned": splice_adata.varm["psi_learned"]}

    # Keep full .obs and .var
    splice_adata_small.obs = splice_adata.obs.copy()
    splice_adata_small.var = splice_adata.var.copy()

    # Clear unnecessary content
    splice_adata_small.X = None
    splice_adata_small.uns.clear()
    splice_adata_small.layers.clear()
    splice_adata_small.obsp.clear()

    # Save
    output_path = os.path.join(DATA_DIR, "splice_adata_PHI_psi_var_obs.h5ad")
    splice_adata_small.write_h5ad(output_path, compression="gzip")
    print(f"Saved reduced AnnData with .obs, .var, X_PHI, and psi_learned to: {output_path}")

    print("\n========================================")
    print("LeafletFA Model Analysis Completed.")
    print("========================================\n")

if __name__ == "__main__":
    main()

# For sbatch execution:
# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025
# sbatch --mem=250G -p dev,cpu --wrap="python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/downstream_analysis/02_leafletFA_regressions.py run_id_placeholder /path/to/model_outputs_dir"
# Example actual command:
# sbatch --mem=250G -p dev,cpu --wrap="python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/downstream_analysis/02_leafletFA_regressions.py 1 /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel/2025-05-13/"