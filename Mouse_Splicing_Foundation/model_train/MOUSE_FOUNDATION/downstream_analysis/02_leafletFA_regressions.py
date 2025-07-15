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

def predict_age_with_factors(splice_adata, NMF_ge_matrix, PLOTS_DIR, DATA_DIR, tissue_col_name="tissue", n_bootstraps=1000, alpha_val=1.0):
    """
    Predict age using factor activities with Ridge regression and calculate CIs for coefficients via bootstrapping.
    Trains multiple models based on combinations of splicing factors, NMF GE factors, and tissue.
    
    Args:
        splice_adata: AnnData object with X_PHI in obsm and age_numeric in obs.
        NMF_ge_matrix: Cell by NMF gene expression matrix (e.g., ge_adata.obsm["X_nmf_standard_mb"]).
        PLOTS_DIR: Directory to save plot files.
        DATA_DIR: Directory to save data files.
        tissue_col_name: Column name in splice_adata.obs for tissue information.
        n_bootstraps: Number of bootstrap samples for CI estimation.
        alpha_val: Alpha value for Ridge regression.

    Returns:
        dict: A dictionary where keys are model configuration names and values are dicts
              containing 'model', 'coefficients_df', 'r2', and 'mse'.
    """
    print("Predicting age with multiple feature sets...")
    
    all_results = {}

    # --- Prepare base feature sets ---
    
    # 1. Splicing Factors
    X_splicing = splice_adata.obsm["X_PHI"]
    splicing_feature_names = [f"splice_factor_{i}" for i in range(X_splicing.shape[1])]

    # 2. NMF Gene Expression Factors
    X_nmf = None
    nmf_feature_names = []
    if NMF_ge_matrix is not None and NMF_ge_matrix.shape[0] == splice_adata.shape[0]:
        X_nmf = NMF_ge_matrix
        nmf_feature_names = [f"nmf_ge_factor_{i}" for i in range(X_nmf.shape[1])]
        print(f"  Using {X_nmf.shape[1]} NMF GE factors.")

    elif NMF_ge_matrix is not None and (NMF_ge_matrix.shape[0] != splice_adata.shape[0]):
         print(f"  Warning: NMF_ge_matrix row count ({NMF_ge_matrix.shape[0]}) does not match splice_adata ({splice_adata.shape[0]}). Skipping NMF features.")
    
    # 3. Tissue Features (Categorical, One-Hot Encoded)
    X_tissue = None
    tissue_feature_names = []
    if tissue_col_name and tissue_col_name in splice_adata.obs.columns:
        tissue_data = splice_adata.obs[tissue_col_name]
        if pd.api.types.is_categorical_dtype(tissue_data) or pd.api.types.is_object_dtype(tissue_data) or tissue_data.nunique() < 20:
            print(f"  Treating '{tissue_col_name}' as categorical and one-hot encoding.")
            try:
                tissue_dummies = pd.get_dummies(tissue_data, prefix=tissue_col_name, drop_first=True)
                if tissue_dummies.shape[1] > 0: # Ensure some columns were created
                    X_tissue = tissue_dummies.values.astype(float)
                    tissue_feature_names = tissue_dummies.columns.tolist()
                    print(f"  Created {len(tissue_feature_names)} tissue features from '{tissue_col_name}'.")
                else:
                    print(f"  Warning: One-hot encoding of '{tissue_col_name}' resulted in zero features (e.g., only one category after drop_first=True). Skipping tissue features.")
            except Exception as e:
                print(f"  Warning: Could not one-hot encode '{tissue_col_name}': {e}. Skipping tissue features.")

        else: # Numerical tissue data - less common, but handle
            print(f"  Treating '{tissue_col_name}' as numerical.")
            # Consider standardizing if it's numerical: StandardScaler().fit_transform(tissue_data.values.reshape(-1, 1))
            X_tissue = tissue_data.values.reshape(-1, 1)
            tissue_feature_names = [tissue_col_name]
    else:
        print(f"  Tissue column '{tissue_col_name}' not found or not specified. Models will not include tissue features explicitly.")

    # Target variable
    y_age = splice_adata.obs["age_numeric"].values

    # --- Define Model Configurations ---
    model_configs = []
    base_features_X = {}
    base_features_names = {}

    if X_splicing is not None and X_splicing.shape[1] > 0:
        base_features_X["splicing"] = X_splicing
        base_features_names["splicing"] = splicing_feature_names
        # model_configs.append({"name": "splicing_factors_only", "keys": ["splicing"]}) # Removed as per user request
        
    if X_nmf is not None and X_nmf.shape[1] > 0:
        base_features_X["nmf"] = X_nmf
        base_features_names["nmf"] = nmf_feature_names
        # model_configs.append({"name": "nmf_ge_factors_only", "keys": ["nmf"]}) # Optional: NMF only

    if X_tissue is not None and X_tissue.shape[1] > 0:
        base_features_X["tissue"] = X_tissue
        base_features_names["tissue"] = tissue_feature_names
        model_configs.append({"name": "tissue_only", "keys": ["tissue"]})

        if "splicing" in base_features_X:
            model_configs.append({"name": "tissue_and_splicing_factors", "keys": ["tissue", "splicing"]})
        if "nmf" in base_features_X:
            model_configs.append({"name": "tissue_and_nmf_ge_factors", "keys": ["tissue", "nmf"]})
        if "splicing" in base_features_X and "nmf" in base_features_X:
            model_configs.append({"name": "tissue_splicing_and_nmf_ge_factors", "keys": ["tissue", "splicing", "nmf"]})
    
    # --- Iterate through configurations, train, and evaluate ---
    for config in model_configs:
        model_name = config["name"]
        print(f"Training model: {model_name}")

        current_X_parts = [base_features_X[key] for key in config["keys"]]
        current_feature_name_parts = [base_features_names[key] for key in config["keys"]]
            
        X_combined = np.hstack(current_X_parts)
        combined_feature_names = sum(current_feature_name_parts, [])

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(X_combined, y_age, test_size=0.2, random_state=42)
        
        # Train Ridge model
        model = Ridge(alpha=alpha_val)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print(f"  {model_name} - Age prediction performance (alpha={alpha_val}): R² = {r2:.3f}, MSE = {mse:.3f}")
        
        coeffs_orig = model.coef_
        
        # Bootstrap for confidence intervals
        print(f"  Performing {n_bootstraps} bootstrap iterations for CI ({model_name})...")
        bootstrapped_coeffs = np.zeros((n_bootstraps, X_train.shape[1]))
        for i in tqdm(range(n_bootstraps), desc=f"Bootstrap ({model_name})", leave=False):
            X_boot, y_boot = resample(X_train, y_train, random_state=i)
            boot_model = Ridge(alpha=alpha_val)
            boot_model.fit(X_boot, y_boot)
            bootstrapped_coeffs[i, :] = boot_model.coef_
        
        ci_lower = np.percentile(bootstrapped_coeffs, 2.5, axis=0)
        ci_upper = np.percentile(bootstrapped_coeffs, 97.5, axis=0)
        
        coef_df = pd.DataFrame({
            "Feature": combined_feature_names,
            "Coefficient": coeffs_orig,
            "CI_Lower": ci_lower,
            "CI_Upper": ci_upper
        })
        coef_df = coef_df.sort_values(by="Coefficient")
        coef_df.to_csv(os.path.join(DATA_DIR, f"age_prediction_coeffs_{model_name}_alpha{alpha_val}.csv"), index=False)
        
        # Save model metrics to file as well 
        model_metrics_df = pd.DataFrame({
            "model_name": [model_name],
            "r2": [r2],
            "mse": [mse]
        })
        model_metrics_df.to_csv(os.path.join(DATA_DIR, f"age_prediction_metrics_{model_name}_alpha{alpha_val}.csv"), index=False)

        # Plot coefficients for the current model
        if PLOTS_DIR and not coef_df.empty:
            plt.figure(figsize=(6, 9))
            errors = [
                coef_df["Coefficient"] - coef_df["CI_Lower"],
                coef_df["CI_Upper"] - coef_df["Coefficient"]
            ]
            
            plt.errorbar(x=coef_df["Coefficient"], y=coef_df["Feature"], 
                         xerr=errors, fmt='o', ecolor='black', capsize=5,
                         markersize=4, 
                         label=f'Coefficient (alpha={alpha_val}) with 95% CI')
            
            plt.axvline(0, color='red', linestyle='--', lw=0.8, label='Zero effect')
            plt.yticks(coef_df["Feature"]) 
            plt.xlabel("Coefficient Value", fontsize=11)
            plt.ylabel("Feature", fontsize=11)
            plt.tick_params(axis='both', which='major', labelsize=8)
            plt.legend(fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"age_prediction_coeffs_{model_name}_alpha{alpha_val}_CIs.pdf"), format='pdf', bbox_inches='tight')
            plt.close()

        all_results[model_name] = {
            "model": model,
            "r2": r2,
            "mse": mse
        }

    # Save all_results as dataframe to file!
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(os.path.join(DATA_DIR, "age_prediction_results.csv"), index=False)
    return all_results

def analyze_gene_expression_nmf_correlations(splice_adata, gene_expression_adata, 
                                             PLOTS_DIR=None, DATA_DIR=None,
                                             obsm_key="X_nmf_standard_mb",
                                             varm_key="nmf_standard_mb_components",
                                             top_k_genes=3,
                                             fdr_thresh=0.05,
                                             rho_thresh=0.2):
    """
    Analyze correlations between gene expression NMF components and splicing factors.

    Args:
        splice_adata: AnnData with splicing data and 'X_PHI' in .obsm
        gene_expression_adata: AnnData with gene expression NMF data
        PLOTS_DIR: directory to save plots
        DATA_DIR: directory to save data
        obsm_key: key in gene_expression_adata.obsm with NMF components (cells × components)
        varm_key: key in gene_expression_adata.varm with NMF loadings (genes × components)
        top_k_genes: number of top genes to show per NMF component
        fdr_thresh: significance threshold for corrected p-values
        rho_thresh: absolute correlation threshold for annotation
    """
    # Ensure alignment
    assert np.all(splice_adata.obs["cell_id"].values == gene_expression_adata.obs["cell_id"].values), \
        "Cell IDs do not match between splicing and gene expression data"

    # Get data
    X_PHI = splice_adata.obsm["X_PHI"]
    X_NMF = gene_expression_adata.obsm[obsm_key]
    nmf_components = gene_expression_adata.varm[varm_key]  # genes × components
    gene_names = gene_expression_adata.var["gene_name"].values

    n_nmf = X_NMF.shape[1]
    n_factors = X_PHI.shape[1]

    # Build factor dataframe
    factor_df = pd.DataFrame(X_PHI, columns=[f"factor{i}" for i in range(n_factors)])
    nmf_names = [f"NMF{i+1}" for i in range(n_nmf)]

    # Get top-k genes per NMF component
    top_genes_dict = {}
    for i, comp in enumerate(nmf_names):
        weights = nmf_components[:, i]
        top_idx = np.argsort(-np.abs(weights))[:top_k_genes]
        top_genes = gene_names[top_idx]
        top_genes_dict[comp] = top_genes.tolist()

    # Compute correlation and p-value matrices
    corr_matrix = np.zeros((n_nmf, n_factors))
    pval_matrix = np.zeros((n_nmf, n_factors))

    for i in range(n_nmf):
        nmf_activity = X_NMF[:, i]
        for j in range(n_factors):
            rho, pval = spearmanr(nmf_activity, X_PHI[:, j])
            corr_matrix[i, j] = rho
            pval_matrix[i, j] = pval

    # FDR correction
    _, pvals_fdr_flat, _, _ = multipletests(pval_matrix.flatten(), method='fdr_bh')
    fdr_matrix = pvals_fdr_flat.reshape(pval_matrix.shape)

    # Annotations for clustermap
    annot_matrix = np.where(
        (fdr_matrix < fdr_thresh) & (np.abs(corr_matrix) > rho_thresh),
        "*",
        ""
    )

    # Build labels with top genes
    nmf_labels = [f"{nmf} ({', '.join(top_genes_dict[nmf])})" for nmf in nmf_names]

    # Convert to DataFrame
    corr_df = pd.DataFrame(corr_matrix, index=nmf_labels, columns=factor_df.columns)
    fdr_df = pd.DataFrame(fdr_matrix, index=nmf_labels, columns=factor_df.columns)

    # Save correlation and FDR matrices
    if DATA_DIR:
        corr_df.to_csv(os.path.join(DATA_DIR, "gene_nmf_factor_correlations.csv"))
        fdr_df.to_csv(os.path.join(DATA_DIR, "gene_nmf_factor_fdr.csv"))

    # Plot
    if PLOTS_DIR:
        g = sns.clustermap(
            corr_df,
            cmap="PRGn",
            center=0,
            linewidths=0.2,
            linecolor="black",
            xticklabels=True,
            yticklabels=True,
            annot=annot_matrix,
            fmt='',
            annot_kws={'size': 6, 'color': 'black'},
            figsize=(6,8)
        )
        g.ax_heatmap.set_xlabel("Splicing Factors", fontsize=12)
        g.ax_heatmap.set_ylabel("Gene NMF Components", fontsize=12)
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, ha="right", fontsize=8)
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
        g.savefig(os.path.join(PLOTS_DIR, "gene_nmf_factor_correlation_clustermap.pdf"), dpi=300, bbox_inches='tight')
        plt.close(g.fig)

    return corr_df, fdr_df, top_genes_dict


def analyze_rbp_correlations(splice_adata, ge_adata, ge_layer_name="predicted_log_norm_tms", PLOTS_DIR=None, DATA_DIR=None):
    """
    Analyze correlations between RBP gene expression and splicing factors
    
    Args:
        splice_adata: AnnData object with splicing data and X_PHI in obsm
        ge_adata: AnnData object with gene expression data
        ge_layer_name: Name of the layer in ge_adata to use for RBP expression
        PLOTS_DIR: Optional directory to save plot files.
        DATA_DIR: Optional directory to save data files.
    """
    print("Analyzing RBP correlations with splicing factors using NMF...")
    
    # Add assert statement to check that the two AnnData objects have the exact same cells in the same order
    assert np.all(splice_adata.obs["cell_id"].values == ge_adata.obs["cell_id"].values)
    
    # Ensure RBP_gene column exists in ge_adata.var
    if "RBP_gene" not in ge_adata.var.columns:
        print("  Error: 'RBP_gene' column not found in ge_adata.var")

    # Get RBP mask
    rbp_mask = ge_adata.var["RBP_gene"].values

    # Get RBP expression from input layer in ge_layer_name 
    rbp_expr = ge_adata.layers[ge_layer_name][:, rbp_mask]
    if scipy.sparse.issparse(rbp_expr):
        rbp_expr = rbp_expr.toarray()

    # Check for and clip negative values
    if np.any(rbp_expr < 0):
        print("  Warning: Negative values found in RBP expression data. Clipping to zero for NMF.")
        rbp_expr = np.clip(rbp_expr, 0, None)
    
    # Filter out genes with all-zero expression after potential clipping
    non_zero_genes_mask = np.any(rbp_expr > 0, axis=0) # NMF requires non-negative, so check for > 0
    rbp_expr = rbp_expr[:, non_zero_genes_mask]
    
    # Get the gene names after filtering
    rbp_gene_names = ge_adata.var["gene_name"].values[rbp_mask][non_zero_genes_mask]
    print(rbp_gene_names)
            
    print(f"  Analyzing {len(rbp_gene_names)} RBP genes with non-zero expression using NMF")
    
    # Perform NMF to get latent representation of RBPs
    n_components_nmf = min(10, rbp_expr.shape[1]) # Ensure n_components is not more than features
    nmf = NMF(n_components=n_components_nmf, init='nndsvda', random_state=42, max_iter=200)
    rbp_latent_nmf = nmf.fit_transform(rbp_expr)
    
    # Create a dictionary to store top RBPs per NMF component
    top_rbps_dict_nmf = {}
    
    # Get NMF components (loadings)
    nmf_components = pd.DataFrame(
        nmf.components_.T,
        index=rbp_gene_names,
        columns=[f"NMF{i+1}" for i in range(nmf.n_components_)]
    )
    
    # Get top RBPs per component
    for comp_col in nmf_components.columns:
        top_rbps = nmf_components[comp_col].abs().sort_values(ascending=False).head(10)
        top_rbps_dict_nmf[comp_col] = top_rbps.index.tolist()
    
    # Convert to a DataFrame
    top_rbps_df_nmf = pd.DataFrame.from_dict(top_rbps_dict_nmf, orient='index')
    top_rbps_df_nmf.columns = [f"TopRBP{i+1}" for i in range(top_rbps_df_nmf.shape[1])]
    
    # Save top RBPs
    if DATA_DIR:
        top_rbps_df_nmf.to_csv(os.path.join(DATA_DIR, "top_rbps_per_nmf_component.csv"))
    
    # Wrap factor activity in a DataFrame
    n_factors = splice_adata.obsm["X_PHI"].shape[1]
    factor_df = pd.DataFrame(
        splice_adata.obsm["X_PHI"],
        index=splice_adata.obs_names,
        columns=[f"factor{i}" for i in range(n_factors)]
    )
    
    # Dimensions
    n_nmf = rbp_latent_nmf.shape[1]
    n_factors = factor_df.shape[1]
    nmf_names = [f"NMF{i+1}" for i in range(n_nmf)]
    factor_names = list(factor_df.columns)

    # Initialize correlation and p-value matrices
    corr_matrix = np.zeros((n_nmf, n_factors))
    pval_matrix = np.zeros((n_nmf, n_factors))

    # Calculate correlations
    for i in range(n_nmf):
        print(f"  Calculating correlations for NMF component {nmf_names[i]}...")
        nmf_activity = rbp_latent_nmf[:, i]
        for j in range(n_factors):
            factor_activity = factor_df.iloc[:, j]
            rho, pval = spearmanr(nmf_activity, factor_activity)
            corr_matrix[i, j] = rho
            pval_matrix[i, j] = pval

    # Build DataFrames
    corr_df = pd.DataFrame(corr_matrix, index=nmf_names, columns=factor_names)
    pval_df = pd.DataFrame(pval_matrix, index=nmf_names, columns=factor_names)

    # FDR correction
    pvals_flat = pval_matrix.flatten()
    _, pvals_fdr_flat, _, _ = multipletests(pvals_flat, method='fdr_bh')
    pvals_fdr_matrix = pvals_fdr_flat.reshape(pval_matrix.shape)
    fdr_df = pd.DataFrame(pvals_fdr_matrix, index=nmf_names, columns=factor_names)

    # Significance: FDR < 0.05 and |rho| > 0.1
    sig_matrix = (fdr_df < 0.05) & (np.abs(corr_df) > 0.2)
    annot_matrix = np.where(sig_matrix, '*', '')

    # Save CSVs
    if DATA_DIR:
        corr_df.to_csv(os.path.join(DATA_DIR, "rbp_nmf_factor_correlations.csv"))
        fdr_df.to_csv(os.path.join(DATA_DIR, "rbp_nmf_factor_pvals_fdr.csv"))
        print(f"  Saved NMF-Factor correlation matrices to {DATA_DIR}")

    # Label rows with top 3 RBPs (if available)
    nmf_labels = []
    for nmf_name in nmf_names:
        if nmf_name in top_rbps_dict_nmf:
            top_rbps = top_rbps_dict_nmf[nmf_name][:3]
            nmf_labels.append(f"{nmf_name}({', '.join(top_rbps)})")
        else:
            print(f"Warning: {nmf_name} not in top_rbps_dict_nmf")
            nmf_labels.append(nmf_name)

    # Plot clustermap with significance stars
    if PLOTS_DIR:
        plt.figure(figsize=(6, 4))
        g = sns.clustermap(
            corr_df,
            cmap="PRGn",
            center=0,
            linewidths=0.2,
            linecolor="black",
            xticklabels=True,
            yticklabels=nmf_labels,
            figsize=(6, 4),
            annot=annot_matrix,
            fmt='',
            annot_kws={'size': 6, 'color': 'black'}
        )
        g.ax_heatmap.set_xlabel("Splicing Factors", fontsize=12)
        g.ax_heatmap.set_ylabel("RBP NMF Components", fontsize=12)
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, ha="right", fontsize=8)
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "rbp_nmf_factor_correlations_clustermap.pdf"),
                    format='pdf', bbox_inches='tight')
        plt.close()
    else:
        print("No PLOTS_DIR set. Skipping heatmap.")
    return corr_df, nmf_components, top_rbps_df_nmf, rbp_latent_nmf

def logistic_regression_feature_prediction(splice_adata, gene_expression_adata, feature, 
                                         covariate_column=None, test_size=0.2,
                                         PLOTS_DIR=None, DATA_DIR=None): 
    """
    Train a logistic regression model to predict cell type or other categorical feature
    using LeafletFA factors and optionally gene expression NMF components.
    
    Args:
        splice_adata: AnnData object with splicing data and X_PHI in obsm
        gene_expression_adata: AnnData object with gene expression data (must have 'X_nmf_standard_mb' in obsm)
        feature: Column name in adata.obs to predict
        covariate_column: Optional column to use as a covariate (e.g., "tissue")
        test_size: Fraction of data to use for testing
        PLOTS_DIR: Directory to save plots.
        DATA_DIR: Directory to save data files.
        
    Returns:
        Tuple of (accuracies, models, label_encoder, coefficients_dfs)
    """
    print(f"Training logistic regression model to predict {feature}...")
    
    # Prepare factor data
    X_phi = splice_adata.obsm["X_PHI"]
    
    # Prepare target variable from aligned splice_adata
    y = splice_adata.obs[feature].values
    
    # Encode target if it's categorical
    le = LabelEncoder()
    if pd.api.types.is_categorical_dtype(splice_adata.obs[feature]) or pd.api.types.is_object_dtype(splice_adata.obs[feature]):
        y_encoded = le.fit_transform(y)
    else: # Assuming numerical target if not categorical/object for logistic regression (might need user to ensure it's appropriate)
        y_encoded = y 
        # For true multiclass logistic regression, LabelEncoder is generally preferred for y
        # If y is already 0, 1, 2..., it might be fine.
        # To be safe, let's still fit_transform if it's not pre-encoded.
        print(f"  Target feature '{feature}' is numeric. Encoding for logistic regression.")
        y_encoded = le.fit_transform(y) # Ensures classes are 0 to n_classes-1

    # Prepare different feature sets
    feature_sets = {
        "factors_only": {"data": X_phi, "names": [f"Factor{i}" for i in range(X_phi.shape[1])]}
    }
    
    # Add gene expression NMF components if available
    X_nmf = gene_expression_adata.obsm['X_nmf_standard_mb']
    nmf_feature_names = [f"NMF{i}" for i in range(X_nmf.shape[1])]
        
    feature_sets["nmf_only"] = {"data": X_nmf, "names": nmf_feature_names}
    feature_sets["factors_and_nmf"] = {
            "data": np.hstack([X_phi, X_nmf]),
            "names": feature_sets["factors_only"]["names"] + nmf_feature_names
        }

    # Add covariate if specified
    if covariate_column and covariate_column in splice_adata.obs.columns:
        print(f"  Including covariate: {covariate_column}")
        cov_data = splice_adata.obs[covariate_column]
        
        if pd.api.types.is_categorical_dtype(cov_data) or pd.api.types.is_object_dtype(cov_data) or cov_data.nunique() < 20:
            print(f"    Treating covariate '{covariate_column}' as categorical and one-hot encoding.")
            cov_dummies = pd.get_dummies(cov_data, prefix=covariate_column, drop_first=True)
            cov_values = cov_dummies.values.astype(float)
            cov_feature_names = cov_dummies.columns.tolist()
        elif pd.api.types.is_numeric_dtype(cov_data):
            print(f"    Treating covariate '{covariate_column}' as numerical. Standardizing.")
            scaler = StandardScaler()
            cov_values = scaler.fit_transform(cov_data.values.reshape(-1, 1))
            cov_feature_names = [covariate_column]
        else:
            print(f"  Warning: Covariate '{covariate_column}' is of an unrecognized type. Skipping covariate.")
            cov_values = None
            cov_feature_names = []

        if cov_values is not None and len(cov_feature_names) > 0:
            # Add to feature sets that include factors
            if "factors_only" in feature_sets:
                base_X = feature_sets["factors_only"]["data"]
                base_names = feature_sets["factors_only"]["names"]
                feature_sets[f"factors_with_cov_{covariate_column}"] = {
                    "data": np.hstack([base_X, cov_values]),
                    "names": base_names + cov_feature_names
                }
            
            if "factors_and_nmf" in feature_sets:
                base_X_fn = feature_sets["factors_and_nmf"]["data"]
                base_names_fn = feature_sets["factors_and_nmf"]["names"]
                feature_sets[f"factors_and_nmf_with_cov_{covariate_column}"] = {
                    "data": np.hstack([base_X_fn, cov_values]),
                    "names": base_names_fn + cov_feature_names
                }
    
    # Train and evaluate models
    accuracies = {}
    models = {}
    coefficients_dfs = {}
    
    for name, H_set_info in feature_sets.items():
        X_current = H_set_info["data"]
        current_feature_names = H_set_info["names"]

        # Split data
        # Ensure y_encoded corresponds to the potentially subsetted/aligned X_current
        if X_current.shape[0] != len(y_encoded):
            print(f"  Error: Mismatch in number of samples for feature set '{name}' ({X_current.shape[0]}) and target ({len(y_encoded)}). Skipping.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(X_current, y_encoded, test_size=test_size, 
                                                            random_state=42, stratify=y_encoded)
        
        # Train model
        model = LogisticRegression(max_iter=500, multi_class='ovr', solver='lbfgs', C=1.0, penalty='l2') # Using 'ovr' for multi-class
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[name] = accuracy
        models[name] = model
        
        print(f"  {name} accuracy for '{feature}': {accuracy:.3f}")
        
        # Extract and save coefficients
        if hasattr(model, "coef_"):
            coefs = model.coef_ # Shape: (n_classes, n_features) for ovr/multinomial, (1, n_features) for binary if n_classes=2 and squeezed
            classes = le.classes_

            # If binary classification and classes are [0, 1], coef_ might be (1, n_features)
            # For ovr with n_classes > 2, coef_ is (n_classes, n_features)
            # If n_classes == 2, le.classes_ are [class0, class1]. coef_ is for class1 vs class0.
            if len(classes) == 2 and coefs.shape[0] == 1:
                 # For binary case, create a frame for the positive class vs negative.
                coef_df = pd.DataFrame(coefs.T, columns=[f"Coef_vs_{classes[0]}"], index=current_feature_names)
                coef_df.index.name = "Feature"
            else: # Multiclass case
                coef_df = pd.DataFrame(coefs.T, columns=classes, index=current_feature_names)
                coef_df.index.name = "Feature"

            coefficients_dfs[name] = coef_df
            
            if DATA_DIR:
                # Save also model metrics 
                model_metrics_df = pd.DataFrame({
                    "model_name": [name],
                    "accuracy": [accuracy]
                })
                model_metrics_df.to_csv(os.path.join(DATA_DIR, f"{feature}_prediction_metrics_{name}.csv"), index=False)
                coef_df.to_csv(os.path.join(DATA_DIR, f"{feature}_prediction_coeffs_{name}.csv"))
            
            # Plot heatmap for multiclass with more than 2 classes and if PLOTS_DIR is provided
            if PLOTS_DIR and coefs.shape[0] > 1 and len(classes) > 2: # Only plot heatmap if truly multiclass and multiple sets of coeffs
                plt.figure(figsize=(7,7))
                sns.clustermap(coef_df, cmap="PRGn", center=0, xticklabels=True, yticklabels=True, linecolor='black', figsize=(7,7))
                plt.xticks(fontsize=8, rotation=45, ha="right")
                plt.yticks(fontsize=8)
                plt.tight_layout()
                plt.savefig(os.path.join(PLOTS_DIR, f"{feature}_prediction_coeffs_heatmap_{name}.pdf"), format='pdf', bbox_inches='tight')
                plt.close()

            elif PLOTS_DIR and len(classes) == 2: # For binary, plot a bar chart of coefficients
                plt.figure(figsize=(10, max(6, len(coef_df) * 0.3)))
                # Select the coefficient for the positive class (assuming le.classes_[1] is the positive class)
                # If coefs.shape[0] == 1, it's already coefficients for class 1 vs class 0.
                # Coef_df will have one column named like "Coef_vs_{classes[0]}" or similar.
                plot_data = coef_df[coef_df.columns[0]].sort_values()
                plot_data.plot(kind='barh')
                plt.axvline(0, color='grey', linestyle='--')
                plt.title(f"Logistic Regression Coefficients ({name})\nfor {feature} (Class: {classes[1]} vs {classes[0]})", fontsize=14)
                plt.xlabel("Coefficient Value", fontsize=12)
                plt.ylabel("Feature", fontsize=12)
                plt.yticks(fontsize=10)
                plt.xticks(fontsize=10)
                plt.tight_layout()
                plt.savefig(os.path.join(PLOTS_DIR, f"{feature}_prediction_coeffs_barchart_{name}.pdf"), format='pdf', bbox_inches='tight')
                plt.close()
    return accuracies, models, le, coefficients_dfs # Return LabelEncoder as le

def correlate_single_rbp_with_factors(splice_adata, ge_adata, rbp_gene_list, ge_layer_name="predicted_log_norm_tms", PLOTS_DIR=None, DATA_DIR=None, top_rbps_df_nmf=None):
    """
    Calculate Spearman correlation between individual RBP gene expression and splicing factor activities.
    If top_rbps_df_nmf is provided, will use those RBPs instead of the full rbp_gene_list.
    Performs FDR correction on p-values and adds significance stars to the clustermap.

    Args:
        splice_adata: AnnData with X_PHI in obsm.
        ge_adata: AnnData with gene expression, must have ge_layer_name and var['gene_name'].
                  Cells must be aligned with splice_adata.
        rbp_gene_list: A list of RBP gene names (must match ge_adata.var['gene_name']).
        ge_layer_name: Layer in ge_adata to use for RBP expression.
        PLOTS_DIR: Directory to save plots.
        DATA_DIR: Directory to save data files.
        top_rbps_df_nmf: Optional DataFrame containing top RBPs from NMF analysis.

    Returns:
        pandas.DataFrame with RBP genes as rows, Factors as columns, and correlation rho as values.
        pandas.DataFrame with RBP genes as rows, Factors as columns, and FDR-adjusted p-values as values.
    """
    print("Correlating single RBP gene expression with splicing factors...")
    
    # Assert cells are aligned in both adata objects using "cell_id" column in .obs 
    assert np.all(splice_adata.obs["cell_id"].values == ge_adata.obs["cell_id"].values), "Cells in splice_adata and ge_adata are not aligned."

    X_phi = splice_adata.obsm["X_PHI"]
    n_factors = X_phi.shape[1]
    factor_names = [f"Factor{i}" for i in range(n_factors)]

    # If top_rbps_df_nmf is provided, use those RBPs instead
    if top_rbps_df_nmf is not None:
        print("  Using top RBPs from NMF analysis...")
        # Extract unique RBP genes from all columns of top_rbps_df_nmf
        unique_rbps = set()
        for col in top_rbps_df_nmf.columns:
            unique_rbps.update(top_rbps_df_nmf[col].dropna().tolist())
        rbp_gene_list = list(unique_rbps)
        print(f"  Found {len(rbp_gene_list)} unique RBPs from NMF analysis")

    valid_rbps = []
    rbp_indices_in_ge = []
    for rbp in rbp_gene_list:
        if rbp in ge_adata.var["gene_name"].values:
            valid_rbps.append(rbp)
            rbp_indices_in_ge.append(ge_adata.var_names[ge_adata.var["gene_name"] == rbp][0])
        else:
            print(f"  Warning: RBP '{rbp}' not found in ge_adata.var['gene_name']. Skipping.")
    
    rbp_expr_matrix = ge_adata[:, rbp_indices_in_ge].layers[ge_layer_name]
    if scipy.sparse.issparse(rbp_expr_matrix):
        rbp_expr_matrix = rbp_expr_matrix.toarray()

    print("  Calculating correlations using vectorized operations...")
    
    # Calculate correlations for all factors and RBPs at once using numpy's vectorized operations
    n_rbps = rbp_expr_matrix.shape[1]
    corr_matrix = np.zeros((n_rbps, n_factors))
    pval_matrix = np.zeros((n_rbps, n_factors))
    
    # For each factor, calculate correlations with all RBPs at once
    for j in range(n_factors):
        print(f"  Calculating correlations for factor {factor_names[j]}...")
        factor_activity = X_phi[:, j]
        # Calculate correlations for all RBPs with this factor
        for i in range(n_rbps):
            rho, pval = spearmanr(factor_activity, rbp_expr_matrix[:, i])
            corr_matrix[i, j] = rho
            pval_matrix[i, j] = pval
    
    # Create DataFrames
    corr_df = pd.DataFrame(corr_matrix, index=valid_rbps, columns=factor_names)

    # Perform FDR correction on p-values
    pvals_flat = pval_matrix.flatten()
    _, pvals_fdr, _, _ = multipletests(pvals_flat, method='fdr_bh')
    pvals_fdr_matrix = pvals_fdr.reshape(pval_matrix.shape)
    fdr_df = pd.DataFrame(pvals_fdr_matrix, index=valid_rbps, columns=factor_names)

    # Create significance matrix (True where both FDR < 0.05 and |cor| > 0.1)
    sig_matrix = (fdr_df < 0.05) & (np.abs(corr_df) > 0.2)

    # Save DataFrames to CSV
    if DATA_DIR:
        corr_df.to_csv(os.path.join(DATA_DIR, "single_rbp_factor_correlations_rho.csv"))
        fdr_df.to_csv(os.path.join(DATA_DIR, "single_rbp_factor_correlations_fdr.csv"))
        print(f"  Saved RBP-Factor correlation matrices to {DATA_DIR}")

    # Plot heatmap with significance stars
    if PLOTS_DIR:
        plt.figure(figsize=(5,6))
        
        # Create annotation matrix with stars for significant correlations
        annot_matrix = np.where(sig_matrix, '*', '')
        
        # Create clustermap
        g = sns.clustermap(
            corr_df, 
            cmap="PRGn", 
            center=0, 
            linewidths=0.2,
            xticklabels=True, 
            yticklabels=True, 
            linecolor='black', 
            figsize=(5,6),
            annot=annot_matrix,
            fmt='',
            annot_kws={'size': 6, 'color': 'black'}
        )

        # Adjust clustermap tick sizes 
        plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=8)
        plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=6)

        plt.tight_layout()
        
        # Add legend for significance
        plt.figtext(0.01, 0.01, '* FDR < 0.05 and |cor| > 0.2', fontsize=5)
        
        plt.savefig(os.path.join(PLOTS_DIR, "single_rbp_factor_correlations_heatmap.pdf"), format='pdf', bbox_inches='tight')
        plt.close()
        print(f"  Saved RBP-Factor correlation heatmap to {PLOTS_DIR}")
    
    return corr_df, fdr_df
                
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

def plot_feature_comparison_celltype_prediction(splice_adata, ge_adata, PLOTS_DIR, min_cells=1000):
    """
    Compare splicing vs expression vs combined features for predicting broad cell types.
    Only includes cell types with at least `min_cells` instances.
    """

    # Filter to only cell types with enough representation
    cell_type_counts = splice_adata.obs["broad_cell_type"].value_counts()
    valid_cell_types = cell_type_counts[cell_type_counts >= min_cells].index.tolist()
    valid_cells = splice_adata.obs["broad_cell_type"].isin(valid_cell_types)

    splice_adata = splice_adata[valid_cells]
    ge_adata = ge_adata[valid_cells]

    # Extract feature matrices
    feature_sets = {
        'Splicing': splice_adata.obsm["X_PHI"],
        'Expression': ge_adata.obsm["X_nmf_standard_mb"][:, :30],
        'Combined': np.hstack([
            splice_adata.obsm["X_PHI"],
            ge_adata.obsm["X_nmf_standard_mb"][:, :30]
        ])
    }

    # Target labels
    labels = splice_adata.obs["broad_cell_type"]
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    all_results = {}
    confusion_matrices = {}

    for feature_name, X in feature_sets.items():
        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.3, stratify=labels, random_state=42
        )

        # Re-encode to match split sets
        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)

        model = LogisticRegression(max_iter=500, multi_class='ovr')
        model.fit(X_train, y_train_enc)
        y_pred_enc = model.predict(X_test)

        # Convert predictions to labels
        y_pred = le.inverse_transform(y_pred_enc)
        y_test_lbl = le.inverse_transform(y_test_enc)

        # Store confusion matrix data
        confusion_matrices[feature_name] = {
            'y_true': y_test_lbl,
            'y_pred': y_pred,
            'labels': le.classes_
        }

        # Compute per-class recall
        report = classification_report(y_test_lbl, y_pred, output_dict=True)

        for cell_type in le.classes_:
            recall = report.get(cell_type, {}).get('recall', 0.0)
            all_results.setdefault(cell_type, {})[feature_name] = recall

    # Convert results to DataFrame
    comparison_df = pd.DataFrame(all_results).T.fillna(0.0)
    comparison_df = comparison_df.sort_values(by='Combined', ascending=True)

    # Plot 1: Horizontal bar chart of recall
    fig, ax = plt.subplots(figsize=(5,7))
    comparison_df.plot.barh(
        ax=ax, width=0.75,
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
        edgecolor='black'
    )

    # Draw a vertical line at y=0.5
    ax.axvline(x=0.5, color='grey', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Prediction Recall', fontsize=14)
    ax.set_ylabel('Cell Type', fontsize=14)
    ax.set_xlim(0, 1.05)
    ax.tick_params(labelsize=10)
    ax.legend(title='Feature Set', fontsize=11, title_fontsize=12)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "celltype_prediction_feature_comparison.pdf"), dpi=300)
    plt.close()

    # Plot 2: Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (feature_name, cm_data) in enumerate(confusion_matrices.items()):
        ax = axes[idx]
        #cm = confusion_matrix(cm_data['y_true'], cm_data['y_pred'], labels=cm_data['labels'])
        cm = confusion_matrix(
            cm_data['y_true'], cm_data['y_pred'], 
            labels=cm_data['labels'], normalize='true'  # Normalize rows
        )

        sns.heatmap(cm * 100, annot=False, cmap='Blues',
            xticklabels=cm_data['labels'], yticklabels=cm_data['labels'], ax=ax,
            cbar_kws={'label': 'Recall (%)'})
        
        ax.set_title(f'{feature_name} Features', fontsize=11)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "celltype_prediction_confusion_matrices.pdf"), dpi=300)
    plt.close()

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

    # Look at RBP-Factor correlations (NMF-based on all RBPs)
    print("   :gear: Analyzing RBP-Factor correlations (NMF-based on all RBPs)...")
    cor_matrix_rbp_nmf, nmf_components_rbp, top_rbps_df_nmf, rbp_latent_nmf_features = analyze_rbp_correlations(
        splice_adata, ge_adata, ge_layer_name=ge_layer_name, PLOTS_DIR=PLOTS_DIR, DATA_DIR=DATA_DIR
    )

    # Get correlation between all gene expression NMF components and splicing factors
    print("   :gear: Analyzing correlation between all gene expression NMF components and splicing factors...")
    corr_df, fdr_df, top_genes_dict = analyze_gene_expression_nmf_correlations(
        splice_adata, ge_adata, PLOTS_DIR=PLOTS_DIR, DATA_DIR=DATA_DIR
    )

    # Run multi-class logistic regression for cell type prediction with gene expression NMF components as features as well 
    print("   :gear: Running logistic regression for cell type prediction...")
    n_nmf_main = ge_adata.obsm["X_nmf_standard_mb"].shape[1] if "X_nmf_standard_mb" in ge_adata.obsm else 0
    target_feature_logreg = "broad_cell_type" 
    logistic_regression_feature_prediction(
        splice_adata, ge_adata, feature=target_feature_logreg, 
        covariate_column="tissue", test_size=0.2, PLOTS_DIR=PLOTS_DIR, DATA_DIR=DATA_DIR)

    # ----------------------------------------------------------
    # Variance Explained Analysis
    print("   :gear: Running variance explained analysis...")
    run_variance_explained_analysis(
        splice_adata=splice_adata, 
        sample_id = "dataset",
        PLOTS_DIR=PLOTS_DIR, 
        DATA_DIR=DATA_DIR
    )

    # Look at individual RBP-Factor correlations
    print("   :gear: Correlating single RBP gene expression with splicing factors...")
    list_of_rbps_for_corr = ge_adata.var.gene_name[ge_adata.var["RBP_gene"]].tolist()
    print(f"   :gear: Correlating single RBP gene expression with splicing factors for {len(list_of_rbps_for_corr)} RBPs")
    correlate_single_rbp_with_factors(
        splice_adata, ge_adata, rbp_gene_list=list_of_rbps_for_corr, 
        ge_layer_name=ge_layer_name, PLOTS_DIR=PLOTS_DIR, DATA_DIR=DATA_DIR, top_rbps_df_nmf=top_rbps_df_nmf)

    # Predict age from factor activities with/without covariate
    print("   :gear: Predicting age from factor activities...")
    
    # Use NMF as the number of NMF components for gene expression
    predict_age_with_factors(
        splice_adata, 
        NMF_ge_matrix=ge_adata.obsm["X_nmf_standard_mb"],
        PLOTS_DIR=PLOTS_DIR, 
        DATA_DIR=DATA_DIR, 
        tissue_col_name="tissue", # Assuming 'tissue' is the column for tissue source
        n_bootstraps=100, # Reduced for faster testing, can be increased
        alpha_val=1.0
    )

    # Combine splicing and gene expression latent space 
    print(f"Running UMAP on combined latent space...")

    # Run some cell type prediction tasks 
    plot_feature_comparison_celltype_prediction(splice_adata, ge_adata, PLOTS_DIR)
    
    # Run UMAP on the combined latent space using NMF components
    # NMF----------------------------------------------------------
    combined = np.concatenate([PHI, ge_adata.obsm["X_nmf_standard_mb"]], axis=1) # cell x (K + n_nmf)
    combined_adata = ad.AnnData(X=combined)
    combined_adata.obs = splice_adata.obs.copy()
    sc.pp.neighbors(combined_adata, use_rep="X")
    sc.tl.umap(combined_adata)
    top_cell_types = combined_adata.obs['broad_cell_type'].value_counts().head(15).index.tolist()
    combined_adata.obs['cell_type_highlighted'] = 'Other'
    combined_adata.obs.loc[combined_adata.obs['broad_cell_type'].isin(top_cell_types), 'cell_type_highlighted'] = \
    combined_adata.obs['broad_cell_type']
    cmap = cm.get_cmap('tab20', len(top_cell_types))
    colors = [cmap(i) for i in range(len(top_cell_types))]
    color_dict = {cell_type: colors[i] for i, cell_type in enumerate(top_cell_types)}
    color_dict['Other'] = [0.9, 0.9, 0.9, 1.0]	  # gray for Other
    plt.figure(figsize=(8, 5))
    sc.pl.umap(
        combined_adata,
        color='cell_type_highlighted',
        palette=color_dict,
        show=False,
        frameon=True,
        legend_fontsize=10,
        legend_loc='right margin'
    )
    plt.title('UMAP by Cell Type (Top 10 Highlighted)')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(PLOTS_DIR, "combined_NMF_GE_AS_umap_cell_type_top10.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

    # scVI----------------------------------------------------------
    combined_adata_scvi = np.concatenate([PHI, ge_adata.obsm["X_scVI_linear"]], axis=1) # cell x (K + n_scvi)
    combined_adata_scvi = ad.AnnData(X=combined_adata_scvi)
    combined_adata_scvi.obs = splice_adata.obs.copy()
    sc.pp.neighbors(combined_adata_scvi, use_rep="X")
    sc.tl.umap(combined_adata_scvi)
    combined_adata_scvi.obs['cell_type_highlighted'] = 'Other'
    combined_adata_scvi.obs.loc[combined_adata_scvi.obs['broad_cell_type'].isin(top_cell_types), 'cell_type_highlighted'] = \
    combined_adata_scvi.obs['broad_cell_type']
    cmap = cm.get_cmap('tab20', len(top_cell_types))
    colors = [cmap(i) for i in range(len(top_cell_types))]
    color_dict = {cell_type: colors[i] for i, cell_type in enumerate(top_cell_types)}
    color_dict['Other'] = [0.9, 0.9, 0.9, 1.0]	  # gray for Other
    plt.figure(figsize=(8, 5))
    sc.pl.umap(
        combined_adata_scvi,
        color='cell_type_highlighted',
        palette=color_dict,
        show=False,
        frameon=True,
        legend_fontsize=10,
        legend_loc='right margin'
    )
    plt.title('UMAP by Cell Type (Top 10 Highlighted)')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(PLOTS_DIR, "combined_scVI_linear_GE_AS_umap_cell_type_top10.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

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