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

# Import utility functions - simple direct import
sys.path.append('/gpfs/commons/home/kisaev/Leaflet-analysis/Multi_Species_Splicing_Foundation/shared_utils/')
from utils import *

# Check if using CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

######################
### Analysis Functions 
######################

def predict_age_with_factors(splice_adata, NMF_ge_matrix, n_nmf_components, PLOTS_DIR, DATA_DIR, tissue_col_name="tissue", n_bootstraps=1000, alpha_val=1.0):
    """
    Predict age using factor activities with Ridge regression and calculate CIs for coefficients via bootstrapping.
    Trains multiple models based on combinations of splicing factors, NMF GE factors, and tissue.
    
    Args:
        splice_adata: AnnData object with X_PHI in obsm and age_numeric in obs.
        NMF_ge_matrix: Cell by NMF gene expression matrix (e.g., ge_adata.obsm["X_nmf_standard_mb"]).
        n_nmf_components: Number of NMF GE components to use.
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
    if NMF_ge_matrix is not None and n_nmf_components > 0 and NMF_ge_matrix.shape[0] == splice_adata.shape[0]:
        if NMF_ge_matrix.shape[1] >= n_nmf_components:
            X_nmf = NMF_ge_matrix[:, :n_nmf_components]
            nmf_feature_names = [f"nmf_ge_factor_{i}" for i in range(X_nmf.shape[1])]
            print(f"  Using {X_nmf.shape[1]} NMF GE factors.")
        else:
            print(f"  Warning: Requested {n_nmf_components} NMF GE components, but only {NMF_ge_matrix.shape[1]} are available. Using all available.")
            X_nmf = NMF_ge_matrix
            nmf_feature_names = [f"nmf_ge_factor_{i}" for i in range(X_nmf.shape[1])]
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
    
    # Build correlation matrix: rows = NMF components, columns = splicing factors
    cor_matrix_nmf = pd.DataFrame(
        index=[f"NMF{i+1}" for i in range(rbp_latent_nmf.shape[1])],
        columns=factor_df.columns
    )
    
    # Calculate correlations
    for i in range(rbp_latent_nmf.shape[1]):
        for factor_col_name in factor_df.columns: # Renamed 'factor' to 'factor_col_name' to avoid conflict
            rho, _ = spearmanr(rbp_latent_nmf[:, i], factor_df[factor_col_name])
            cor_matrix_nmf.loc[f"NMF{i+1}", factor_col_name] = rho # Use factor_col_name here
    
    # Convert to float type
    cor_matrix_rbp_nmf = cor_matrix_nmf.astype(float)
    
    # Save correlation matrix
    if DATA_DIR:
        cor_matrix_rbp_nmf.to_csv(os.path.join(DATA_DIR, "rbp_nmf_factor_correlations.csv"))
    
    # Create clustermap
    if not cor_matrix_rbp_nmf.empty:
        # Reduced width, kept height reasonable for a clustermap
        g = sns.clustermap(
            cor_matrix_rbp_nmf, 
            annot=False, 
            cmap="PRGn", 
            center=0,
            figsize=(5, 5), 
            linewidths=.5,
            linecolor='black',
            xticklabels=True,  # Ensure x tick labels are shown
            yticklabels=True   # Ensure y tick labels are shown
        )
                
        # Set axis labels with increased font size for the heatmap part
        g.ax_heatmap.set_xlabel("Splicing Factors", fontsize=12)
        g.ax_heatmap.set_ylabel("RBP NMF Components", fontsize=12)
        
        # Force all ticks to be shown by explicitly setting them with correct clustering order
        g.ax_heatmap.set_xticks(range(len(cor_matrix_rbp_nmf.columns)))
        g.ax_heatmap.set_xticklabels(cor_matrix_rbp_nmf.columns[g.dendrogram_col.reordered_ind], 
                                     rotation=45, ha='right', fontsize=10)
        g.ax_heatmap.set_yticks(range(len(cor_matrix_rbp_nmf.index)))
        g.ax_heatmap.set_yticklabels(cor_matrix_rbp_nmf.index[g.dendrogram_row.reordered_ind], 
                                     rotation=0, fontsize=10)
        
        # Force all ticks to be visible
        g.ax_heatmap.tick_params(axis='x', which='major', labelsize=8, length=3)
        g.ax_heatmap.tick_params(axis='y', which='major', labelsize=8, length=3)
        
        g.fig.tight_layout(rect=[0, 0, 1, 0.99]) # Adjust rect to make space for suptitle
        if PLOTS_DIR:
            g.savefig(os.path.join(PLOTS_DIR, "rbp_nmf_factor_correlations_clustermap.pdf"), format='pdf', bbox_inches='tight')
        plt.close(g.fig) # Close the figure associated with ClusterGrid
    else:
        print("Correlation matrix is empty. Skipping clustermap.")
        
    return cor_matrix_rbp_nmf, nmf_components, top_rbps_df_nmf, rbp_latent_nmf

def logistic_regression_feature_prediction(splice_adata, gene_expression_adata, feature, 
                                         n_nmf_components=10, # Added n_nmf_components
                                         covariate_column=None, test_size=0.2,
                                         PLOTS_DIR=None, DATA_DIR=None): 
    """
    Train a logistic regression model to predict cell type or other categorical feature
    using LeafletFA factors and optionally gene expression NMF components.
    
    Args:
        splice_adata: AnnData object with splicing data and X_PHI in obsm
        gene_expression_adata: AnnData object with gene expression data (must have 'X_nmf_standard_mb' in obsm)
        feature: Column name in adata.obs to predict
        n_nmf_components: Number of gene expression NMF components to use
        covariate_column: Optional column to use as a covariate (e.g., "tissue")
        test_size: Fraction of data to use for testing
        make_umap: Whether to make a UMAP of the data
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
    X_nmf = gene_expression_adata.obsm['X_nmf_standard_mb'][:, :n_nmf_components]
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
                sns.clustermap(coef_df, cmap="PRGn", center=0, xticklabels=True, yticklabels=True, linecolor='black')
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

def correlate_single_rbp_with_factors(splice_adata, ge_adata, rbp_gene_list, ge_layer_name="predicted_log_norm_tms", PLOTS_DIR=None, DATA_DIR=None):
    """
    Calculate Spearman correlation between individual RBP gene expression and splicing factor activities.

    Args:
        splice_adata: AnnData with X_PHI in obsm.
        ge_adata: AnnData with gene expression, must have ge_layer_name and var['gene_name'].
                  Cells must be aligned with splice_adata.
        rbp_gene_list: A list of RBP gene names (must match ge_adata.var['gene_name']).
        ge_layer_name: Layer in ge_adata to use for RBP expression.
        PLOTS_DIR: Directory to save plots.
        DATA_DIR: Directory to save data files.

    Returns:
        pandas.DataFrame with RBP genes as rows, Factors as columns, and correlation rho as values.
        pandas.DataFrame with RBP genes as rows, Factors as columns, and p-values as values.
    """
    print("Correlating single RBP gene expression with splicing factors...")
    
    # Assert cells are aligned in both adata objects using "cell_id" column in .obs 
    assert np.all(splice_adata.obs["cell_id"].values == ge_adata.obs["cell_id"].values), "Cells in splice_adata and ge_adata are not aligned."

    X_phi = splice_adata.obsm["X_PHI"]
    n_factors = X_phi.shape[1]
    factor_names = [f"Factor{i}" for i in range(n_factors)]

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

    # Initialize matrices to store correlation coefficients and p-values
    corr_matrix = np.zeros((len(valid_rbps), n_factors))
    pval_matrix = np.zeros((len(valid_rbps), n_factors))
    
    # Calculate correlation for each factor with each RBP individually
    for j in range(n_factors):
        factor_activity = X_phi[:, j]
        
        # Calculate correlation for each RBP separately
        for i, rbp_idx in enumerate(range(rbp_expr_matrix.shape[1])):
            rbp_expr = rbp_expr_matrix[:, rbp_idx]
            
            # Calculate correlation between this factor and this RBP
            rho, pval = spearmanr(factor_activity, rbp_expr)
            
            # Store in matrices
            corr_matrix[i, j] = rho
            pval_matrix[i, j] = pval
    
    # Create DataFrames
    corr_df = pd.DataFrame(corr_matrix, index=valid_rbps, columns=factor_names)
    pval_df = pd.DataFrame(pval_matrix, index=valid_rbps, columns=factor_names)

    # Save DataFrames to CSV
    corr_df.to_csv(os.path.join(DATA_DIR, "single_rbp_factor_correlations_rho.csv"))
    pval_df.to_csv(os.path.join(DATA_DIR, "single_rbp_factor_correlations_pval.csv"))
    print(f"  Saved RBP-Factor correlation matrices to {DATA_DIR}")

    # Select top N RBPs by absolute correlation sum for plotting to keep heatmap manageable
    top_n_rbps_plot = min(50, len(valid_rbps))
    if len(valid_rbps) > top_n_rbps_plot:
        # Sum of absolute correlations for each RBP across factors
        rbp_corr_sum_abs = corr_df.abs().sum(axis=1)
        top_rbps_for_plot = rbp_corr_sum_abs.nlargest(top_n_rbps_plot).index
        plot_corr_df = corr_df.loc[top_rbps_for_plot]
        plot_title = f"Top {top_n_rbps_plot} Single RBP-Splicing Factor Correlations (Spearman ρ)"
    else:
        plot_corr_df = corr_df
        plot_title = f"All {len(valid_rbps)} Single RBP-Splicing Factor Correlations (Spearman ρ)"

    plt.figure(figsize=(7,7))
    sns.clustermap(plot_corr_df, cmap="PRGn", center=0, xticklabels=True, yticklabels=True, linecolor='black')
    plt.xlabel("Splicing Factor", fontsize=10)
    plt.ylabel("RBP Gene", fontsize=10)
    plt.xticks(fontsize=7, rotation=45, ha='right')
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "single_rbp_factor_correlations_heatmap.pdf"), format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved RBP-Factor correlation heatmap to {PLOTS_DIR}")
    return corr_df, pval_df
                
def run_variance_explained_analysis(
    splice_adata, 
    ge_adata, 
    sample_id,
    rbp_nmf_latent_features,
    n_rbp_nmf_to_use=5,
    n_aging_nmf_to_use=5,
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

    # --- Pre-computed RBP NMF Components ---
    rbp_nmf_cov_names = []
    if rbp_nmf_latent_features.shape[0] == analysis_df.shape[0]:
        actual_n_rbp_nmf = min(n_rbp_nmf_to_use, rbp_nmf_latent_features.shape[1])
        if actual_n_rbp_nmf > 0:
            for i in range(actual_n_rbp_nmf):
                cov_name = f"RBP_NMF{i+1}"
                analysis_df[cov_name] = rbp_nmf_latent_features[:, i]
                rbp_nmf_cov_names.append(cov_name)
            print(f"  Added {actual_n_rbp_nmf} pre-computed RBP NMF components.")

    # --- Aging Gene NMF from ge_adata ---
    aging_nmf_cov_names = []
    if "Aging_gene" in ge_adata.var.columns and "predicted_log_norm_tms" in ge_adata.layers and n_aging_nmf_to_use > 0:
        aging_mask = ge_adata.var["Aging_gene"].values
        if np.any(aging_mask):
            aging_expr = ge_adata[:, aging_mask].layers["predicted_log_norm_tms"]
            if scipy.sparse.issparse(aging_expr):
                aging_expr = aging_expr.toarray()
            aging_expr = np.clip(aging_expr, 0, None) # NMF requires non-negative
            
            # Filter out genes with all-zero expression after clipping
            non_zero_aging_genes_mask = np.any(aging_expr > 1e-6, axis=0) # Check for > small epsilon
            aging_expr_filtered = aging_expr[:, non_zero_aging_genes_mask]

            if aging_expr_filtered.shape[1] > 0:
                actual_n_aging_nmf = min(n_aging_nmf_to_use, aging_expr_filtered.shape[1], aging_expr_filtered.shape[0])
                if actual_n_aging_nmf > 0:
                    nmf_aging = NMF(n_components=actual_n_aging_nmf, init='nndsvda', random_state=42, max_iter=200)
                    aging_nmf_data = nmf_aging.fit_transform(aging_expr_filtered)
                    for i in range(actual_n_aging_nmf):
                        cov_name = f"Aging_NMF{i+1}"
                        analysis_df[cov_name] = aging_nmf_data[:, i]
                        aging_nmf_cov_names.append(cov_name)
                    print(f"  Added {actual_n_aging_nmf} Aging Gene NMF components.")

    # 2. Define Covariate List for Formula (using the approach that worked before)
    base_formula_covariates = [
        'C(cell_type)', 
        'C(tissue)', 
        'C(sex)', 
        'C(dataset)', 
        'age'
    ]
    
    # Add NMF components as regular column names (no backticks)
    all_covariates = base_formula_covariates + rbp_nmf_cov_names + aging_nmf_cov_names
    
    # Define interaction term
    interaction_terms = ['C(cell_type):age', 'C(tissue):age']

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
        
    # 5. Plot Heatmap    
    min_prop_threshold = 0.00 
    # Map: factor name → label with R²
    factor_r2_map = {
        row['factor']: f"{row['factor']} (R²={row['r2_overall']:.2f})"
        for _, row in r2_df.set_index("factor").loc[anova_prop.index].reset_index().iterrows()
    }
    
    # Ensure plot_data has finite values for masking operations
    finite_anova_prop = anova_prop.fillna(0) # Use a copy filled with 0 for mask calculations
    factors_to_plot_mask = (finite_anova_prop.abs() > min_prop_threshold).any(axis=1)
    covariates_to_plot_mask = (finite_anova_prop.loc[factors_to_plot_mask].abs() > min_prop_threshold).any(axis=0)
        
    plot_data = anova_prop.loc[factors_to_plot_mask, covariates_to_plot_mask]
    plot_data = plot_data.fillna(0) 
        
    height = 5
    width = 6

    # Ensure all data is float for clustermap
    plot_data = plot_data.astype(float)
    # Rename index of plot_data (rows of the clustermap)
    plot_data_annotated = plot_data.rename(index=factor_r2_map)

    # Ensure all data is float
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
        annot_kws={"size": 5},  # Smaller font size for annotations
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
    print("Variance explained analysis complete.")

def calculate_age_acceleration_by_cell_type(splice_adata, ge_adata, n_nmf_components=30, 
                                          PLOTS_DIR=None, DATA_DIR=None):
    """
    Calculate age acceleration for different cell types using splicing factors, 
    gene expression factors, and combined features.
    
    Args:
        splice_adata: AnnData with X_PHI in obsm and age_numeric in obs
        ge_adata: AnnData with X_nmf_standard_mb in obsm  
        n_nmf_components: Number of NMF components to use from gene expression
        PLOTS_DIR: Directory to save plots
        DATA_DIR: Directory to save data files
        
    Returns:
        dict: Age acceleration results for each feature set and cell type
    """
    print("Calculating age acceleration by cell type...")
    
    # Prepare feature sets
    X_splicing = splice_adata.obsm["X_PHI"]
    X_nmf = ge_adata.obsm["X_nmf_standard_mb"][:, :n_nmf_components]
    X_combined = np.hstack([X_splicing, X_nmf])
    
    feature_sets = {
        "splicing_factors": X_splicing,
        "expression_nmf": X_nmf, 
        "combined": X_combined
    }
    
    # Get metadata
    cell_types = splice_adata.obs["broad_cell_type"].unique()
    age_values = splice_adata.obs["age_numeric"].values
    cell_type_values = splice_adata.obs["broad_cell_type"].values
    
    # Store results
    age_acceleration_results = {}
    
    for feature_name, X_features in feature_sets.items():
        print(f"  Processing {feature_name}...")
        
        feature_results = {}
        
        for cell_type in cell_types:
            # Subset to this cell type
            mask = cell_type_values == cell_type
            
            if np.sum(mask) < 20:  # Skip cell types with too few cells
                continue
                
            ct_features = X_features[mask]
            ct_ages = age_values[mask]
            
            # Train age prediction model on this cell type
            try:
                model = Ridge(alpha=1.0)
                model.fit(ct_features, ct_ages)
                
                # Calculate age acceleration
                predicted_ages = model.predict(ct_features)
                age_acceleration = predicted_ages - ct_ages
                
                # Store results
                feature_results[cell_type] = {
                    'mean_age_acceleration': np.mean(age_acceleration),
                    'std_age_acceleration': np.std(age_acceleration),
                    'median_age_acceleration': np.median(age_acceleration),
                    'r2_age_prediction': model.score(ct_features, ct_ages),
                    'n_cells': np.sum(mask),
                    'age_acceleration_values': age_acceleration  # Store individual values
                }
                
            except Exception as e:
                print(f"    Warning: Could not fit model for {cell_type}: {e}")
                continue
        
        age_acceleration_results[feature_name] = feature_results
    
    # Create summary DataFrames and plots
    _plot_age_acceleration_results(age_acceleration_results, PLOTS_DIR, DATA_DIR)
    filtered_results = _plot_age_acceleration_results(
    results=age_acceleration_results, 
    PLOTS_DIR=PLOTS_DIR, 
    DATA_DIR=DATA_DIR,
    adata=splice_adata,  
    min_age_range=2,    # 3+ month span required
    min_cells_per_age=5  # 5+ cells per age required
    )
    return age_acceleration_results, filtered_results

def _plot_age_acceleration_results(results, PLOTS_DIR, DATA_DIR, adata=None, min_age_range=3, min_cells_per_age=5):
    """
    Helper function to plot and save age acceleration results with proper filtering.
    
    Args:
        results: Age acceleration results dictionary
        PLOTS_DIR: Directory for plots
        DATA_DIR: Directory for data
        adata: AnnData object to check age distributions (optional but recommended)
        min_age_range: Minimum age range (months) required for reliable prediction
        min_cells_per_age: Minimum cells per age group required
    """    
    # Create summary DataFrame
    summary_data = []
    for feature_type, cell_type_results in results.items():
        for cell_type, metrics in cell_type_results.items():
            summary_data.append({
                'feature_type': feature_type,
                'cell_type': cell_type,
                'mean_age_acceleration': metrics['mean_age_acceleration'],
                'std_age_acceleration': metrics['std_age_acceleration'], 
                'median_age_acceleration': metrics['median_age_acceleration'],
                'r2_age_prediction': metrics['r2_age_prediction'],
                'n_cells': metrics['n_cells']
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # ---- CRITICAL: Filter out unreliable cell types ----
    
    if adata is not None:
        print(" Analyzing age distribution by cell type...")
        
        # Get age distribution for each cell type
        age_analysis = []
        for cell_type in summary_df['cell_type'].unique():
            cell_mask = adata.obs['broad_cell_type'] == cell_type
            if cell_mask.sum() == 0:
                continue
                
            ages = adata.obs.loc[cell_mask, 'age_numeric']  
            age_range = ages.max() - ages.min()
            n_age_groups = ages.nunique()
            min_cells_in_age = ages.value_counts().min()
            max_cells_in_age = ages.value_counts().max()
            
            age_analysis.append({
                'cell_type': cell_type,
                'total_cells': cell_mask.sum(),
                'age_range_months': age_range,
                'n_age_groups': n_age_groups,
                'min_age': ages.min(),
                'max_age': ages.max(),
                'min_cells_per_age': min_cells_in_age,
                'max_cells_per_age': max_cells_in_age,
                'age_distribution': dict(ages.value_counts().sort_index())
            })
        
        age_df = pd.DataFrame(age_analysis)
        
        # Define reliability criteria (adjusted for 4 total age groups: 2,3,18,24 months)
        reliable_mask = (
            (age_df['age_range_months'] >= min_age_range) & 
            (age_df['n_age_groups'] >= 2) &  # At least 2 different ages (reasonable with only 4 total)
            (age_df['min_cells_per_age'] >= min_cells_per_age)
        )
        
        reliable_cell_types = age_df.loc[reliable_mask, 'cell_type'].tolist()
        unreliable_cell_types = age_df.loc[~reliable_mask, 'cell_type'].tolist()
        
        print(f"✅ Reliable cell types ({len(reliable_cell_types)}): {reliable_cell_types[:5]}...")
        print(f"❌ Unreliable cell types ({len(unreliable_cell_types)}): {unreliable_cell_types[:5]}...")
        
        # Filter summary_df to only reliable cell types
        summary_df_filtered = summary_df[summary_df['cell_type'].isin(reliable_cell_types)].copy()
        summary_df_filtered['reliability'] = 'Reliable'
        
        # Keep unreliable for comparison but mark them
        summary_df_unreliable = summary_df[summary_df['cell_type'].isin(unreliable_cell_types)].copy()
        summary_df_unreliable['reliability'] = 'Unreliable'
        
        # Combine for full dataset
        summary_df_annotated = pd.concat([summary_df_filtered, summary_df_unreliable], ignore_index=True)
        
        # Save age analysis
        if DATA_DIR:
            age_df.to_csv(os.path.join(DATA_DIR, "cell_type_age_analysis.csv"), index=False)
            summary_df_annotated.to_csv(os.path.join(DATA_DIR, "age_acceleration_summary_annotated.csv"), index=False)
    else:
        print("⚠️ No adata provided - cannot filter by age distribution")
        summary_df_filtered = summary_df.copy()
        summary_df_annotated = summary_df.copy()
        summary_df_annotated['reliability'] = 'Unknown'
        reliable_cell_types = summary_df['cell_type'].unique().tolist()
    
    # Save summary
    if DATA_DIR:
        summary_df_filtered.to_csv(os.path.join(DATA_DIR, "age_acceleration_summary_filtered.csv"), index=False)
    
        # ---- Plot 2: Clustermap of mean age acceleration (FILTERED) ----
        
        pivot_mean = summary_df_filtered.pivot(index='cell_type', columns='feature_type', values='mean_age_acceleration')
        
        if not pivot_mean.empty:            
            g = sns.clustermap(
                pivot_mean, 
                annot=False, 
                xticklabels=True, yticklabels=True,
                cmap='PRGn', 
                center=0,
                cbar_kws={'label': 'Mean Age Acceleration', 'orientation': 'horizontal'},
                figsize=(5, 6),
                linewidths=0.5
            )
            
            plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=10)
            plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=10)
                        
            g.savefig(os.path.join(PLOTS_DIR, "age_acceleration_clustermap_filtered.pdf"), 
                     bbox_inches='tight', dpi=300)
            plt.close(g.fig)
        
        # ---- Plot 2: Improved horizontal bar chart with reliability annotation ----
        # Use summary_df_filtered for this plot (filtered)
        pivot_r2 = summary_df_filtered.pivot(index='cell_type', columns='feature_type', values='r2_age_prediction')
        
        # Add reliability information only if reliability column exists
        if 'reliability' in summary_df_filtered.columns:
            reliability_map = summary_df_filtered.drop_duplicates('cell_type').set_index('cell_type')['reliability']
            
            # Sort by the "combined" feature type R² values, then by reliability
            if 'combined' in pivot_r2.columns:
                sort_key = pivot_r2['combined'].fillna(0)
            else:
                sort_key = pivot_r2.mean(axis=1, skipna=True)
            
            # Secondary sort by reliability (reliable first)
            sort_df = pd.DataFrame({'r2': sort_key, 'reliability': reliability_map}).fillna({'reliability': 'Unknown'})
            sort_df['reliable_first'] = (sort_df['reliability'] == 'Reliable').astype(int)
            sort_df = sort_df.sort_values(['reliable_first', 'r2'], ascending=[False, True])
            
            pivot_r2_sorted = pivot_r2.loc[sort_df.index]
        else:
            # Simple sort by combined R² if no reliability info available
            if 'combined' in pivot_r2.columns:
                sort_key = pivot_r2['combined'].fillna(0)
            else:
                sort_key = pivot_r2.mean(axis=1, skipna=True)
            
            pivot_r2_sorted = pivot_r2.loc[sort_key.sort_values(ascending=True).index]
        
        # Create the plot with reliability colors
        fig, ax = plt.subplots(figsize=(5, 7))
        
        # Plot bars
        pivot_r2_sorted.plot(
            kind='barh',
            width=0.8,
            color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(pivot_r2_sorted.columns)],
            edgecolor='black',
            linewidth=0.5,
            ax=ax
        )
        
        ax.set_xlabel('Age R² Score', fontsize=10, fontweight='bold')
        ax.set_ylabel('Cell Type', fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=10)
        ax.legend(title='Predictor', fontsize=10, title_fontsize=11)
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "age_prediction_r2_by_celltype_annotated.pdf"), 
                   bbox_inches='tight', dpi=300)
        plt.close()
            
    return summary_df_filtered if adata is not None else summary_df


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

#    # Add value labels
#    for container in ax.containers:
#        ax.bar_label(container, fmt="%.2f", label_type='edge', fontsize=8, padding=2)

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
    print("LeafletFA Model Analysis - Mouse Splicing Foundation")
    print("========================================\n")
    
    ############################
    # 1. Load Data and Model
    ############################
    print("\n>> Loading data and model...")
    
    splice_adata = ad.read_h5ad(ATSE_ANNDATA_PATH)
    ge_adata = ad.read_h5ad(GE_ANNDATA_scVI_PATH)
    ge_adata_nmf = ad.read_h5ad(GE_ANNDATA_NMF_PATH)

    # Load aging gene lists
    aging_genes_mouse, aging_genes_human = load_aging_genes(AGING_GENES_PATH)
    
    # Load RBP genes
    rbps = load_rbp_genes(RBP_FILE_PATH)

    # if "mouse.id" is in splice_adata.obs rename it to donor_id 
    if "mouse.id" in splice_adata.obs.columns:
        print(f"Renaming mouse.id to donor_id in splice_adata.obs")
        splice_adata.obs.rename(columns={"mouse.id": "donor_id"}, inplace=True)
        # Update gene annotations
        splice_adata.var["RBP_gene"] = splice_adata.var["gene_name"].isin(rbps["mouse_gene_name"])
        splice_adata.var["Aging_gene"] = splice_adata.var["gene_name"].isin(aging_genes_mouse)
        ge_adata.var["RBP_gene"] = ge_adata.var["gene_name"].isin(rbps["mouse_gene_name"])
        ge_adata.var["Aging_gene"] = ge_adata.var["gene_name"].isin(aging_genes_mouse)
    
    else:
        splice_adata.var["RBP_gene"] = splice_adata.var["gene_name"].isin(rbps["gene_name"]) # when running with Human data... 
        splice_adata.var["Aging_gene"] = splice_adata.var["gene_name"].isin(aging_genes_human)
        ge_adata.var["RBP_gene"] = ge_adata.var["gene_name"].isin(rbps["gene_name"])
        ge_adata.var["Aging_gene"] = ge_adata.var["gene_name"].isin(aging_genes_human)
    
    splice_adata.var["gene_id"] = splice_adata.var["gene_id"].str.split(".").str[0]

    assert np.all(ge_adata.obs_names == ge_adata_nmf.obs_names), "Cell IDs in ge_adata and ge_adata_nmf do not match or are not in the same order."
    assert np.all(ge_adata.var_names == ge_adata_nmf.var_names), "Gene names in ge_adata and ge_adata_nmf do not match or are not in the same order."
    ge_adata.obsm["X_nmf_standard_mb"] = ge_adata_nmf.obsm["X_nmf_standard_mb"]
    ge_adata.varm["nmf_standard_mb_components"] = ge_adata_nmf.varm["nmf_standard_mb_components"]
    
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

    PHI = leaflet_model["assign_post"]
    K_factors_model = leaflet_model["K"]
    print(f"  ✓ Extracted {K_factors_model} factors from the model")

    splice_adata.obsm["X_PHI"] = PHI
    if "psi_learned" in leaflet_model and 'PSI_CELLS' not in splice_adata.layers:
        PSI_CELLS = np.dot(PHI, leaflet_model["psi_learned"])
        splice_adata.layers["PSI_CELLS"] = PSI_CELLS

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
        splice_adata, ge_adata, ge_layer_name="predicted_log_norm_tms", PLOTS_DIR=PLOTS_DIR, DATA_DIR=DATA_DIR
    )

    # Run multi-class logistic regression for cell type prediction with gene expression NMF components as features as well 
    print("   :gear: Running logistic regression for cell type prediction...")
    n_nmf_main = ge_adata.obsm["X_nmf_standard_mb"].shape[1] if "X_nmf_standard_mb" in ge_adata.obsm else 0
    target_feature_logreg = "broad_cell_type" 
    logistic_regression_feature_prediction(
        splice_adata, ge_adata, feature=target_feature_logreg, 
        n_nmf_components=min(30, n_nmf_main) if n_nmf_main > 0 else 0, 
        covariate_column="tissue", test_size=0.2, PLOTS_DIR=PLOTS_DIR, DATA_DIR=DATA_DIR)

    print("   :gear: Calculating age acceleration by cell type...")
    age_accel_results, filtered_results = calculate_age_acceleration_by_cell_type(
        splice_adata, 
        ge_adata, 
        n_nmf_components=min(30, n_nmf_main) if n_nmf_main > 0 else 0,
        PLOTS_DIR=PLOTS_DIR, 
        DATA_DIR=DATA_DIR
    )

    # ----------------------------------------------------------
    # Variance Explained Analysis
    print("   :gear: Running variance explained analysis...")
    run_variance_explained_analysis(
        splice_adata=splice_adata, 
        ge_adata=ge_adata, 
        sample_id = "dataset",
        rbp_nmf_latent_features=rbp_latent_nmf_features, # Pass the RBP NMF features
        n_rbp_nmf_to_use=10,          # Number of RBP NMF components to use
        n_aging_nmf_to_use=10,        # Number of NMF components for aging genes
        PLOTS_DIR=PLOTS_DIR, 
        DATA_DIR=DATA_DIR
    )

    # Look at individual RBP-Factor correlations
    print("   :gear: Correlating single RBP gene expression with splicing factors...")
    list_of_rbps_for_corr = ge_adata.var_names[ge_adata.var["RBP_gene"]].tolist()
    print(f"   :gear: Correlating single RBP gene expression with splicing factors for {len(list_of_rbps_for_corr)} RBPs")
    correlate_single_rbp_with_factors(
        splice_adata, ge_adata, rbp_gene_list=list_of_rbps_for_corr, 
        ge_layer_name="predicted_log_norm_tms", PLOTS_DIR=PLOTS_DIR, DATA_DIR=DATA_DIR)

    # Predict age from factor activities with/without covariate
    print("   :gear: Predicting age from factor activities...")
    
    # Use NMF as the number of NMF components for gene expression
    predict_age_with_factors(
        splice_adata, 
        NMF_ge_matrix=ge_adata.obsm["X_nmf_standard_mb"],
        n_nmf_components=min(10, n_nmf_main) if n_nmf_main > 0 else 0,
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