#!/usr/bin/env python
"""
LeafletFA Utilities - Mouse Splicing Foundation

This module contains utility functions used across multiple analysis scripts:
- Loading models and reference data
- Processing AnnData objects
- Creating visualizations
- Statistical analysis helpers
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
from sklearn.utils import resample
import torch
import gzip
import pickle
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

# Configure plotting defaults
#sns.set_theme(style="whitegrid")
#plt.rc('font', size=12)
#plt.rc('axes', titlesize=14)
#plt.rc('axes', labelsize=12)
#plt.rc('xtick', labelsize=10)
#plt.rc('ytick', labelsize=10)
#plt.rc('legend', fontsize=10)
#plt.rc('figure', titlesize=16)

#############################
### Data Loading Functions ##
#############################

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

def add_gene_symbols_to_var(var_df, species="human"):
    """
    Adds gene_name column to a DataFrame using Ensembl gene_id.

    Parameters
    ----------
    var_df : pd.DataFrame
        DataFrame with a 'gene_id' column (e.g., splice_adata.var).
    species : str
        Species for gene ID mapping (default is "human").

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with a new 'gene_name' column.
    """
    if "gene_id" not in var_df.columns:
        print("WARNING: gene_id column not found. Gene symbols will not be added.")
        return var_df

    try:
        from mygene import MyGeneInfo

        print("Querying MyGeneInfo to map Ensembl gene IDs to gene symbols...")

        # Clean gene_id (remove version suffix)
        clean_ids = var_df["gene_id"].dropna().apply(lambda x: x.split('.')[0])
        unique_gene_ids = clean_ids.unique().tolist()

        if unique_gene_ids:
            mg = MyGeneInfo()
            query_result = mg.querymany(
                unique_gene_ids,
                scopes="ensembl.gene",
                fields="symbol",
                species=species,
                as_dataframe=True,
                df_index=True,
                returnall=False
            )

            # Create mapping from clean gene_id to symbol
            gene_symbol_map = query_result["symbol"].to_dict()

            # Apply mapping
            var_df["gene_name"] = var_df["gene_id"].apply(
                lambda x: gene_symbol_map.get(x.split('.')[0]) if pd.notnull(x) else None
            )

            print(f"Mapped {var_df['gene_name'].notnull().sum()} gene IDs to gene symbols.")

    except ImportError:
        print("WARNING: mygene module not available. Gene symbols will not be added.")
    except Exception as e:
        print(f"WARNING: Error during gene symbol mapping: {e}")

    return var_df

def load_aging_genes(filepath):
    """Load aging-related genes list"""
    # Load global aging genes
    global_aging_genes = pd.read_csv(filepath, sep='\t')
    aging_genes_mouse = global_aging_genes["global_aging_genes"].unique()
    print(f"Loaded {len(aging_genes_mouse)} aging-related genes (mouse)")
    # Convert to uppercase
    aging_genes_human = [gene.upper() for gene in aging_genes_mouse]
    print(f"Loaded {len(aging_genes_human)} aging-related genes (human)")
    return aging_genes_mouse, aging_genes_human

def load_rbp_genes(filepath):
    """Load RNA binding protein gene list"""
    # Load RBP file
    rbps = pd.read_excel(filepath, header=1, index_col=None)
    
    # Rename columns for clarity
    rbps.rename(columns={rbps.columns[0]: 'gene_name', rbps.columns[1]: 'gene_id'}, inplace=True)
    
    # Convert gene names to mouse format (capitalize only first letter)
    rbps['mouse_gene_name'] = rbps['gene_name'].str.capitalize()
    
    print(f"Loaded {len(rbps)} RNA binding proteins")
    return rbps

def process_dataset(splice_adata, ge_adata, leaflet_model, rbps, aging_genes):
    """
    Process datasets and extract model features
    
    Args:
        splice_adata: AnnData object with splicing data
        ge_adata: AnnData object with gene expression data
        leaflet_model: Loaded LeafletFA model
        rbps: DataFrame with RBP gene information
        aging_genes: List of aging-related genes
        
    Returns:
        Tuple of processed AnnData objects and model parameters
    """
    # Extract hyperparameters
    alpha_pi = leaflet_model["alpha_pi"]
    bb_conc = leaflet_model["bb_conc"]
    dir_conc = leaflet_model["dir_conc"]
    
    # Extract factor activities and usage
    PHI = leaflet_model["assign_post"]
    PI = leaflet_model["pi"]
    PSI = leaflet_model["psi_learned"]
    K = leaflet_model["K"]
    
    # Update gene annotations
    splice_adata.var["gene_id"] = splice_adata.var["gene_id"].str.split(".").str[0]
    splice_adata.var["RBP_gene"] = splice_adata.var["gene_name"].isin(rbps["mouse_gene_name"])
    splice_adata.var["Aging_gene"] = splice_adata.var["gene_name"].isin(aging_genes)
    
    # Format PSI as DataFrame for easier processing
    PSI_df = pd.DataFrame(PSI.T)
    PSI_df.columns = [f"Factor_{i}" for i in range(K)]
    PSI_df.index = splice_adata.var.index
    PSI_df["PSI_variance"] = PSI_df.var(axis=1)
    
    # Add imputed junction usage ratios via MODEL 
    PSI_CELLS = np.dot(PHI, PSI)
    splice_adata.layers["PSI_CELLS"] = PSI_CELLS
    
    # Add PHI to adata
    splice_adata.obsm["X_PHI"] = PHI
    
    # Calculate cell perplexity/entropy 
    PHI_safe = np.clip(PHI, 1e-10, 1)  # Prevent log(0) errors
    entropy = -np.sum(PHI_safe * np.log(PHI_safe), axis=1)
    perplexity = np.exp(entropy)
    splice_adata.obs["perplexity"] = perplexity
    splice_adata.obs["entropy"] = entropy
    
    # Create numeric age values (extract the numbers from strings like "18m")
    age_numeric = pd.to_numeric(splice_adata.obs["age"].astype(str).str.replace("m", ""))
    splice_adata.obs["age_numeric"] = age_numeric
    
    # Get model parameters
    model_params = {
        "alpha_pi": alpha_pi,
        "bb_conc": bb_conc,
        "dir_conc": dir_conc,
        "K": K,
        "PHI": PHI,
        "PI": PI,
        "PSI": PSI,
        "PSI_df": PSI_df
    }
    
    print(f"Processed data: {K} factors extracted")
    return splice_adata, ge_adata, model_params

#############################
### Visualization Functions #
#############################

def plot_correlation_matrix(PHI, PLOTS_DIR, fdr_threshold=0.01):
    """Plot simple correlation matrix with significance stars"""
    
    n_factors = PHI.shape[1]
    factor_labels = [f"SP {i+1}" for i in range(n_factors)]
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(PHI.T)
    
    # Compute p-values for all pairs
    p_matrix = np.ones((n_factors, n_factors))
    for i in range(n_factors):
        for j in range(n_factors):
            if i != j:
                _, p = pearsonr(PHI[:, i], PHI[:, j])
                p_matrix[i, j] = p
    
    # FDR correction on off-diagonal elements
    off_diag_mask = ~np.eye(n_factors, dtype=bool)
    p_values = p_matrix[off_diag_mask]
    _, p_adj_flat, _, _ = multipletests(p_values, method="fdr_bh")
    
    # Put corrected p-values back into matrix
    p_adj_matrix = np.ones((n_factors, n_factors))
    p_adj_matrix[off_diag_mask] = p_adj_flat
    
    # Create significance annotations
    annot = np.full((n_factors, n_factors), "", dtype=object)
    for i in range(n_factors):
        for j in range(n_factors):
            if i != j:  # Skip diagonal
                is_significant = (p_adj_matrix[i, j] < fdr_threshold) & (abs(corr_matrix[i, j]) > 0.2)
                if is_significant:
                    annot[i, j] = "*"
    
    # Create DataFrame for clustermap
    df_corr = pd.DataFrame(corr_matrix, index=factor_labels, columns=factor_labels)
    
    # Plot with clustermap
    g = sns.clustermap(
        df_corr,
        cmap="PRGn",
        center=0,
        figsize=(8, 8),
        xticklabels=True,
        yticklabels=True,
        linewidths=0.5,
        cbar_pos=(0.02, 0.83, 0.03, 0.15),
        dendrogram_ratio=0.15
    )
    
    # Set publication-ready font sizes
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=14)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=14)
    
    # Add significance annotations after clustering
    ax = g.ax_heatmap
    # Get the reordered indices from clustering
    row_order = g.dendrogram_row.reordered_ind
    col_order = g.dendrogram_col.reordered_ind
    
    for i, orig_i in enumerate(row_order):
        for j, orig_j in enumerate(col_order):
            if annot[orig_i, orig_j] == "*":
                ax.text(j + 0.5, i + 0.5, "*", ha='center', va='center', 
                       color='black', fontsize=16, weight='bold')
    
    # Save with high DPI for publication
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_pdf = os.path.join(PLOTS_DIR, "correlation_matrix_PHI.pdf")
    g.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.close()
    
    return pd.DataFrame(corr_matrix, index=factor_labels, columns=factor_labels)

def plot_factor_pi_distribution(PI, alpha_pi, PLOTS_DIR):
    """Plot the distribution of factor PI values"""
    PI_df = pd.DataFrame(PI, columns=["PI"])
    PI_df["Factor"] = [f"Factor {i}" for i in range(len(PI))]  # Create Factor X labels
    PI_df = PI_df.sort_values(by="PI", ascending=False)
    
    plt.figure(figsize=(8, 5))  # Less wide, slightly taller
    sns.barplot(
        x="Factor", 
        y="PI", 
        data=PI_df, 
        order=PI_df["Factor"],
        palette="Greys_r"  # Reversed viridis colormap
    )
    
    plt.title(f"Factor PI Probabilities (Sorted) - learned α: {alpha_pi:.4f}", fontsize=14)
    plt.xlabel("Factor", fontsize=14)
    plt.ylabel("π", fontsize=14)  # Use π symbol for y-axis
    plt.axhline(0.01, color="red", linestyle="--", alpha=0.7)
    
    # Adjust tick labels
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=14)
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "factor_pi_distribution.pdf"), format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved PI distribution plot to {PLOTS_DIR}/factor_pi_distribution.pdf")
    return PI_df

def plot_factor_distribution_by_variable(adata, factor_idx, variable_col, continuous=False, 
                                        n_top_groups=8, output_file=None, figsize=(10, 6)):
    """
    Plot factor activity distribution colored by a categorical or continuous variable
    
    Args:
        adata: AnnData object with X_PHI in obsm
        factor_idx: Index of factor to plot
        variable_col: Column name in adata.obs for coloring
        continuous: Whether the variable is continuous or categorical
        n_top_groups: Number of top groups to display for categorical variables
        output_file: Path to save the figure, if None, the plot is displayed
        figsize: Figure size tuple
    """
    # Get factor values
    factor_values = adata.obsm["X_PHI"][:, factor_idx]
    
    # Create plot
    plt.figure(figsize=figsize)
    
    if continuous:
        # For continuous variables, create scatter plot
        variable_values = adata.obs[variable_col].values
        
        # Sort by variable value for better visualization
        sort_idx = np.argsort(variable_values)
        x_sorted = variable_values[sort_idx]
        y_sorted = factor_values[sort_idx]
        
        # Create scatter plot
        sc = plt.scatter(x_sorted, y_sorted, c=x_sorted, cmap="viridis", alpha=0.7, s=30)
        
        # Add colorbar
        cbar = plt.colorbar(sc)
        cbar.set_label(variable_col)
        
        # Add linear regression line
        x_line = np.linspace(min(x_sorted), max(x_sorted), 100)
        m, b = np.polyfit(x_sorted, y_sorted, 1)
        plt.plot(x_line, m*x_line + b, color='red', linestyle='--')
        
        # Calculate correlation
        corr, pval = stats.spearmanr(x_sorted, y_sorted)
        
        # Add correlation annotation
        plt.text(0.05, 0.95, f"Spearman ρ = {corr:.3f} (p = {pval:.3e})", 
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.xlabel(variable_col)
        plt.ylabel(f"Factor {factor_idx} Activity")
        
    else:
        # For categorical variables, create violin plot
        
        # Get value counts and select top groups
        value_counts = adata.obs[variable_col].value_counts()
        if len(value_counts) > n_top_groups:
            top_groups = value_counts.head(n_top_groups).index
            
            # Create a version of the variable with 'Other' for non-top groups
            grouped_variable = adata.obs[variable_col].copy()
            grouped_variable[~grouped_variable.isin(top_groups)] = 'Other'
        else:
            grouped_variable = adata.obs[variable_col]
            top_groups = value_counts.index
        
        # Create DataFrame for seaborn
        plot_df = pd.DataFrame({
            'Factor Activity': factor_values,
            variable_col: grouped_variable
        })
        
        # Order groups by median factor activity
        group_medians = plot_df.groupby(variable_col)['Factor Activity'].median().sort_values(ascending=False)
        ordered_groups = group_medians.index
        
        # Create violin plot
        ax = sns.violinplot(x=variable_col, y='Factor Activity', data=plot_df, 
                         order=ordered_groups, palette='Set2')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Perform ANOVA to test for significant differences
        from scipy.stats import f_oneway
        
        # Get groups for ANOVA
        groups_for_anova = []
        for group in ordered_groups:
            group_values = plot_df[plot_df[variable_col] == group]['Factor Activity']
            if len(group_values) > 1:  # Need at least 2 points for ANOVA
                groups_for_anova.append(group_values)
        
        # Perform ANOVA if we have at least 2 groups
        if len(groups_for_anova) >= 2:
            f_stat, p_val = f_oneway(*groups_for_anova)
            plt.text(0.05, 0.95, f"ANOVA: F = {f_stat:.2f}, p = {p_val:.3e}", 
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title(f"Factor {factor_idx} Activity Distribution by {variable_col}")
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved distribution plot to {output_file}")
    else:
        plt.show()
    
    return plt.gcf()

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

def run_variance_explained_analysis(
    splice_adata, 
    sample_id,
    cell_type_col,
    PLOTS_DIR=None, 
    DATA_DIR=None
):
    """
    Run variance explained analysis for splicing factors using OLS regression and ANOVA.
    Uses pre-computed RBP NMF features and derives NMF features for aging genes.
    """
    print("Running variance explained analysis...")
        
    # 1. Prepare analysis DataFrame
    X_phi = splice_adata.obsm["X_PHI"]
    factor_names = [f"Factor_{i+1}" for i in range(X_phi.shape[1])]
    analysis_df = pd.DataFrame(X_phi, index=splice_adata.obs_names, columns=factor_names)

    # print breakdown by cell type, tissue, sex, and dataset
    print(splice_adata.obs[cell_type_col].value_counts())
    print(splice_adata.obs["tissue"].value_counts())
    print(splice_adata.obs["sex"].value_counts())
    print(splice_adata.obs[sample_id].value_counts())

    # --- Basic Covariates from splice_adata.obs ---
    obs_cols_to_copy = {
        cell_type_col: "cell_type", 
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
    anova_df_combined = pd.concat(anova_results_list)
    
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
    
    # Save add cell_type_col to filename    
    plot_path = os.path.join(PLOTS_DIR, f"variance_explained_heatmap_{cell_type_col}.pdf")
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
    factor_label_df[["factor_id", "r2_str"]] = factor_label_df["factor"].str.extract(r"(Factor_\d+)\s+\(R²=(.*)\)")
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
    return factor_label_df 
