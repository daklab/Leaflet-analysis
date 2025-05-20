#!/usr/bin/env python
"""
LeafletFA Model Analysis - Mouse Splicing Foundation

This script:
1. Loads trained LeafletFA model outputs and associated data
2. Analyzes splicing factor distributions and correlations with metadata
3. Performs differential splicing analysis across cell types and ages
4. Explores relationships between splicing factors and gene expression
5. Generates visualizations and statistical summaries
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
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

# Configure plotting styles
#sns.set_theme()
#sc.set_figure_params(figsize=(7, 7), frameon=True, dpi=80, facecolor='white')

# Import utility functions - simple direct import
sys.path.append('/gpfs/commons/home/kisaev/Leaflet-analysis/Multi_Species_Splicing_Foundation/shared_utils/')
from utils import *

# Check if using CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#############################
### Configuration Section ###
#############################

# Input/Output paths - all these need to be parameters from the command line or it's fine to just edit here with human paths
BASE_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
ATSE_ANNDATA_PATH = f"{BASE_DIR}/MODEL_INPUT/052025/MOUSE_SPLICING_FOUNDATION_Anndata_ATSE_counts_with_waypoints_20250513_073829.h5ad"
GE_ANNDATA_PATH = f"{BASE_DIR}/MODEL_INPUT/052025/aligned_gene_expression_data_20250513_035938.h5ad"
ATSE_FILE_PATH = f"{BASE_DIR}/ATSE_mapper/ATSE_files/MOUSE_FOUNDATION_ATSE_FILE_unanno_also_2025-04-26_19-55-26.txt.gz"

MODEL_OUTPUTS_DIR = f"{BASE_DIR}/Leaflet/leafletFAmodel/2025-05-13/"
GENOME_DB_PATH = "/gpfs/commons/home/kisaev/Leaflet-private/src/clustering/gencodeVM19"

# Reference gene lists
RBP_FILE_PATH = "/gpfs/commons/groups/knowles_lab/Karin/VanNostrand_2020_supptable1_41586_2020_2077_MOESM3_ESM.xlsx"
AGING_GENES_PATH = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/27857814"

######################
### Helper Functions #
######################

def compute_factor_activity_by_group(adata, groupby="broad_cell_type", DATA_DIR=None, PLOTS_DIR=None):
    """
    Compute mean and median factor expression across different cell types
    
    Args:
        adata: AnnData object with X_PHI in obsm
        groupby: Column name to group cells by (default: "broad_cell_type")
        output_dir: Directory to save the data files
        plots_dir: Directory to save the plot files
        
    Returns:
        Tuple of (mean_df, median_df) DataFrames with cell types as rows and factors as columns
    """
    print(f"   :gear: Computing factor expression summary by {groupby}...")
    
    # Extract PHI matrix
    X_PHI = adata.obsm["X_PHI"]
    n_factors = X_PHI.shape[1]
    
    # Create DataFrame with factor values and grouping variable
    factor_cols = [f"factor_{i}" for i in range(n_factors)]
    df = pd.DataFrame(X_PHI, columns=factor_cols)
    df[groupby] = adata.obs[groupby].values
    
    # Compute mean and median by group
    mean_expression = df.groupby(groupby, observed=True)[factor_cols].mean()
    median_expression = df.groupby(groupby, observed=True)[factor_cols].median()
    
    # Add cell count for reference
    cell_counts = df.groupby(groupby, observed=True).size().rename("cell_count")
    mean_expression = mean_expression.join(cell_counts)
    median_expression = median_expression.join(cell_counts)
    
    # Sort by cell count (descending)
    mean_expression = mean_expression.sort_values("cell_count", ascending=False)
    median_expression = median_expression.sort_values("cell_count", ascending=False)
    
    # Save CSV files to DATA_DIR
    mean_filename = os.path.join(DATA_DIR, f"mean_factor_expression_by_{groupby}.csv")
    median_filename = os.path.join(DATA_DIR, f"median_factor_expression_by_{groupby}.csv")
    
    mean_expression.to_csv(mean_filename)
    median_expression.to_csv(median_filename)
    
    print(f"   ✓ Mean factor expression saved to: {mean_filename}")
    print(f"   ✓ Median factor expression saved to: {median_filename}")
    
    # Create clustermap for mean expression
    mean_expr_for_plot = mean_expression.drop(columns=["cell_count"])
    
    plt.figure(figsize=(14, 10))
    sns.clustermap(
        mean_expr_for_plot,
        cmap="viridis",
        figsize=(14, 10),
        xticklabels=True,
        yticklabels=True
    )
    plt.suptitle(f"Mean Factor Expression by {groupby}", y=0.98, fontsize=14)
    plt.savefig(os.path.join(PLOTS_DIR, f"mean_factor_expression_by_{groupby}_clustermap.png"), dpi=300)
    plt.close()
    
    print(f"   ✓ Clustermap saved to: {os.path.join(PLOTS_DIR, f'mean_factor_expression_by_{groupby}_clustermap.png')}")
    return mean_expression, median_expression

def compute_variance_components(adata, groupby="broad_cell_type", pi_values=None, DATA_DIR=None, PLOTS_DIR=None):
    """
    Compute variance components for factors (between/within group variance)
    
    This analysis answers: How much of Factor Activity Variations Among cells is 
    explained by cell type, tissue, dataset, etc.?
    
    Args:
        adata: AnnData object with X_PHI in obsm
        groupby: Column name to group cells by (e.g., "broad_cell_type")
        pi_values: Array of PI values from the LeafletFA model
        DATA_DIR: Directory to save the output data
        PLOTS_DIR: Directory to save the output plots

    Returns:
        DataFrame with variance components for each factor
    """
    print(f"   :gear: Computing variance components explained by {groupby}...")
    
    # Extract PHI matrix
    X_PHI = adata.obsm["X_PHI"]
    n_factors = X_PHI.shape[1]
    
    # Use provided PI values
    PI = pi_values
    
    # Initialize output
    explained_variances = []
    
    # Get the group labels, ensure they're not categorical
    if pd.api.types.is_categorical_dtype(adata.obs[groupby]):
        groups = adata.obs[groupby].astype(str).values
    else:
        groups = adata.obs[groupby].values
    
    # Process each factor using one-way ANOVA decomposition
    for factor_idx in range(n_factors):
        # Get factor values
        factor_values = X_PHI[:, factor_idx]
        
        # Calculate total variance
        total_variance = np.var(factor_values, ddof=1)
        
        # Create a DataFrame for ANOVA
        anova_df = pd.DataFrame({
            'factor_values': factor_values,
            'group': groups
        })
        
        # Calculate group means and counts
        group_stats = anova_df.groupby('group')['factor_values'].agg(['mean', 'count'])
        
        # Overall mean
        overall_mean = np.mean(factor_values)
        
        # Calculate between-group sum of squares
        # Formula: Σ n_i * (mean_i - overall_mean)²
        between_ss = np.sum(group_stats['count'] * (group_stats['mean'] - overall_mean)**2)
        
        # Degrees of freedom
        n_groups = len(group_stats)
        n_samples = len(factor_values)
        df_between = n_groups - 1
        df_within = n_samples - n_groups
        
        # Between-group variance
        between_variance = between_ss / df_between if df_between > 0 else 0
        
        # Calculate within-group sum of squares
        # For each group, calculate: Σ (x - group_mean)²
        within_ss = 0
        for group_name, group_data in anova_df.groupby('group'):
            group_mean = group_stats.loc[group_name, 'mean']
            within_ss += np.sum((group_data['factor_values'] - group_mean)**2)
        
        # Within-group variance
        within_variance = within_ss / df_within if df_within > 0 else 0
        
        # Calculate proportion of variance explained (R²)
        # This is equivalent to: 1 - (within_ss / total_ss)
        total_ss = between_ss + within_ss
        variance_explained = between_ss / total_ss if total_ss > 0 else 0
        
        # Store results
        explained_variances.append({
            "Factor": f"factor_{factor_idx}",
            "Factor_Index": factor_idx,
            "PI": PI[factor_idx],
            "Explained_Variance": variance_explained,
            "Between_Variance": between_variance,
            "Within_Variance": within_variance,
            "Total_Variance": total_variance,
            "Group_Variable": groupby,
            "N_Groups": n_groups,
            "N_Samples": n_samples
        })
    
    # Convert to DataFrame
    anova_df = pd.DataFrame(explained_variances)
    data_file = os.path.join(DATA_DIR, f"variance_components_by_{groupby}.csv")
    anova_df.to_csv(data_file, index=False)

    # Create plots
    print(f"   :gear: Creating variance component visualizations for {groupby}...")
        
    # 1. Total Variance vs Explained Variance, colored by PI
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        anova_df["Total_Variance"], 
        anova_df["Explained_Variance"],
        c=anova_df["PI"],
        cmap="viridis",
        alpha=0.7,
        s=100 * anova_df["PI"] / anova_df["PI"].max(),  # Size points by PI
        edgecolors='k'
    )
        
    # Add factor labels to points
    for i, row in anova_df.iterrows():
        plt.annotate(
            f"{row['Factor_Index']}",
            (row["Total_Variance"], row["Explained_Variance"]),
            xytext=(5, 0),
            textcoords='offset points',
            fontsize=9,
            weight='bold'
        )
        
    plt.colorbar(scatter, label="Factor PI (Global Assignment Probability)")
    plt.xlabel("Total Variance")
    plt.ylabel(f"Variance Explained by {groupby}")
    plt.title(f"Factor Variance Explained by {groupby} vs Total Variance")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    variance_plot_file = os.path.join(PLOTS_DIR, f"variance_explained_by_{groupby}_scatter.png")
    plt.savefig(variance_plot_file, dpi=300)
    plt.close()
        
    # 2. Barplot of factors sorted by explained variance
    plt.figure(figsize=(14, 7))
        
    # Sort by explained variance (descending)
    sorted_df = anova_df.sort_values("Explained_Variance", ascending=False)
        
    # Create color mapping based on explained variance levels
    def get_color(explained_var):
        if explained_var > 0.5:
            return "darkgreen"  # High explained variance
        elif explained_var > 0.25:
            return "olivedrab"  # Medium explained variance
        else:
            return "lightgreen"  # Low explained variance
    
    # Apply color mapping
    bar_colors = [get_color(val) for val in sorted_df["Explained_Variance"]]
    
    # Create barplot with factors ordered by explained variance
    bars = plt.bar(
        range(len(sorted_df)), 
        sorted_df["Explained_Variance"],
        color=bar_colors
    )
        
    # Set x-axis labels to factor indices in sorted order
    plt.xticks(
        range(len(sorted_df)),
        sorted_df["Factor_Index"],
        rotation=0
    )
    
    # Add PI values as text on bars
    for bar, pi in zip(bars, sorted_df["PI"]):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.02,
            f'PI={pi:.3f}',
            ha='center', 
            va='bottom',
            rotation=90,
            fontsize=8
        )
    
    plt.xlabel("Factor Index (ordered by explained variance)")
    plt.ylabel(f"Variance Explained by {groupby}")
    plt.title(f"Proportion of Factor Variance Explained by {groupby}")
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.7, label="High (>0.5): Strong cell type marker")
    plt.axhline(0.25, color='orange', linestyle='--', alpha=0.7, label="Medium (>0.25): Partial cell type association")
    plt.legend(loc='upper right')
    plt.ylim(0, 1)
    plt.tight_layout()
    
    barplot_file = os.path.join(PLOTS_DIR, f"variance_explained_by_{groupby}_barplot.png")
    plt.savefig(barplot_file, dpi=300)
    plt.close()
            
    print(f"   ✓ Variance component visualizations saved to {PLOTS_DIR}")
    print(f"   ✓ Variance component data saved to {DATA_DIR}")
    
    return anova_df

def plot_factor_distribution_by_group(adata, factors, group_column, n_groups=5, DATA_DIR=None, PLOTS_DIR=None):
    """Plot factor activity distribution across groups"""
    # Get factor matrix
    phi = adata.obsm["X_PHI"]
    
    # Ensure factors is a list
    if not isinstance(factors, list):
        factors = [factors]
    
    # Limit to top N groups by size
    top_groups = adata.obs[group_column].value_counts().head(n_groups).index
    
    # Set up the figure
    n_plots = len(factors)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
    
    # Handle the case where there's only one factor
    if n_plots == 1:
        axes = [axes]

    # Plot each factor
    for i, factor_idx in enumerate(factors):
        ax = axes[i]
        
        # Create data for violin plot
        plot_data = []
        plot_groups = []
        
        for group in top_groups:
            mask = adata.obs[group_column] == group
            values = phi[mask, factor_idx]
            plot_data.extend(values)
            plot_groups.extend([group] * len(values))
        
        plot_df = pd.DataFrame({
            "Factor Activity": plot_data,
            group_column: plot_groups
        })
        
        # Create violin plot
        sns.violinplot(x=group_column, y="Factor Activity", data=plot_df, ax=ax)
        
        # Add title and adjust layout
        ax.set_title(f"Factor {factor_idx} Distribution by {group_column}")
        ax.set_xlabel(group_column)
        ax.set_ylabel("Factor Activity")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    plt.tight_layout()
    
    # Save or show
    factors_str = "_".join(str(f) for f in factors)
    output_file = os.path.join(PLOTS_DIR, f"factor_{factors_str}_distribution_by_{group_column}.png")
    plt.savefig(output_file, dpi=300)
    plt.close()


def visualize_cell_perplexity(adata, color_by=None, n_bins=50, DATA_DIR=None, PLOTS_DIR=None):
    """
    Create visualizations of cell perplexity distribution
    
    Args:
        adata: AnnData object with 'perplexity' in obs
        color_by: Optional metadata column to color the distribution by, for example "broad_cell_type"
        n_bins: Number of bins for the histogram
        DATA_DIR: Directory to save the output data
        PLOTS_DIR: Directory to save the output plots
    """
    # Create perplexity directory inside the PLOTS_DIR
    perplexity_dir = os.path.join(PLOTS_DIR, "perplexity")
    os.makedirs(perplexity_dir, exist_ok=True)      
    
    print(f"   :gear: Visualizing cell perplexity{' by '+color_by if color_by else ''}...")
    
    # 1. Basic histogram of perplexity (only create once)
    if color_by is None:
        plt.figure(figsize=(10, 6))
        sns.histplot(adata.obs['perplexity'], bins=n_bins, kde=True)
        plt.title('Distribution of Cell Perplexity')
        plt.xlabel('Perplexity')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(perplexity_dir, "perplexity_histogram.png"), dpi=300)
        plt.close()
        
        # 2. UMAP colored by perplexity if UMAP coordinates exist (only create once)
        if 'X_umap' in adata.obsm:
            plt.figure(figsize=(10, 8))
            sc.pl.umap(
                adata, 
                color='perplexity', 
                cmap='viridis',
                show=False, 
                frameon=True
            )
            plt.title('UMAP Colored by Cell Perplexity')
            plt.tight_layout()
            plt.savefig(os.path.join(perplexity_dir, "umap_perplexity.png"), dpi=300)
            plt.close()
    
    # 3. Perplexity distribution colored by metadata (if specified)
    if color_by and color_by in adata.obs.columns:
        # Create a DataFrame with perplexity and the color variable
        perplexity_df = pd.DataFrame({
            'perplexity': adata.obs['perplexity'],
            color_by: adata.obs[color_by]
        })
        
        # Get top categories by frequency
        if perplexity_df[color_by].nunique() > 20:
            top_categories = perplexity_df[color_by].value_counts().head(20).index
            perplexity_df['color_category'] = perplexity_df[color_by].apply(
                lambda x: x if x in top_categories else 'Other'
            )
            color_column = 'color_category'
        else:
            color_column = color_by
        
        # Violin plot by category
        plt.figure(figsize=(14, 8))
        order = perplexity_df.groupby(color_column)['perplexity'].median().sort_values(ascending=False).index
        sns.violinplot(
            data=perplexity_df,
            x=color_column,
            y='perplexity',
            order=order
        )
        plt.title(f'Cell Perplexity Distribution by {color_by}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(perplexity_dir, f"perplexity_violin_by_{color_by}.png"), dpi=300)
        plt.close()
    
    print(f"   ✓ Cell perplexity visualizations saved to {perplexity_dir}")

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
    
    # Load splicing data
    splice_adata = ad.read_h5ad(ATSE_ANNDATA_PATH)

    # if "mouse.id" is in splice_adata.obs rename it to donor_id 
    if "mouse.id" in splice_adata.obs.columns:
        splice_adata.obs.rename(columns={"mouse.id": "donor_id"}, inplace=True)
    
    # Load gene expression data
    ge_adata = ad.read_h5ad(GE_ANNDATA_PATH)
        
    # Load aging gene lists
    aging_genes = load_aging_genes(AGING_GENES_PATH)
    
    # Load RBP genes
    rbps = load_rbp_genes(RBP_FILE_PATH)
    
    # Choose model based on param_id or best performance
    model_path = f"{MODEL_OUTPUTS_DIR}/run_{param_id}/leafletfa_model.pkl.xz"
    print(f"Using specified model: run_{param_id} with model path {model_path}")
    
    # Create output directory
    from datetime import datetime
    import os 
    # ensure os is imported
    print(f"os is imported: {os}")
    timestamp = datetime.now().strftime("%Y-%m-%d")
    OUTPUT_DIR = f"/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/results/{timestamp}/param_id_{param_id}"
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
    DATA_DIR = os.path.join(OUTPUT_DIR, "data")
    
    os.makedirs(PLOTS_DIR, exist_ok=True); os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Load the model
    leaflet_model = load_model(model_path)
    
    # Update gene annotations
    splice_adata.var["gene_id"] = splice_adata.var["gene_id"].str.split(".").str[0]
    splice_adata.var["RBP_gene"] = splice_adata.var["gene_name"].isin(rbps["mouse_gene_name"])
    splice_adata.var["Aging_gene"] = splice_adata.var["gene_name"].isin(aging_genes)
    
    print(f"   ✓ Data loaded successfully")
    
    ############################
    # 2. Extract Model Parameters
    ############################

    print("\n>> Extracting model parameters...")
    
    # Extract hyperparameters
    alpha_pi = leaflet_model["alpha_pi"]
    bb_conc = leaflet_model["bb_conc"]
    dir_conc = leaflet_model["dir_conc"]
    
    # Extract factor activities and usage
    PHI = leaflet_model["assign_post"]
    PI = leaflet_model["pi"]
    PSI = leaflet_model["psi_learned"]
    K = leaflet_model["K"]
            
    # Add PHI to adata
    splice_adata.obsm["X_PHI"] = PHI
    
    # Calculate cell perplexity/entropy 
    print("   :gear: Calculating cell perplexity...")
    PHI_safe = np.clip(PHI, 1e-10, 1)  # Prevent log(0) errors
    entropy = -np.sum(PHI_safe * np.log(PHI_safe), axis=1)
    perplexity = np.exp(entropy)
    splice_adata.obs["perplexity"] = perplexity
    splice_adata.obs["entropy"] = entropy
    
    # Print model parameters
    print(f"   ✓ Extracted {K} factors from the model")
    print(alpha_pi, bb_conc, dir_conc)
#    print(f"   ✓ Model parameters: alpha_pi={alpha_pi:.4f}, bb_conc={bb_conc:.4f}, dir_conc={dir_conc:.4f}")
    
    # Save model parameters to file
    model_params = {
        "alpha_pi": alpha_pi,
        "bb_conc": bb_conc,
        "dir_conc": dir_conc,
        "K": K,
        "run_id": param_id
    }

    pd.DataFrame([model_params]).to_csv(os.path.join(DATA_DIR, "model_parameters.csv"), index=False)
    
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
    # 3. Basic Visualizations
    ############################
    print("\n>> Generating basic visualizations...")
    
    # Plot PI distribution
    plot_factor_pi_distribution(PI, alpha_pi, PLOTS_DIR)
    
    # Generate UMAP from PHI
    sc.pp.neighbors(splice_adata, use_rep="X_PHI", n_neighbors=8)
    sc.tl.umap(splice_adata)
    
    # Plot UMAPs
    print("   :gear: Generating UMAPs...")

    # Cell type UMAP
    # Identify top 15 cell types
    top_cell_types = splice_adata.obs['broad_cell_type'].value_counts().head(10).index.tolist()

    # Create highlighted column
    splice_adata.obs['cell_type_highlighted'] = 'Other'
    splice_adata.obs.loc[splice_adata.obs['broad_cell_type'].isin(top_cell_types), 'cell_type_highlighted'] = \
        splice_adata.obs['broad_cell_type']

    # Generate a distinct color for each cell type using tab20 (supports 20)
    cmap = cm.get_cmap('tab20', len(top_cell_types))
    colors = [cmap(i) for i in range(len(top_cell_types))]
    color_dict = {cell_type: colors[i] for i, cell_type in enumerate(top_cell_types)}
    color_dict['Other'] = [0.9, 0.9, 0.9, 1.0]	  # gray for Other

    # Plot UMAP with legend in right margin
    plt.figure(figsize=(8, 5))
    sc.pl.umap(
        splice_adata,
        color='cell_type_highlighted',
        palette=color_dict,
        show=False,
        frameon=True,
        legend_fontsize=10,
        legend_loc='right margin'
    )
    plt.title('UMAP by Cell Type (Top 10 Highlighted)')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(PLOTS_DIR, "umap_cell_type_top10.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Get top five tissues by frequency
    top_tissues = splice_adata.obs['tissue'].value_counts().head(5).index.tolist()

    # Create a temporary column for coloring tissues
    splice_adata.obs['tissue_highlighted'] = 'Other'
    for tissue in top_tissues:
        mask = splice_adata.obs['tissue'] == tissue
        splice_adata.obs.loc[mask, 'tissue_highlighted'] = tissue

    # Create a custom colormap for tissues
    colors = cm.tab10(np.linspace(0, 1, len(top_tissues) + 1))
    tissue_color_dict = {tissue: colors[i] for i, tissue in enumerate(top_tissues)}
    tissue_color_dict['Other'] = [0.9, 0.9, 0.9, 1.0]	  # Light gray for "Other"

    # Plot the tissue UMAP
    plt.figure(figsize=(6, 6))  # Increase figure size
    sc.pl.umap(splice_adata, color='tissue_highlighted', palette=tissue_color_dict,
               show=False, frameon=True, legend_fontsize=12)
    plt.title('UMAP by Tissue (Top 5 Highlighted)')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add space for title
    plt.savefig(os.path.join(PLOTS_DIR, "umap_tissue_top5.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Make a umap colored by dataset 
    plt.figure(figsize=(6, 6))  # Increase figure size
    sc.pl.umap(splice_adata, color='dataset', 
               show=False, frameon=True, legend_fontsize=12)
    plt.title('UMAP by Dataset')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add space for title
    plt.savefig(os.path.join(PLOTS_DIR, "umap_dataset.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Add visualization for cell perplexity
    # First call creates the basic plots (histogram and UMAP)
    visualize_cell_perplexity(splice_adata, PLOTS_DIR=PLOTS_DIR, DATA_DIR=DATA_DIR)

    # Additional calls create violin plots by different metadata columns
    for color_var in ['broad_cell_type', 'tissue', 'dataset', 'age_numeric']:
        if color_var in splice_adata.obs.columns:
            visualize_cell_perplexity(splice_adata, color_by=color_var, PLOTS_DIR=PLOTS_DIR, DATA_DIR=DATA_DIR)

    # Test factor activity distribution by cell type (this will be more useful in a notebook/interactive environment)
    plot_factor_distribution_by_group(splice_adata, 3, "broad_cell_type", n_groups=5, DATA_DIR=DATA_DIR, PLOTS_DIR=PLOTS_DIR)   

    ############################
    # 4. Advanced Factor Analysis
    ############################
    print("\n>> Performing advanced factor analysis...")

    # Compute average factor activities across cells in different cell types, tissues and datasets...
    mean_exp_by_celltype, median_exp_by_celltype = compute_factor_activity_by_group(
        splice_adata, groupby="broad_cell_type", DATA_DIR=DATA_DIR, PLOTS_DIR=PLOTS_DIR
    )   

    mean_exp_by_tissue, median_exp_by_tissue = compute_factor_activity_by_group(
        splice_adata, groupby="tissue", DATA_DIR=DATA_DIR, PLOTS_DIR=PLOTS_DIR
    )   

    mean_exp_by_dataset, median_exp_by_dataset = compute_factor_activity_by_group(
        splice_adata, groupby="dataset", DATA_DIR=DATA_DIR, PLOTS_DIR=PLOTS_DIR
    )

    # Compute variance components
    print("   :gear: Computing variance components...")
    # Compute variance components by cell type using the PI values
    variance_components_celltype = compute_variance_components(
        splice_adata, 
        groupby="broad_cell_type",
        pi_values=PI,  # Pass the PI values you already have
        DATA_DIR=DATA_DIR,
        PLOTS_DIR=PLOTS_DIR
    )

    # Compute variance components by cell type using the PI values
    variance_components_tissue = compute_variance_components(
        splice_adata, 
        groupby="tissue",
        pi_values=PI,  # Pass the PI values you already have
        DATA_DIR=DATA_DIR,
        PLOTS_DIR=PLOTS_DIR
    )

    # Compute variance components by cell type using the PI values
    variance_components_dataset = compute_variance_components(
        splice_adata, 
        groupby="dataset",
        pi_values=PI,  # Pass the PI values you already have
        DATA_DIR=DATA_DIR,
        PLOTS_DIR=PLOTS_DIR
    )
    
if __name__ == "__main__":
    main()

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025
# sbatch --mem=250G -p dev,cpu --wrap="python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/downstream_analysis/01_evaluate_leafletFA_results.py"