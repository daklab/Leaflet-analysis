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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from scipy import stats

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
from matplotlib.colors import LinearSegmentedColormap

# Configure plotting styles
#sns.set_theme()
#sc.set_figure_params(figsize=(7, 7), frameon=True, dpi=80, facecolor='white')

# Import utility functions - simple direct import
sys.path.append('/gpfs/commons/home/kisaev/Leaflet-analysis/Multi_Species_Splicing_Foundation/shared_utils/')
from utils import *

# Check if using CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

######################
### Helper Functions #
######################

def compute_factor_activity_by_group(adata, groupby="broad_cell_type", DATA_DIR=None, PLOTS_DIR=None):
    """
    Compute non-zero-aware mean and median factor activity across cell groups.
    Visualize mean activity in a clustermap, with outline strength reflecting how many cells had non-zero activity.
    
    Returns:
        Tuple of (mean_expression, median_expression) DataFrames.
    """

    print(f"   :gear: Computing factor expression summary by {groupby}...")
    
    # Extract PHI matrix
    X_PHI = adata.obsm["X_PHI"]
    n_factors = X_PHI.shape[1]
    factor_cols = [f"factor_{i}" for i in range(n_factors)]
    
    # Build DataFrame
    df = pd.DataFrame(X_PHI, columns=factor_cols)
    df[groupby] = adata.obs[groupby].values
    
    # Grouping
    grouped = df.groupby(groupby, observed=True)
    
    # Compute metrics
    mean_expression = grouped[factor_cols].apply(lambda x: x.where(x != 0).mean())
    median_expression = grouped[factor_cols].median()
    nonzero_counts = grouped[factor_cols].apply(lambda x: (x > 0.01).sum())
    total_counts = grouped.size()
    nonzero_fraction = nonzero_counts.div(total_counts, axis=0)
    
    # Clean up any NaN values that might cause issues
    mean_expression = mean_expression.fillna(0)
    median_expression = median_expression.fillna(0)
    nonzero_fraction = nonzero_fraction.fillna(0)
    
    # Attach metadata
    mean_expression_with_meta = mean_expression.copy()
    median_expression_with_meta = median_expression.copy()
    mean_expression_with_meta["cell_count"] = total_counts
    median_expression_with_meta["cell_count"] = total_counts
    
    if DATA_DIR:
        os.makedirs(DATA_DIR, exist_ok=True)
        mean_expression_with_meta.to_csv(os.path.join(DATA_DIR, f"mean_factor_expression_by_{groupby}.csv"))
        median_expression_with_meta.to_csv(os.path.join(DATA_DIR, f"median_factor_expression_by_{groupby}.csv"))
        nonzero_fraction.to_csv(os.path.join(DATA_DIR, f"nonzero_fraction_by_{groupby}.csv"))
        print(f"✓ Saved to {DATA_DIR}")
    
    if PLOTS_DIR:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        
        # --------- Clustermap 1: Mean Activity ---------
        plot_data = mean_expression.copy()  # Use version without metadata
        
        # Check for any remaining issues with the data
        print(f"Plot data shape: {plot_data.shape}")
        print(f"Data range: {plot_data.min().min():.3f} to {plot_data.max().max():.3f}")
        
        # Create clustermap with error handling
        g = sns.clustermap(
            plot_data,
            cmap="PRGn",
            center=0,
            figsize=(6, 6),  # Made larger for better visibility
            xticklabels=True,
            yticklabels=True,
            cbar_kws={'label': 'Mean Activity'},
            linewidths=0.2,
            linecolor='gray'
             )
        
        # Improve label formatting
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
        
        # Save first plot
        path1 = os.path.join(PLOTS_DIR, f"mean_factor_expression_by_{groupby}_clustermap.pdf")
        g.savefig(path1, bbox_inches='tight', dpi=300)
        print(f"✓ Mean activity clustermap saved to: {path1}")
        plt.close(g.fig)

        # Make one also using median expression
        g2 = sns.clustermap(
            median_expression,
            cmap="PRGn",
            center=0,
            figsize=(6, 6),  # Made larger for better visibility
            xticklabels=True,
            yticklabels=True,
            cbar_kws={'label': 'Median Activity'},
            linewidths=0.2,
            linecolor='gray'
        )   
        plt.setp(g2.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(g2.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
        path2 = os.path.join(PLOTS_DIR, f"median_factor_expression_by_{groupby}_clustermap.pdf")
        g2.savefig(path2, bbox_inches='tight', dpi=300)
        print(f"✓ Median activity clustermap saved to: {path2}")
        plt.close(g2.fig)

        # Get normalized clustermap to see relative factor activities across cell types
        # Z-score normalize each factor (column) across cell types (rows)
        mean_zscore = mean_expression.apply(lambda x: stats.zscore(x), axis=0).fillna(0)
        
        g3 = sns.clustermap(
            mean_zscore,
            cmap="PRGn",
            center=0,
            figsize=(6, 6),  # Made larger for better visibility
            xticklabels=True,
            yticklabels=True,
            cbar_kws={'label': 'Relative Activity (Z-score)'},
            linewidths=0.2,
            linecolor='gray',
            metric='euclidean'
        )
        plt.setp(g3.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        plt.setp(g3.ax_heatmap.get_yticklabels(), rotation=0, fontsize=10)
        path3 = os.path.join(PLOTS_DIR, f"zscore_factor_expression_by_{groupby}_clustermap.pdf")
        g3.savefig(path3, bbox_inches='tight', dpi=300)
        print(f"✓ Z-score normalized clustermap saved to: {path3}")
        plt.close(g3.fig)


        # --- Delta: median_old - median_young across cell groups ---
        print("Computing delta median factor activity (old - young)...")

        # Sanity check
        assert "age_group" in adata.obs.columns, "Missing 'age_group' column in .obs"
        assert set(adata.obs["age_group"].unique()) == {"young", "old"}, "Expected 'young' and 'old' groups only"

        df_all = pd.DataFrame(X_PHI, columns=factor_cols)
        df_all[groupby] = adata.obs[groupby].values
        df_all["age_group"] = adata.obs["age_group"].values

        # Group by groupby + age_group
        median_by_group_and_age = df_all.groupby([groupby, "age_group"], observed=True)[factor_cols].median()

        # Pivot into two matrices: one for young, one for old
        median_young = median_by_group_and_age.xs("young", level="age_group")
        median_old = median_by_group_and_age.xs("old", level="age_group")

        # Align index (cell types) and subtract
        delta_median = median_old - median_young
        delta_median = delta_median.fillna(0)

        # Save delta matrix
        if DATA_DIR:
            delta_median.to_csv(os.path.join(DATA_DIR, f"delta_median_old_minus_young_by_{groupby}.csv"))

        # Reorder delta_median based on median_expression clustermap (g2)
        reordered_rows = median_expression.index[g2.dendrogram_row.reordered_ind]
        reordered_cols = median_expression.columns[g2.dendrogram_col.reordered_ind]

        delta_median_reordered = delta_median.loc[reordered_rows, reordered_cols]

        # Plot delta clustermap (same row/col order as mean_expression)
        g4 = sns.clustermap(
            delta_median_reordered,
            cmap="BrBG",
            center=0,
            figsize=(7, 5),
            xticklabels=True,
            yticklabels=True,
            cbar_kws={'label': 'Δ (Old - Young)'},
            linewidths=0.2,
            linecolor='gray',
            row_cluster=False,
            col_cluster=False
        )
        # Use same tick label formatting
        plt.setp(g4.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(g4.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)

        path4 = os.path.join(PLOTS_DIR, f"delta_median_old_minus_young_by_{groupby}_clustermap.pdf")
        g4.savefig(path4, bbox_inches='tight', dpi=300)
        print(f"✓ Δ median activity (old-young) clustermap saved to: {path4}")
        plt.close(g4.fig)

    return mean_expression_with_meta, median_expression_with_meta

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

    # In 01_evaluate_leafletFA_results.py, inside main() after PI is defined
    np.save(os.path.join(DATA_DIR, "PI_values.npy"), PI)
    print(f"   ✓ PI values saved to: {os.path.join(DATA_DIR, 'PI_values.npy')}")
    
    # Initialize output
    explained_variances = []
    
    # Get the group labels, ensure they're not categorical
    if pd.api.types.is_categorical_dtype(adata.obs[groupby]):
        groups = adata.obs[groupby].astype(str).values
    else:
        groups = adata.obs[groupby].values

    # If groupby is numeric, split into three quantile-based groups
    if groupby == "age_numeric" or groupby == "age":
        lower_q = np.quantile(groups, 1/3)
        upper_q = np.quantile(groups, 2/3)
        print(f"   :information_source: '{groupby}' is numeric — splitting into tertiles at {lower_q:.2f} and {upper_q:.2f}")
        
        def assign_tertile(val):
            if val <= lower_q:
                return "group1"  # young
            elif val <= upper_q:
                return "group2"  # middle
            else:
                return "group3"  # old

        groups = np.array([assign_tertile(v) for v in groups])
    
    # Process each factor using one-way ANOVA decomposition
    for factor_idx in range(n_factors):
        # Get factor values
        factor_values = X_PHI[:, factor_idx]
        
        # Create a DataFrame for ANOVA
        anova_df = pd.DataFrame({
            'factor_values': factor_values,
            'group': groups
        })

        # Group stats
        group_stats = anova_df.groupby('group')['factor_values'].agg(['mean', 'count'])

        # Overall mean
        overall_mean = np.mean(factor_values)

        # Between-group sum of squares: Σ n_i * (mean_i - overall_mean)^2
        between_ss = np.sum(group_stats['count'] * (group_stats['mean'] - overall_mean)**2)
               
        # Within-group sum of squares: Σ (x_ij - group_mean)^2
        within_ss = 0
        for group_name, group_data in anova_df.groupby('group'):
            group_mean = group_stats.loc[group_name, 'mean']
            within_ss += np.sum((group_data['factor_values'] - group_mean) ** 2)

        # Degrees of freedom
        n_groups = len(group_stats)
        n_samples = len(factor_values)
        df_between = n_groups - 1
        df_within = n_samples - n_groups        

        # Variance components (mean square errors)
        between_variance = between_ss / df_between if df_between > 0 else 0
        within_variance = within_ss / df_within if df_within > 0 else 0
    
        # Total sum of squares (for consistency)
        total_ss = between_ss + within_ss
        total_variance = total_ss / (n_samples - 1)  # same denominator as np.var(..., ddof=1)
    
        # Proportion of variance explained (R²)
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
    plt.figure(figsize=(5,5))
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
            fontsize=12,
            weight='bold'
        )
        
    # Add colorbar with smaller tick labels
    cbar = plt.colorbar(scatter, label="Factor PI (Global Assignment Probability)")
    cbar.ax.tick_params(labelsize=8)  # Reduce legend tick label size
    
    # Increase x-axis tick label size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.xlabel("Total Variance", fontsize=14)
    plt.ylabel(f"Variance Explained by {groupby}", fontsize=14)
    plt.title(f"Factor Variance Explained by {groupby} vs Total Variance", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    variance_plot_file = os.path.join(PLOTS_DIR, f"variance_explained_by_{groupby}_scatter.pdf")
    plt.savefig(variance_plot_file, format='pdf', bbox_inches='tight')
    plt.close()
        
    # 2. Barplot of factors sorted by explained variance
    plt.figure(figsize=(6, 5))
        
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
        [f"Factor {idx}" for idx in sorted_df["Factor_Index"]],
        rotation=45,
        fontsize=8,
        ha='right'
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
    
    plt.xlabel("Factor Index (ordered by explained variance)", fontsize=10)
    plt.ylabel(f"Variance Explained by {groupby}", fontsize=10)
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.7, label="High (>0.5): Higher proportion of variance explained")
    plt.axhline(0.25, color='orange', linestyle='--', alpha=0.7, label="Medium (>0.25): Lower proportion of variance explained")
    plt.legend(loc='upper right', fontsize=10)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    barplot_file = os.path.join(PLOTS_DIR, f"variance_explained_by_{groupby}_barplot.pdf")
    plt.savefig(barplot_file, format='pdf', bbox_inches='tight')
    plt.close()
            
    print(f"✓ Variance component visualizations saved to {PLOTS_DIR}")
    print(f"✓ Variance component data saved to {DATA_DIR}")
    return anova_df

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
    
    print(f"Visualizing cell perplexity{' by '+color_by if color_by else ''}...")
    
    # 1. Basic histogram of perplexity (only create once)
    if color_by is None:
        plt.figure(figsize=(5, 4))
        # Add line to indicate median perplexity
        plt.axvline(adata.obs['perplexity'].median(), color='red', linestyle='--', label='Median Perplexity')
        sns.histplot(adata.obs['perplexity'], bins=n_bins, kde=True, color='lightgray')
        plt.xlabel('# of Effective Factors/Cell (Perplexity)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(perplexity_dir, "perplexity_histogram.pdf"), format='pdf', bbox_inches='tight')
        plt.close()

        # 2. UMAP colored by perplexity if UMAP coordinates exist (only create once)
        if 'X_umap' in adata.obsm:
            plt.figure(figsize=(8, 8))
            sc.pl.umap(
                adata, 
                color='perplexity', 
                cmap='viridis',
                show=False, 
                frameon=True
            )
            plt.title('UMAP Colored by Cell Perplexity')
            plt.tight_layout()
            plt.savefig(os.path.join(perplexity_dir, "umap_perplexity.pdf"), format='pdf', bbox_inches='tight')
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
        plt.figure(figsize=(6, 4))
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
        plt.savefig(os.path.join(perplexity_dir, f"perplexity_violin_by_{color_by}.pdf"), format='pdf', bbox_inches='tight')
        plt.close()

        # Create pie chart for high perplexity cells
        # Calculate 90th percentile threshold
        perplexity_threshold = np.percentile(perplexity_df['perplexity'], 90)
        
        # Get cells above threshold
        high_perplexity_cells = perplexity_df[perplexity_df['perplexity'] >= perplexity_threshold]
        
        # Count categories
        category_counts = high_perplexity_cells[color_column].value_counts()
        
        # Create pie chart
        plt.figure(figsize=(4, 4))
        plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%',
               colors=plt.cm.tab20(np.linspace(0, 1, len(category_counts))),
               textprops={'fontsize': 8})
        plt.title(f'{color_by.replace("_", " ").title()} Distribution for Top 10% Perplexity Cells\n(Threshold: {perplexity_threshold:.2f})')
        plt.tight_layout()
        plt.savefig(os.path.join(perplexity_dir, f"high_perplexity_{color_by}_pie.pdf"), 
                   format='pdf', bbox_inches='tight')
        plt.close()
    
    # Plot scatter plot of perplexity vs library size
    plt.figure(figsize=(6, 6))
    scatter = sns.scatterplot(
        x='library_size', 
        y='perplexity', 
        data=adata.obs,
        hue='age_numeric',
        palette='viridis',
        alpha=0.6,
        s=20  # smaller point size
    )
    
    # Add colorbar label
    plt.colorbar(scatter.collections[0]).set_label('Age')
    
    plt.title('Perplexity vs Library Size', fontsize=14)
    plt.xlabel('Library Size', fontsize=12)
    plt.ylabel('Perplexity', fontsize=12)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "perplexity_vs_library_size.pdf"), 
                format='pdf', 
                bbox_inches='tight',
                dpi=300)
    plt.close()    
    print(f"✓ Cell perplexity visualizations saved to {perplexity_dir}")

###########################
### Main Analysis Script ##
###########################

# Get param_id and MODEL_OUTPUTS_DIR from command line if provided
if len(sys.argv) > 1:
    param_id = sys.argv[1]
    MODEL_OUTPUTS_DIR = sys.argv[2]
    ATSE_ANNDATA_PATH = sys.argv[3]
    GE_ANNDATA_scVI_PATH = sys.argv[4]
    AGING_GENES_PATH = sys.argv[5]
    RBP_FILE_PATH = sys.argv[6]
    OUTPUT_DIR = sys.argv[7]
    print(f"Using specified param_id: {param_id}")
    print(f"Using specified MODEL_OUTPUTS_DIR: {MODEL_OUTPUTS_DIR}")
    print(f"Using specified ATSE_ANNDATA_PATH: {ATSE_ANNDATA_PATH}")
    print(f"Using specified GE_ANNDATA_scVI_PATH: {GE_ANNDATA_scVI_PATH}")
    print(f"Using specified AGING_GENES_PATH: {AGING_GENES_PATH}")
    print(f"Using specified RBP_FILE_PATH: {RBP_FILE_PATH}")
    print(f"Using output directory: {OUTPUT_DIR}")

def main():
    print("\n========================================")
    print("LeafletFA Model Evaluate 01")
    print("========================================\n")
    
    ############################
    # 1. Load Data and Model
    ############################
    print("\n>> Loading data and model...")

    final_cells = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/062025/filtered_cell_ids.txt"
    with open(final_cells, "r") as f:
        final_cells = f.read().splitlines()
    
    # Load splicing data
    splice_adata = ad.read_h5ad(ATSE_ANNDATA_PATH)
    ge_adata = ad.read_h5ad(GE_ANNDATA_scVI_PATH)

    # If ge_adata.obs doesn't have cell_id make it from cell_id_clean
    if "cell_id" not in ge_adata.obs.columns:
        ge_adata.obs["cell_id"] = ge_adata.obs["cell_id_clean"]
        splice_adata.obs["cell_id"] = splice_adata.obs["cell_id_clean"] 

    assert np.all(ge_adata.obs["cell_id"].values == splice_adata.obs["cell_id"].values), "Cell IDs in ge_adata and splice_adata do not match or are not in the same order."

    # Only subset by final_cells if in mouse so check if "MOUSE_" is in ATSE_ANNDATA_PATH
    if "MOUSE_" in ATSE_ANNDATA_PATH:
        print("   :gear: Subsetting to final cells (outlier removal)...")
        # Subset both anndatas to only include cells in final_cells
        splice_adata = splice_adata[splice_adata.obs["cell_id"].isin(final_cells)].copy()
        ge_adata = ge_adata[ge_adata.obs["cell_id"].isin(final_cells)].copy()

    assert np.all(ge_adata.obs["cell_id"].values == splice_adata.obs["cell_id"].values), "Cell IDs in ge_adata and splice_adata do not match or are not in the same order."

    # Add "library_size" from ge_adata to splice_adata
    splice_adata.obs["library_size"] = ge_adata.obs["library_size"]
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

    # Choose model based on param_id or best performance
    model_path = f"{MODEL_OUTPUTS_DIR}/run_{param_id}/leafletfa_model.pkl.xz"
    print(f"Using specified model: run_{param_id} with model path {model_path}")
    
    # if OUTPUT_DIR does not exist, create it
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Create output directory
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
    DATA_DIR = os.path.join(OUTPUT_DIR, "data")
    
    os.makedirs(PLOTS_DIR, exist_ok=True); os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Load the model
    leaflet_model = load_model(model_path)
        
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
    # Subset PHI based on cell_id_index in splice_adata.obs
    PHI = PHI[splice_adata.obs.cell_id_index, :]
    # assert shape of PHI matches shape of splice_adata.obs
    assert PHI.shape == (len(splice_adata.obs), leaflet_model["K"]), "PHI shape does not match the number of cells and factors."
    PI = leaflet_model["pi"]
    K = leaflet_model["K"]
            
    # Add PHI to adata
    splice_adata.obsm["X_PHI"] = PHI
    print(f"   ✓ PHI added to adata")
    
    # Calculate cell perplexity/entropy 
    print("   :gear: Calculating cell perplexity...")
    PHI_safe = np.clip(PHI, 1e-10, 1)  # Prevent log(0) errors
    entropy = -np.sum(PHI_safe * np.log(PHI_safe), axis=1)
    perplexity = np.exp(entropy)
    splice_adata.obs["perplexity"] = perplexity
    splice_adata.obs["entropy"] = entropy

    # In 01_evaluate_leafletFA_results.py, inside main() after perplexity is calculated
    median_perplexity = splice_adata.obs["perplexity"].median()
    pd.DataFrame({"median_cell_perplexity": [median_perplexity]}).to_csv(os.path.join(DATA_DIR, "median_cell_perplexity.csv"), index=False)
    print(f"   ✓ Median cell perplexity ({median_perplexity:.2f}) saved to: {os.path.join(DATA_DIR, 'median_cell_perplexity.csv')}")
    
    # --- New snippet to save cell-level data with perplexity ---
    print(f"  Preparing to save cell-level metadata with perplexity...")

    # Define the columns you want from splice_adata.obs
    cols_to_select = ['broad_cell_type', 'tissue', 'age_numeric', 'dataset', 'perplexity', 'library_size']

    # Create a DataFrame with these columns
    # First, check which of the desired columns actually exist in splice_adata.obs
    existing_cols_to_select = [col for col in cols_to_select if col in splice_adata.obs.columns]
    cell_meta_df = splice_adata.obs[existing_cols_to_select].copy()

    # Add cell IDs - assuming they are in the AnnData index splice_adata.obs_names
    # If you have a specific 'cell_id' column, you could add it to cols_to_select instead
    cell_meta_df['cell_id'] = splice_adata.obs["cell_id"]
    # Reorder columns to have 'cell_id' first, then perplexity, then others
    final_columns_order = ['cell_id'] + \
                          (['perplexity'] if 'perplexity' in cell_meta_df.columns else []) + \
                          [col for col in existing_cols_to_select if col != 'perplexity']
    
    # Filter out any columns that might have been duplicated or are not in the df
    final_columns_order = [col for col in final_columns_order if col in cell_meta_df.columns]
    cell_meta_df = cell_meta_df[final_columns_order]
    # Define the output file path
    output_filename = f"cell_metadata_with_perplexity_param_id_{param_id}.csv.gz" # param_id should be defined in your script 01
    output_file_path = os.path.join(DATA_DIR, output_filename)
    # Save to a gzipped CSV
    cell_meta_df.to_csv(output_file_path, index=False, compression='gzip')
    print(f"   ✓ Cell metadata with perplexity saved to: {output_file_path}")
     # Print model parameters
    print(f"   ✓ Extracted {K} factors from the model")
    print(alpha_pi, bb_conc, dir_conc)
    
    # Save model parameters to file
    model_params = {
        "alpha_pi": alpha_pi,
        "bb_conc": bb_conc,
        "dir_conc": dir_conc,
        "K": K,
        "run_id": param_id
    }

    pd.DataFrame([model_params]).to_csv(os.path.join(DATA_DIR, "model_parameters.csv"), index=False)
    
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
    top_cell_types = splice_adata.obs['broad_cell_type'].value_counts().head(15).index.tolist()

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
    plt.figure(figsize=(5, 5))
    sc.pl.umap(
        splice_adata,
        color='cell_type_highlighted',
        palette=color_dict,
        show=False,
        frameon=True,
        legend_fontsize=6,
        legend_loc='right margin'
    )
    plt.title('UMAP by Cell Type (Top 15 Highlighted)')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(PLOTS_DIR, "umap_cell_type_top15.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

    # Get top five tissues by frequency
    top_tissues = splice_adata.obs['tissue'].value_counts().head(10).index.tolist()

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
    plt.figure(figsize=(5, 5))  # Increase figure size
    sc.pl.umap(splice_adata, color='tissue_highlighted', palette=tissue_color_dict,
               show=False, frameon=True, legend_fontsize=12)
    plt.title('UMAP by Tissue (Top 5 Highlighted)')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add space for title
    plt.savefig(os.path.join(PLOTS_DIR, "umap_tissue_top10.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

    # Make a umap colored by dataset 
    plt.figure(figsize=(5, 5))  # Increase figure size
    sc.pl.umap(splice_adata, color='dataset', 
               show=False, frameon=True, legend_fontsize=12)
    plt.title('UMAP by Dataset')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add space for title
    plt.savefig(os.path.join(PLOTS_DIR, "umap_dataset.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

    # Add visualization for cell perplexity
    # First call creates the basic plots (histogram and UMAP)
    visualize_cell_perplexity(splice_adata, PLOTS_DIR=PLOTS_DIR, DATA_DIR=DATA_DIR)

    # Additional calls create violin plots by different metadata columns
    for color_var in ['broad_cell_type', 'tissue', 'dataset', 'age_numeric']:
        if color_var in splice_adata.obs.columns:
            visualize_cell_perplexity(splice_adata, color_by=color_var, PLOTS_DIR=PLOTS_DIR, DATA_DIR=DATA_DIR)

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
    
    # Compute variance components by numeric age
    variance_components_age = compute_variance_components(
        splice_adata, 
        groupby="age_numeric",
        pi_values=PI,  # Pass the PI values you already have
        DATA_DIR=DATA_DIR,
        PLOTS_DIR=PLOTS_DIR
    )

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

