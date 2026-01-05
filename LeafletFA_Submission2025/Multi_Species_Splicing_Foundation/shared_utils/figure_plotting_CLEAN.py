# =============================================================================
# IMPORTS AND SETUP
# =============================================================================
import anndata as ad
from pathlib import Path
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy.stats import spearmanr, pearsonr
import scipy.sparse as sp
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib.ticker import ScalarFormatter
import scanpy as sc
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import defaultdict
import warnings
from adjustText import adjust_text
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def relabel_covariate(row):
    covs = row['top_covariates']
    has_celltype = 'C(cell_type)' in covs
    has_age = 'age' in covs
    has_tissue = 'C(tissue)' in covs
    has_sex = 'C(sex)' in covs
    has_interaction = 'C(cell_type):age' in covs
    labels = []
    if has_interaction:
        labels.append('cell_type:age')
    if has_celltype:
        labels.append('cell_type')
    if has_age:
        labels.append('age')
    if has_tissue:
        labels.append('tissue')
    if has_sex:
        labels.append('sex')
    return ' + '.join(labels) if labels else 'other'

# =============================================================================
# PLOTTING UTILITIES AND BASE CLASSES
# =============================================================================
class PlotConfig:
    """Centralized plotting configuration"""
    def __init__(self):
        self.default_figsize = (6, 4)
        self.default_dpi = 300
        self.color_palettes = {
            'factors': 'YlGnBu',
            'correlations': 'viridis',
            'age_groups': ['lightcoral', 'lightgreen', 'skyblue', 'gold']
        }
    
    def setup_plot(self, figsize=None, title=None):
        """Consistent plot setup"""
        figsize = figsize or self.default_figsize
        fig, ax = plt.subplots(figsize=figsize)
        if title:
            ax.set_title(title)
        return fig, ax

# Global plot config instance
PLOT_CONFIG = PlotConfig()

# =============================================================================
# FACTOR ACTIVITY ANALYSIS
# =============================================================================
def plot_covariates_factor_labels(pi, factor_labels, figsize=(4, 6), save_path=None, show_plot=True):
    """
    Create a bar plot showing R² values for factors with PI annotations.
    """ 
    # Create DataFrame for pi values
    pi_df = pd.DataFrame({
        "factor_id": [f"factor_{i}" for i in range(len(pi))],
        "PI": pi
    })
    
    # Merge with factor_labels by factor_id
    df = factor_labels.merge(pi_df, on="factor_id")
    df = df.sort_values("r2", ascending=False).copy()
    df["rank"] = range(1, len(df) + 1)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=df, y="factor_id", x="r2", hue="combined_label", dodge=False, ax=ax)
    # Add dashed red line at R^2 = 0.5
    ax.axvline(x=0.5, color='grey', linestyle='--', linewidth=1)    
    # Annotate each bar with PI value
    for i, (r2_val, pi_val) in enumerate(zip(df["r2"], df["PI"])):
        ax.text(
            x=r2_val + 0.01,
            y=i,
            s=f"PI={pi_val:.2f}",
            va='center',
            ha='left',
            fontsize=8
        )    
    # Set labels
    ax.set_xlabel("R² (Explained Variance)")
    ax.set_ylabel("Factor")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf")
    if show_plot:
        plt.show()
    return fig, ax


def plot_factor_violin_sorted(
    adata,
    factor_list,
    groupby="broad_cell_type",
    include_groups=None,
    width=8,       # ← fixed width
    height=4,
    cmap="YlGnBu",
    top_n=10,
    save_prefix=None
):
    """
    Plot violin plots of factor activity grouped by a metadata field,
    showing top_n groups by median activity and collapsing the rest as 'Other'.
    """
    obs_df = adata.obs.copy()
    if include_groups is not None:
        obs_df = obs_df[obs_df[groupby].isin(include_groups)]

    for factor in factor_list:
        plot_df = obs_df[[groupby, factor]].dropna()

        # Compute medians and get top_n groups
        medians = plot_df.groupby(groupby)[factor].median().sort_values(ascending=False)
        top_groups = medians.head(top_n).index.tolist()

        # Map rest to "Other"
        plot_df[groupby] = plot_df[groupby].apply(lambda x: x if x in top_groups else "Other")

        # Define plot order: top groups + Other
        plot_order = top_groups + ["Other"]

        # Categorical for ordering
        plot_df[groupby] = pd.Categorical(plot_df[groupby], categories=plot_order, ordered=True)

        # Build color palette (add grey for "Other")
        base_palette = sns.color_palette(cmap, len(top_groups))
        color_map = dict(zip(top_groups, base_palette))
        color_map["Other"] = "lightgrey"

        plt.figure(figsize=(width, height))
        sns.violinplot(
            data=plot_df,
            x=groupby,
            y=factor,
            order=plot_order,
            palette=color_map,
            inner="box",
            cut=0
        )
        plt.title(f"{factor} Activity by Top {top_n} {groupby} (+Other)")
        plt.xlabel(groupby)
        plt.ylabel("Factor Activity")
        plt.xticks(rotation=90)
        plt.tight_layout()
        if save_prefix:
            plt.savefig(f"{save_prefix}_{factor}_by_{groupby}_top{top_n}_with_other.pdf", format="pdf")
        plt.show()
        
def plot_factor_age_trends(
    df,
    factor,
    groupby="broad_cell_type",
    age_col="age_numeric",
    top_n=7,
    figsize=(4, 4),
    title_prefix=None,
    save_path=None
):
    """
    Plot median factor activity over age, labeled by top N cell types.
    """

    # Step 1: rank groups by median factor activity
    overall_median = (
        df.groupby(groupby)[factor]
        .median()
        .sort_values(ascending=False)
    )
    top_groups = overall_median.head(top_n).index.tolist()

    # Step 2: subset data
    filtered_df = df[df[groupby].isin(top_groups)]

    # Step 3: summarize median by age within group
    summary = (
        filtered_df.groupby([groupby, age_col])[factor]
        .median()
        .reset_index()
    )

    # Step 4: plot
    plt.figure(figsize=figsize)
    ax = sns.lineplot(
        data=summary,
        x=age_col,
        y=factor,
        hue=groupby,
        marker="o",
        legend=False
    )

    # Step 5: label last point per group
    for group in summary[groupby].unique():
        data_ct = summary[summary[groupby] == group]
        first = data_ct.sort_values(age_col).iloc[0]
        ax.text(
            x=first[age_col] - 0.2,
            y=first[factor],
            s=group,
            fontsize=7,
            va='center'
        )

    # Final formatting
    title = f"{factor} Median Activity by Age"
    if title_prefix:
        title = f"{title_prefix}: {title}"
    plt.title(title)
    plt.xlabel("Age (months)")
    plt.ylabel("Median Factor Activity")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf")
    plt.show()


# =============================================================================
# AGE PREDICTION AND CLASSIFICATION  
# =============================================================================
class AgeAnalysisHelper:
    """Helper class for common age analysis tasks"""
    
    @staticmethod
    def create_age_groups(ages, method="percentile", n_groups=3, thresholds=None):
        """Standardized age group creation"""
        if thresholds is not None:
            # Use custom thresholds
            def assign_group(age):
                for i, threshold in enumerate(thresholds):
                    if age <= threshold:
                        return i
                return len(thresholds)
            return ages.apply(assign_group)
        else:
            # Use percentiles
            percentiles = np.linspace(0, 100, n_groups + 1)[1:-1]
            thresholds = [np.percentile(ages, p) for p in percentiles]
            return AgeAnalysisHelper.create_age_groups(ages, thresholds=thresholds)
    
    @staticmethod
    def validate_inputs(splice_adata, ge_adata, required_cols):
        """Common input validation"""
        for col in required_cols:
            if col not in splice_adata.obs.columns:
                raise KeyError(f"Missing column: {col}")
        
        if len(splice_adata) != len(ge_adata):
            raise ValueError("AnnData objects must have same number of cells")
        
        return True

def compute_age_r2_with_cv_linear_regression(splice_adata, ge_adata, **kwargs):
    # Your existing function - but now can use AgeAnalysisHelper methods
    pass

def compute_three_group_age_classification(splice_adata, ge_adata, **kwargs):
    # Your existing function
    pass

# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================
def plot_gene_vs_factor_correlation(splice_adata, ge_adata, gene_name, factor_col, **kwargs):
    # Your existing function
    pass

def correlate_junction_psi_with_genes_efficient(splice_adata, ge_adata, diff_spl, factor_idx, **kwargs):
    # Your existing function
    pass

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================  
def plot_cv_r2_scatter_with_labels(results_df, **kwargs):
    # Your existing function
    pass

def plot_global_results(results_df, detailed_scores, **kwargs):
    # Your existing function
    pass

