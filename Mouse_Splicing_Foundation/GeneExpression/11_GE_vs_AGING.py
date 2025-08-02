#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import sys
from datetime import datetime

# Import utility functions
sys.path.append('/gpfs/commons/home/kisaev/Leaflet-analysis/Multi_Species_Splicing_Foundation/shared_utils/')
from utils import *

# Setup
today = datetime.now().strftime("%Y%m%d")
BASE_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
OUTPUT_DIR = f"/gpfs/commons/home/kisaev/Leaflet-analysis/Multi_Species_Splicing_Foundation/GE_PLOTS/MOUSE/{today}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File paths
GE_ANNDATA_scVI_PATH = f"{BASE_DIR}/scVI/ge_adata_with_both_scvi_models_2025-07-30.h5ad"
GE_ANNDATA_NMF_PATH = f"{BASE_DIR}/NMF/ge_adata_with_NMF_standard_50_1024_2025-07-30.h5ad"
AGING_GENES_PATH = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/27857814"
RBP_FILE_PATH = "/gpfs/commons/groups/knowles_lab/Karin/VanNostrand_2020_supptable1_41586_2020_2077_MOESM3_ESM.xlsx"

# Load data
print("Loading data...")
ge_adata = ad.read_h5ad(GE_ANNDATA_scVI_PATH)
ge_adata_nmf = ad.read_h5ad(GE_ANNDATA_NMF_PATH)
print(f"Done reading in the gene expression data!")

# Data preprocessing
if "cell_id" not in ge_adata.obs.columns:
    ge_adata.obs["cell_id"] = ge_adata.obs["cell_id_clean"]
    ge_adata_nmf.obs["cell_id"] = ge_adata_nmf.obs["cell_id_clean"]

if "gene_name" not in ge_adata.var.columns:
    ge_adata.var["gene_name"] = ge_adata.var_names

# Load gene sets
aging_genes_mouse, aging_genes_human = load_aging_genes(AGING_GENES_PATH)
rbps = load_rbp_genes(RBP_FILE_PATH)
rbps["mouse_gene_name"] = rbps["mouse_gene_name"].str.upper()
aging_genes_mouse = [g.upper() for g in aging_genes_mouse]

# Annotate genes
ge_adata.var["RBP_gene"] = ge_adata.var["gene_name"].isin(rbps["mouse_gene_name"])
ge_adata.var["Aging_gene"] = ge_adata.var["gene_name"].isin(aging_genes_mouse)

# Merge NMF results
assert np.all(ge_adata.obs_names == ge_adata_nmf.obs_names), "Cell order mismatch"
assert np.all(ge_adata.var_names == ge_adata_nmf.var_names), "Gene order mismatch"
ge_adata.obsm["X_nmf_standard_mb"] = ge_adata_nmf.obsm["X_nmf_standard_mb"]
ge_adata.varm["nmf_standard_mb_components"] = ge_adata_nmf.varm["nmf_standard_mb_components"]
print(f"Done merging NMF results!")

# Clean age data - remove 'm' and convert to integer
if 'age' in ge_adata.obs.columns:
    ge_adata.obs["age_numeric"] = ge_adata.obs["age"].astype(str).str.replace("m", "").astype(int)
    print(f"Age values: {ge_adata.obs['age_numeric'].unique()}")
print(f"Done cleaning age data!")

def test_aging_association_by_celltype(adata, factor_key, method_name="Factor", 
                                       celltype_col="broad_cell_type", min_cells=500, min_age_groups=2):
    """Test linear association between latent factors and age within each cell type"""
    if 'age_numeric' not in adata.obs.columns:
        print("No age_numeric column found")
        return None
    
    all_results = []
    valid_cell_types = []
    
    for ct in adata.obs[celltype_col].unique():
        mask = adata.obs[celltype_col] == ct
        sub_adata = adata[mask]
        
        # Filter by minimum cells and age groups
        if len(sub_adata) < min_cells:
            print(f"Skipping {ct}: only {len(sub_adata)} cells (< {min_cells})")
            continue
        if sub_adata.obs["age_numeric"].nunique() < min_age_groups:
            print(f"Skipping {ct}: only {sub_adata.obs['age_numeric'].nunique()} age groups (< {min_age_groups})")
            continue
            
        valid_cell_types.append(ct)
        print(f"Analyzing {ct}: {len(sub_adata)} cells, {sub_adata.obs['age_numeric'].nunique()} age groups")
        
        factors = pd.DataFrame(sub_adata.obsm[factor_key], index=sub_adata.obs.index)
        age = sub_adata.obs['age_numeric'].values
        
        for i, factor_col in enumerate(factors.columns):
            factor_vals = factors.iloc[:, i].values
            
            # Linear regression
            lr = LinearRegression()
            lr.fit(age.reshape(-1, 1), factor_vals)
            r_squared = lr.score(age.reshape(-1, 1), factor_vals)
            
            # Pearson correlation
            corr, p_val = stats.pearsonr(age, factor_vals)
            
            all_results.append({
                'cell_type': ct,
                'factor': f"{method_name}_{i}",
                'correlation': corr,
                'p_value': p_val,
                'r_squared': r_squared,
                'slope': lr.coef_[0],
                'n_cells': len(sub_adata),
                'n_age_groups': sub_adata.obs['age_numeric'].nunique()
            })
    
    print(f"Valid cell types analyzed: {valid_cell_types}")
    return pd.DataFrame(all_results)

def create_factor_loadings(adata):
    """Create factor loading DataFrames with informative headers"""
    gene_names = adata.var["gene_name"].values
    
    # NMF loadings
    nmf_loadings = pd.DataFrame(
        adata.varm["nmf_standard_mb_components"],
        index=gene_names,
        columns=[f"NMF_Factor_{i}" for i in range(adata.varm["nmf_standard_mb_components"].shape[1])]
    )
    
    # scVI loadings  
    scvi_loadings = pd.DataFrame(
        adata.varm["scVI_linear_gene_loadings"],
        index=gene_names,
        columns=[f"scVI_Latent_{i}" for i in range(adata.varm["scVI_linear_gene_loadings"].shape[1])]
    )
    
    return nmf_loadings, scvi_loadings

def plot_aging_associations_clustermaps(aging_results, output_dir, method_name, p_threshold=0.05):
    """Create clustermaps of aging association coefficients"""
    if aging_results is None or aging_results.empty:
        return
    
    # Create pivot tables - TRANSPOSED so factors are columns (easier to see global patterns)
    pivot_slope = aging_results.pivot(index='cell_type', columns='factor', values='slope')
    pivot_pval = aging_results.pivot(index='cell_type', columns='factor', values='p_value')
    pivot_corr = aging_results.pivot(index='cell_type', columns='factor', values='correlation')
    
    # Create significance mask
    sig_mask = pivot_pval <= p_threshold
    
    # 1. Clustermap of regression slopes (coefficients)
    plt.figure(figsize=(16, 10))
    
    # For clustering, fill non-significant values with 0 instead of NaN
    slope_for_clustering = pivot_slope.copy()
    slope_for_clustering[~sig_mask] = 0
    
    # Create display mask where non-significant values will appear white
    display_mask = ~sig_mask
    
    # Calculate vmax for symmetric colorbar using only significant values
    sig_values = pivot_slope[sig_mask].values
    sig_values = sig_values[~np.isnan(sig_values)]
    if len(sig_values) > 0:
        vmax = np.max(np.abs(sig_values))
    else:
        vmax = 1
    
    g1 = sns.clustermap(
        slope_for_clustering,
        cmap='RdBu_r',
        center=0,
        vmin=-vmax,
        vmax=vmax,
        figsize=(16, 10),
        cbar_kws={'label': 'Regression Coefficient (Slope)', 'shrink': 0.8},
        linewidths=0.5,
        mask=display_mask,
        dendrogram_ratio=0.15,
        cbar_pos=(0.02, 0.83, 0.03, 0.15)
    )
    
    g1.ax_heatmap.set_title(f'{method_name}: Age-Factor Regression Coefficients\n(Only p<{p_threshold} shown, Factors as columns)', 
                           fontsize=14, pad=20)
    g1.ax_heatmap.set_xlabel('Factors', fontsize=12)
    g1.ax_heatmap.set_ylabel('Cell Types', fontsize=12)
    
    plt.savefig(os.path.join(output_dir, f"{method_name.lower()}_aging_slopes_clustermap.pdf"), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Clustermap of correlations for comparison
    plt.figure(figsize=(16, 10))
    
    # For clustering, fill non-significant values with 0 instead of NaN
    corr_for_clustering = pivot_corr.copy()
    corr_for_clustering[~sig_mask] = 0
    
    g2 = sns.clustermap(
        corr_for_clustering,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        figsize=(16, 10),
        cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
        linewidths=0.5,
        mask=display_mask,
        dendrogram_ratio=0.15,
        cbar_pos=(0.02, 0.83, 0.03, 0.15)
    )
    
    g2.ax_heatmap.set_title(f'{method_name}: Age-Factor Correlations\n(Only p<{p_threshold} shown, Factors as columns)', 
                           fontsize=14, pad=20)
    g2.ax_heatmap.set_xlabel('Factors', fontsize=12)
    g2.ax_heatmap.set_ylabel('Cell Types', fontsize=12)
    
    plt.savefig(os.path.join(output_dir, f"{method_name.lower()}_aging_correlations_clustermap.pdf"), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Summary heatmap showing number of significant associations per factor
    factor_sig_counts = sig_mask.sum(axis=0).sort_values(ascending=False)  # Changed axis=0 for transposed data
    
    plt.figure(figsize=(max(8, len(factor_sig_counts) * 0.4), 6))
    
    # Create color map for bars based on values
    import matplotlib.cm as cm
    colors = cm.viridis(factor_sig_counts.values / factor_sig_counts.max())
    
    bars = plt.barh(range(len(factor_sig_counts)), factor_sig_counts.values, color=colors)
    plt.yticks(range(len(factor_sig_counts)), factor_sig_counts.index, fontsize=10)
    plt.xlabel('Number of Cell Types with Significant Age Association')
    plt.ylabel('Factors')
    plt.title(f'{method_name}: Factor Age Associations Across Cell Types')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, factor_sig_counts.values)):
        plt.text(count + 0.1, i, str(count), va='center', fontsize=9)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f"{method_name.lower()}_factor_significance_summary.pdf"), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return pivot_slope, pivot_corr, sig_mask

def identify_top_aging_factors(aging_results, method_name, n_top=3):
    """Identify top aging factors based on breadth and magnitude of effects"""
    if aging_results is None or aging_results.empty:
        return None
    
    # Calculate metrics per factor
    factor_metrics = aging_results.groupby('factor').agg({
        'p_value': lambda x: (x < 0.05).sum(),  # Number of significant cell types
        'slope': ['mean', 'std'],
        'correlation': ['mean', 'std'],
        'r_squared': 'mean'
    }).round(4)
    
    factor_metrics.columns = ['n_sig_celltypes', 'mean_slope', 'std_slope', 
                             'mean_corr', 'std_corr', 'mean_r2']
    
    # Calculate composite score: breadth * average magnitude
    factor_metrics['breadth_magnitude_score'] = (
        factor_metrics['n_sig_celltypes'] * np.abs(factor_metrics['mean_slope'])
    )
    
    # Sort by composite score
    top_factors = factor_metrics.sort_values('breadth_magnitude_score', ascending=False).head(n_top)
    
    print(f"\n{method_name} Top {n_top} Aging Factors (by breadth × magnitude):")
    print(top_factors)
    
    return top_factors

def plot_top_factor_by_age_groups(adata, factor_key, top_factor_name, method_name, output_dir,
                                 celltype_col="broad_cell_type", min_cells=100):
    """Create boxplots showing top aging factor activity across age groups by cell type"""
    
    if 'age_numeric' not in adata.obs.columns:
        print("No age_numeric column found")
        return
    
    # Extract factor index from name (e.g., "NMF_0" -> 0)
    try:
        factor_idx = int(top_factor_name.split('_')[-1])
    except:
        print(f"Could not parse factor index from {top_factor_name}")
        return
    
    # Get factor values
    factor_values = pd.DataFrame(adata.obsm[factor_key], index=adata.obs.index)
    
    # Create plotting dataframe
    plot_data = []
    valid_cell_types = []
    
    for ct in adata.obs[celltype_col].unique():
        mask = adata.obs[celltype_col] == ct
        sub_adata = adata[mask]
        
        if len(sub_adata) < min_cells:
            continue
            
        valid_cell_types.append(ct)
        
        for idx, cell in enumerate(sub_adata.obs.index):
            plot_data.append({
                'cell_type': ct,
                'age': sub_adata.obs.loc[cell, 'age_numeric'],
                'factor_value': factor_values.loc[cell, factor_idx],
                'age_group': f"{sub_adata.obs.loc[cell, 'age_numeric']}m"
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    if plot_df.empty:
        print("No data available for plotting")
        return
    
    # Create figure with subplots for each cell type
    n_celltypes = len(valid_cell_types)
    n_cols = min(4, n_celltypes)
    n_rows = (n_celltypes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_celltypes == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten() if n_celltypes > 1 else axes
    
    # Get age groups for consistent ordering
    age_groups = sorted(plot_df['age'].unique())
    age_group_labels = [f"{age}m" for age in age_groups]
    
    for i, ct in enumerate(valid_cell_types):
        ax = axes_flat[i]
        ct_data = plot_df[plot_df['cell_type'] == ct]
        
        # Create boxplot
        sns.boxplot(data=ct_data, x='age_group', y='factor_value', 
                   order=age_group_labels, ax=ax, palette='viridis')
        
        # Add trend line
        ages_numeric = ct_data['age'].values
        factor_vals = ct_data['factor_value'].values
        
        # Fit linear regression for trend
        lr = LinearRegression()
        lr.fit(ages_numeric.reshape(-1, 1), factor_vals)
        
        # Get correlation and p-value
        corr, p_val = stats.pearsonr(ages_numeric, factor_vals)
        
        # Plot trend line
        age_range = np.linspace(ages_numeric.min(), ages_numeric.max(), 100)
        trend_line = lr.predict(age_range.reshape(-1, 1))
        
        # Map age to x-axis positions for trend line
        age_positions = [age_group_labels.index(f"{age}m") for age in ages_numeric]
        ax2 = ax.twinx()
        ax2.plot(age_positions, factor_vals, 'o', alpha=0.3, color='red', markersize=1)
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks([])
        
        # Add statistics to title
        ax.set_title(f'{ct}\nr={corr:.3f}, p={p_val:.3e}\nslope={lr.coef_[0]:.4f}', 
                    fontsize=10)
        ax.set_xlabel('Age Group')
        ax.set_ylabel(f'{top_factor_name} Activity')
        ax.tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for i in range(n_celltypes, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.suptitle(f'{method_name} {top_factor_name}: Activity Across Age Groups by Cell Type', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, f"{method_name.lower()}_{top_factor_name.lower()}_age_boxplots.pdf"), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return plot_df

def identify_global_vs_specific_factors(aging_results, method_name, p_threshold=0.05):
    """Identify factors that are globally vs cell-type specifically associated with aging"""
    if aging_results is None or aging_results.empty:
        return None, None
    
    # Count significant associations per factor
    sig_results = aging_results[aging_results['p_value'] < p_threshold]
    factor_counts = sig_results.groupby('factor').agg({
        'cell_type': 'count',
        'correlation': ['mean', 'std'],
        'slope': ['mean', 'std']
    }).round(3)
    
    factor_counts.columns = ['n_sig_celltypes', 'mean_correlation', 'std_correlation', 'mean_slope', 'std_slope']
    factor_counts = factor_counts.sort_values('n_sig_celltypes', ascending=False)
    
    # Global factors: significant in multiple cell types with consistent direction
    global_factors = factor_counts[
        (factor_counts['n_sig_celltypes'] >= 3) & 
        (factor_counts['std_correlation'] < 0.2)
    ]
    
    # Cell-type specific: significant in 1-2 cell types
    specific_factors = factor_counts[factor_counts['n_sig_celltypes'] <= 2]
    
    print(f"\n{method_name} Global Aging Factors (consistent across ≥3 cell types):")
    print(global_factors)
    
    print(f"\n{method_name} Cell-Type Specific Aging Factors (≤2 cell types):")
    print(specific_factors.head(10))
    
    return global_factors, specific_factors

# Main analysis
print("Creating factor loadings...")
nmf_loadings, scvi_loadings = create_factor_loadings(ge_adata)

print("Testing aging associations by cell type...")
nmf_aging = test_aging_association_by_celltype(ge_adata, "X_nmf_standard_mb", "NMF")
scvi_aging = test_aging_association_by_celltype(ge_adata, "X_scVI_linear", "scVI")

# Create enhanced visualizations with clustermaps
if nmf_aging is not None:
    print("Creating NMF clustermaps...")
    nmf_slopes, nmf_corrs, nmf_sig_mask = plot_aging_associations_clustermaps(nmf_aging, OUTPUT_DIR, "NMF")
    nmf_global, nmf_specific = identify_global_vs_specific_factors(nmf_aging, "NMF")
    
    # Identify and plot top aging factors
    nmf_top_factors = identify_top_aging_factors(nmf_aging, "NMF")
    if nmf_top_factors is not None and len(nmf_top_factors) > 0:
        top_nmf_factor = nmf_top_factors.index[0]  # Get the name of the top factor
        print(f"Creating detailed plots for top NMF factor: {top_nmf_factor}")
        nmf_plot_data = plot_top_factor_by_age_groups(
            ge_adata, "X_nmf_standard_mb", top_nmf_factor, "NMF", OUTPUT_DIR
        )
    
if scvi_aging is not None:
    print("Creating scVI clustermaps...")
    scvi_slopes, scvi_corrs, scvi_sig_mask = plot_aging_associations_clustermaps(scvi_aging, OUTPUT_DIR, "scVI")
    scvi_global, scvi_specific = identify_global_vs_specific_factors(scvi_aging, "scVI")
    
    # Identify and plot top aging factors
    scvi_top_factors = identify_top_aging_factors(scvi_aging, "scVI")
    if scvi_top_factors is not None and len(scvi_top_factors) > 0:
        top_scvi_factor = scvi_top_factors.index[0]  # Get the name of the top factor
        print(f"Creating detailed plots for top scVI factor: {top_scvi_factor}")
        scvi_plot_data = plot_top_factor_by_age_groups(
            ge_adata, "X_scVI_linear", top_scvi_factor, "scVI", OUTPUT_DIR
        )

# Save results
print("Saving results...")
nmf_loadings.to_csv(os.path.join(OUTPUT_DIR, "nmf_loadings.tsv.gz"), sep="\t", compression="gzip")
scvi_loadings.to_csv(os.path.join(OUTPUT_DIR, "scvi_loadings.tsv.gz"), sep="\t", compression="gzip")

# Save detailed results by cell type
if nmf_aging is not None:
    nmf_aging.to_csv(os.path.join(OUTPUT_DIR, "nmf_aging_by_celltype.tsv"), sep="\t", index=False)
    # Save coefficient matrices
    if 'nmf_slopes' in locals():
        nmf_slopes.to_csv(os.path.join(OUTPUT_DIR, "nmf_aging_slopes_matrix.tsv"), sep="\t")
    # Save top factors analysis
    if 'nmf_top_factors' in locals() and nmf_top_factors is not None:
        nmf_top_factors.to_csv(os.path.join(OUTPUT_DIR, "nmf_top_aging_factors.tsv"), sep="\t")
    # Save boxplot data
    if 'nmf_plot_data' in locals():
        nmf_plot_data.to_csv(os.path.join(OUTPUT_DIR, "nmf_top_factor_age_data.tsv"), sep="\t", index=False)
        
if scvi_aging is not None:
    scvi_aging.to_csv(os.path.join(OUTPUT_DIR, "scvi_aging_by_celltype.tsv"), sep="\t", index=False)
    # Save coefficient matrices
    if 'scvi_slopes' in locals():
        scvi_slopes.to_csv(os.path.join(OUTPUT_DIR, "scvi_aging_slopes_matrix.tsv"), sep="\t")
    # Save top factors analysis
    if 'scvi_top_factors' in locals() and scvi_top_factors is not None:
        scvi_top_factors.to_csv(os.path.join(OUTPUT_DIR, "scvi_top_aging_factors.tsv"), sep="\t")
    # Save boxplot data
    if 'scvi_plot_data' in locals():
        scvi_plot_data.to_csv(os.path.join(OUTPUT_DIR, "scvi_top_factor_age_data.tsv"), sep="\t", index=False)

# Save global vs specific factor summaries
if nmf_aging is not None:
    nmf_global, nmf_specific = identify_global_vs_specific_factors(nmf_aging, "NMF")
    if nmf_global is not None:
        nmf_global.to_csv(os.path.join(OUTPUT_DIR, "nmf_global_aging_factors.tsv"), sep="\t")
    if nmf_specific is not None:
        nmf_specific.to_csv(os.path.join(OUTPUT_DIR, "nmf_specific_aging_factors.tsv"), sep="\t")

if scvi_aging is not None:
    scvi_global, scvi_specific = identify_global_vs_specific_factors(scvi_aging, "scVI")
    if scvi_global is not None:
        scvi_global.to_csv(os.path.join(OUTPUT_DIR, "scvi_global_aging_factors.tsv"), sep="\t")
    if scvi_specific is not None:
        scvi_specific.to_csv(os.path.join(OUTPUT_DIR, "scvi_specific_aging_factors.tsv"), sep="\t")

print(f"Analysis complete. Enhanced visualizations saved to {OUTPUT_DIR}")

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION
# script=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/GeneExpression/11_GE_vs_AGING.py
# sbatch --mem=350G -p cpu,bigmem -J "MUS_GE_vs_AGING" --wrap="python $script"