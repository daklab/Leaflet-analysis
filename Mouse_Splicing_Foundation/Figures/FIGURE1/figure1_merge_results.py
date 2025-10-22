#!/usr/bin/env python
"""
FIGURE 1 - STEP 3: Merge results and create final visualizations
Combines all cell type results and generates figures
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
from datetime import datetime

# Plotting configuration
sns.set_theme(context="notebook", style="white")
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 13,
    'axes.labelsize': 13,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'figure.titlesize': 12,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'axes.grid': False,
})


def plot_global_aging_effects(adata, output_dir, today):
    """Show overall distribution of age-related splicing changes"""
    young_mask = adata.obs["age_group"] == "young"
    old_mask = adata.obs["age_group"] == "old"
    
    psi_young = np.nanmean(adata[young_mask].layers["psi"].toarray(), axis=0)
    psi_old = np.nanmean(adata[old_mask].layers["psi"].toarray(), axis=0)
    delta_psi_global = psi_old - psi_young
    delta_psi_global = delta_psi_global[~np.isnan(delta_psi_global)]
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.hist(delta_psi_global, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, label='No change')
    ax.axvline(0.05, color='red', linestyle=':', alpha=0.5)
    ax.axvline(-0.05, color='red', linestyle=':', alpha=0.5)
    
    n_increased = np.sum(delta_psi_global > 0.05)
    n_decreased = np.sum(delta_psi_global < -0.05)
    ax.text(0.02, 0.95, f"Increased (ΔPSI > 0.05): {n_increased:,}",
            transform=ax.transAxes, fontsize=10)
    ax.text(0.02, 0.90, f"Decreased (ΔPSI < -0.05): {n_decreased:,}",
            transform=ax.transAxes, fontsize=10)
    
    ax.set_xlabel("ΔPSI (Old - Young)")
    ax.set_ylabel("Number of Junctions")
    ax.set_title("Global Age-Related Splicing Changes")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig1b_global_delta_psi_{today}.pdf", bbox_inches="tight")
    plt.close()
    
    return delta_psi_global


def plot_celltype_specific_changes(df, output_dir, today, top_n=30):
    """Visualize cell type-specific splicing changes"""
    summary = df.groupby('cell_type').agg({
        'delta_psi': ['mean', 'median', lambda x: (x.abs() > 0.05).sum()],
        'n_cells_young': 'first',
        'n_cells_old': 'first'
    }).reset_index()
    
    summary.columns = ['cell_type', 'mean_dpsi', 'median_dpsi', 'n_changed',
                       'n_cells_young', 'n_cells_old']
    summary['total_cells'] = summary['n_cells_young'] + summary['n_cells_old']
    
    summary_top = summary.nlargest(top_n, 'total_cells')
    summary_top = summary_top.sort_values('n_changed', ascending=False)
    
    print(f"Plotting top {top_n} cell types out of {len(summary)} total")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, ct in enumerate(summary_top['cell_type']):
        ct_data = df[df['cell_type'] == ct]['delta_psi']
        y_pos = [i] * len(ct_data)
        colors = ['#2ecc71' if d > 0.05 else '#9b59b6' if d < -0.05 else 'lightgray'
                  for d in ct_data]
        ax.scatter(ct_data, y_pos, alpha=0.4, s=2, c=colors, rasterized=True)
    
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.8, label='No change')
    ax.axvline(0.05, color='#2ecc71', linestyle=':', alpha=0.7, linewidth=1.5,
               label='ΔPSI > 0.05')
    ax.axvline(-0.05, color='#9b59b6', linestyle=':', alpha=0.7, linewidth=1.5,
               label='ΔPSI < -0.05')
    
    labels = [f"{ct} (Δ={n:,}, cells={n_cells:,})"
              for ct, n, n_cells in zip(summary_top['cell_type'],
                                       summary_top['n_changed'],
                                       summary_top['total_cells'])]
    
    ax.set_yticks(range(len(summary_top)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("ΔPSI (Old - Young)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Cell Type", fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig1c_celltype_delta_psi_top{top_n}_{today}.pdf",
                bbox_inches="tight", dpi=300)
    plt.close()
    
    return summary


def make_plot_input_from_junction_df(junction_df, adata, groupby_col="tissue_celltype",
                                     metric="delta_pb", young_label="young", old_label="old"):
    """Create plotting input from junction DataFrame"""
    if junction_df.empty:
        return pd.DataFrame(columns=["cell_type", "delta_psi", "n_cells_young", "n_cells_old"])
    if metric not in junction_df.columns:
        raise ValueError(f"metric '{metric}' not found in junction_df")
    
    ct_age = adata.obs.groupby([groupby_col, "age_group"]).size().unstack(fill_value=0)
    ct_sizes = ct_age.rename(columns={young_label: "n_cells_young", old_label: "n_cells_old"})
    ct_sizes = ct_sizes[["n_cells_young", "n_cells_old"]].reset_index().rename(
        columns={groupby_col: "cell_type"}
    )
    
    df = junction_df[["cell_type", metric]].rename(columns={metric: "delta_psi"}).copy()
    df = df.merge(ct_sizes, on="cell_type", how="left")
    return df


def main():
    today = datetime.now().strftime("%Y%m%d")
    output_dir = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/FIGURES/FIGURE1"
    run_dir = f"{output_dir}/figure1_20251019"
    intermediate_dir = f"{run_dir}/intermediate"
    
    print(f"\n{'='*60}")
    print("MERGING RESULTS FROM ALL CELL TYPES")
    print(f"{'='*60}\n")
    
    # Find all intermediate junction files
    junction_files = glob.glob(f"{intermediate_dir}/junction_*.csv")
    atse_files = glob.glob(f"{intermediate_dir}/atse_*.csv")
    
    print(f"Found {len(junction_files)} junction files")
    print(f"Found {len(atse_files)} ATSE files")
    
    if len(junction_files) == 0:
        print("ERROR: No junction files found!")
        return
    
    # Merge all junction results
    print("\nMerging junction files...")
    all_junction_dfs = []
    for f in junction_files:
        df = pd.read_csv(f)
        all_junction_dfs.append(df)
        print(f"  Loaded {len(df):,} rows from {os.path.basename(f)}")
    
    junction_df_all = pd.concat(all_junction_dfs, ignore_index=True)
    print(f"\nTotal junctions: {len(junction_df_all):,}")
    
    # Merge all ATSE results
    print("\nMerging ATSE files...")
    all_atse_dfs = []
    for f in atse_files:
        df = pd.read_csv(f)
        all_atse_dfs.append(df)
    
    atse_df_all = pd.concat(all_atse_dfs, ignore_index=True)
    print(f"Total ATSEs: {len(atse_df_all):,}")
    
    # Load preprocessed data for plotting
    print("\nLoading preprocessed data for plotting...")
    splice_adata = ad.read_h5ad(f"{run_dir}/preprocessed_data_20251019.h5ad")
    
    # Generate global aging effects plot
    print("\nGenerating global aging effects plot...")
    delta_psi_global = plot_global_aging_effects(splice_adata, run_dir, today)
    
    # Create cell type-specific plot
    print("\nGenerating cell type-specific plot...")
    plot_df = make_plot_input_from_junction_df(
        junction_df_all, splice_adata,
        groupby_col="tissue_celltype",
        metric="delta_psi_mean"
    )
    
    summary = plot_celltype_specific_changes(plot_df, run_dir, today, top_n=30)
    
    # Save final merged results
    print("\nSaving final results...")
    plot_df.to_csv(f"{run_dir}/celltype_delta_psi_{today}.csv", index=False)
    junction_df_all.to_csv(f"{run_dir}/junction_df_all_{today}.csv", index=False)
    atse_df_all.to_csv(f"{run_dir}/atse_df_all_{today}.csv", index=False)
    summary.to_csv(f"{run_dir}/celltype_summary_{today}.csv", index=False)
    
    print(f"\nSaved final results:")
    print(f"  {run_dir}/celltype_delta_psi_{today}.csv")
    print(f"  {run_dir}/junction_df_all_{today}.csv ({len(junction_df_all):,} rows)")
    print(f"  {run_dir}/atse_df_all_{today}.csv ({len(atse_df_all):,} rows)")
    print(f"  {run_dir}/celltype_summary_{today}.csv")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total cell types processed: {junction_df_all['cell_type'].nunique()}")
    print(f"Total junctions analyzed: {len(junction_df_all):,}")
    print(f"Total ATSEs analyzed: {len(atse_df_all):,}")
    print(f"Junctions with |ΔPSI| > 0.05: {(junction_df_all['delta_psi_mean'].abs() > 0.05).sum():,}")
    print(f"\nAnalysis complete!")


if __name__ == "__main__":
    main()