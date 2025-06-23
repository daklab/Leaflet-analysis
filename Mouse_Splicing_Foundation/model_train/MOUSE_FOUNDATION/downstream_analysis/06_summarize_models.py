#!/usr/bin/env python
"""
LeafletFA Model Summary and Comparison (Initial Version)
"""

import os
import glob
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler

# --- Configuration ---
#MODEL_CONFIG_MAPPING_FILE = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel/2025-05-13/parameter_combinations.csv"
#DATE_RESULTS_TO_SUMMARIZE = "2025-05-13/2025-06-14" #date model was trained/date models were summarized for plotting 
#BASE_RESULTS_DIR = f"/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/results/{DATE_RESULTS_TO_SUMMARIZE}"

MODEL_CONFIG_MAPPING_FILE = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/Leaflet/leafletFAmodel/2025-06-11/parameter_combinations.csv"
DATE_RESULTS_TO_SUMMARIZE = "2025-06-11/2025-06-14" #date model was trained/date models were summarized for plotting 
BASE_RESULTS_DIR = f"/gpfs/commons/home/kisaev/Leaflet-analysis/Human_Splicing_Foundation/model_train/HUMAN_FOUNDATION/results/{DATE_RESULTS_TO_SUMMARIZE}"

OUTPUT_SUMMARY_DIR = os.path.join(BASE_RESULTS_DIR, "comparison_summary" + pd.Timestamp.now().strftime("%Y%m%d"))
os.makedirs(OUTPUT_SUMMARY_DIR, exist_ok=True)

def analyze_factor_deltas_by_cell_type(summary_df, cell_type_col="cell_type_grouped"):
    """
    Analyze factor deltas between young and old samples within each cell type.
    
    Parameters:
    - summary_df: DataFrame containing factor activities and metadata
    - cell_type_col: Column name containing cell type information
    
    Returns:
    - DataFrame with delta values and statistics
    """
    # Create a pivot table for delta_median and avg_factor_activity
    delta_median_pivot = summary_df.pivot(index=cell_type_col, columns='factor', values='delta_median')
    avg_activity_pivot = summary_df.pivot(index=cell_type_col, columns='factor', values='avg_factor_activity')
    
    # Create figure for two side-by-side heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot delta median heatmap
    sns.heatmap(delta_median_pivot, cmap="seismic", ax=ax1, cbar=True, 
                annot=False, center=0, vmin=-0.05, vmax=0.05, xticklabels=True, yticklabels=True)
    
    # Plot average activity heatmap
    sns.heatmap(avg_activity_pivot, cmap="coolwarm", ax=ax2, cbar=True, 
                annot=False, center=0, xticklabels=True, yticklabels=False)
    
    # Set titles and labels
    ax1.set_title('Delta Median Factor Activity (Old - Young)', fontsize=10)
    ax2.set_title('Average Factor Activity in Cell Type', fontsize=10)
    
    # Set axis labels
    ax1.set_xticklabels(delta_median_pivot.columns, rotation=90, fontsize=10)
    ax1.set_yticklabels(delta_median_pivot.index, rotation=0, fontsize=10)
    ax1.set_ylabel("Cell Type")
    ax1.set_xlabel("Factor")
    
    ax2.set_xticklabels(avg_activity_pivot.columns, rotation=90, fontsize=10)
    ax2.set_ylabel("Cell Type")
    ax2.set_xlabel("Factor")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    current_date = pd.Timestamp.now().strftime("%Y%m%d")
    plt.savefig(os.path.join(OUTPUT_SUMMARY_DIR, f"factor_deltas_by_cell_type_{current_date}.pdf"))
    plt.close()
    
    return delta_median_pivot, avg_activity_pivot

def main():
    summary_df = pd.read_csv(os.path.join(OUTPUT_SUMMARY_DIR, "summary_metrics_across_models.csv"))
    # Replace NaN in delta_fixed with "learned"
    summary_df["delta_fixed"] = summary_df["delta_fixed"].fillna("learned")
    # Create a combo string to group model design choices
    summary_df["prior_combo"] = (
        summary_df["likelihood_type"].astype(str) + "_" +
        summary_df["delta_fixed"].astype(str) + "_" +
        summary_df["junc_specific_prior"].astype(str)
    )

    # List of key metrics to plot across prior_combo
    metrics_to_plot = [
        "range_pi", "alpha_pi",
        "aging_r2__tissue_and_splicing_factors",
        "aging_r2__tissue_splicing_and_nmf_ge_factors",
        "celltype_classification_accuracy",
        "diversity_mean",
        "n_dataset_var_above_0.5",
        "n_celltype_var_above_0.5",
        "n_tissue_var_above_0.5",
        "diversity_median",
        "diversity_mean", # diversity across all factors in terms of three main features how variance across them is distributed 
        "max_dataset_var",
        "max_celltype_var",
        "max_tissue_var",
        "global_r2_max", # from global anova analysis maximum value across factors 
        "median_corr_imputed_vs_observed",
        "median_cell_perplexity"
    ]

    # Generate a sorted barplot for each metric by prior_combo
    for metric in metrics_to_plot:
        print(f"Plotting {metric} by model parameters...")
        sorted_combos = summary_df.groupby("prior_combo")[metric].median().sort_values(ascending=False).index
        plt.figure(figsize=(6, 6))
        sns.barplot(
            data=summary_df, x="prior_combo", y=metric,
            order=sorted_combos,
            color='grey'
        )
        plt.xticks(rotation=50, ha="right", fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(f"{metric.replace('_', ' ').title()} by Model Design Combo")
        plt.xlabel("Model parameters used", fontsize=14)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_SUMMARY_DIR, f"{metric}_by_model_combo.pdf"))
        plt.close()

    # --- Heatmap of Metrics by prior_combo ---
    sns.set(font_scale=1.5)

    # Normalize across each metric (row-wise normalization)
    heatmap_df = summary_df.groupby("prior_combo")[metrics_to_plot].median().T
    scaler = MinMaxScaler()
    heatmap_df_scaled = pd.DataFrame(
        scaler.fit_transform(heatmap_df.T).T,
        index=heatmap_df.index,
        columns=heatmap_df.columns
    )

    # Plot clustermap (metrics as rows, prior_combo as columns)
    sns.clustermap(heatmap_df_scaled, annot=False, 
        cmap="YlGnBu", yticklabels=True, xticklabels=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_SUMMARY_DIR, "model_performance_heatmap_by_metric.pdf"))
    plt.close()

    # save summary_df
    summary_df.to_csv(os.path.join(OUTPUT_SUMMARY_DIR, "summary_metrics_across_models.csv"), index=False)
if __name__ == "__main__":
    main()
