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
DATE_RESULTS_TO_SUMMARIZE = "2025-05-21"
BASE_RESULTS_DIR = "/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/results/"
MODEL_CONFIG_MAPPING_FILE = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel/2025-05-13/parameter_combinations.csv"

OUTPUT_SUMMARY_DIR = os.path.join(BASE_RESULTS_DIR, "comparison_summary_v1_" + pd.Timestamp.now().strftime("%Y%m%d"))
os.makedirs(OUTPUT_SUMMARY_DIR, exist_ok=True)

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

    # Generate a sorted boxplot for each metric by prior_combo
    for metric in metrics_to_plot:
        print(f"Plotting {metric} by model parameters...")
        sorted_combos = summary_df.groupby("prior_combo")[metric].median().sort_values(ascending=False).index
        plt.figure(figsize=(8, 6))
        sns.boxplot(
            data=summary_df, x="prior_combo", y=metric,
            order=sorted_combos
        )
        sns.stripplot(
            data=summary_df, x="prior_combo", y=metric,
            order=sorted_combos,
            color="black", size=5, jitter=True
        )
        plt.xticks(rotation=50, ha="right", fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(f"{metric.replace('_', ' ').title()} by Model Design Combo")
        plt.xlabel("Model parameters used", fontsize=14)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_SUMMARY_DIR, f"{metric}_by_model_combo.png"))
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
    plt.savefig(os.path.join(OUTPUT_SUMMARY_DIR, "model_performance_heatmap_by_metric.png"))
    plt.close()

    # save summary_df
    summary_df.to_csv(os.path.join(OUTPUT_SUMMARY_DIR, "summary_metrics_across_models.csv"), index=False)
if __name__ == "__main__":
    main()
