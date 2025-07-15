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

# --- Configuration ---
#MODEL_CONFIG_MAPPING_FILE = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel/2025-07-06/parameter_combinations.csv"
#DATE_RESULTS_TO_SUMMARIZE = "2025-07-06/2025-07-07" #date model was trained/date models were summarized for plotting 
#BASE_RESULTS_DIR = f"/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/results/{DATE_RESULTS_TO_SUMMARIZE}"

MODEL_CONFIG_MAPPING_FILE = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/Leaflet/leafletFAmodel/2025-07-06/parameter_combinations.csv"
DATE_RESULTS_TO_SUMMARIZE = "2025-07-06/2025-07-07" #date model was trained/date models were summarized for plotting 
BASE_RESULTS_DIR = f"/gpfs/commons/home/kisaev/Leaflet-analysis/Human_Splicing_Foundation/model_train/HUMAN_FOUNDATION/results/{DATE_RESULTS_TO_SUMMARIZE}"

OUTPUT_SUMMARY_DIR = os.path.join(BASE_RESULTS_DIR, "comparison_summary" + pd.Timestamp.now().strftime("%Y%m%d"))
os.makedirs(OUTPUT_SUMMARY_DIR, exist_ok=True)

def main():
    model_params_df = pd.read_csv(MODEL_CONFIG_MAPPING_FILE)
    model_params_df["likelihood_type"] = "Beta-Binomial"
    model_params_df.loc[model_params_df["input_conc"] == np.inf, "likelihood_type"] = "Binomial"
    print(model_params_df.head())
    model_output_dirs = glob.glob(os.path.join(BASE_RESULTS_DIR, "*"))
    # only keep the directories that contain "param_id" in the name
    model_output_dirs = [d for d in model_output_dirs if "param_id" in os.path.basename(d)]
    print(f"Found {len(model_output_dirs)} model output directories")

    model_summaries = []
    summary_table_rows = []

    for model_output_dir in model_output_dirs:
        param_id = os.path.basename(model_output_dir).split("_")[2]
        data_dir = os.path.join(model_output_dir, "data")

        model_params = model_params_df.iloc[int(param_id)]

        learned_params_path = os.path.join(data_dir, "model_parameters.csv")
        learned_params_df = pd.read_csv(learned_params_path)

        pi = np.load(os.path.join(data_dir, "PI_values.npy"))
        
        perplexity_df = pd.read_csv(os.path.join(data_dir, "median_cell_perplexity.csv"))

        #global_var_r2_df = pd.read_csv(os.path.join(data_dir, "variance_explained_r_squared.csv"))
        aging_r2 = pd.read_csv(os.path.join(data_dir, "age_prediction_results.csv")).iloc[1:].T
        aging_r2.columns = ["r2", "mse"]
        aging_r2["features"] = aging_r2.index
        aging_r2 = aging_r2.reset_index(drop=True)

        cell_type_classification_df = pd.read_csv(os.path.join(data_dir, "broad_cell_type_prediction_metrics_factors_only.csv"))

        # Variance components
        vc_df = pd.read_csv(os.path.join(data_dir, "variance_components_by_dataset.csv"))
        vc_df = vc_df.rename(columns={"Explained_Variance": "dataset_variance_explained"})[["Factor", "PI", "dataset_variance_explained"]]
        vc_cell = pd.read_csv(os.path.join(data_dir, "variance_components_by_broad_cell_type.csv"))
        vc_cell = vc_cell.rename(columns={"Explained_Variance": "cell_type_variance_explained"})[["Factor", "PI", "cell_type_variance_explained"]]
        vc_tissue = pd.read_csv(os.path.join(data_dir, "variance_components_by_tissue.csv"))
        vc_tissue = vc_tissue.rename(columns={"Explained_Variance": "tissue_variance_explained"})[["Factor", "PI", "tissue_variance_explained"]]

        vc_df = vc_df.merge(vc_cell, on=["Factor", "PI"]).merge(vc_tissue, on=["Factor", "PI"])
       
        # Get min, max and median PI value as well 
        range_pi = vc_df["PI"].max() - vc_df["PI"].min()
        median_pi = vc_df["PI"].median()

        # Variance summaries
        mean_dataset_var = vc_df["dataset_variance_explained"].mean()
        max_dataset_var = vc_df["dataset_variance_explained"].max()
        top3_dataset_var = vc_df.nlargest(3, "dataset_variance_explained")["dataset_variance_explained"].mean()
        n_dataset_var_above_0_5 = (vc_df["dataset_variance_explained"] > 0.5).sum()

        mean_celltype_var = vc_df["cell_type_variance_explained"].mean()
        max_celltype_var = vc_df["cell_type_variance_explained"].max()
        top3_celltype_var = vc_df.nlargest(3, "cell_type_variance_explained")["cell_type_variance_explained"].mean()
        n_celltype_var_above_0_5 = (vc_df["cell_type_variance_explained"] > 0.5).sum()

        mean_tissue_var = vc_df["tissue_variance_explained"].mean()
        max_tissue_var = vc_df["tissue_variance_explained"].max()
        top3_tissue_var = vc_df.nlargest(3, "tissue_variance_explained")["tissue_variance_explained"].mean()
        n_tissue_var_above_0_5 = (vc_df["tissue_variance_explained"] > 0.5).sum()

        var_fractions = vc_df[["dataset_variance_explained", "cell_type_variance_explained", "tissue_variance_explained"]].div(
            vc_df[["dataset_variance_explained", "cell_type_variance_explained", "tissue_variance_explained"]].sum(axis=1), axis=0)
        vc_df["variance_diversity"] = var_fractions.apply(lambda row: entropy(row.values), axis=1)

        diversity_mean = vc_df["variance_diversity"].mean()
        diversity_median = vc_df["variance_diversity"].median()
        diversity_min = vc_df["variance_diversity"].min()
        diversity_max = vc_df["variance_diversity"].max()

        #global_r2_median = global_var_r2_df["r2_overall"].median()
        #global_r2_max = global_var_r2_df["r2_overall"].max()

        sanity_check_df = pd.read_csv(os.path.join(data_dir, "PSI_imputed_vs_observed_correlations_summary.txt"))
        sanity_check_dict = {row.iloc[0].split(":")[0].strip(): float(row.iloc[0].split(":")[1].strip()) for _, row in sanity_check_df.iterrows()}
        sanity_check_df = pd.DataFrame([sanity_check_dict])

        aging_metrics_dict = {}
        for _, row in aging_r2.iterrows():
            key_prefix = row["features"].lower().replace(" ", "_")
            aging_metrics_dict[f"aging_r2__{key_prefix}"] = float(row["r2"])
            aging_metrics_dict[f"aging_mse__{key_prefix}"] = float(row["mse"])

        summary_row = {
            "param_id": param_id,
            "lr": model_params.get("lr", np.nan), 
            "K": model_params.get("K", np.nan),
            "new_K":len(pi), 
            "gamma": model_params.get("gamma", np.nan), 
            "range_pi": range_pi,
            "median_pi": median_pi,
            "delta_fixed": model_params.get("delta_fixed", np.nan),  # now this is used as input dir_conc
            "junc_specific_prior": model_params.get("junc_specific_prior", np.nan),
            "alpha_pi": learned_params_df.loc[0, "alpha_pi"] if "alpha_pi" in learned_params_df.columns else np.nan,
            "bb_conc": learned_params_df.loc[0, "bb_conc"] if "bb_conc" in learned_params_df.columns else np.nan,
            "learned_dir_conc": learned_params_df.loc[0, "dir_conc"] if "dir_conc" in learned_params_df.columns else np.nan,
            "likelihood_type": model_params.get("likelihood_type", "unknown"),
            "median_cell_perplexity": perplexity_df["median_cell_perplexity"].values[0],
            "celltype_classification_accuracy": float(cell_type_classification_df["accuracy"].values[0]),
      #      "global_r2_median": global_r2_median,
      #      "global_r2_max": global_r2_max,
            "mean_dataset_var": mean_dataset_var,
            "max_dataset_var": max_dataset_var,
            "top3_dataset_var": top3_dataset_var,
            "n_dataset_var_above_0.5": n_dataset_var_above_0_5,
            "mean_celltype_var": mean_celltype_var,
            "max_celltype_var": max_celltype_var,
            "top3_celltype_var": top3_celltype_var,
            "n_celltype_var_above_0.5": n_celltype_var_above_0_5,
            "mean_tissue_var": mean_tissue_var,
            "max_tissue_var": max_tissue_var,
            "top3_tissue_var": top3_tissue_var,
            "n_tissue_var_above_0.5": n_tissue_var_above_0_5,
            "diversity_mean": diversity_mean,
            "diversity_median": diversity_median,
            "diversity_min": diversity_min,
            "diversity_max": diversity_max,
            "mean_corr_imputed_vs_observed": sanity_check_df["mean_corr"].values[0],
            "std_corr_imputed_vs_observed": sanity_check_df["std_corr"].values[0],
            "median_corr_imputed_vs_observed": sanity_check_df["median_corr"].values[0],
        }

        summary_row.update(aging_metrics_dict)
        summary_table_rows.append(summary_row)
        model_summaries.append({"param_id": param_id})

    summary_df = pd.DataFrame(summary_table_rows)
    summary_df.to_csv(os.path.join(OUTPUT_SUMMARY_DIR, "summary_metrics_across_models.csv"), index=False)

    print(f"Saved summary to {OUTPUT_SUMMARY_DIR}/summary_metrics_across_models.csv")
    print(f"Number of models summarized: {len(model_summaries)}")

if __name__ == "__main__":
    main()