import mudata as mu
import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.decomposition import MiniBatchNMF
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import csr_matrix
import pickle
import os

MAX_JUNCTIONS = 5
NMF_COMPONENTS_LIST = [20, 30, 50]  # Test multiple component values
NMF_BATCH_SIZE = 2048
anndata_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/splicing_adata_train_20251026_145221.h5ad"
BASE_TEST = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/"
TEST_PATHS = [BASE_TEST + "MASKED_25_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_sparsity_fixed.h5mu",
              BASE_TEST + "MASKED_50_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_sparsity_fixed.h5mu",
              BASE_TEST + "MASKED_75_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_sparsity_fixed.h5mu"]

# Calculate metrics function
def calculate_metrics(true_values, predicted_values):
    """Calculate various metrics comparing true and predicted values."""
    
    # Remove any NaN or infinite values
    valid_mask = ~(np.isnan(true_values) | np.isnan(predicted_values) | 
                   np.isinf(true_values) | np.isinf(predicted_values))
    
    true_clean = true_values[valid_mask]
    pred_clean = predicted_values[valid_mask]
    
    if len(true_clean) == 0:
        print("No valid values to compare!")
        return None
    
    metrics = {}
    
    # Pearson correlation
    if len(true_clean) > 1:
        pearson_corr, pearson_p = pearsonr(true_clean, pred_clean)
        metrics['pearson_r'] = pearson_corr
        metrics['pearson_p'] = pearson_p
    else:
        metrics['pearson_r'] = np.nan
        metrics['pearson_p'] = np.nan
    
    # Spearman correlation
    if len(true_clean) > 1:
        spearman_corr, spearman_p = spearmanr(true_clean, pred_clean)
        metrics['spearman_r'] = spearman_corr
        metrics['spearman_p'] = spearman_p
    else:
        metrics['spearman_r'] = np.nan
        metrics['spearman_p'] = np.nan
    
    # L1 (Mean Absolute Error)
    metrics['l1_mae'] = mean_absolute_error(true_clean, pred_clean)
    
    # RMSE (Root Mean Squared Error)
    metrics['rmse'] = np.sqrt(mean_squared_error(true_clean, pred_clean))
    
    return metrics

# Read in the training data
print("Loading training data...")
full_ad = ad.read_h5ad(anndata_file)
full_ad = full_ad[:, full_ad.var["num_junctions"] <= MAX_JUNCTIONS].copy()
X_train = full_ad.layers["junc_ratio"]
print(f"Training data shape: {X_train.shape}")

# Initialize results collection
all_results = []

# Loop through different NMF component values
for n_components in NMF_COMPONENTS_LIST:
    print(f"\n{'='*60}")
    print(f"Training NMF model with {n_components} components...")
    print(f"{'='*60}")
    
    # Train NMF model
    model = MiniBatchNMF(n_components=n_components, init='nndsvda', batch_size=NMF_BATCH_SIZE, random_state=42)
    
    # Fit model on training data
    W = model.fit_transform(X_train)
    H = model.components_
    print(f"Done training NMF model with {n_components} components")
    
    # Loop through each test dataset
    for TEST_PATH in TEST_PATHS:
        # Extract mask percentage from filename
        if "25_PERCENT" in TEST_PATH:
            mask_percent = "25%"
        elif "50_PERCENT" in TEST_PATH:
            mask_percent = "50%"
        elif "75_PERCENT" in TEST_PATH:
            mask_percent = "75%"
        else:
            mask_percent = "unknown"
            
        print(f"\nProcessing {mask_percent} masked data...")
        
        # Load test data
        masked_ad = ad.read_h5ad(TEST_PATH)
        masked_ad = masked_ad[:, masked_ad.var["num_junctions"] <= MAX_JUNCTIONS].copy()
        print(f"Test data shape: {masked_ad.shape}")
        
        X_test = masked_ad.layers["junc_ratio"]  # Keep as sparse for transform
        
        # Transform test data
        W_test = model.transform(X_test)
        print(f"Done transforming test data")
        
        # Reconstruct
        X_reconstructed = W_test @ H
        print(f"Done reconstructing test data")
        
        # Get the mask and original values for this specific test dataset
        mask = masked_ad.layers["junc_ratio_masked_bin_mask"]
        original_values = masked_ad.layers["junc_ratio_masked_original"]
        
        # Convert to dense if needed for easier manipulation
        if hasattr(mask, 'toarray'):
            mask_dense = mask.toarray().astype(bool)
        else:
            mask_dense = mask.astype(bool)
        
        if hasattr(original_values, 'toarray'):
            original_dense = original_values.toarray()
        else:
            original_dense = original_values
        
        # Extract only the masked positions for comparison
        # The mask indicates which values were masked (1 = masked, 0 = not masked)
        masked_positions = mask_dense == 1
        
        # Get original and imputed values at masked positions
        original_masked_values = original_dense[masked_positions]
        imputed_masked_values = X_reconstructed[masked_positions]
        
        # Filter out zero values from original (since these weren't actually masked)
        non_zero_mask = original_masked_values != 0
        original_masked_values_filtered = original_masked_values[non_zero_mask]
        imputed_masked_values_filtered = imputed_masked_values[non_zero_mask]
        
        print(f"Number of masked values to evaluate: {len(original_masked_values_filtered)}")
        
        # Calculate metrics for this combination
        metrics = calculate_metrics(original_masked_values_filtered, imputed_masked_values_filtered)
        
        if metrics:
            # Add metadata to metrics
            result_row = {
                'n_components': n_components,
                'mask_percent': mask_percent,
                'test_path': os.path.basename(TEST_PATH),
                'n_masked_values': len(original_masked_values_filtered),
                **metrics
            }
            all_results.append(result_row)
            
            print(f"\nMetrics for {n_components} components, {mask_percent} masking:")
            print(f"  Pearson r: {metrics['pearson_r']:.4f}")
            print(f"  Spearman r: {metrics['spearman_r']:.4f}")
            print(f"  MAE: {metrics['l1_mae']:.6f}")
            print(f"  RMSE: {metrics['rmse']:.6f}")
        else:
            print(f"Failed to calculate metrics for {n_components} components, {mask_percent} masking")

# Convert results to DataFrame and save
print(f"\n{'='*60}")
print("FINAL RESULTS SUMMARY")
print(f"{'='*60}")

if all_results:
    results_df = pd.DataFrame(all_results)
    
    # Reorder columns for better readability
    column_order = ['n_components', 'mask_percent', 'n_masked_values', 
                   'pearson_r', 'pearson_p', 'spearman_r', 'spearman_p', 
                   'l1_mae', 'rmse', 'test_path']
    results_df = results_df[column_order]
    
    # Display summary
    print(f"\nTotal experiments completed: {len(results_df)}")
    print(f"Components tested: {sorted(results_df['n_components'].unique())}")
    print(f"Mask percentages tested: {sorted(results_df['mask_percent'].unique())}")
    
    # Show the results table
    print(f"\nDetailed Results:")
    print(results_df.to_string(index=False, float_format='%.6f'))
    
    # Save results to multiple formats
    results_df.to_csv('nmf_imputation_results_comprehensive.csv', index=False)
    results_df.to_pickle('nmf_imputation_results_comprehensive.pkl')
    
    # Also save as a more readable summary
    summary_df = results_df[['n_components', 'mask_percent', 'pearson_r', 'spearman_r', 'l1_mae', 'rmse']].copy()
    summary_df.to_csv('nmf_imputation_results_summary.csv', index=False)
    
    print(f"\nResults saved to:")
    print(f"  - nmf_imputation_results_comprehensive.csv (full results)")
    print(f"  - nmf_imputation_results_comprehensive.pkl (full results, pickle)")
    print(f"  - nmf_imputation_results_summary.csv (key metrics only)")
    
    # Print best performing configurations
    print(f"\nBest performing configurations:")
    for metric in ['pearson_r', 'spearman_r']:
        best_idx = results_df[metric].idxmax()
        best_row = results_df.loc[best_idx]
        print(f"  Best {metric}: {best_row[metric]:.4f} "
              f"(components={best_row['n_components']}, mask={best_row['mask_percent']})")
    
    for metric in ['l1_mae', 'rmse']:
        best_idx = results_df[metric].idxmin()
        best_row = results_df.loc[best_idx]
        print(f"  Best {metric}: {best_row[metric]:.6f} "
              f"(components={best_row['n_components']}, mask={best_row['mask_percent']})")
        
else:
    print("No results were generated. Check for errors in the processing.")

print(f"\n{'='*60}")
print("ANALYSIS COMPLETE")
print(f"{'='*60}")

"""
#Submit with:
conda activate LeafletSC
script=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/Figures/FIGURE2/02_NMF_baseline.py
cd /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/Figures/FIGURE2/
sbatch --job-name=nmf_baseline \
       --partition=bigmem,cpu \
       --mem=800G \
       --time=12:00:00 \
       --output=nmf_baseline_%j.out \
       --error=nmf_baseline_%j.err \
       --wrap="python $script"
"""
