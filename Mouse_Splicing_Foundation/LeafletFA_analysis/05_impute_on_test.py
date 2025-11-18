#!/usr/bin/env python
"""
LeafletFA Model Evaluation Script - Fixed Batched Evaluation
Evaluates trained models on test data using mini-batches with proper mask handling.
"""

import os, sys, glob, pickle, gzip
import numpy as np
import pandas as pd
import torch
import mudata as mu
from scipy import sparse
from scipy.stats import spearmanr
from datetime import datetime
import anndata as ad

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
torch.set_default_tensor_type("torch.cuda.FloatTensor" if torch.cuda.is_available() else "torch.FloatTensor")
torch.manual_seed(0)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

sys.path.append("/gpfs/commons/home/kisaev/Leaflet-private/src/")
import BetaDirichletFactor.LeafletFA as LeafletFA

def load_model(model_file):
    """Load model from pickle"""
    model = {}
    with gzip.open(model_file, "rb") as f:
        while True:
            try:
                model.update(pickle.load(f))
            except EOFError:
                break
    return model

def evaluate_batched(model_path, ad, output_dir, batch_size=4096):
    """Evaluate model using batches with proper mask handling"""
    print(f"\n{'='*80}\nEvaluating: {model_path}\n{'='*80}")
    
    # Clear GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        # Load model
        model = load_model(model_path)
        K = model["psis_loc"].shape[0]
        print(f"K={K} factors")
        
        # Sample PSI
        psi = 1 / (1 + np.exp(-(model["psis_loc"] + 
                                 model["psis_scale"] * np.random.randn(*model["psis_loc"].shape))))
        
        # Batch setup
        n_cells = ad.n_obs
        n_batches = int(np.ceil(n_cells / batch_size))
        print(f"Processing {n_cells:,} cells in {n_batches} batches of {batch_size:,}")
        
        all_imputed = []
        
        # Process batches
        for i in range(n_batches):
            start, end = i * batch_size, min((i + 1) * batch_size, n_cells)
            batch_size_actual = end - start
            print(f"  Batch {i+1}/{n_batches} (cells {start}-{end}, n={batch_size_actual})...", end=" ")
            
            # Create batch with proper slicing
            ad_batch = ad[start:end, :].copy()
            
            # Check if mask layers exist and slice them accordingly
            if "junc_ratio_masked_original" in ad.layers:
                ad_batch.layers["junc_ratio_masked_original"] = ad.layers["junc_ratio_masked_original"][start:end, :]
            
            if "junc_ratio_masked_bin_mask" in ad.layers:
                ad_batch.layers["junc_ratio_masked_bin_mask"] = ad.layers["junc_ratio_masked_bin_mask"][start:end, :]
            
            # Ensure other required layers are properly copied
            for layer_name in ["cell_by_junction_matrix", "cell_by_cluster_matrix"]:
                if layer_name in ad.layers:
                    ad_batch.layers[layer_name] = ad.layers[layer_name][start:end, :]
            
            # Initialize and train with adjusted parameters
            batch_model = LeafletFA.LeafletFA(
                adata=ad_batch, K=K, 
                fixed_psi=torch.tensor(psi),
                pi_init=torch.tensor(model["pi"]),
                alpha_pi_init=torch.tensor(model["alpha_pi"]),
                junc_specific_prior=model["junc_specific_prior"],
                waypoints_use=False, input_conc_prior=np.inf,
                delta_fixed=torch.tensor(model["dir_conc"]),
                num_epochs=100, 
                print_epochs=5, ELBO_num_particles=10,
                lr=0.01, gamma=0.05, min_delta=10, num_samples=100,
                patience=10, output_dir=output_dir, log_wandb=False
            )
            
            # Initialize model from anndata
            batch_model.from_anndata()
            
            # Try to initialize mask, with error handling
            try:
                batch_model.initialize_triton_mask()
            except Exception as e:
                print(f"\n  Warning: Mask initialization failed: {e}")
                # Alternative: Try to proceed without mask or with a default mask
                print("  Attempting to continue without custom mask...")
            
            # Train the model
            batch_model.train(num_initializations=1)
            batch_model.get_all_variables()
            
            # Get predictions
            all_imputed.append(batch_model.assign_post @ psi)
            print("✓")
            
            # Cleanup
            del batch_model, ad_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Combine and evaluate
        print("Combining predictions...")
        imputed = np.vstack(all_imputed)
        
        # Get the masked values for evaluation
        masked_orig = sparse.csr_matrix(ad.layers["junc_ratio_masked_original"])
        bin_mask = sparse.csr_matrix(ad.layers["junc_ratio_masked_bin_mask"])
        rows, cols = bin_mask.nonzero()
        
        orig_vals = masked_orig[rows, cols].A1
        pred_vals = imputed[rows, cols]
        
        # Calculate metrics
        pearson = np.corrcoef(orig_vals, pred_vals)[0, 1]
        spearman = spearmanr(orig_vals, pred_vals, nan_policy="omit")[0]
        l1_loss = np.mean(np.abs(orig_vals - pred_vals))
        rmse = np.sqrt(np.mean((orig_vals - pred_vals)**2))
        
        print(f"[RESULT] Pearson: {pearson:.4f}, Spearman: {spearman:.4f}, L1 Loss: {l1_loss:.4f}, RMSE: {rmse:.4f}\n")
        
        return {
            'model_path': model_path,
            'pearson': pearson,
            'spearman': spearman,
            'num_masked_values': len(orig_vals),
            'K_used': K,
            'num_batches': n_batches,
            'l1_loss': l1_loss,
            'rmse': rmse
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'model_path': model_path,
            'pearson': np.nan,
            'spearman': np.nan,
            'l1_loss': np.nan,
            'rmse': np.nan,
            'error': str(e)
        }
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    # Paths
    MODEL_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel/2025-10-25"
    OUTPUT_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_OUTPUT/batch_evaluation_1125"
    BATCH_SIZE = 4096  # Can be adjusted based on GPU memory
    MAX_JUNCTIONS = 5
    BASE_PATH = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025" 
    TEST_PATHS = ["MASKED_75_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_sparsity_fixed.h5mu",
    "MASKED_50_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_sparsity_fixed.h5mu", 
    "MASKED_25_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_sparsity_fixed.h5mu"]
    MASKING_PERCENTAGES = [75, 50, 25]

    # Load params
    params_df = pd.read_csv(os.path.join(MODEL_DIR, "parameter_combinations.csv"))
    print(f"Found {len(params_df)} parameter sets")
    
    # Find models
    run_dirs = sorted(glob.glob(os.path.join(MODEL_DIR, "run_*")))
    print(f"Found {len(run_dirs)} run directories\n")
    
    # Evaluate all
    all_results = []
    for run_dir in run_dirs:
        print(f"Running {run_dir}")
        run_name = os.path.basename(run_dir)
        run_idx = int(run_name.split('_')[1])
        model_file = os.path.join(run_dir, "leafletfa_model.pkl.gz")
        
        if not os.path.exists(model_file):
            print(f"Skipping {run_name}: no model file")
            continue
        
        print(f"{'#'*80}\n{run_name} (index {run_idx})")
        params = params_df.iloc[run_idx].to_dict() if run_idx < len(params_df) else {}
        print(f"Params: K={params.get('K')}, passes={params.get('num_passes')}, "
              f"gamma={params.get('gamma')}, lr={params.get('lr')}")
        
        # Run batched evaluation for all test paths
        masking_results = []

        for i, test_path in enumerate(TEST_PATHS):
            test_path = os.path.join(BASE_PATH, test_path)
            test_path_output = os.path.join(OUTPUT_DIR, test_path.replace(".h5mu", ""))
            os.makedirs(test_path_output, exist_ok=True)
            print(f"Evaluating model on {test_path} with output directory {test_path_output}")
            test_ad = ad.read_h5ad(test_path)
            test_ad = test_ad[:, test_ad.var["num_junctions"] <= MAX_JUNCTIONS].copy()
            test_ad.var["junction_id_index"] = np.arange(test_ad.shape[1])
            print(f"The number of junctions in the test adata object is: {test_ad.shape[1]}") 
            result = evaluate_batched(model_file, test_ad, test_path_output, BATCH_SIZE)
            masking_results.append(result)
            
            # Save individual masking result for this test file
            individual_result_df = pd.DataFrame([result])
            individual_result_df["masking_percentage"] = MASKING_PERCENTAGES[i]
            individual_result_df["run_name"] = run_name
            individual_result_df["run_idx"] = run_idx
            
            # Add the parameters to the individual result
            for col in params_df.columns:
                individual_result_df[col] = params.get(col, np.nan)
            
            # Save individual result file
            individual_result_file = os.path.join(test_path_output, f"{run_name}_masking_{MASKING_PERCENTAGES[i]}pct_result.csv")
            individual_result_df.to_csv(individual_result_file, index=False)
            print(f"Saved individual result to: {individual_result_file}")

        # convert to dataframe and add masking percentage
        masking_results_df = pd.DataFrame(masking_results)
        masking_results_df["masking_percentage"] = MASKING_PERCENTAGES
        masking_results_df["run_name"] = run_name
        masking_results_df["run_idx"] = run_idx

        # Add the parameters to the dataframe
        for col in params_df.columns:
            masking_results_df[col] = params.get(col, np.nan)

        print(masking_results_df)
        
        # Save combined results for this run
        run_output_dir = os.path.join(OUTPUT_DIR, run_name)
        os.makedirs(run_output_dir, exist_ok=True)
        combined_result_file = os.path.join(run_output_dir, f"{run_name}_combined_masking_results.csv")
        masking_results_df.to_csv(combined_result_file, index=False)
        print(f"Saved combined results for {run_name} to: {combined_result_file}")

        all_results.append(masking_results_df)

    all_results_df = pd.concat(all_results)
    all_results_df.to_csv(os.path.join(OUTPUT_DIR, f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), index=False)

if __name__ == "__main__":
    main()

"""
#Submit with:
conda activate LeafletSC
script=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/LeafletFA_analysis/05_impute_on_test.py
cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_OUTPUT/
sbatch --job-name=leaflet_eval \
       --partition=gpu \
       --gres=gpu:1 \
       --mem=300G \
       --time=24:00:00 \
       --output=leaflet_eval_%j.out \
       --error=leaflet_eval_%j.err \
       --wrap="python $script"
"""