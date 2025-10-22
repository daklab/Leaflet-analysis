#!/usr/bin/env python
"""
LeafletFA Model Evaluation Script - Batched Evaluation
Evaluates trained models on test data using mini-batches to avoid OOM.
"""

import os, sys, glob, pickle, gzip
import numpy as np
import pandas as pd
import torch
import mudata as mu
from scipy import sparse
from scipy.stats import spearmanr
from datetime import datetime

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

def fix_sparsity(ad):
    """Fix sparsity pattern mismatch"""
    junc = ad.layers["cell_by_junction_matrix"].tocoo()
    clust = ad.layers["cell_by_cluster_matrix"].tocoo()
    missing = set(zip(clust.row, clust.col)) - set(zip(junc.row, junc.col))
    
    if missing:
        mr, mc = zip(*missing)
        all_r = np.concatenate([junc.row, mr])
        all_c = np.concatenate([junc.col, mc])
        all_v = np.concatenate([junc.data, np.zeros(len(missing))])
        ad.layers["cell_by_junction_matrix"] = sparse.csr_matrix(
            (all_v, (all_r, all_c)), shape=junc.shape, dtype=junc.dtype)
    return ad

def evaluate_batched(model_path, ad, output_dir, batch_size=4096):
    """Evaluate model using batches"""
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
            print(f"  Batch {i+1}/{n_batches} (cells {start}-{end})...", end=" ")
            
            ad_batch = ad[start:end, :].copy()
            
            # Initialize and train
            batch_model = LeafletFA.LeafletFA(
                adata=ad_batch, K=K, 
                fixed_psi=torch.tensor(psi),
                pi_init=torch.tensor(model["pi"]),
                alpha_pi_init=torch.tensor(model["alpha_pi"]),
                junc_specific_prior=model["junc_specific_prior"],
                waypoints_use=False, input_conc_prior=np.inf,
                delta_fixed=torch.tensor(model["dir_conc"]),
                num_epochs=20, print_epochs=5, ELBO_num_particles=10,
                lr=0.6, gamma=0.005, min_delta=10, num_samples=100,
                patience=10, output_dir=output_dir, log_wandb=False
            )
            
            batch_model.from_anndata()
            batch_model.initialize_triton_mask()
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
        
        masked_orig = sparse.csr_matrix(ad.layers["junc_ratio_masked_original"])
        bin_mask = sparse.csr_matrix(ad.layers["junc_ratio_masked_bin_mask"])
        rows, cols = bin_mask.nonzero()
        
        orig_vals = masked_orig[rows, cols].A1
        pred_vals = imputed[rows, cols]
        
        pearson = np.corrcoef(orig_vals, pred_vals)[0, 1]
        spearman = spearmanr(orig_vals, pred_vals, nan_policy="omit")[0]
        
        print(f"[RESULT] Pearson: {pearson:.4f}, Spearman: {spearman:.4f}\n")
        
        return {
            'model_path': model_path,
            'pearson': pearson,
            'spearman': spearman,
            'num_masked_values': len(orig_vals),
            'K_used': K,
            'num_batches': n_batches
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    # Paths
    TEST_PATH = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/MASKED_0.2_test_30_70_ge_splice_combined_20250730_164104.h5mu"
    MODEL_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel/2025-09-22"
    OUTPUT_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_OUTPUT/batch_evaluation"
    BATCH_SIZE = 4096
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print(f"Loading test data...")
    mdata = mu.read_h5mu(TEST_PATH)
    ad = mdata["splicing"]
    ad.obs.reset_index(drop=True, inplace=True)
    ad.obs["cell_id_index"] = ad.obs.index
    ad = fix_sparsity(ad)
    print(f"Data shape: {ad.shape}")
    
    # Load params
    params_df = pd.read_csv(os.path.join(MODEL_DIR, "parameter_combinations.csv"))
    print(f"Found {len(params_df)} parameter sets")
    
    # Find models
    run_dirs = sorted(glob.glob(os.path.join(MODEL_DIR, "run_*")))
    print(f"Found {len(run_dirs)} run directories\n")
    
    # Evaluate all
    results = []
    for run_dir in run_dirs:
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
        
        try:
            result = evaluate_batched(model_file, ad.copy(), OUTPUT_DIR, BATCH_SIZE)
            result.update({'run_name': run_name, 'run_idx': run_idx})
            for col in params_df.columns:
                result[f'param_{col}'] = params.get(col, np.nan)
            results.append(result)
            
            # Save intermediate
            pd.DataFrame(results).to_csv(
                os.path.join(OUTPUT_DIR, "evaluation_results_intermediate.csv"), index=False)
            
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                'model_path': model_file, 'run_name': run_name, 'run_idx': run_idx,
                'pearson': np.nan, 'spearman': np.nan, 'error': str(e)
            })
    
    # Final results
    if results:
        results_df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(OUTPUT_DIR, f"evaluation_results_{timestamp}.csv")
        results_df.to_csv(results_file, index=False)
        
        print(f"\n{'='*80}\nEVALUATION SUMMARY\n{'='*80}")
        valid = results_df.dropna(subset=['pearson'])
        
        if len(valid) > 0:
            sorted_res = valid.sort_values('pearson', ascending=False)
            print("\nTop 10 models:")
            print(sorted_res[['run_name', 'pearson', 'spearman', 'param_K', 
                              'param_num_passes', 'param_gamma', 'param_lr']].head(10))
            
            print(f"\nStats:")
            print(f"  Mean Pearson: {valid['pearson'].mean():.4f} ± {valid['pearson'].std():.4f}")
            print(f"  Best Pearson: {valid['pearson'].max():.4f} ({valid.loc[valid['pearson'].idxmax(), 'run_name']})")
        
        print(f"\nTotal: {len(results)}, Success: {len(valid)}, Failed: {len(results)-len(valid)}")
        print(f"Results saved to: {results_file}")
    else:
        print("\n❌ No results")

if __name__ == "__main__":
    main()

"""
Submit with:
script=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/LeafletFA_analysis/05_impute_on_test.py
cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_OUTPUT/
sbatch --job-name=leaflet_eval \
       --partition=gpu \
       --gres=gpu:1 \
       --mem=300G \
       --time=7-00:00:00 \
       --output=leaflet_eval_%j.out \
       --error=leaflet_eval_%j.err \
       --wrap="python $script"
"""