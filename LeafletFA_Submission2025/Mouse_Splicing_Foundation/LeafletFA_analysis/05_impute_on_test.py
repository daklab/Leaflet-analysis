#!/usr/bin/env python
"""
Parallel LeafletFA Model Evaluation System
Submits individual SLURM jobs for each model evaluation
"""

import os
import sys
import glob
import pandas as pd
import subprocess
from datetime import datetime
import argparse

def create_submission_script():
    """Create the main parallel submission script"""
    
    script_content = '''#!/usr/bin/env python
"""
Parallel LeafletFA Model Evaluation - Job Submission Script
"""

import os
import glob
import pandas as pd
import subprocess
from datetime import datetime

# Configuration
MODEL_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel/2025-10-25"
OUTPUT_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_OUTPUT/batch_evaluation_parallel"
LOG_DIR = os.path.join(OUTPUT_DIR, "slurm_logs")

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Load parameter combinations
params_df = pd.read_csv(os.path.join(MODEL_DIR, "parameter_combinations.csv"))
print(f"Found {len(params_df)} parameter sets")

# Find all run directories
run_dirs = sorted(glob.glob(os.path.join(MODEL_DIR, "run_*")))
print(f"Found {len(run_dirs)} run directories")

# Track job IDs for dependency management
job_ids = []
job_info = []

# Submit individual jobs for each run
for run_dir in run_dirs:
    run_name = os.path.basename(run_dir)
    run_idx = int(run_name.split('_')[1])
    model_file = os.path.join(run_dir, "leafletfa_model.pkl.gz")
    
    if not os.path.exists(model_file):
        print(f"Skipping {run_name}: no model file")
        continue
    
    # Create SLURM job script
    job_script = f"""#!/bin/bash
#SBATCH --job-name=eval_{run_name}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --output={LOG_DIR}/{run_name}_%j.out
#SBATCH --error={LOG_DIR}/{run_name}_%j.err

# Load environment
source activate LeafletSC

# Run evaluation for this specific model
python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/LeafletFA_analysis/evaluate_single_model.py \\
    --model_path {model_file} \\
    --run_name {run_name} \\
    --run_idx {run_idx} \\
    --output_dir {OUTPUT_DIR}
"""
    
    # Write job script
    job_file = os.path.join(LOG_DIR, f"job_{run_name}.sh")
    with open(job_file, "w") as f:
        f.write(job_script)
    
    # Submit job
    cmd = f"sbatch {job_file}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Extract job ID from output
        job_id = result.stdout.strip().split()[-1]
        job_ids.append(job_id)
        job_info.append({
            'run_name': run_name,
            'run_idx': run_idx,
            'job_id': job_id,
            'model_path': model_file
        })
        print(f"Submitted {run_name} -> Job ID: {job_id}")
    else:
        print(f"Failed to submit {run_name}: {result.stderr}")

# Save job tracking information
job_df = pd.DataFrame(job_info)
job_df.to_csv(os.path.join(OUTPUT_DIR, f"job_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), index=False)
print(f"\\nSubmitted {len(job_ids)} jobs")
print(f"Job tracking saved to: {OUTPUT_DIR}/job_tracking_*.csv")

# Create aggregation job that depends on all evaluation jobs
if job_ids:
    dependency_string = ":".join(job_ids)
    agg_script = f"""#!/bin/bash
#SBATCH --job-name=aggregate_results
#SBATCH --partition=bigmem
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output={LOG_DIR}/aggregate_%j.out
#SBATCH --error={LOG_DIR}/aggregate_%j.err
#SBATCH --dependency=afterok:{dependency_string}

source activate LeafletSC

python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/LeafletFA_analysis/aggregate_results.py \\
    --output_dir {OUTPUT_DIR}
"""
    
    agg_job_file = os.path.join(LOG_DIR, "job_aggregate.sh")
    with open(agg_job_file, "w") as f:
        f.write(agg_script)
    
    result = subprocess.run(f"sbatch {agg_job_file}", shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        agg_job_id = result.stdout.strip().split()[-1]
        print(f"\\nSubmitted aggregation job: {agg_job_id}")
        print("This job will run after all evaluation jobs complete successfully")
'''
    
    return script_content


def create_single_model_evaluator():
    """Create the single model evaluation script"""
    
    script_content = '''#!/usr/bin/env python
"""
Single Model Evaluation Script
Evaluates one model on all test datasets
"""

import os
import sys
import pickle
import gzip
import numpy as np
import pandas as pd
import torch
import anndata as ad
from scipy import sparse
from scipy.stats import spearmanr
import argparse

# Setup paths
sys.path.append("/gpfs/commons/home/kisaev/Leaflet-private/src/")
import BetaDirichletFactor.LeafletFA as LeafletFA

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
torch.set_default_tensor_type("torch.cuda.FloatTensor" if torch.cuda.is_available() else "torch.FloatTensor")
torch.manual_seed(0)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
    """Evaluate model using batches"""
    print(f"Evaluating on {ad.n_obs:,} cells, {ad.n_vars:,} junctions")
    
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
        print(f"Processing in {n_batches} batches")
        
        all_imputed = []
        
        # Process batches
        for i in range(n_batches):
            start, end = i * batch_size, min((i + 1) * batch_size, n_cells)
            print(f"  Batch {i+1}/{n_batches}...", end=" ")
            
            # Create batch
            ad_batch = ad[start:end, :].copy()
            
            # Copy layers
            for layer in ["junc_ratio_masked_original", "junc_ratio_masked_bin_mask", 
                         "cell_by_junction_matrix", "cell_by_cluster_matrix"]:
                if layer in ad.layers:
                    ad_batch.layers[layer] = ad.layers[layer][start:end, :]
            
            # Initialize model
            batch_model = LeafletFA.LeafletFA(
                adata=ad_batch, K=K,
                fixed_psi=torch.tensor(psi),
                pi_init=torch.tensor(model["pi"]),
                alpha_pi_init=torch.tensor(model["alpha_pi"]),
                junc_specific_prior=model["junc_specific_prior"],
                waypoints_use=False, input_conc_prior=np.inf,
                delta_fixed=torch.tensor(model["dir_conc"]),
                num_epochs=100, print_epochs=5, ELBO_num_particles=10,
                lr=0.01, gamma=0.05, min_delta=10, num_samples=100,
                patience=10, output_dir=output_dir, log_wandb=False
            )
            
            batch_model.from_anndata()
            try:
                batch_model.initialize_triton_mask()
            except:
                pass
            
            batch_model.train(num_initializations=1)
            batch_model.get_all_variables()
            
            all_imputed.append(batch_model.assign_post @ psi)
            print("✓")
            
            del batch_model, ad_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Evaluate
        imputed = np.vstack(all_imputed)
        masked_orig = sparse.csr_matrix(ad.layers["junc_ratio_masked_original"])
        bin_mask = sparse.csr_matrix(ad.layers["junc_ratio_masked_bin_mask"])
        rows, cols = bin_mask.nonzero()
        
        orig_vals = masked_orig[rows, cols].A1
        pred_vals = imputed[rows, cols]
        
        # Metrics
        pearson = np.corrcoef(orig_vals, pred_vals)[0, 1]
        spearman = spearmanr(orig_vals, pred_vals, nan_policy="omit")[0]
        l1_loss = np.mean(np.abs(orig_vals - pred_vals))
        rmse = np.sqrt(np.mean((orig_vals - pred_vals)**2))
        
        print(f"Results: Pearson={pearson:.4f}, Spearman={spearman:.4f}, L1={l1_loss:.4f}, RMSE={rmse:.4f}")
        
        return {
            'pearson': pearson,
            'spearman': spearman,
            'l1_loss': l1_loss,
            'rmse': rmse,
            'num_masked_values': len(orig_vals),
            'K_used': K
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        return {'error': str(e)}
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--run_name', required=True)
    parser.add_argument('--run_idx', type=int, required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    
    # Configuration
    BATCH_SIZE = 4096
    MAX_JUNCTIONS = 5
    BASE_PATH = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025"
    MODEL_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel/2025-10-25"
    
    TEST_FILES = [
        "MASKED_75_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_sparsity_fixed.h5mu",
        "MASKED_50_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_sparsity_fixed.h5mu",
        "MASKED_25_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_sparsity_fixed.h5mu"
    ]
    MASKING_PERCENTAGES = [75, 50, 25]
    
    # Load parameters
    params_df = pd.read_csv(os.path.join(MODEL_DIR, "parameter_combinations.csv"))
    params = params_df.iloc[args.run_idx].to_dict() if args.run_idx < len(params_df) else {}
    
    print(f"Evaluating {args.run_name} (idx={args.run_idx})")
    print(f"Parameters: {params}")
    
    # Evaluate on all test sets
    results = []
    for i, test_file in enumerate(TEST_FILES):
        test_path = os.path.join(BASE_PATH, test_file)
        print(f"\\nLoading test set {i+1}/3: {MASKING_PERCENTAGES[i]}% masked")
        
        test_ad = ad.read_h5ad(test_path)
        test_ad = test_ad[:, test_ad.var["num_junctions"] <= MAX_JUNCTIONS].copy()
        test_ad.var["junction_id_index"] = np.arange(test_ad.shape[1])
        
        result = evaluate_batched(args.model_path, test_ad, args.output_dir, BATCH_SIZE)
        result['masking_percentage'] = MASKING_PERCENTAGES[i]
        result['run_name'] = args.run_name
        result['run_idx'] = args.run_idx
        result['model_path'] = args.model_path
        
        # Add parameters
        for col, val in params.items():
            result[col] = val
        
        results.append(result)
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = os.path.join(args.output_dir, f"{args.run_name}_results.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
'''
    
    return script_content


def create_aggregator_script():
    """Create the results aggregation script"""
    
    script_content = '''#!/usr/bin/env python
"""
Aggregate Results from Parallel Evaluation Jobs
"""

import os
import glob
import pandas as pd
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    
    print("Aggregating results from parallel jobs...")
    
    # Find all result files
    result_files = glob.glob(os.path.join(args.output_dir, "run_*_results.csv"))
    print(f"Found {len(result_files)} result files")
    
    if not result_files:
        print("No results found!")
        return
    
    # Load and combine
    all_results = []
    for file in result_files:
        df = pd.read_csv(file)
        all_results.append(df)
        print(f"  Loaded {os.path.basename(file)}: {len(df)} rows")
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save combined results
    output_file = os.path.join(args.output_dir, f"combined_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    combined_df.to_csv(output_file, index=False)
    print(f"\\nCombined results saved to: {output_file}")
    
    # Print summary statistics
    print("\\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Group by masking percentage
    for mask_pct in sorted(combined_df['masking_percentage'].unique()):
        subset = combined_df[combined_df['masking_percentage'] == mask_pct]
        print(f"\\nMasking {mask_pct}%:")
        print(f"  Best L1 Loss: {subset['l1_loss'].min():.4f} ({subset.loc[subset['l1_loss'].idxmin(), 'run_name']})")
        print(f"  Best RMSE: {subset['rmse'].min():.4f} ({subset.loc[subset['rmse'].idxmin(), 'run_name']})")
        print(f"  Best Spearman: {subset['spearman'].max():.4f} ({subset.loc[subset['spearman'].idxmax(), 'run_name']})")
    
    print("\\n" + "="*60)
    print("Aggregation complete!")

if __name__ == "__main__":
    main()
'''
    
    return script_content


# Main execution
if __name__ == "__main__":
    print("Creating parallel evaluation system scripts...")
    
    # Create output directory
    script_dir = "/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/LeafletFA_analysis"
    os.makedirs(script_dir, exist_ok=True)
    
    # Write submission script
    with open(os.path.join(script_dir, "submit_parallel_evaluation.py"), "w") as f:
        f.write(create_submission_script())
    print("Created: submit_parallel_evaluation.py")
    
    # Write single model evaluator
    with open(os.path.join(script_dir, "evaluate_single_model.py"), "w") as f:
        f.write(create_single_model_evaluator())
    print("Created: evaluate_single_model.py")
    
    # Write aggregator
    with open(os.path.join(script_dir, "aggregate_results.py"), "w") as f:
        f.write(create_aggregator_script())
    print("Created: aggregate_results.py")
    
    print("\nTo run the parallel evaluation system:")
    print("1. cd /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/LeafletFA_analysis")
    print("2. python submit_parallel_evaluation.py")
    print("\nThis will submit individual jobs for each model and an aggregation job that runs after all complete.")
    
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