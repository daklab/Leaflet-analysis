#!/usr/bin/env python
"""
LeafletFA Model Training Script with Multi-pass Mini-batch Training
Optimized for GPU with large datasets
"""

# %%
# Load libraries and set up environment
import os 
import sys
import datetime
import numpy as np
import pandas as pd
import anndata as ad    
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc 
import json
import pickle
import lzma
import wandb
import gzip
import gc
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import TruncatedSVD
import pyro
from pyro.infer.autoguide import AutoGuideList, AutoDiagonalNormal
from pyro import poutine
from pyro.infer.autoguide.initialization import init_to_value

# Ensure CUDA is available and if not use CPU
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device found")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

float_type = {"device": device, "dtype": torch.float}
torch.set_default_tensor_type("torch.FloatTensor" if device.type == "cpu" else "torch.cuda.FloatTensor")

# Set seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Configure plotting styles
sns.set_theme()
sc.set_figure_params(figsize=(7, 7), frameon=True, dpi=80, facecolor='white')

# Define module paths
src_path = "/gpfs/commons/home/kisaev/Leaflet-private/src/"

# Add to sys.path if not already present
if src_path not in sys.path:
    sys.path.append(src_path)

# Import custom modules
import BetaDirichletFactor.LeafletFA as LeafletFA
import BetaDirichletFactor.utils as utils
import BetaDirichletFactor.waypoints as wayp

# Get arguments from command line
param_id = int(sys.argv[1])
base_output_dir = sys.argv[2]
ATSE_anndata_file = sys.argv[3]

print(f"Loading parameter set {param_id}...")
print(f"Base output directory: {base_output_dir}")
print(f"Anndata file: {ATSE_anndata_file}")

# Load parameters
param_file = os.path.join(base_output_dir, "parameter_combinations.json")
with open(param_file, "r") as f:
    param_list = json.load(f)
params = param_list[param_id] 

# Convert 'inf' string to torch.tensor(np.inf)
params["input_conc"] = None if params["input_conc"] is None else torch.tensor(np.inf)

# Define output directory
output_dir = os.path.join(base_output_dir, f"run_{param_id}")
os.makedirs(output_dir, exist_ok=True)
print(f"All outputs will be saved in {output_dir}")

# Get today's date and time 
today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Starting run at: {today}")
today_date = datetime.datetime.now().strftime("%Y-%m-%d")

# Initialize wandb
wandb.init(
    project=f"LeafletFA-MouseFoundation_{today_date}",
    config=params,
    name=f"run_{param_id}",
    dir=output_dir,
    group="MouseFoundation",
    notes=f"Parameter set {param_id}, K={params['K']}, waypoints={params['waypoints_use']}, multi-pass minibatch"
)

# Also log additional parameters
wandb.config.update({
    "param_id": param_id,
    "data_source": "MouseFoundation",
    "anndata_file": ATSE_anndata_file,
})

print(f"Loading Anndata file: {ATSE_anndata_file}")
adata = ad.read_h5ad(ATSE_anndata_file)
print(f"Anndata file loaded successfully.")
print(adata.obs.head()) 
print(f"Anndata object contains {adata.n_obs} cells and {adata.n_vars} genes.")

# =============================================================================
# Filter ATSEs by Junction Count (if specified)
# =============================================================================

# Get max junctions parameter (default to 3 if not specified)
MAX_JUNCTIONS = params.get("max_junctions", 3)

if "num_junctions" in adata.var.columns and MAX_JUNCTIONS is not None:
    print(f"\nFiltering ATSEs to those with <= {MAX_JUNCTIONS} junctions...")
    original_shape = adata.shape
    
    # Filter to ATSEs with <= max_junctions
    adata = adata[:, adata.var["num_junctions"] <= MAX_JUNCTIONS].copy()
    
    # Reset junction indices
    adata.var["junction_id_index"] = np.arange(adata.shape[1])
    
    print(f"✓ Reduced data from {original_shape} to {adata.shape}")
    print(f"  Filtered out {original_shape[1] - adata.shape[1]} ATSEs with > {MAX_JUNCTIONS} junctions")
    print(f"  Retained {adata.shape[1]} ATSEs ({adata.shape[1]/original_shape[1]*100:.1f}% of original)")
    
    # Log filtering info to wandb
    wandb.log({
        "original_n_junctions": original_shape[1],
        "filtered_n_junctions": adata.shape[1],
        "max_junctions_threshold": MAX_JUNCTIONS,
        "junction_retention_rate": adata.shape[1]/original_shape[1],
    })
else:
    print("Note: 'num_junctions' column not found in adata.var or max_junctions not specified, skipping junction filtering")

# Log basic dataset info (after filtering)
wandb.log({
    "dataset_cells": adata.shape[0],
    "dataset_junctions": adata.shape[1],
    "initial_learning_rate": params["lr"],
    "gamma_decay": params["gamma"],
    "inital_K": params["K"], 
    "delta_fixed": params["delta_fixed"],
})

# =============================================================================
# Multi-pass Mini-batch Training Configuration
# =============================================================================

# Always use mini-batch for GPU
BATCH_SIZE = params.get("batch_size", 5000)  # 5000 cells for GPU
NUM_PASSES = params.get("num_passes", 10)  # Number of times each cell is seen
NUM_EPOCHS_FIRST = params.get("num_epochs_first", 50)  # Epochs for very first batch
NUM_EPOCHS_LATER = params.get("num_epochs_later", 20)  # Epochs for subsequent batches

n_cells = adata.n_obs
n_junctions = adata.n_vars
n_batches_per_pass = int(np.ceil(n_cells / BATCH_SIZE))
total_batches = n_batches_per_pass * NUM_PASSES

print("\n" + "="*60)
print("MULTI-PASS MINI-BATCH TRAINING MODE")
print(f"  Device: {device}")
print(f"  Total cells: {n_cells:,}")
print(f"  Total junctions: {n_junctions:,}")
print(f"  Batch size: {BATCH_SIZE:,}")
print(f"  Batches per pass: {n_batches_per_pass}")
print(f"  Number of passes: {NUM_PASSES}")
print(f"  Total batches: {total_batches}")
print(f"  Each cell seen: {NUM_PASSES} times")
print("="*60)

# =============================================================================
# Initialize storage for global parameters
# =============================================================================

# Global parameters that get updated across batches
global_psi = None
global_psi_loc = None  # For variational parameters
global_psi_scale = None
global_pi = None
global_alpha_pi = None
global_dir_conc = params.get("delta_fixed", None)
global_bb_conc = None
global_a = None
global_b = None
global_a_shape = None
global_a_rate = None
global_b_shape = None
global_b_rate = None

# Storage for all cells' PHI assignments (gets updated each pass)
all_cell_assignments = np.zeros((n_cells, params["K"]))
cell_assignment_counts = np.zeros(n_cells)  # Track how many times each cell was updated

# Track training metrics
all_batch_results = []
pass_results = []

# Track pi evolution across passes
pi_evolution = []  # Will store pi after each complete pass through data

# =============================================================================
# Multi-pass Training Loop
# =============================================================================

batch_counter = 0

for pass_idx in range(NUM_PASSES):
    print(f"\n{'='*60}")
    print(f"PASS {pass_idx + 1}/{NUM_PASSES}")
    print(f"{'='*60}")
    
    # Shuffle cells for each pass
    cell_indices = np.random.permutation(n_cells)
    pass_elbo = []
    
    for batch_in_pass in range(n_batches_per_pass):
        batch_counter += 1
        
        print(f"\n{'-'*50}")
        print(f"Pass {pass_idx + 1}, Batch {batch_in_pass + 1}/{n_batches_per_pass} (Overall: {batch_counter}/{total_batches})")
        print(f"{'-'*50}")
        
        # Get batch indices
        start_idx = batch_in_pass * BATCH_SIZE
        end_idx = min((batch_in_pass + 1) * BATCH_SIZE, n_cells)
        batch_cell_indices = cell_indices[start_idx:end_idx]
        batch_cell_indices = np.sort(batch_cell_indices)  # Sort for consistency
        
        print(f"Processing {len(batch_cell_indices)} cells")
        
        # Create batch adata
        batch_adata = adata[batch_cell_indices, :].copy()
        
        # =============================================================================
        # Special handling for very first batch with waypoint initialization
        # =============================================================================
        
        if batch_counter == 1 and params.get("waypoints_use", False):
            print("\n*** FIRST BATCH: Computing waypoint initialization ***")
            
            # Compute waypoints on first batch
            N_WAYPOINTS = params.get("n_waypoints", params["K"])
            N_PCA_COMPONENTS = min(params.get("n_pca_components", 20), len(batch_cell_indices) - 1, n_junctions - 1)
            N_DIM_COMPONENTS = min(params.get("n_dim_components", 20), N_PCA_COMPONENTS)
            METACELL_SIZE = min(params.get("metacell_size", 200), len(batch_cell_indices) // N_WAYPOINTS)
            
            # Choose data for PCA
            if "junc_ratio_centered" in batch_adata.layers:
                data_for_pca = batch_adata.layers["junc_ratio_centered"]
            elif "junc_ratio" in batch_adata.layers:
                data_for_pca = batch_adata.layers["junc_ratio"]
            else:
                data_for_pca = batch_adata.X
            
            # Run PCA
            print(f"Running PCA with {N_PCA_COMPONENTS} components...")
            svd = TruncatedSVD(n_components=N_PCA_COMPONENTS, random_state=42)
            U = svd.fit_transform(data_for_pca)
            S = svd.singular_values_
            batch_adata.obsm['X_pca'] = U * S
            
            # Find waypoints
            print(f"Finding {N_WAYPOINTS} waypoints...")
            waypoints = wayp.max_min_sampling(
                batch_adata.obsm['X_pca'], 
                N_WAYPOINTS,
                num_components=N_DIM_COMPONENTS,
                seed=42
            )
            
            # Create metacells
            metacell_dict = wayp.assign_nearest_cells(
                waypoints,
                batch_adata.obsm['X_pca'],
                num_nearest=METACELL_SIZE
            )
            
            # Generate initializations
            if "junc_ratio" in batch_adata.layers:
                rho_hat = batch_adata.layers["junc_ratio"]
            else:
                rho_hat = batch_adata.X
                
            psi_inits, phi_inits = wayp.generate_initializations(
                rho_hat,
                {N_WAYPOINTS: waypoints},
                {N_WAYPOINTS: metacell_dict},
                epsilon=0.001
            )
            
            # Store in batch_adata
            if isinstance(psi_inits[0], torch.Tensor):
                batch_adata.varm[f'psi_init_{N_WAYPOINTS}_waypoints'] = psi_inits[0].cpu().numpy()
            else:
                batch_adata.varm[f'psi_init_{N_WAYPOINTS}_waypoints'] = psi_inits[0]
                
            if isinstance(phi_inits[0], torch.Tensor):
                batch_adata.obsm[f'phi_init_{N_WAYPOINTS}_waypoints'] = phi_inits[0].cpu().numpy()
            else:
                batch_adata.obsm[f'phi_init_{N_WAYPOINTS}_waypoints'] = phi_inits[0]
            
            print(f"✓ Waypoint initialization complete")
        
        # =============================================================================
        # Initialize LeafletFA model for this batch
        # =============================================================================
        
        if batch_counter == 1:
            # Very first batch: learn from scratch (possibly with waypoints)
            print("Initializing model from scratch (first batch ever)...")
            
            leaflet_model = LeafletFA.LeafletFA(
                adata=batch_adata,
                K=params["K"],
                junc_specific_prior=params["junc_specific_prior"],
                waypoints_use=params.get("waypoints_use", False),
                input_conc_prior=params.get("input_conc", None),
                delta_fixed=params.get("delta_fixed", None),
                num_epochs=NUM_EPOCHS_FIRST,
                print_epochs=5,
                ELBO_num_particles=params["ELBO_num_particles"],
                lr=params["lr"],
                gamma=params["gamma"],
                min_delta=params["min_delta"],
                num_samples=params["num_samples"],
                patience=params["patience"],
                output_dir=output_dir,
                log_wandb=True,
                enable_cpu_optimization=False  # We're on GPU
            )
            
        else:
            # Subsequent batches: initialize with previous batch's PSI but continue learning
            print("Initializing model with previous batch parameters...")
            
            # Create waypoint-like initialization from previous batch
            # Store previous PSI as initialization
            batch_adata.varm[f'psi_init_{params["K"]}_waypoints'] = global_psi.T  # Transpose for correct shape
            
            # Initialize PHI for these cells based on previous global PSI if available
            if pass_idx > 0 and all_cell_assignments[batch_cell_indices].sum() > 0:
                # Use previous PHI for these cells as initialization
                batch_adata.obsm[f'phi_init_{params["K"]}_waypoints'] = all_cell_assignments[batch_cell_indices]
            else:
                # Random initialization for PHI
                phi_init = np.random.dirichlet(np.ones(params["K"]), size=len(batch_cell_indices))
                batch_adata.obsm[f'phi_init_{params["K"]}_waypoints'] = phi_init
            
            leaflet_model = LeafletFA.LeafletFA(
                adata=batch_adata,
                K=params["K"],
                junc_specific_prior=params["junc_specific_prior"],
                waypoints_use=True,  # Use the "waypoint" initialization we just created
                input_conc_prior=params.get("input_conc", None),
                delta_fixed=torch.tensor(global_dir_conc, device=device) if global_dir_conc is not None else None,
                num_epochs=NUM_EPOCHS_LATER,
                print_epochs=5,
                ELBO_num_particles=params["ELBO_num_particles"],
                lr=params["lr"] * (0.95 ** pass_idx),  # Decay learning rate across passes
                gamma=params["gamma"],
                min_delta=params["min_delta"],
                num_samples=params["num_samples"],
                patience=params["patience"],
                output_dir=output_dir,
                log_wandb=True,
                enable_cpu_optimization=False
            )
        
        # Extract sparse tensors
        print("Extracting sparse tensors...")
        leaflet_model.from_anndata()
        
        # Initialize Triton mask for GPU
        if device.type == 'cuda':
            leaflet_model.initialize_triton_mask()
        
        # Train on this batch
        print(f"Training batch {batch_counter}...")
        leaflet_model.train(num_initializations=1)
        
        # Extract learned parameters
        print("Extracting learned parameters...")
        leaflet_model.get_all_variables()
        
        # Update global parameters (PSI continues to evolve)
        # Ensure all parameters are numpy arrays, not tensors
        if hasattr(leaflet_model, 'psi'):
            global_psi = leaflet_model.psi if isinstance(leaflet_model.psi, np.ndarray) else leaflet_model.psi.cpu().numpy()
        if hasattr(leaflet_model, 'pi'):
            global_pi = leaflet_model.pi if isinstance(leaflet_model.pi, np.ndarray) else leaflet_model.pi.cpu().numpy()
        if hasattr(leaflet_model, 'alpha_pi'):
            global_alpha_pi = leaflet_model.alpha_pi if isinstance(leaflet_model.alpha_pi, np.ndarray) else leaflet_model.alpha_pi.cpu().numpy()
        if hasattr(leaflet_model, 'dir_conc'):
            if isinstance(leaflet_model.dir_conc, torch.Tensor):
                global_dir_conc = leaflet_model.dir_conc.cpu().item()  # Convert to scalar
            else:
                global_dir_conc = leaflet_model.dir_conc
        if hasattr(leaflet_model, 'bb_conc'):
            global_bb_conc = leaflet_model.bb_conc
        
        # Store additional parameters if available
        if hasattr(leaflet_model, 'psis_loc'):
            global_psi_loc = leaflet_model.psis_loc
            global_psi_scale = leaflet_model.psis_scale
        if hasattr(leaflet_model, 'a'):
            global_a = leaflet_model.a
            global_b = leaflet_model.b
            global_a_shape = leaflet_model.a_shape
            global_a_rate = leaflet_model.a_rate
            global_b_shape = leaflet_model.b_shape
            global_b_rate = leaflet_model.b_rate
        
        # Update PHI assignments for this batch of cells
        if hasattr(leaflet_model, 'assign_post'):
            all_cell_assignments[batch_cell_indices, :] = leaflet_model.assign_post
            cell_assignment_counts[batch_cell_indices] += 1
        
        # Track results
        batch_elbo = leaflet_model.best_elbo if hasattr(leaflet_model, 'best_elbo') else None
        pass_elbo.append(batch_elbo)
        
        all_batch_results.append({
            'pass': pass_idx + 1,
            'batch_in_pass': batch_in_pass + 1,
            'overall_batch': batch_counter,
            'batch_size': len(batch_cell_indices),
            'elbo': batch_elbo,
            'learning_rate': params["lr"] * (0.95 ** pass_idx)
        })
        
        print(f"Batch {batch_counter} complete. ELBO: {batch_elbo:.4e}")
        
        # Log to wandb
        wandb.log({
            "batch": batch_counter,
            "pass": pass_idx + 1,
            "batch_elbo": batch_elbo,
            "learning_rate": params["lr"] * (0.95 ** pass_idx)
        })
        
        # Clean up memory
        del batch_adata, leaflet_model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # End of pass summary
    avg_pass_elbo = np.mean([e for e in pass_elbo if e is not None])
    pass_results.append({
        'pass': pass_idx + 1,
        'avg_elbo': avg_pass_elbo,
        'min_elbo': np.min([e for e in pass_elbo if e is not None]),
        'max_elbo': np.max([e for e in pass_elbo if e is not None])
    })
    
    # Store pi evolution after each complete pass
    if global_pi is not None:
        pi_snapshot = {
            'pass': pass_idx + 1,
            'pi': global_pi.copy(),  # Make a copy to preserve current state
            'avg_elbo': avg_pass_elbo
        }
        pi_evolution.append(pi_snapshot)
        print(f"Pi after pass {pass_idx + 1}: {global_pi[:5]}... (showing first 5)")  # Show preview
    
    print(f"\nPass {pass_idx + 1} complete. Average ELBO: {avg_pass_elbo:.4e}")
    print(f"Cells seen frequency: min={cell_assignment_counts.min()}, max={cell_assignment_counts.max()}, mean={cell_assignment_counts.mean():.1f}")

# =============================================================================
# Final Model Assembly
# =============================================================================

print("\n" + "="*60)
print("TRAINING COMPLETE - ASSEMBLING FINAL MODEL")
print("="*60)

# Verify all cells were seen
print(f"Cell coverage: {(cell_assignment_counts > 0).sum()}/{n_cells} cells updated")
print(f"Average updates per cell: {cell_assignment_counts.mean():.1f}")

# Create final model structure (compatible with downstream analysis)
class FinalModel:
    pass

leaflet_model = FinalModel()
leaflet_model.psi = global_psi
leaflet_model.psi_learned = global_psi  # For compatibility
leaflet_model.assign_post = all_cell_assignments
leaflet_model.pi = global_pi
leaflet_model.alpha_pi = global_alpha_pi
leaflet_model.dir_conc = global_dir_conc
leaflet_model.bb_conc = global_bb_conc
leaflet_model.K = params["K"]
leaflet_model.best_elbo = np.mean([r['elbo'] for r in all_batch_results if r['elbo'] is not None])
leaflet_model.junc_specific_prior = params["junc_specific_prior"]
leaflet_model.input_conc_prior = params.get("input_conc", None)
leaflet_model.gamma = params["gamma"]
leaflet_model.ELBO_num_particles = params["ELBO_num_particles"]

# Add variational parameters if available
if global_psi_loc is not None:
    leaflet_model.psis_loc = global_psi_loc
    leaflet_model.psis_scale = global_psi_scale
if global_a is not None:
    leaflet_model.a = global_a
    leaflet_model.b = global_b
    leaflet_model.a_shape = global_a_shape
    leaflet_model.a_rate = global_a_rate
    leaflet_model.b_shape = global_b_shape
    leaflet_model.b_rate = global_b_rate

# Prune K if needed (remove factors with very low pi)
print("\nPruning factors...")
original_K = leaflet_model.K
if leaflet_model.pi is not None:
    keep_factors = leaflet_model.pi > 0.01
    if keep_factors.sum() < original_K:
        leaflet_model.K = keep_factors.sum()
        leaflet_model.psi = leaflet_model.psi[keep_factors, :]
        leaflet_model.assign_post = leaflet_model.assign_post[:, keep_factors]
        leaflet_model.assign_post = leaflet_model.assign_post / leaflet_model.assign_post.sum(axis=1, keepdims=True)
        leaflet_model.pi = leaflet_model.pi[keep_factors]
        leaflet_model.pi = leaflet_model.pi / leaflet_model.pi.sum()
        print(f"Pruned K from {original_K} to {leaflet_model.K}")
    else:
        print(f"No pruning needed, keeping all {original_K} factors")

new_K = leaflet_model.K

# =============================================================================
# Save Results
# =============================================================================

# Save latent variables to adata
adata.obsm[f"X_leafletFA_K{new_K}"] = leaflet_model.assign_post

# Log final metrics to wandb
wandb.log({
    "final_K": new_K,
    "original_K": original_K,
    "final_elbo": leaflet_model.best_elbo,
    "num_passes": NUM_PASSES,
    "total_batches": total_batches,
    "cells_coverage": (cell_assignment_counts > 0).sum() / n_cells,
    "avg_updates_per_cell": cell_assignment_counts.mean(),
})

# Log pi evolution summary if available
if pi_evolution:
    # Calculate stability metrics
    if len(pi_evolution) > 1:
        # Calculate pi stability (how much it changed between passes)
        pi_changes = []
        for i in range(1, len(pi_evolution)):
            change = np.linalg.norm(pi_evolution[i]['pi'] - pi_evolution[i-1]['pi'])
            pi_changes.append(change)
        
        wandb.log({
            "pi_final_stability": pi_changes[-1] if pi_changes else 0,
            "pi_avg_change": np.mean(pi_changes) if pi_changes else 0,
            "pi_convergence_trend": pi_changes[-1] / pi_changes[0] if len(pi_changes) > 0 and pi_changes[0] > 0 else 1,
        })
        
        print(f"\nPi stability metrics:")
        print(f"  Final change (last pass): {pi_changes[-1]:.6f}")
        print(f"  Average change per pass: {np.mean(pi_changes):.6f}")
        print(f"  Convergence ratio: {pi_changes[-1] / pi_changes[0] if pi_changes[0] > 0 else 1:.3f}")

# Save factor assignment probabilities
if hasattr(leaflet_model, 'pi') and leaflet_model.pi is not None:
    PI = leaflet_model.pi
    PI_df = pd.DataFrame(PI, columns=["PI"])
    PI_df["Factor"] = PI_df.index
    PI_df["Factor"] = PI_df["Factor"].astype(str)
    PI_df = PI_df.sort_values(by="PI", ascending=False)
    print(f"\nFactor probabilities:")
    print(PI_df.head(10))
    PI_df.to_csv(os.path.join(output_dir, "factor_assignment_probabilities.csv"), index=False)

# Save batch results
batch_results_df = pd.DataFrame(all_batch_results)
batch_results_df.to_csv(os.path.join(output_dir, "batch_training_results.csv"), index=False)

pass_results_df = pd.DataFrame(pass_results)
pass_results_df.to_csv(os.path.join(output_dir, "pass_summary_results.csv"), index=False)

# Save pi evolution
if pi_evolution:
    # Create a DataFrame for pi evolution
    pi_evolution_data = []
    for snapshot in pi_evolution:
        for k in range(len(snapshot['pi'])):
            pi_evolution_data.append({
                'pass': snapshot['pass'],
                'factor': k,
                'pi_value': snapshot['pi'][k],
                'avg_elbo': snapshot['avg_elbo']
            })
    
    pi_evolution_df = pd.DataFrame(pi_evolution_data)
    pi_evolution_df.to_csv(os.path.join(output_dir, "pi_evolution.csv"), index=False)
    
    # Create a visualization of pi evolution
    plt.figure(figsize=(12, 8))
    for k in range(min(10, params["K"])):  # Plot top 10 factors
        factor_data = pi_evolution_df[pi_evolution_df['factor'] == k]
        plt.plot(factor_data['pass'], factor_data['pi_value'], 
                marker='o', label=f'Factor {k}', linewidth=2)
    
    plt.xlabel('Pass Number', fontsize=12)
    plt.ylabel('Pi Value (Factor Probability)', fontsize=12)
    plt.title('Evolution of Factor Probabilities (Pi) Across Training Passes', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pi_evolution_plot.png"), dpi=100, bbox_inches='tight')
    plt.close()
    
    # Also create a heatmap
    pi_matrix = np.zeros((NUM_PASSES, params["K"]))
    for snapshot in pi_evolution:
        pi_matrix[snapshot['pass']-1, :] = snapshot['pi']
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(pi_matrix.T, annot=False, cmap='coolwarm', 
                xticklabels=range(1, NUM_PASSES+1),
                yticklabels=range(params["K"]),
                cbar_kws={'label': 'Pi Value'})
    plt.xlabel('Pass Number', fontsize=12)
    plt.ylabel('Factor Index', fontsize=12)
    plt.title('Heatmap of Pi Evolution Across Training', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pi_evolution_heatmap.png"), dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved pi evolution tracking to CSV and visualizations")

# Create results summary
results_df = pd.DataFrame([{
    "param_id": param_id,
    "K": params["K"],
    "pruned_K": new_K,
    "junc_specific_prior": params["junc_specific_prior"],
    "dir_conc": params["delta_fixed"],
    "waypoints_use": params["waypoints_use"],
    "best_elbo": leaflet_model.best_elbo,
    "input_conc": leaflet_model.bb_conc,
    "num_epochs_first": NUM_EPOCHS_FIRST,
    "num_epochs_later": NUM_EPOCHS_LATER,
    "num_passes": NUM_PASSES,
    "batch_size": BATCH_SIZE,
    "total_batches": total_batches,
    "lr": params["lr"],
    "gamma": params["gamma"],
    "num_samples": params["num_samples"],
    "ELBO_num_particles": params["ELBO_num_particles"],
    "cells_coverage": (cell_assignment_counts > 0).sum() / n_cells,
    "avg_updates_per_cell": cell_assignment_counts.mean(),
}])

results_file = os.path.join(output_dir, "run_summary.csv")
results_df.to_csv(results_file, index=False)
print(f"\nSaved run summary to {results_file}")

# Save model
model_file = os.path.join(output_dir, "leafletfa_model.pkl.gz")

# Save essential attributes
only_things_we_need = [
    'ELBO_num_particles', 'K', 'assign_post', 'best_elbo', 
    'gamma', 'input_conc_prior', 'junc_specific_prior', 
    'pi', 'psi', 'psi_learned', 'dir_conc', 'bb_conc', 'alpha_pi'
]

# Add variational parameters if available
if hasattr(leaflet_model, 'psis_loc'):
    only_things_we_need.extend(['psis_loc', 'psis_scale'])
if hasattr(leaflet_model, 'a'):
    only_things_we_need.extend(['a', 'a_rate', 'a_shape', 'b', 'b_rate', 'b_shape'])

with gzip.open(model_file, "wb") as f:
    print(f"\nSaving model to {model_file}...")
    
    metadata = {
        "_meta": {
            "param_id": param_id,
            "data_shape": adata.shape,
            "waypoints_used": params.get("waypoints_use", False),
            "training_mode": "multi_pass_minibatch",
            "num_passes": NUM_PASSES,
            "batch_size": BATCH_SIZE,
            "total_batches": total_batches,
            "training_date": today,
            "best_elbo": leaflet_model.best_elbo,
            "original_K": original_K,
            "pruned_K": new_K,
        }
    }
    pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    for attr_name in only_things_we_need:
        if hasattr(leaflet_model, attr_name):
            attr_value = getattr(leaflet_model, attr_name)
            if attr_value is not None:
                print(f"  Saving {attr_name}...")
                if isinstance(attr_value, torch.Tensor):
                    attr_value = attr_value.detach().cpu().numpy()
                pickle.dump({attr_name: attr_value}, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Model saved. Size: {os.path.getsize(model_file) / (1024**2):.2f} MB")

# Final logging
wandb.log({
    "training_complete": True,
    "final_K": new_K,
    "model_saved": True,
})

wandb.finish()

# Print final summary
print("\n" + "="*60)
print("RUN COMPLETE")
print(f"  Training mode: Multi-pass mini-batch")
print(f"  Device: {device}")
print(f"  Total passes: {NUM_PASSES}")
print(f"  Total batches: {total_batches}")
print(f"  Final K: {new_K} (pruned from {original_K})")
print(f"  Final ELBO: {leaflet_model.best_elbo:.4e}")
print(f"  Total time: {(datetime.datetime.now() - datetime.datetime.strptime(today, '%Y-%m-%d %H:%M:%S')).total_seconds()/60:.2f} minutes")
print("="*60)