# run_leaflet.py

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
import anndata as ad
import scanpy as sc 
import json
import pickle
import lzma
import wandb  # Import wandb
import gzip

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
    project=f"LeafletFA-HumanFoundation-GPU-{today_date}",  # Your project name
    config=params,  # Config parameters for this run
    # add time to run name 
    name=f"run_{param_id}",  # Name of this run
    dir=output_dir,  # Directory to store wandb files
    # Optional: Add a group for easier organization
    group="HumanFoundation",
    # Optional: Add notes
    notes=f"Parameter set {param_id}, K={params['K']}, waypoints={params['waypoints_use']}"
)

# Also log additional parameters
wandb.config.update({
    "param_id": param_id,
    "data_source": "HumanFoundation",
    "anndata_file": ATSE_anndata_file,
})

print(f"Loading Anndata file: {ATSE_anndata_file}")
adata = ad.read_h5ad(ATSE_anndata_file)
print(f"Anndata file loaded successfully.")
print(adata.obs.head()) 
print(f"Anndata object contains {adata.n_obs} cells and {adata.n_vars} junctions.")

# Log basic dataset info
wandb.log({
    "dataset_cells": adata.shape[0],
    "dataset_junctions": adata.shape[1],
    "initial_learning_rate": params["lr"],
    "gamma_decay": params["gamma"],
    "inital_K": params["K"], 
    "delta_fixed": params["delta_fixed"],
})

# Initialize model
print("Initializing LeafletFA model...")
leaflet_model = LeafletFA.LeafletFA(
    adata=adata, 
    K=params["K"], 
    junc_specific_prior=params["junc_specific_prior"], 
    waypoints_use=params["waypoints_use"], 
    input_conc_prior=params["input_conc"], 
    delta_fixed=params["delta_fixed"],
    num_epochs=params["num_epochs"], 
    print_epochs=5, 
    ELBO_num_particles=params["ELBO_num_particles"], 
    lr=params["lr"], 
    gamma=params["gamma"], 
    min_delta=params["min_delta"],
    num_samples=params["num_samples"], 
    patience=params["patience"],
    output_dir=output_dir,
    log_wandb=True  # Log to wandb
)

# Print confirm that model has dir_conc 
print(f"Model initialized with dir_conc: {leaflet_model.dir_conc}")

# Train model
print(f"Extracting sparse tensors from anndata object")
leaflet_model.from_anndata()

print(f"Obtaining mask for sparse operations")
leaflet_model.initialize_triton_mask()

print("Training LeafletFA model...")
leaflet_model.train(num_initializations=params["num_inits"])

print("Training complete, extracting results...")
leaflet_model.get_all_variables()

# Prune K! 
print("Pruning K...")
leaflet_model.prune_K(threshold=0.01)
new_K = leaflet_model.K 

# Calculate correlations between initializations if more than 2 
assign_matrices = [result["summary_stats"]["assign"]["mean"] for result in leaflet_model.latent_results]
if params["num_inits"] > 1:
    avg_corr, median_corr, min_corr = utils.calculate_and_plot_correlations(assign_matrices)
else: 
    avg_corr, median_corr, min_corr = None, None, None  # Set default values when there's only one initialization

# Save latent variables
adata.obsm[f"X_leafletFA_K{new_K}"] = leaflet_model.assign_post

# Log silhouette scores to wandb
wandb.log({
    "new_K": new_K,
})

# Make a quick barplot of PI and add to wandb log 
alpha_pi=leaflet_model.alpha_pi
PI = leaflet_model.pi
PI_df = pd.DataFrame(PI, columns=["PI"])
PI_df["Factor"] = PI_df.index
# Ensure 'Factor' is treated as categorical
PI_df["Factor"] = PI_df["Factor"].astype(str)
PI_df = PI_df.sort_values(by="PI", ascending=False)
print(f"Original K is {leaflet_model.K} and reduced K is {len(PI_df)}")
PI_df.to_csv(os.path.join(output_dir, "factor_assignment_probabilities.csv"), index=False)
print(f"Saved factor assignment probabilities to {os.path.join(output_dir, 'factor_assignment_probabilities.csv')}")

# Log to wandb
wandb.log({
    "alpha_pi": alpha_pi,
    "dir_conc": leaflet_model.dir_conc,
    "bb_conc": leaflet_model.bb_conc
})

# Create a dataframe to store results
results_df = pd.DataFrame([{
    "param_id": param_id,
    "K": params["K"],
    "junc_specific_prior": params["junc_specific_prior"],
    "dir_conc": params["delta_fixed"], 
    "waypoints_use": params["waypoints_use"],
    "best_elbo": leaflet_model.best_elbo,
    "input_conc": leaflet_model.bb_conc,
    "num_epochs": params["num_epochs"],
    "num_inits": params["num_inits"],
    "avg_corr": avg_corr,
    "median_corr": median_corr,
    "min_corr": min_corr,
    "lr": params["lr"],
    "num_samples": params["num_samples"],
    "ELBO_num_particles": params["ELBO_num_particles"],
    "pruned_K": len(leaflet_model.pi)  # Number of factors retained after pruning
}])

# Save results dataframe
results_file = os.path.join(output_dir, "run_summary.csv")
results_df.to_csv(results_file, index=False)
print(f"Saved run summary to {results_file}")

# Save leafletfa model object 
model_file = os.path.join(output_dir, "leafletfa_model.pkl.xz")

# Save the trained LeafletFA model (without the adata object)
leaflet_model.adata = None
# Remove tensors also 
leaflet_model.y = None
leaflet_model.total_counts = None

# List of attributes to keep
only_things_we_need = [
    'ELBO_num_particles', 'K', 'a', 'a_rate', 'a_shape', 'assign_post', 'b', 
    'b_rate', 'b_shape', 'bb_conc', 'best_elbo', 'best_init', 'gamma', "alpha_pi", 
    'input_conc_prior', 'junc_specific_prior', 'losses', 'phi_samples', 
    'pi', 'psi_learned', 'psi_samples', 'dir_conc', 
    'psis_loc', 'psis_scale'
]

# Iterate over all attributes in the model
for attr_name in dir(leaflet_model):
    attr_value = getattr(leaflet_model, attr_name)
    
    # Check if it's NOT in the whitelist and is NOT a method
    if not attr_name.startswith('__') and attr_name not in only_things_we_need:
        if not callable(attr_value):  # Skip methods
            delattr(leaflet_model, attr_name)

# Check remaining attributes
print(f"Remaining attributes: {dir(leaflet_model)}")

import gc
gc.collect()  # Free memory
print(f"Trying to free up memory before saving model...")

# Save only if num_epochs > 20
if params.get("num_epochs", 0) > 20:
    with gzip.open(model_file, "wb") as f:
        print(f"Starting model save to {model_file}...")

        for attr_name in only_things_we_need:
            attr_value = getattr(leaflet_model, attr_name, None)
            if attr_value is not None:
                print(f"Saving {attr_name}...")  # Track progress
                pickle.dump({attr_name: attr_value}, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Model saved to {model_file}")
else:
    print(f"Model not saved to {model_file} because num_epochs <= 20.")

print("Run complete.")
