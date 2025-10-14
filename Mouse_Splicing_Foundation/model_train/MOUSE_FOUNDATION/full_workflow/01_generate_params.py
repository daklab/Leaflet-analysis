# generate_params.py
import itertools
import json
import os
import datetime
import pandas as pd

# Define output directory
# Define base output directory
base_output_dir = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel/"

# Create output directory if it doesn't exist with today's date inside base_output_dir
today = datetime.datetime.now().strftime("%Y-%m-%d")
base_output_dir = os.path.join(base_output_dir, today)
os.makedirs(base_output_dir, exist_ok=True)
print(f"All outputs will be saved in {base_output_dir}")

# Define parameter grid
param_grid = {
    "input_conc": [None],  # 'inf' will be converted to torch.tensor(np.inf)
    "junc_specific_prior": [True, False],
    "delta_fixed": [None],
    "K": [20],
    "waypoints_use": [True],  # Test both with and without waypoints
    
    # Multi-pass mini-batch parameters
    "batch_size": [4096],  # GPU batch size
    "num_passes": [5, 20],  # Number of times each cell is seen
    "num_epochs_first": [500],  # Epochs for very first batch
    "num_epochs_later": [500],  # Epochs for subsequent batches
    
    # Training parameters
    "ELBO_num_particles": [5],
    "num_samples": [100],
    'gamma': [0.01],  # Learning rate decay
    'min_delta': [100],
    "lr": [0.01, 0.05],  # Initial learning rate
    "patience": [5],
    
    # Data filtering
    "max_junctions": [5],  # Maximum number of junctions per ATSE
}

# Generate all parameter combinations
param_combinations = list(itertools.product(*param_grid.values()))

# Convert to list of dictionaries
param_list = []
for values in param_combinations:
    params = dict(zip(param_grid.keys(), values))
    
    # Add waypoint-specific parameters when waypoints_use is True
    if params["waypoints_use"]:
        # Set n_waypoints to match K
        params["n_waypoints"] = params["K"]
        
        # Set other waypoint parameters
        params["n_pca_components"] = min(params["K"], 20)  # Use K or 20, whichever is smaller
        params["n_dim_components"] = min(params["K"], 20)  # Use K or 20, whichever is smaller
        params["metacell_size"] = 30  # Fixed metacell size
        params["recompute_pca"] = True  # Don't recompute if already exists
    
    # Calculate total batches for reference
    estimated_cells = 150000
    batches_per_pass = int(estimated_cells / params["batch_size"]) + 1
    params["estimated_total_batches"] = batches_per_pass * params["num_passes"]
    
    # Add deprecated parameters for backward compatibility (set to None)
    params["num_inits"] = 1  # Always 1 for mini-batch
    params["num_epochs"] = None  # Not used in mini-batch mode
    
    param_list.append(params)

# Save parameter combinations to JSON
param_file = os.path.join(base_output_dir, "parameter_combinations.json")
with open(param_file, "w") as f:
    json.dump(param_list, f, indent=4)

# Also save as a CSV for easy viewing
param_df = pd.DataFrame(param_list)
param_csv = os.path.join(base_output_dir, "parameter_combinations.csv")
param_df.to_csv(param_csv, index=False)

print(f"Generated {len(param_list)} parameter sets.")
print(f"Parameter JSON saved to: {param_file}")
print(f"Parameter CSV saved to: {param_csv}")

# Print a sample of the parameters to verify
print("\nSample parameter sets:")
for i in range(min(3, len(param_list))):
    print(f"\nParameter set {i}:")
    print(f"  K: {param_list[i]['K']}")
    print(f"  batch_size: {param_list[i]['batch_size']}")
    print(f"  num_passes: {param_list[i]['num_passes']}")
    print(f"  num_epochs_first: {param_list[i]['num_epochs_first']}")
    print(f"  num_epochs_later: {param_list[i]['num_epochs_later']}")
    if param_list[i]["waypoints_use"]:
        print(f"  n_waypoints: {param_list[i]['n_waypoints']}")
        print(f"  n_pca_components: {param_list[i]['n_pca_components']}")
    print(f"  lr: {param_list[i]['lr']}")
    print(f"  gamma: {param_list[i]['gamma']}")
    print(f"  estimated_total_batches: {param_list[i]['estimated_total_batches']}")

# Create a summary of training configurations
training_summary = pd.DataFrame([
    {
        "K": p["K"],
        "waypoints": "Yes" if p["waypoints_use"] else "No",
        "batch_size": p["batch_size"],
        "num_passes": p["num_passes"],
        "epochs_first": p["num_epochs_first"],
        "epochs_later": p["num_epochs_later"],
        "total_batches": p["estimated_total_batches"],
        "lr": p["lr"],
        "gamma": p["gamma"]
    }
    for p in param_list
]).drop_duplicates()

print("\n" + "="*50)
print("Training Configuration Summary:")
print("="*50)
print(f"Total unique configurations: {len(training_summary)}")
print("\nSample configurations:")
print(training_summary.head(10).to_string(index=False))

# Calculate estimated training time
print("\n" + "="*50)
print("Training Time Estimates:")
print("="*50)
for k in param_grid["K"]:
    for passes in param_grid["num_passes"]:
        for epochs_first in param_grid["num_epochs_first"]:
            for epochs_later in param_grid["num_epochs_later"]:
                batches_per_pass = int(estimated_cells / param_grid["batch_size"][0]) + 1
                total_batches = batches_per_pass * passes
                # Rough estimate: first batch takes longer
                estimated_minutes = (epochs_first * 2) + ((total_batches - 1) * epochs_later * 0.5)
                print(f"K={k}, passes={passes}, epochs={epochs_first}/{epochs_later}: ~{estimated_minutes:.0f} minutes")
                break  # Just show one example per K/passes combo
            break
        
# Save summary
summary_file = os.path.join(base_output_dir, "training_summary.csv")
training_summary.to_csv(summary_file, index=False)
print(f"\nTraining summary saved to: {summary_file}")

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel/
# python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/full_workflow/01_generate_params.py