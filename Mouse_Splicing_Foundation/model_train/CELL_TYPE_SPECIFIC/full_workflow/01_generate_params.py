# generate_params_celltype.py
import itertools
import json
import os
import datetime
import pandas as pd
import glob

# Define base output directory
base_output_dir = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel_celltype/"

# Create output directory if it doesn't exist with today's date inside base_output_dir
today = datetime.datetime.now().strftime("%Y-%m-%d")
base_output_dir = os.path.join(base_output_dir, today)
os.makedirs(base_output_dir, exist_ok=True)
print(f"All outputs will be saved in {base_output_dir}")

# Define the directory where per-cell-type AnnData files are stored
# Update this path to match your actual output from the modified script
input_data_dir = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/062025"

# Find all cell type directories (look for the most recent per_celltype folder)
celltype_dirs = glob.glob(os.path.join(input_data_dir, "per_celltype_*"))
if not celltype_dirs:
    raise FileNotFoundError(f"No per_celltype directories found in {input_data_dir}")

# Use the most recent directory
celltype_base_dir = max(celltype_dirs, key=os.path.getctime)
print(f"Using cell type data from: {celltype_base_dir}")

# Find all cell type subdirectories and their corresponding AnnData files
cell_type_files = {}
for cell_type_dir in os.listdir(celltype_base_dir):
    cell_type_path = os.path.join(celltype_base_dir, cell_type_dir)
    if os.path.isdir(cell_type_path):
        # Look for .h5ad files in this directory
        h5ad_files = glob.glob(os.path.join(cell_type_path, "*.h5ad"))
        if h5ad_files:
            cell_type_files[cell_type_dir] = h5ad_files[0]  # Take the first (should be only one)

print(f"Found {len(cell_type_files)} cell types with AnnData files:")
for cell_type, file_path in cell_type_files.items():
    print(f"  - {cell_type}: {os.path.basename(file_path)}")

# Define parameter grid
param_grid = {
    "input_conc": [None, 'inf'],  # 'inf' will be converted to torch.tensor(np.inf)
    "junc_specific_prior": [True, False],
    "delta_fixed": [1, None],
    "K": [30],  # This might need to be adjusted per cell type based on waypoints
    "waypoints_use": [True],
    "num_inits": [1],
    "ELBO_num_particles": [5],
    "num_samples": [100],
    'gamma': [0.0001],
    'min_delta': [100],
    "lr": [0.9],
    "num_epochs": [1000],
    "patience": [5],
}

# Generate all parameter combinations
param_combinations = list(itertools.product(*param_grid.values()))

# Convert to list of dictionaries and add cell type information
param_list = []
for cell_type, anndata_file in cell_type_files.items():
    for values in param_combinations:
        param_dict = dict(zip(param_grid.keys(), values))
        param_dict['cell_type'] = cell_type
        param_dict['anndata_file'] = anndata_file
        param_list.append(param_dict)

# Save parameter combinations to JSON
param_file = os.path.join(base_output_dir, "parameter_combinations_celltype.json")
with open(param_file, "w") as f:
    json.dump(param_list, f, indent=4)

# Also save as a CSV
param_df = pd.DataFrame(param_list)
param_df.to_csv(os.path.join(base_output_dir, "parameter_combinations_celltype.csv"), index=False)

# Create a summary of cell types and parameter combinations
summary_df = param_df.groupby('cell_type').size().reset_index(name='num_param_combinations')
summary_df.to_csv(os.path.join(base_output_dir, "celltype_summary.csv"), index=False)

print(f"Generated {len(param_list)} parameter sets across {len(cell_type_files)} cell types.")
print(f"Parameter JSON saved to: {param_file}")
print(f"Parameter CSV saved to: {os.path.join(base_output_dir, 'parameter_combinations_celltype.csv')}")
print(f"Cell type summary saved to: {os.path.join(base_output_dir, 'celltype_summary.csv')}")

# Create a mapping file for easy reference
mapping_file = os.path.join(base_output_dir, "celltype_file_mapping.json")
with open(mapping_file, "w") as f:
    json.dump(cell_type_files, f, indent=4)

print(f"Cell type to file mapping saved to: {mapping_file}")

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel_celltype/
# python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/CELL_TYPE_SPECIFIC/full_workflow/01_generate_params.py