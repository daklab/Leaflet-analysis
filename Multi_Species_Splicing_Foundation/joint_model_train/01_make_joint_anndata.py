# Load packages 
import anndata as ad 
import pandas as pd 
import numpy as np

# To-do:
# Make sure num_junctions column exists 
# Make sure junction counts and ATSE counts consistent 
# RuntimeError: indices and values must have same nnz, but got nnz from indices: 10906620, nnz from values: 11613541

# Paths 
orthologous_junctions_path = "/gpfs/commons/home/kisaev/Leaflet-analysis/Multi_Species_Splicing_Foundation/plots_2025-08-14/junction_mapping_mouse_human_with_annotations.csv"
orthologous_junctions = pd.read_csv(orthologous_junctions_path)

# Load mouse anndata 
mouse_anndata_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/aligned_splicing_data_20250730_164104.h5ad"
mouse = ad.read_h5ad(mouse_anndata_file)

# Load human anndata 
human_anndata_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/072025/aligned_splicing_data_20250731_212313.h5ad"
human = ad.read_h5ad(human_anndata_file)

mouse_present = orthologous_junctions["mouse_junction_id"].isin(mouse.var["junction_id"])
human_present = orthologous_junctions["human_junction_id"].isin(human.var["junction_id"])

# Subset orthologous junctions to only include junctions that are present in mouse and human
orthologous_junctions = orthologous_junctions[mouse_present & human_present]

# Filter mouse anndata for orthologous junctions
mouse = mouse[:, mouse.var["junction_id"].isin(orthologous_junctions["mouse_junction_id"])].copy()

# Filter human anndata for orthologous junctions
human = human[:, human.var["junction_id"].isin(orthologous_junctions["human_junction_id"])].copy()

# Create universal junction id hash for orthologous junctions
orthologous_junctions["joint_junction_id"] = orthologous_junctions["mouse_junction_id"] + "_" + orthologous_junctions["human_junction_id"]

human_ortho = orthologous_junctions[["human_junction_id", "joint_junction_id"]]
human_ortho.columns = ["junction_id", "joint_junction_id"]

mouse_ortho = orthologous_junctions[["mouse_junction_id", "joint_junction_id"]]
mouse_ortho.columns = ["junction_id", "joint_junction_id"]

# Merge mouse.var and human.var with orthologous_junctions
mouse.var = mouse.var.merge(mouse_ortho, on="junction_id", how="left")
human.var = human.var.merge(human_ortho, on="junction_id", how="left")

# Order mouse and human anndatas by joint junction id
mouse.var = mouse.var.sort_values(by="joint_junction_id")
human.var = human.var.sort_values(by="joint_junction_id")

# Assert that order of joint junction id is the same in mouse and human anndatas
assert np.all(mouse.var["joint_junction_id"].values == human.var["joint_junction_id"].values), "ERROR: Order of joint junction id is not the same in mouse and human anndatas!"

# Now we gotta clean up the .obs to ensure similar columns 
mouse.obs["species"] = "mouse"
human.obs["species"] = "human"

# Standardize .obs columns for concatenation
print("\nStandardizing .obs columns...")

# Columns to drop
cols_to_drop = ['AgingScore_unweighted', 'AgingScore_pos', 'AgingScore_neg']

# Drop columns from mouse.obs if they exist
mouse_cols_to_drop = [col for col in cols_to_drop if col in mouse.obs.columns]
if mouse_cols_to_drop:
    mouse.obs = mouse.obs.drop(columns=mouse_cols_to_drop)
    print(f"Dropped from mouse: {mouse_cols_to_drop}")

# Drop columns from human.obs if they exist
human_cols_to_drop = [col for col in cols_to_drop if col in human.obs.columns]
if human_cols_to_drop:
    human.obs = human.obs.drop(columns=human_cols_to_drop)
    print(f"Dropped from human: {human_cols_to_drop}")

# Rename columns for consistency
mouse_rename_dict = {
    'cell_ontology_class': 'cell_type',
    'mouse.id': 'donor'
}
mouse.obs = mouse.obs.rename(columns=mouse_rename_dict)
print(f"Renamed in mouse: {mouse_rename_dict}")

human_rename_dict = {
    'cell_id_clean': 'cell_clean'
}
human.obs = human.obs.rename(columns=human_rename_dict)
print(f"Renamed in human: {human_rename_dict}")

print("\n.obs standardization complete.")

# Display cleaned columns for verification
print("\nMouse OBS columns after cleaning:")
print(mouse.obs.columns.tolist())

print("\nHuman OBS columns after cleaning:")
print(human.obs.columns.tolist())

# Find common columns for verification
common_cols = list(set(mouse.obs.columns) & set(human.obs.columns))
print(f"\nFound {len(common_cols)} common columns between mouse and human:")
print(sorted(common_cols))

# Columns unique to mouse
mouse_unique = list(set(mouse.obs.columns) - set(human.obs.columns))
print(f"\n{len(mouse_unique)} columns unique to mouse:")
print(sorted(mouse_unique))

# Columns unique to human
human_unique = list(set(human.obs.columns) - set(mouse.obs.columns))
print(f"\n{len(human_unique)} columns unique to human:")
print(sorted(human_unique))

# Subset mouse and human anndatas to only include common columns
mouse.obs = mouse.obs[common_cols]
human.obs = human.obs[common_cols]

# Make sure the columns in both .obs are the same order 
assert np.all(mouse.obs.columns.values == human.obs.columns.values), "ERROR: Columns in .obs are not the same order in mouse and human anndatas!"

# Convert mouse age to numeric values   
mouse.obs["age"] = mouse.obs["age"].str.replace("m", "").astype(float)
mouse.obs["age_units"] = "months"
human.obs["age_units"] = "years"

# Try concatenating mouse and human anndatas    
combined = ad.concat([mouse, human], axis=0, join='outer', 
                     index_unique='-', fill_value=0)

print(f"Combined shape: {combined.shape}")
print(f"Mouse cells: {sum(combined.obs['species'] == 'mouse')}")
print(f"Human cells: {sum(combined.obs['species'] == 'human')}")
print(f"Layers: {list(combined.layers.keys())}")

# Save into anndata file
output_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/CROSS_SPECIES_AGING/Leaflet/input_files/joint_anndata_20250903.h5ad"
combined.write_h5ad(output_file)