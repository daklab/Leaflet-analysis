#!/usr/bin/env python
"""
LeafletFA Input Preparation - Mouse Splicing Foundation (Per Cell Type)

This script:
1. Loads splicing data from aligned AnnData
2. Filters ATSEs based on quality scores
3. For each cell type:
   - Creates a subset of data
   - Computes dimensionality reduction (PCA) on junction ratios
   - Identifies waypoints and creates metacells
   - Generates initializations for LeafletFA model training
   - Saves prepared AnnData with initializations
"""

import os
import sys
import pandas as pd
import numpy as np
import anndata as ad
import datetime
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import TruncatedSVD
import torch
from tqdm import tqdm

# Define module paths for custom modules
src_path = "/gpfs/commons/home/kisaev/Leaflet-private/src/"
if src_path not in sys.path:
    sys.path.append(src_path)

# Import custom modules
import BetaDirichletFactor.waypoints as wayp

# Configuration
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
BASE_OUTPUT_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/062025"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"per_celltype_{timestamp}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}", flush=True)

# Input file paths
SPLICE_INPUT = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/062025/aligned_splicing_data_20250625_182138.h5ad"
print(f"The input file is: {SPLICE_INPUT}")

# Model configuration
N_WAYPOINTS = 30
N_PCA_COMPONENTS = 30
N_DIM_COMPONENTS = 30
METACELL_SIZE = 200

# ATSE filtering parameters
ATSE_FILTER_PERCENTILE = 0.2  # Filter out ATSEs below this percentile

# Cell type filtering parameters
MIN_CELLS_PER_TYPE = 1000  # Minimum number of cells required for a cell type to be processed

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)
if device == torch.device('cuda'):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def load_data():
    """Load splicing data and ATSE file"""
    print("\n>> Loading datasets...")
    
    try:
        # Load splicing data
        print(f"   ⚙️ Loading splicing AnnData from {SPLICE_INPUT}")
        splice_adata = ad.read_h5ad(SPLICE_INPUT)
        splice_adata.obs.reset_index(drop=True, inplace=True)
        splice_adata.obs["cell_id_index"] = splice_adata.obs.index 
        print(f"   ✓ Loaded splicing data with {splice_adata.shape[0]} cells and {splice_adata.shape[1]} junctions")
                
        # Print summary of cell types (already standardized in step 05)
        print(f"   ✓ Data contains {splice_adata.obs['broad_cell_type'].nunique()} standardized cell types")
        print(f"   ✓ Cell type counts: {dict(splice_adata.obs['broad_cell_type'].value_counts())}")
        
        return splice_adata
        
    except Exception as e:
        print(f"   ❌ Error loading datasets: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def compute_atse_scores(splice_adata):
    """Compute quality scores for ATSEs to determine which to keep"""
    print("\n>> Computing ATSE quality scores...")
    
    try:
        # Calculate counts and proportions
        print("   ⚙️ Computing junction expression statistics...")
        splice_adata.var["non_zero_count_cells"] = np.array((splice_adata.X > 0).sum(axis=0)).flatten()
        splice_adata.var["non_zero_cell_prop"] = splice_adata.var["non_zero_count_cells"] / splice_adata.shape[0]
        
        # Calculate component scores
        print("   ⚙️ Computing component quality scores...")
        splice_adata.var["annotation_status_score"] = splice_adata.var["annotation_status"].map(
            {"both": 0.25, "five_prime": 0.5, "three_prime": 0.5, "unannotated": 0.25}
        ) * 2  # Weight more heavily
        
        splice_adata.var["non_zero_cell_prop_score"] = (splice_adata.var["non_zero_cell_prop"] > 0.01).astype(int) * 1.5
        
        # Group by event_id and calculate scores
        print("   ⚙️ Aggregating scores by ATSE...")
        atse_scores = splice_adata.var.groupby("event_id")[
            ["annotation_status_score", "non_zero_cell_prop_score"]
        ].sum()
        
        # Normalize by junction counts
        junction_counts = splice_adata.var["event_id"].value_counts().rename("junction_count")
        atse_scores["atse_score"] = atse_scores.sum(axis=1)
        atse_scores["number_of_junctions"] = junction_counts
        atse_scores["normalized_atse_score"] = atse_scores["atse_score"] / junction_counts
        
        # Calculate percentiles
        score_percentiles = atse_scores["normalized_atse_score"].describe(
            percentiles=[0.1, 0.5, 0.6, 0.9]
        )
        print(f"   ✓ ATSE score percentiles: {dict(score_percentiles)}")
        
        # Filter ATSEs by percentile
        filter_threshold = atse_scores["normalized_atse_score"].quantile(ATSE_FILTER_PERCENTILE)
        atse_scores_filtered = atse_scores[atse_scores["normalized_atse_score"] > filter_threshold]
        
        print(f"   ✓ Filtered to {len(atse_scores_filtered)} ATSEs (top {100-ATSE_FILTER_PERCENTILE*100}%)")
        
        return atse_scores_filtered
        
    except Exception as e:
        print(f"   ❌ Error computing ATSE scores: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def get_valid_cell_types(splice_adata):
    """Get cell types that have sufficient cells for processing"""
    print("\n>> Identifying valid cell types...")
    
    try:
        cell_type_counts = splice_adata.obs['broad_cell_type'].value_counts()
        valid_cell_types = cell_type_counts[cell_type_counts >= MIN_CELLS_PER_TYPE].index.tolist()
        
        print(f"   ✓ Found {len(valid_cell_types)} cell types with >= {MIN_CELLS_PER_TYPE} cells:")
        for cell_type in valid_cell_types:
            print(f"      - {cell_type}: {cell_type_counts[cell_type]} cells")
        
        if len(valid_cell_types) == 0:
            print(f"   ❌ No cell types have >= {MIN_CELLS_PER_TYPE} cells")
            sys.exit(1)
            
        return valid_cell_types
        
    except Exception as e:
        print(f"   ❌ Error identifying valid cell types: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def filter_and_process_junctions(splice_adata, atse_scores_filtered):
    """Filter junctions by ATSE scores and process for model input"""
    print("   ⚙️ Filtering junctions by ATSE scores...")
    
    try:
        # Filter junctions by ATSE scores
        splice_adata_filtered = splice_adata[:, splice_adata.var["event_id"].isin(atse_scores_filtered.index)]
        print(f"   ✓ Filtered to {splice_adata_filtered.shape[1]} junctions in {len(atse_scores_filtered)} ATSEs")
                
        # Reset indices and update junction index
        splice_adata_filtered.var.reset_index(drop=True, inplace=True)
        if 'junction_id_index' in splice_adata_filtered.var.columns:
            splice_adata_filtered.var.rename(columns={'junction_id_index': 'old_junction_id_index'}, inplace=True)
        splice_adata_filtered.var['junction_id_index'] = splice_adata_filtered.var.index
        
        return splice_adata_filtered
        
    except Exception as e:
        print(f"   ❌ Error filtering and processing junctions: {str(e)}")
        traceback.print_exc()
        raise

def compute_dimensionality_reduction(splice_adata_subset):
    """Compute PCA on junction ratio data for a cell type subset"""
    print("   ⚙️ Computing dimensionality reduction...")
    
    try:
        # Use the already calculated centered junction ratios (junc_ratio)
        print("      Using pre-calculated centered PSI values...")
        
        # Perform PCA using sparse data
        print(f"      Computing PCA with {N_PCA_COMPONENTS} components...")
        svd = TruncatedSVD(n_components=min(N_PCA_COMPONENTS, splice_adata_subset.shape[0]-1, splice_adata_subset.shape[1]), random_state=42)
        
        # Fit and transform the junction ratio data
        U = svd.fit_transform(splice_adata_subset.layers["junc_ratio"])
        
        # Get the singular values
        S = svd.singular_values_
        
        # Multiply U by S to get U * S
        U_by_S = U * S
        
        # Store the PCA results in the obsm attribute
        splice_adata_subset.obsm['X_pca'] = U_by_S
        
        # Store explained variance for future reference
        splice_adata_subset.uns['pca_explained_variance_ratio'] = svd.explained_variance_ratio_
        
        print(f"      ✓ PCA complete. Top 5 explained variance: {svd.explained_variance_ratio_[:5]}")
        
        return splice_adata_subset
        
    except Exception as e:
        print(f"      ❌ Error computing dimensionality reduction: {str(e)}")
        traceback.print_exc()
        raise

def identify_waypoints_and_metacells(splice_adata_subset):
    """Identify waypoints and create metacells for LeafletFA initialization"""
    print("   ⚙️ Identifying waypoints and creating metacells...")
    
    try:
        # Dictionary to store waypoints and metacell assignments
        waypoints_dict = {}
        metacell_dicts = {}
        
        # Extract PCA components
        pca_components = splice_adata_subset.obsm["X_pca"]
        
        # Adjust number of waypoints and metacell size based on available cells
        n_cells = splice_adata_subset.shape[0]
        adjusted_waypoints = min(N_WAYPOINTS, n_cells // 50)  # At least 50 cells per waypoint
        adjusted_metacell_size = min(METACELL_SIZE, n_cells // adjusted_waypoints)
        
        if adjusted_waypoints < 5:
            print(f"      ⚠️ Warning: Only {adjusted_waypoints} waypoints possible with {n_cells} cells")
        
        # Generate random seed
        random_seed = np.random.randint(0, 10000 + 1)
        
        # Identify waypoints using max-min sampling
        print(f"      Finding {adjusted_waypoints} waypoints from the PCA components...")
        waypoints = wayp.max_min_sampling(
            pca_components, 
            adjusted_waypoints, 
            num_components=min(N_DIM_COMPONENTS, pca_components.shape[1]), 
            seed=random_seed
        )
        
        # Store waypoints
        waypoints_dict[adjusted_waypoints] = waypoints
        
        # Assign nearest cells to each waypoint (metacells)
        print(f"      Assigning {adjusted_metacell_size} nearest cells to each waypoint...")
        metacell_dict = wayp.assign_nearest_cells(
            waypoints, 
            pca_components, 
            num_nearest=adjusted_metacell_size
        )
        
        # Store metacell dictionaries
        metacell_dicts[adjusted_waypoints] = metacell_dict
        
        print(f"      ✓ Created {adjusted_waypoints} waypoints and metacells")
        
        return waypoints_dict, metacell_dicts
        
    except Exception as e:
        print(f"      ❌ Error identifying waypoints: {str(e)}")
        traceback.print_exc()
        raise

def generate_and_store_initializations(splice_adata_subset, waypoints_dict, metacell_dicts):
    """Generate and store LeafletFA initializations in AnnData"""
    print("   ⚙️ Generating LeafletFA initializations...")
    
    try:
        # Get centered junction ratios
        rho_hat = splice_adata_subset.layers["junc_ratio"]
        
        # Generate initializations
        print("      Computing Phi and Psi initializations...")
        psi_initializations, phi_initializations = wayp.generate_initializations(
            rho_hat, 
            waypoints_dict, 
            metacell_dicts, 
            epsilon=0.001
        )
        
        # Store initializations in AnnData
        for i, n_waypoints in enumerate(waypoints_dict.keys()):
            print(f"      Storing initializations for {n_waypoints} waypoints...")
            
            # Extract initializations
            psi = psi_initializations[i]
            phi = phi_initializations[i]
            
            # Convert to NumPy arrays if they are torch tensors
            if isinstance(psi, torch.Tensor):
                psi = psi.cpu().numpy()
            if isinstance(phi, torch.Tensor):
                phi = phi.cpu().numpy()
            
            # Store in AnnData
            splice_adata_subset.varm[f'psi_init_{n_waypoints}_waypoints'] = psi
            splice_adata_subset.obsm[f'phi_init_{n_waypoints}_waypoints'] = phi
        
        print(f"      ✓ Successfully stored initializations")
        
        return splice_adata_subset
        
    except Exception as e:
        print(f"      ❌ Error generating initializations: {str(e)}")
        traceback.print_exc()
        raise

def process_cell_type(splice_adata_filtered, cell_type):
    """Process a single cell type"""
    print(f"\n>> Processing cell type: {cell_type}")
    
    try:
        # Subset data for this cell type
        cell_mask = splice_adata_filtered.obs['broad_cell_type'] == cell_type
        splice_adata_subset = splice_adata_filtered[cell_mask, :].copy()
        
        # Reset cell indices
        splice_adata_subset.obs.reset_index(drop=True, inplace=True)
        splice_adata_subset.obs["cell_id_index"] = splice_adata_subset.obs.index
        
        print(f"   ✓ Subset contains {splice_adata_subset.shape[0]} cells and {splice_adata_subset.shape[1]} junctions")

        # Reset junction indices
        splice_adata_subset.var.reset_index(drop=True, inplace=True)
        if 'old_junction_id_index' in splice_adata_subset.var.columns:
            splice_adata_subset.var.drop(columns=['old_junction_id_index'], inplace=True)
        splice_adata_subset.var['junction_id_index'] = splice_adata_subset.var.index
                
        # Skip if too few cells or junctions
        if splice_adata_subset.shape[0] < 100 or splice_adata_subset.shape[1] < 100:
            print(f"   ⚠️ Skipping {cell_type}: insufficient cells ({splice_adata_subset.shape[0]}) or junctions ({splice_adata_subset.shape[1]})")
            return None
        
        # Compute dimensionality reduction
        splice_adata_subset = compute_dimensionality_reduction(splice_adata_subset)
        
        # Identify waypoints and create metacells
        waypoints_dict, metacell_dicts = identify_waypoints_and_metacells(splice_adata_subset)
        
        # Generate and store initializations
        splice_adata_subset = generate_and_store_initializations(splice_adata_subset, waypoints_dict, metacell_dicts)
        
        return splice_adata_subset
        
    except Exception as e:
        print(f"   ❌ Error processing cell type {cell_type}: {str(e)}")
        traceback.print_exc()
        return None

def save_cell_type_anndata(splice_adata_subset, cell_type):
    """Save the prepared AnnData object for a specific cell type"""
    print(f"   ⚙️ Saving AnnData for {cell_type}...")
    
    try:
        # Create cell type specific directory
        cell_type_dir = os.path.join(OUTPUT_DIR, cell_type.replace(" ", "_").replace("/", "_"))
        os.makedirs(cell_type_dir, exist_ok=True)
        
        # Define output filename
        output_filename = f"MOUSE_SPLICING_FOUNDATION_{cell_type.replace(' ', '_').replace('/', '_')}_Anndata_ATSE_counts_with_waypoints_{timestamp}.h5ad"
        output_path = os.path.join(cell_type_dir, output_filename)
        
        # Save AnnData object
        splice_adata_subset.write_h5ad(output_path, compression='gzip')
        
        print(f"   ✓ Successfully saved AnnData for {cell_type} to {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"   ❌ Error saving AnnData for {cell_type}: {str(e)}")
        traceback.print_exc()
        return None

def create_summary_file(processed_cell_types, output_paths):
    """Create a summary file with information about processed cell types"""
    try:
        summary_data = []
        for cell_type, output_path in zip(processed_cell_types, output_paths):
            if output_path is not None:
                summary_data.append({
                    'cell_type': cell_type,
                    'output_path': output_path,
                    'status': 'success'
                })
            else:
                summary_data.append({
                    'cell_type': cell_type,
                    'output_path': None,
                    'status': 'failed'
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(OUTPUT_DIR, f"processing_summary_{timestamp}.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n✓ Summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"\n❌ Error creating summary file: {str(e)}")

# Main execution
print("\n========================================")
print("LeafletFA Input Preparation - Mouse Splicing Foundation (Per Cell Type)")
print("========================================\n")

# Load data
splice_adata = load_data()

# Compute ATSE scores and filter
atse_scores_filtered = compute_atse_scores(splice_adata)

# Get valid cell types
valid_cell_types = get_valid_cell_types(splice_adata)

# Filter junctions globally (this will be applied to all cell types)
print("\n>> Filtering junctions globally...")
splice_adata_filtered = filter_and_process_junctions(splice_adata, atse_scores_filtered)

# Process each cell type
processed_cell_types = []
output_paths = []

for cell_type in tqdm(valid_cell_types, desc="Processing cell types"):
    splice_adata_subset = process_cell_type(splice_adata_filtered, cell_type)
    
    if splice_adata_subset is not None:
        output_path = save_cell_type_anndata(splice_adata_subset, cell_type)
        processed_cell_types.append(cell_type)
        output_paths.append(output_path)
    else:
        processed_cell_types.append(cell_type)
        output_paths.append(None)

# Create summary file
create_summary_file(processed_cell_types, output_paths)

print("\n========================================")
print("LeafletFA input preparation complete!")
print(f"Results saved to: {OUTPUT_DIR}")
print(f"Successfully processed {sum(1 for path in output_paths if path is not None)} out of {len(valid_cell_types)} cell types")
print("========================================\n")

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/062025
# sbatch --mem=400G -p cpu,bigmem -J "prep_initialized_AnnData_per_celltype" --wrap="python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/LeafletFA_analysis/02_generate_cell_type_specific_inputs.py"