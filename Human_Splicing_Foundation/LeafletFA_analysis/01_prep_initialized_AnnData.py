#!/usr/bin/env python
"""
LeafletFA Input Preparation - Human Splicing Foundation

This script:
1. Loads splicing data from aligned AnnData
2. Filters ATSEs based on quality scores
3. Computes dimensionality reduction (PCA) on junction ratios
4. Identifies waypoints and creates metacells
5. Generates initializations for LeafletFA model training
6. Saves prepared AnnData with initializations
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
OUTPUT_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/062025"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}", flush=True)

# Input file paths
SPLICE_INPUT = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/062025/splice_adata_matched_2025-06-06.h5ad"
ATSE_FILE = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/ATSE_mapper/ATSE_files/stella_gtf/TMS_atse_file_unanno_also_2025-05-11_06-23-05.txt.gz"

# Model configuration
N_WAYPOINTS = 50
N_PCA_COMPONENTS = 50
N_DIM_COMPONENTS = 30
METACELL_SIZE = 200

# ATSE filtering parameters
ATSE_FILTER_PERCENTILE = 0.3  # Filter out ATSEs below this percentile

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)
if device == torch.device('cuda'):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def load_data():
    """Load splicing data and ATSE file"""
    print("\n>> Loading datasets...")
    # Load splicing data
    print(f"   Loading splicing AnnData from {SPLICE_INPUT}")
    splice_adata = ad.read_h5ad(SPLICE_INPUT)
    splice_adata.obs.reset_index(drop=True, inplace=True)
    splice_adata.obs["cell_id_index"] = splice_adata.obs.index 
    print(f"   Loaded splicing data with {splice_adata.shape[0]} cells and {splice_adata.shape[1]} junctions")
    
    # Load ATSE file
    print(f"   Loading ATSE information from {ATSE_FILE}")
    atses = pd.read_csv(ATSE_FILE, sep="\t")
    print(f"   Loaded ATSE info with {len(atses['event_id'].unique())} unique events")

    print(splice_adata.obs.dataset.value_counts())
    print(splice_adata.obs.tissue.value_counts())
    print(splice_adata.obs["age"].value_counts())

    # Assign sequencing technology based on source
    splice_adata.obs["seqtech"] = "single_nuclei"
    splice_adata.obs.loc[splice_adata.obs["dataset"] == "tabula_sapiens", "seqtech"] = "single_cell"
    print(splice_adata.obs["seqtech"].value_counts())
 
    # Print summary of cell types (already standardized in step 05)
    print(f"   Data contains {splice_adata.obs['broad_cell_type'].nunique()} standardized cell types")
    print(f"   Top 5 cell types: {dict(splice_adata.obs['broad_cell_type'].value_counts().head(5))}")
    return splice_adata, atses
        
def compute_atse_scores(splice_adata):
    """Compute quality scores for ATSEs to determine which to keep"""
    print("\n>> Computing ATSE quality scores...")
    
    try:
        # Calculate counts and proportions
        print("   Computing junction expression statistics...")
        splice_adata.var["non_zero_count_cells"] = np.array((splice_adata.X > 0).sum(axis=0)).flatten()
        splice_adata.var["non_zero_cell_prop"] = splice_adata.var["non_zero_count_cells"] / splice_adata.shape[0]
        
        # Calculate component scores
        print("   Computing component quality scores...")
        splice_adata.var["annotation_status_score"] = splice_adata.var["annotation_status"].map(
            {"both": 1, "five_prime": 0.5, "three_prime": 0.5, "unannotated": 0}
        ) * 2  # Weight more heavily
        
        splice_adata.var["non_zero_cell_prop_score"] = (splice_adata.var["non_zero_cell_prop"] > 0.01).astype(int) * 1.5
        
        # Group by event_id and calculate scores
        print("   Aggregating scores by ATSE...")
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
        print(f"   ATSE score percentiles: {dict(score_percentiles)}")
        
        # Filter ATSEs by percentile
        filter_threshold = atse_scores["normalized_atse_score"].quantile(ATSE_FILTER_PERCENTILE)
        atse_scores_filtered = atse_scores[atse_scores["normalized_atse_score"] > filter_threshold]
        
        print(f"   Filtered to {len(atse_scores_filtered)} ATSEs (top {100-ATSE_FILTER_PERCENTILE*100}%)")
        
        return atse_scores_filtered
        
    except Exception as e:
        print(f"   Error computing ATSE scores: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def filter_and_process_junctions(splice_adata, atses, atse_scores_filtered):
    """Filter junctions by ATSE scores and process for model input"""
    print("\n>> Filtering and processing junctions...")
    
    try:
        # Filter junctions by ATSE scores
        print("   Filtering junctions by ATSE scores...")
        splice_adata = splice_adata[:, splice_adata.var["event_id"].isin(atse_scores_filtered.index)]
        print(f"   ✓ Filtered to {splice_adata.shape[1]} junctions in {len(atse_scores_filtered)} ATSEs")
        
        # Merge ATSE metadata
        print("   Merging ATSE metadata...")
        splice_adata.var = splice_adata.var.merge(
            atses[["gene_id", "gene_name", "junction_id", "annotation_status", 
                  "position_off_5_prime", "position_off_3_prime"]], 
            on=["junction_id", "gene_id", "annotation_status", "gene_name", 
               "position_off_5_prime", "position_off_3_prime"]
        )
        
        # Reset indices and update junction index
        splice_adata.var.reset_index(drop=True, inplace=True)
        if 'junction_id_index' in splice_adata.var.columns:
            splice_adata.var.rename(columns={'junction_id_index': 'old_junction_id_index'}, inplace=True)
        splice_adata.var['junction_id_index'] = splice_adata.var.index
        
        print(f"   ✓ Junction information processed for model input")
        
        return splice_adata
        
    except Exception as e:
        print(f"   Error filtering and processing junctions: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def compute_dimensionality_reduction(splice_adata):
    """Compute PCA on junction ratio data"""
    print("\n>> Computing dimensionality reduction...")
    
    # Use the already calculated centered junction ratios (junc_ratio) or
    # check first if junc_ratio layer exists
    if "junc_ratio" not in splice_adata.layers:
        # Need to get sparse centered PSI values
        print("   Computing sparse centered PSI values...")
        # Update junction_counts and cluster_counts
        junction_counts = splice_adata.layers["cell_by_junction_matrix"].tocoo()
        cluster_counts = splice_adata.layers["cell_by_cluster_matrix"].tocoo()
        # Get sparse centered PSI values 
        splice_adata.layers["junc_ratio"] = wayp.calculate_centered_psi(junction_counts, cluster_counts)
        print(f"Done getting sparse centered PSI values!", flush=True)
        
    # Perform PCA using sparse data
    print(f"  Computing PCA with {N_PCA_COMPONENTS} components...")
    svd = TruncatedSVD(n_components=N_PCA_COMPONENTS, random_state=42)
    U = svd.fit_transform(splice_adata.layers["junc_ratio"])
    # Get the singular values
    S = svd.singular_values_
    # Multiply U by S to get U * S
    U_by_S = U * S
    # Store the PCA results in the obsm attribute
    splice_adata.obsm['X_pca'] = U_by_S
    # Store explained variance for future reference
    splice_adata.uns['pca_explained_variance_ratio'] = svd.explained_variance_ratio_
    print(f" PCA complete. Top 5 explained variance: {svd.explained_variance_ratio_[:5]}")
    return splice_adata
        
def identify_waypoints_and_metacells(splice_adata):
    """Identify waypoints and create metacells for LeafletFA initialization"""
    print("\n>> Identifying waypoints and creating metacells...")
    
    try:
        # Dictionary to store waypoints and metacell assignments
        waypoints_dict = {}
        metacell_dicts = {}
        
        # Extract PCA components
        pca_components = splice_adata.obsm["X_pca"]
        
        # Generate random seed
        random_seed = np.random.randint(0, 10000 + 1)
        
        # Identify waypoints using max-min sampling
        print(f"   Finding {N_WAYPOINTS} waypoints from the PCA components...")
        waypoints = wayp.max_min_sampling(
            pca_components, 
            N_WAYPOINTS, 
            num_components=N_DIM_COMPONENTS, 
            seed=random_seed
        )
        
        # Store waypoints
        waypoints_dict[N_WAYPOINTS] = waypoints
        
        # Assign nearest cells to each waypoint (metacells)
        print(f"  Assigning {METACELL_SIZE} nearest cells to each waypoint...")
        metacell_dict = wayp.assign_nearest_cells(
            waypoints, 
            pca_components, 
            num_nearest=METACELL_SIZE
        )
        
        # Store metacell dictionaries
        metacell_dicts[N_WAYPOINTS] = metacell_dict
        
        print(f"   ✓ Created {N_WAYPOINTS} waypoints and metacells")
        
        return waypoints_dict, metacell_dicts
        
    except Exception as e:
        print(f"  Error identifying waypoints: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def generate_and_store_initializations(splice_adata, waypoints_dict, metacell_dicts):
    """Generate and store LeafletFA initializations in AnnData"""
    print("\n>> Generating LeafletFA initializations...")
    
    try:
            
        # Get centered junction ratios
        rho_hat = splice_adata.layers["junc_ratio"]
        
        # Generate initializations
        print("  Computing Phi and Psi initializations...")
        psi_initializations, phi_initializations = wayp.generate_initializations(
            rho_hat, 
            waypoints_dict, 
            metacell_dicts, 
            epsilon=0.001
        )
        
        # Store initializations in AnnData
        for i, n_waypoints in enumerate(waypoints_dict.keys()):
            print(f" Storing initializations for {n_waypoints} waypoints...")
            
            # Extract initializations
            psi = psi_initializations[i]
            phi = phi_initializations[i]
            
            # Convert to NumPy arrays if they are torch tensors
            if isinstance(psi, torch.Tensor):
                psi = psi.cpu().numpy()
            if isinstance(phi, torch.Tensor):
                phi = phi.cpu().numpy()
            
            # Store in AnnData
            splice_adata.varm[f'psi_init_{n_waypoints}_waypoints'] = psi
            splice_adata.obsm[f'phi_init_{n_waypoints}_waypoints'] = phi
        
        print(f" Successfully stored initializations")
        
        return splice_adata
        
    except Exception as e:
        print(f" Error generating initializations: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def save_prepared_anndata(splice_adata):
    """Save the prepared AnnData object for LeafletFA training"""
    print("\n>> Saving prepared AnnData object...")
    
    try:
        # Define output filename specify what waypoints were used 
        # N_WAYPOINTS is just a number, not a list
        waypoint_str = "_".join(str(N_WAYPOINTS))
        output_filename = f"HUMAN_SPLICING_FOUNDATION_Anndata_ATSE_counts_{waypoint_str}waypoints_{timestamp}.h5ad"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Remove unnecessary columns from var
        if 'old_junction_id_index' in splice_adata.var.columns:
            splice_adata.var.drop(columns=['old_junction_id_index'], inplace=True)
        
        for key in splice_adata.layers.keys():
            if isinstance(splice_adata.layers[key], coo_matrix):
                splice_adata.layers[key] = splice_adata.layers[key].tocsr()

        # Save AnnData object
        print(f" Saving AnnData to {output_path}...")
        splice_adata.write_h5ad(output_path, compression='lzf')
        
        print(f" Successfully saved prepared AnnData")
        return True
        
    except Exception as e:
        print(f"  Error saving AnnData: {str(e)}")
        traceback.print_exc()
        return False

# Main execution
print("\n========================================")
print("LeafletFA Input Preparation - Mouse Splicing Foundation")
print("========================================\n")

# Load data
splice_adata, atses = load_data()

# Compute ATSE scores and filter
atse_scores_filtered = compute_atse_scores(splice_adata)

# Filter junctions and process
splice_adata = filter_and_process_junctions(splice_adata, atses, atse_scores_filtered)

# Compute dimensionality reduction
splice_adata = compute_dimensionality_reduction(splice_adata)

# Identify waypoints and create metacells
waypoints_dict, metacell_dicts = identify_waypoints_and_metacells(splice_adata)

# Generate and store initializations
splice_adata = generate_and_store_initializations(splice_adata, waypoints_dict, metacell_dicts)

# Save prepared AnnData
save_prepared_anndata(splice_adata)

print("\n========================================")
print("LeafletFA input preparation complete!")
print(f"Results saved to: {OUTPUT_DIR}")
print("========================================\n")

# Submission command for reference
# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/062025
# sbatch --mem=400G -p cpu,bigmem -J "prep_initialized_AnnData" --wrap="python /gpfs/commons/home/kisaev/Leaflet-analysis/Human_Splicing_Foundation/LeafletFA_analysis/01_prep_initialized_AnnData.py"
