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
OUTPUT_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/072025"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}", flush=True)

# Input file paths
SPLICE_INPUT = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/072025/aligned_splicing_data_20250706_193026.h5ad"
print(f"The input file is: {SPLICE_INPUT}")

# Junction ortho mapping
junc_orthos = "/gpfs/commons/home/kisaev/Leaflet-analysis/Multi_Species_Splicing_Foundation/plots_2025-07-05/junction_mapping_mouse_human_with_annotations.csv"
junc_orthos = pd.read_csv(junc_orthos)

# Model configuration
N_WAYPOINTS = 30
N_PCA_COMPONENTS = 30
N_DIM_COMPONENTS = 30
METACELL_SIZE = 100

# ATSE filtering parameters
ATSE_FILTER_PERCENTILE = 0.8  # Filter out ATSEs below this percentile

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
    return splice_adata
        
def compute_atse_scores(splice_adata):
    """Compute quality scores for ATSEs to determine which to keep"""
    print("\n>> Computing ATSE quality scores...")
    
    try:
        # Ensure junc_ratio is available for variability calculations
        splice_adata = get_junc_ratio(splice_adata)
        
        print("   Computing junction statistics...")
        junction_matrix = splice_adata.layers["cell_by_junction_matrix"]
        junc_ratio_matrix = splice_adata.layers["junc_ratio"]
        
        # Basic expression statistics
        splice_adata.var["non_zero_count_cells"] = np.array((junction_matrix > 0).sum(axis=0)).flatten()
        splice_adata.var["non_zero_cell_prop"] = splice_adata.var["non_zero_count_cells"] / splice_adata.shape[0]
        
        # Total read counts across all cells for each junction
        splice_adata.var["total_read_counts"] = np.array(junction_matrix.sum(axis=0)).flatten()
        
        # Mean read counts per expressing cell
        splice_adata.var["mean_counts_per_cell"] = np.where(
            splice_adata.var["non_zero_count_cells"] > 0,
            splice_adata.var["total_read_counts"] / splice_adata.var["non_zero_count_cells"],
            0
        )
        
        # Calculate junction variability with proper filtering
        print("   Computing junction variability...")
        junction_variability = []
        min_cells = 10
        min_mean_ratio = 0.01
        
        if hasattr(junc_ratio_matrix, 'toarray'):
            junc_ratio_csc = junc_ratio_matrix.tocsc()
            for j in range(junc_ratio_csc.shape[1]):
                col_data = junc_ratio_csc[:, j].toarray().flatten()
                non_zero_values = col_data[col_data > 0]
                
                # Apply filters for meaningful variability
                if len(non_zero_values) < min_cells or np.mean(non_zero_values) < min_mean_ratio:
                    junction_variability.append(0.0)
                else:
                    cv = np.std(non_zero_values) / np.mean(non_zero_values)
                    junction_variability.append(min(cv, 5.0))  # Cap at 5.0
        else:
            for j in range(junc_ratio_matrix.shape[1]):
                values = junc_ratio_matrix[:, j]
                non_zero_values = values[values > 0]
                
                if len(non_zero_values) < min_cells or np.mean(non_zero_values) < min_mean_ratio:
                    junction_variability.append(0.0)
                else:
                    cv = np.std(non_zero_values) / np.mean(non_zero_values)
                    junction_variability.append(min(cv, 5.0))
        
        splice_adata.var["junction_variability"] = junction_variability
        
        print("   Computing component scores...")
        
        # 1. Annotation score
        annotation_weights = {"unannotated": 1.0, "three_prime": 0.8, "five_prime": 0.8, "both": 0.5}
        splice_adata.var["annotation_status_score"] = splice_adata.var["annotation_status"].map(annotation_weights)
        
        # 2. Expression breadth score
        splice_adata.var["expression_breadth_score"] = np.where(
            splice_adata.var["non_zero_cell_prop"] >= 0.05,
            np.log1p(splice_adata.var["non_zero_cell_prop"] * 100),
            0.0
        )
        
        # 3. Read count score (log scale for total reads)
        splice_adata.var["read_count_score"] = np.where(
            splice_adata.var["total_read_counts"] >= 50,  # Minimum 50 total reads
            np.log1p(splice_adata.var["total_read_counts"]),
            0.0
        )
        
        # 4. Variability score (quantile-based)
        var_percentiles = np.percentile([v for v in junction_variability if v > 0], [50, 75, 95])
        if len(var_percentiles) == 3:  # Check if we have enough variable junctions
            splice_adata.var["variability_score"] = np.where(
                splice_adata.var["junction_variability"] >= var_percentiles[2], 2.0,
                np.where(splice_adata.var["junction_variability"] >= var_percentiles[1], 1.5,
                        np.where(splice_adata.var["junction_variability"] >= var_percentiles[0], 1.0, 0.5))
            )
        else:
            splice_adata.var["variability_score"] = 0.5  # Default if no variability
        
        # 5. Conservation score
        conservation_weights = {"conserved": 1.0, "not_conserved": 0.0}
        splice_adata.var["conservation_score"] = splice_adata.var["junction_conserved"].map(conservation_weights)
        
        # Print conservation distribution
        conservation_dist = splice_adata.var["junction_conserved"].value_counts()
        print(f"   Conservation distribution:")
        for status, count in conservation_dist.items():
            print(f"     {status}: {count:,} ({100*count/len(splice_adata.var):.1f}%)")
        
        # Aggregate scores by ATSE
        print("   Aggregating scores by ATSE...")
        score_columns = ["annotation_status_score", "expression_breadth_score", 
                        "read_count_score", "variability_score", "conservation_score"]
        
        atse_scores = splice_adata.var.groupby("event_id")[score_columns].agg({
            "annotation_status_score": ["max", "mean"],
            "expression_breadth_score": "mean",
            "read_count_score": "mean",          # Average read support
            "variability_score": "max",
            "conservation_score": "max"
        })
        
        # Flatten column names
        atse_scores.columns = ['_'.join(col).strip() for col in atse_scores.columns.values]
        
        # Calculate composite score with updated weights
        weights = {
            "annotation": 0.25,     # Annotation novelty
            "expression": 0.15,     # Expression breadth
            "read_count": 0.15,     # Read support
            "variability": 0.20,    # PSI variability
            "conservation": 0.25    # Evolutionary conservation
        }
        
        annotation_component = (0.7 * atse_scores["annotation_status_score_max"] + 
                               0.3 * atse_scores["annotation_status_score_mean"])
        
        atse_scores["composite_score"] = (
            weights["annotation"] * annotation_component +
            weights["expression"] * atse_scores["expression_breadth_score_mean"] +
            weights["read_count"] * atse_scores["read_count_score_mean"] +
            weights["variability"] * atse_scores["variability_score_max"] +
            weights["conservation"] * atse_scores["conservation_score_max"]
        )
        
        # Add junction count and filter
        junction_counts = splice_adata.var["event_id"].value_counts()
        atse_scores["number_of_junctions"] = junction_counts
        
        filter_threshold = atse_scores["composite_score"].quantile(ATSE_FILTER_PERCENTILE)
        atse_scores_filtered = atse_scores[atse_scores["composite_score"] > filter_threshold]
        
        # Print diagnostics
        print(f"   Filtered to {len(atse_scores_filtered):,} ATSEs from {len(atse_scores):,}")
        
        filtered_junctions = splice_adata.var[splice_adata.var["event_id"].isin(atse_scores_filtered.index)]
        print(f"   Remaining junctions: {len(filtered_junctions):,}")
        
        # Show distributions
        for col, name in [("annotation_status", "annotation"), ("junction_conserved", "conservation")]:
            dist = filtered_junctions[col].value_counts()
            print(f"   {name.title()} distribution:")
            for status, count in dist.items():
                print(f"     {status}: {count:,} ({100*count/len(filtered_junctions):.1f}%)")
        
        # ATSEs with conserved junctions
        conserved_atses = atse_scores_filtered[atse_scores_filtered["conservation_score_max"] > 0]
        print(f"   ATSEs with conserved junctions: {len(conserved_atses):,}/{len(atse_scores_filtered):,} "
              f"({100*len(conserved_atses)/len(atse_scores_filtered):.1f}%)")
        
        return atse_scores_filtered
        
    except Exception as e:
        print(f"   Error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def filter_and_process_junctions(splice_adata, atse_scores_filtered):
    """Filter junctions by ATSE scores and process for model input"""
    print("\n>> Filtering and processing junctions...")
    
    try:
        # Filter junctions by ATSE scores - force a copy to avoid view issues
        print("   Filtering junctions by ATSE scores...")
        splice_adata = splice_adata[:, splice_adata.var["event_id"].isin(atse_scores_filtered.index)].copy()
        print(f"   ✓ Filtered to {splice_adata.shape[1]} junctions in {len(atse_scores_filtered)} ATSEs")
        
        # Now reset indices safely
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

def get_junc_ratio(splice_adata):
    """Enhanced version with error checking"""
    if "junc_ratio" not in splice_adata.layers:
        print("   Computing sparse centered PSI values...")
        try:
            # Update junction_counts and cluster_counts
            junction_counts = splice_adata.layers["cell_by_junction_matrix"]
            cluster_counts = splice_adata.layers["cell_by_cluster_matrix"]
            
            # Convert to COO if needed for the wayp function
            if not isinstance(junction_counts, coo_matrix):
                junction_counts = junction_counts.tocoo()
            if not isinstance(cluster_counts, coo_matrix):
                cluster_counts = cluster_counts.tocoo()
            
            # Get sparse centered PSI values  
            junc_ratio = wayp.calculate_centered_psi(junction_counts, cluster_counts)
            
            # Convert result to CSR format immediately
            if isinstance(junc_ratio, coo_matrix):
                splice_adata.layers["junc_ratio"] = junc_ratio.tocsr()
            else:
                splice_adata.layers["junc_ratio"] = junc_ratio
                
            print(f"   ✓ Successfully computed junc_ratio layer (CSR format)", flush=True)
            
        except Exception as e:
            print(f"   Error computing junc_ratio: {str(e)}")
            # Fallback: create dummy ratios if PSI calculation fails
            splice_adata.layers["junc_ratio"] = csr_matrix(splice_adata.X.shape)
            print("   Warning: Using dummy junc_ratio values")
    else:
        print("   junc_ratio layer already exists")
        # Ensure existing junc_ratio is also CSR
        if isinstance(splice_adata.layers["junc_ratio"], coo_matrix):
            splice_adata.layers["junc_ratio"] = splice_adata.layers["junc_ratio"].tocsr()
            print("   Converted existing junc_ratio from COO to CSR")
    
    return splice_adata

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
        waypoint_str = str(N_WAYPOINTS)
        # Add number of junctions used to the filename
        num_junctions = splice_adata.shape[1]
        output_filename = f"HUMAN_SPLICING_FOUNDATION_Anndata_ATSE_counts_{num_junctions}_junctions_{waypoint_str}_waypoints_{timestamp}.h5ad"
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
print("LeafletFA Input Preparation - Human Splicing Foundation")
print("========================================\n")

# Load data
splice_adata = load_data()

# Keep only small size ATSEs (<= 5)
small_atses_only = True 

if small_atses_only: 
    print(small_atses_only)
    print("Keeping only ATSEs with <=5 junctions to maintain interpretability!")
    splice_adata = splice_adata[:, splice_adata.var["num_junctions"] <=5]

# Initialize the column with default value
splice_adata.var["junction_conserved"] = "not_conserved"

# Create set of conserved junction IDs for fast lookup
conserved_junction_ids = set(junc_orthos["human_junction_id"])

print(f"Total junctions in splice_adata: {splice_adata.var.shape[0]:,}")
print(f"Conserved junctions in mapping: {len(conserved_junction_ids):,}")

# Mark conserved junctions
conserved_mask = splice_adata.var["junction_id"].isin(conserved_junction_ids)
splice_adata.var.loc[conserved_mask, "junction_conserved"] = "conserved"

# Summary statistics
conserved_count = (splice_adata.var["junction_conserved"] == "conserved").sum()
not_conserved_count = (splice_adata.var["junction_conserved"] == "not_conserved").sum()

print(f"\nConservation labeling results:")
print(f"  Conserved junctions: {conserved_count:,} ({conserved_count/len(splice_adata.var)*100:.1f}%)")
print(f"  Not conserved: {not_conserved_count:,} ({not_conserved_count/len(splice_adata.var)*100:.1f}%)")

# Convert COO matrices to CSR immediately after loading
print("\n>> Converting COO matrices to CSR format...")
for layer_name, layer_data in splice_adata.layers.items():
    if isinstance(layer_data, coo_matrix):
        splice_adata.layers[layer_name] = layer_data.tocsr()
        print(f"   Converted {layer_name} from COO to CSR")

# Compute ATSE scores and filter
atse_scores_filtered = compute_atse_scores(splice_adata)

# Filter junctions and process
splice_adata = filter_and_process_junctions(splice_adata, atse_scores_filtered)

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
# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/072025
# sbatch --mem=200G -p cpu,bigmem -J "prep_initialized_AnnData" --wrap="python /gpfs/commons/home/kisaev/Leaflet-analysis/Human_Splicing_Foundation/LeafletFA_analysis/01_prep_initialized_AnnData.py"
