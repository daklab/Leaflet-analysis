#!/usr/bin/env python
"""
LeafletFA Input Preparation - Mouse Splicing Foundation (Per Cell Type)

This script:
1. Loads splicing data from aligned AnnData
2. For each cell type:
   - Creates a subset of data
   - Computes ATSE quality scores specific to that cell type
   - Filters ATSEs based on cell-type-specific quality scores
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
import gc
import umap

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
SPLICE_INPUT = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/062025/aligned_splicing_data_20250704_231739.h5ad"
print(f"The input file is: {SPLICE_INPUT}")

# Model configuration
N_WAYPOINTS = 30
N_PCA_COMPONENTS = 30
N_DIM_COMPONENTS = 30
METACELL_SIZE = 20

# ATSE filtering parameters
ATSE_FILTER_PERCENTILE = 0.6  # Filter out ATSEs below this percentile
MIN_JUNCTIONS_PER_ATSE = 2    # Minimum number of junctions an ATSE must have
MIN_CELLS_EXPRESSING = 10      # Minimum number of cells expressing an ATSE

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

def compute_atse_scores_for_celltype(splice_adata_subset):
    """Compute quality scores for ATSEs specific to a cell type subset"""
    print("      ⚙️ Computing cell-type-specific ATSE quality scores...")
    
    try:
        # Calculate counts and proportions for this cell type
        splice_adata_subset.var["non_zero_count_cells"] = np.array((splice_adata_subset.X > 0).sum(axis=0)).flatten()
        splice_adata_subset.var["non_zero_cell_prop"] = splice_adata_subset.var["non_zero_count_cells"] / splice_adata_subset.shape[0]
        
        # Calculate mean expression per junction
        if hasattr(splice_adata_subset.X, 'toarray'):
            mean_expr = np.array(splice_adata_subset.X.mean(axis=0)).flatten()
        else:
            mean_expr = splice_adata_subset.X.mean(axis=0)
        splice_adata_subset.var["mean_expression"] = mean_expr
        
        # Calculate component scores with cell-type-specific weighting
        splice_adata_subset.var["annotation_status_score"] = splice_adata_subset.var["annotation_status"].map(
            {"both": 0.4, "five_prime": 0.5, "three_prime": 0.5, "unannotated": 0.4}
        )
        
        # Score based on expression in this cell type
        splice_adata_subset.var["expression_score"] = (
            (splice_adata_subset.var["non_zero_cell_prop"] > 0.05).astype(float) * 0.3 +
            (splice_adata_subset.var["non_zero_cell_prop"] > 0.1).astype(float) * 0.3 +
            (splice_adata_subset.var["non_zero_cell_prop"] > 0.2).astype(float) * 0.4
        )
        
        # Add variability score (coefficient of variation)
        if hasattr(splice_adata_subset.X, 'toarray'):
            X_dense = splice_adata_subset.X.toarray()
        else:
            X_dense = splice_adata_subset.X
        
        var_expr = np.var(X_dense, axis=0)
        cv = np.divide(np.sqrt(var_expr), mean_expr, out=np.zeros_like(var_expr), where=mean_expr != 0)
        splice_adata_subset.var["cv_score"] = (cv > np.percentile(cv[cv > 0], 25)).astype(float) * 0.2
        
        # Group by event_id and calculate scores
        atse_scores = splice_adata_subset.var.groupby("event_id").agg({
            "annotation_status_score": "sum",
            "expression_score": "sum",
            "cv_score": "sum",
            "non_zero_count_cells": "sum",
            "junction_id": "count"
        }).rename(columns={"junction_id": "junction_count"})
        
        # Calculate total score
        atse_scores["atse_score"] = (
            atse_scores["annotation_status_score"] + 
            atse_scores["expression_score"] + 
            atse_scores["cv_score"]
        )
        
        # Normalize by junction count
        atse_scores["normalized_atse_score"] = atse_scores["atse_score"] / atse_scores["junction_count"]
        
        # Apply minimum requirements
        atse_scores = atse_scores[
            (atse_scores["junction_count"] >= MIN_JUNCTIONS_PER_ATSE) &
            (atse_scores["non_zero_count_cells"] >= MIN_CELLS_EXPRESSING)
        ]
        
        # Filter by percentile
        if len(atse_scores) > 0:
            filter_threshold = atse_scores["normalized_atse_score"].quantile(ATSE_FILTER_PERCENTILE)
            atse_scores_filtered = atse_scores[atse_scores["normalized_atse_score"] > filter_threshold]
            
            print(f"      ✓ Filtered to {len(atse_scores_filtered)} ATSEs (top {100-ATSE_FILTER_PERCENTILE*100}%)")
            print(f"      ✓ Score range: [{atse_scores_filtered['normalized_atse_score'].min():.3f}, {atse_scores_filtered['normalized_atse_score'].max():.3f}]")
        else:
            atse_scores_filtered = atse_scores
            print(f"      ⚠️ No ATSEs passed minimum requirements")
        
        return atse_scores_filtered
        
    except Exception as e:
        print(f"      ❌ Error computing ATSE scores: {str(e)}")
        traceback.print_exc()
        raise

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

def filter_junctions_by_atse_scores(splice_adata_subset, atse_scores_filtered):
    """Filter junctions by cell-type-specific ATSE scores"""
    print("      ⚙️ Filtering junctions by ATSE scores...")
    
    try:
        # Filter junctions by ATSE scores
        junction_mask = splice_adata_subset.var["event_id"].isin(atse_scores_filtered.index)
        splice_adata_filtered = splice_adata_subset[:, junction_mask].copy()
        
        print(f"      ✓ Filtered to {splice_adata_filtered.shape[1]} junctions in {len(atse_scores_filtered)} ATSEs")
        
        # Reset junction indices
        splice_adata_filtered.var.reset_index(drop=True, inplace=True)
        if 'junction_id_index' in splice_adata_filtered.var.columns:
            splice_adata_filtered.var.rename(columns={'junction_id_index': 'old_junction_id_index'}, inplace=True)
        splice_adata_filtered.var['junction_id_index'] = splice_adata_filtered.var.index
        
        return splice_adata_filtered
        
    except Exception as e:
        print(f"      ❌ Error filtering junctions: {str(e)}")
        traceback.print_exc()
        raise

def compute_dimensionality_reduction(splice_adata_subset, cell_type):
    """Compute PCA and UMAP on junction ratio data for a cell type subset"""
    print("   ⚙️ Computing dimensionality reduction...")
    
    try:
        # Use the already calculated centered junction ratios (junc_ratio)
        print("      Using pre-calculated centered PSI values...")
        
        # Perform PCA using sparse data
        n_components = min(N_PCA_COMPONENTS, splice_adata_subset.shape[0]-1, splice_adata_subset.shape[1])
        print(f"      Computing PCA with {n_components} components...")
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        
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
        
        print(f"      ✓ PCA complete. Total explained variance: {svd.explained_variance_ratio_.sum():.3f}")
        
        # Compute UMAP
        print("      Computing UMAP embedding...")
        umap_reducer = umap.UMAP(
            n_neighbors=30,
            min_dist=0.3,
            n_components=2,
            random_state=42,
            metric='euclidean'
        )
        
        # Use the PCA components as input to UMAP
        umap_embedding = umap_reducer.fit_transform(U_by_S)
        splice_adata_subset.obsm['X_umap'] = umap_embedding
        
        print("      ✓ UMAP complete")
        
        # Create UMAP visualization
        print("      Creating UMAP visualization...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: UMAP colored by broad_cell_type (should be uniform for this subset)
        ax1.scatter(umap_embedding[:, 0], umap_embedding[:, 1], 
                   c='steelblue', alpha=0.6, s=10)
        ax1.set_xlabel('UMAP1')
        ax1.set_ylabel('UMAP2')
        ax1.set_title(f'UMAP - {cell_type}\n({splice_adata_subset.shape[0]} cells)')
        
        # Plot 2: UMAP colored by a metadata feature if available
        if 'dataset' in splice_adata_subset.obs.columns:
            # Color by dataset
            datasets = splice_adata_subset.obs['dataset'].astype('category')
            colors = plt.cm.tab20(np.linspace(0, 1, len(datasets.cat.categories)))
            
            for i, dataset in enumerate(datasets.cat.categories):
                mask = datasets == dataset
                ax2.scatter(umap_embedding[mask, 0], umap_embedding[mask, 1], 
                           c=[colors[i]], label=dataset, alpha=0.6, s=10)
            ax2.set_xlabel('UMAP1')
            ax2.set_ylabel('UMAP2')
            ax2.set_title(f'UMAP colored by dataset - {cell_type}')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        else:
            # If no dataset column, show density plot
            from matplotlib.colors import LogNorm
            h = ax2.hist2d(umap_embedding[:, 0], umap_embedding[:, 1], 
                          bins=50, cmap='viridis', norm=LogNorm())
            ax2.set_xlabel('UMAP1')
            ax2.set_ylabel('UMAP2')
            ax2.set_title(f'UMAP density - {cell_type}')
            cbar = plt.colorbar(h[3], ax=ax2)
            cbar.set_label('Cell density')
        
        plt.tight_layout()
        
        # Save the figure
        cell_type_clean = cell_type.replace(" ", "_").replace("/", "_")
        plot_dir = os.path.join(OUTPUT_DIR, cell_type_clean)
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"UMAP_{cell_type_clean}_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      ✓ UMAP plot saved to: {plot_path}")
        
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

def process_cell_type(splice_adata, cell_type):
    """Process a single cell type with cell-type-specific ATSE filtering"""
    print(f"\n>> Processing cell type: {cell_type}")
    
    try:
        # Subset data for this cell type
        cell_mask = splice_adata.obs['broad_cell_type'] == cell_type
        splice_adata_subset = splice_adata[cell_mask, :].copy()
        
        # Reset cell indices
        splice_adata_subset.obs.reset_index(drop=True, inplace=True)
        splice_adata_subset.obs["cell_id_index"] = splice_adata_subset.obs.index
        
        print(f"   ✓ Subset contains {splice_adata_subset.shape[0]} cells and {splice_adata_subset.shape[1]} junctions")
        
        # Skip if too few cells
        if splice_adata_subset.shape[0] < 100:
            print(f"   ⚠️ Skipping {cell_type}: insufficient cells ({splice_adata_subset.shape[0]})")
            return None
        
        # Compute cell-type-specific ATSE scores
        atse_scores_filtered = compute_atse_scores_for_celltype(splice_adata_subset)
        
        # Filter junctions by ATSE scores
        splice_adata_subset = filter_junctions_by_atse_scores(splice_adata_subset, atse_scores_filtered)
        
        # Skip if too few junctions after filtering
        if splice_adata_subset.shape[1] < 100:
            print(f"   ⚠️ Skipping {cell_type}: insufficient junctions after filtering ({splice_adata_subset.shape[1]})")
            return None
        
        # Store ATSE filtering stats
        splice_adata_subset.uns['atse_filter_stats'] = {
            'n_atses_kept': len(atse_scores_filtered),
            'n_junctions_kept': splice_adata_subset.shape[1],
            'filter_percentile': ATSE_FILTER_PERCENTILE,
            'min_score': float(atse_scores_filtered['normalized_atse_score'].min()),
            'max_score': float(atse_scores_filtered['normalized_atse_score'].max()),
            'mean_score': float(atse_scores_filtered['normalized_atse_score'].mean())
        }
        
        # Compute dimensionality reduction
        splice_adata_subset = compute_dimensionality_reduction(splice_adata_subset, cell_type)
        
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

def create_summary_file(processed_cell_types, output_paths, stats_list):
    """Create a summary file with information about processed cell types"""
    try:
        summary_data = []
        for cell_type, output_path, stats in zip(processed_cell_types, output_paths, stats_list):
            if output_path is not None and stats is not None:
                summary_data.append({
                    'cell_type': cell_type,
                    'output_path': output_path,
                    'status': 'success',
                    'n_cells': stats['n_cells'],
                    'n_junctions': stats['n_junctions'],
                    'n_atses': stats['n_atses']
                })
            else:
                summary_data.append({
                    'cell_type': cell_type,
                    'output_path': None,
                    'status': 'failed',
                    'n_cells': None,
                    'n_junctions': None,
                    'n_atses': None
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(OUTPUT_DIR, f"processing_summary_{timestamp}.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n✓ Summary saved to: {summary_path}")
        
        # Print summary statistics
        successful = summary_df[summary_df['status'] == 'success']
        if len(successful) > 0:
            print(f"\n📊 Summary Statistics:")
            print(f"   - Successfully processed: {len(successful)}/{len(summary_df)} cell types")
            print(f"   - Average cells per type: {successful['n_cells'].mean():.0f}")
            print(f"   - Average junctions per type: {successful['n_junctions'].mean():.0f}")
            print(f"   - Average ATSEs per type: {successful['n_atses'].mean():.0f}")
        
        # Create a combined UMAP figure showing all cell types
        if len(successful) > 0:
            print("\n📊 Creating combined UMAP visualization...")
            create_combined_umap_plot(processed_cell_types, output_paths)
        
    except Exception as e:
        print(f"\n❌ Error creating summary file: {str(e)}")

def create_combined_umap_plot(processed_cell_types, output_paths):
    """Create a combined plot showing UMAPs for all successfully processed cell types"""
    try:
        successful_types = [(ct, op) for ct, op in zip(processed_cell_types, output_paths) if op is not None]
        
        if len(successful_types) == 0:
            return
        
        # Calculate grid dimensions
        n_plots = len(successful_types)
        n_cols = min(4, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot each cell type's UMAP
        for idx, (cell_type, output_path) in enumerate(successful_types):
            # Load the saved AnnData to get UMAP coordinates
            try:
                adata_path = output_path
                adata = ad.read_h5ad(adata_path)
                
                if 'X_umap' in adata.obsm:
                    umap_coords = adata.obsm['X_umap']
                    ax = axes[idx]
                    ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                              alpha=0.5, s=1, c='steelblue')
                    ax.set_title(f'{cell_type}\n({adata.shape[0]} cells)', fontsize=10)
                    ax.set_xlabel('UMAP1', fontsize=8)
                    ax.set_ylabel('UMAP2', fontsize=8)
                    ax.tick_params(labelsize=6)
                    
                del adata  # Free memory
                
            except Exception as e:
                print(f"   ⚠️ Could not load UMAP for {cell_type}: {str(e)}")
                ax = axes[idx]
                ax.text(0.5, 0.5, 'Error loading data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(cell_type, fontsize=10)
        
        # Hide extra subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        combined_plot_path = os.path.join(OUTPUT_DIR, f"combined_UMAPs_{timestamp}.png")
        plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Combined UMAP plot saved to: {combined_plot_path}")
        
    except Exception as e:
        print(f"   ⚠️ Error creating combined UMAP plot: {str(e)}")

def compute_junction_scores_for_celltype(splice_adata_subset, target_junctions=15000):
    """Compute quality scores for individual junctions specific to a cell type subset"""
    print(f"      ⚙️ Computing cell-type-specific junction quality scores (target: {target_junctions})...")
    
    try:
        # Ensure junc_ratio is available for variability calculations
        splice_adata_subset = get_junc_ratio(splice_adata_subset)
        
        # Calculate basic junction statistics using sparse matrix directly
        print("         Computing junction expression statistics...")
        junction_matrix = splice_adata_subset.layers["cell_by_junction_matrix"]
        
        # Calculate non-zero counts efficiently with sparse matrix
        splice_adata_subset.var["non_zero_count_cells"] = np.array((junction_matrix > 0).sum(axis=0)).flatten()
        splice_adata_subset.var["non_zero_cell_prop"] = splice_adata_subset.var["non_zero_count_cells"] / splice_adata_subset.shape[0]
        
        # Calculate junction variability using junc_ratio directly
        print("         Computing junction variability scores...")
        junc_ratio_matrix = splice_adata_subset.layers["junc_ratio"]
        
        # Convert to dense only for variability calculation if needed
        if hasattr(junc_ratio_matrix, 'toarray'):
            # For sparse matrices, calculate variability more efficiently
            junction_variability = []
            junc_ratio_csc = junc_ratio_matrix.tocsc()  # Convert to CSC for efficient column access
            
            for j in range(junc_ratio_csc.shape[1]):
                # Extract column as dense array
                col_data = junc_ratio_csc[:, j].toarray().flatten()
                non_zero_values = col_data[col_data != 0]
                
                if len(non_zero_values) > 1:
                    cv = np.std(non_zero_values) / (np.abs(np.mean(non_zero_values)) + 1e-8)
                    junction_variability.append(cv)
                else:
                    junction_variability.append(0.0)
        else:
            # If already dense
            junction_variability = []
            for j in range(junc_ratio_matrix.shape[1]):
                values = junc_ratio_matrix[:, j]
                non_zero_values = values[values != 0]
                if len(non_zero_values) > 1:
                    cv = np.std(non_zero_values) / (np.abs(np.mean(non_zero_values)) + 1e-8)
                    junction_variability.append(cv)
                else:
                    junction_variability.append(0.0)
        
        splice_adata_subset.var["junction_variability"] = junction_variability
        
        # Calculate individual junction scores
        print("         Computing individual junction quality scores...")
        
        # 1. Annotation status score - prioritize novel but don't exclude annotated
        annotation_weights = {
            "unannotated": 1.0,      # Highest priority for novel junctions
            "three_prime": 0.8,      # Partial annotation still interesting
            "five_prime": 0.8,       # Partial annotation still interesting  
            "both": 0.6              # Fully annotated but still valuable
        }
        splice_adata_subset.var["annotation_status_score"] = splice_adata_subset.var["annotation_status"].map(annotation_weights)
        
        # 2. Expression breadth score - more cells expressing = better
        splice_adata_subset.var["expression_breadth_score"] = np.where(
            splice_adata_subset.var["non_zero_cell_prop"] >= 0.05,  # At least 5% of cells
            np.log1p(splice_adata_subset.var["non_zero_cell_prop"] * 100),  # Log scale for diminishing returns
            0.0
        )
        
        # 3. Variability score - prioritize junctions that vary across cells
        # Use quantile-based scoring to make it robust
        variability_percentiles = np.percentile(splice_adata_subset.var["junction_variability"], [25, 75, 95])
        splice_adata_subset.var["variability_score"] = np.where(
            splice_adata_subset.var["junction_variability"] >= variability_percentiles[2],  # Top 5%
            2.0,
            np.where(
                splice_adata_subset.var["junction_variability"] >= variability_percentiles[1],  # Top 25%
                1.5,
                np.where(
                    splice_adata_subset.var["junction_variability"] >= variability_percentiles[0],  # Above median
                    1.0,
                    0.5
                )
            )
        )
        
        # Calculate composite junction score with cell-type-specific weights
        weights = {
            "annotation": 0.25,     # 25% weight to annotation novelty
            "expression": 0.35,     # 35% weight to expression breadth  
            "variability": 0.40     # 40% weight to variability (most important)
        }
        
        splice_adata_subset.var["junction_composite_score"] = (
            weights["annotation"] * splice_adata_subset.var["annotation_status_score"] +
            weights["expression"] * splice_adata_subset.var["expression_breadth_score"] +
            weights["variability"] * splice_adata_subset.var["variability_score"]
        )
        
        # Step 1: Identify top junctions directly
        print(f"         Identifying top {target_junctions} junctions...")
        n_junctions_available = len(splice_adata_subset.var)
        actual_target = min(target_junctions, n_junctions_available)
        
        top_junction_indices = splice_adata_subset.var["junction_composite_score"].nlargest(actual_target).index
        top_junctions = splice_adata_subset.var.loc[top_junction_indices]
        
        # Step 2: Get all ATSEs that contain at least one of these top junctions
        print("         Finding ATSEs containing top junctions...")
        top_atses = set(top_junctions["event_id"].unique())
        print(f"         Top {actual_target} junctions belong to {len(top_atses)} ATSEs")
        
        # Step 3: Keep ALL junctions from these ATSEs (not just the top ones)
        print("         Keeping all junctions from selected ATSEs...")
        selected_junctions_mask = splice_adata_subset.var["event_id"].isin(top_atses)
        final_junction_count = selected_junctions_mask.sum()
        
        print(f"         Final selection: {final_junction_count} junctions from {len(top_atses)} ATSEs")
        
        # Show annotation status distribution in final selection
        final_annotation_dist = splice_adata_subset.var[selected_junctions_mask]["annotation_status"].value_counts()
        print(f"         Junction annotation distribution in final selection:")
        for status, count in final_annotation_dist.items():
            percentage = 100 * count / final_annotation_dist.sum()
            print(f"           {status}: {count} ({percentage:.1f}%)")
        
        # Show some statistics about the selection
        selected_scores = splice_adata_subset.var.loc[selected_junctions_mask, "junction_composite_score"]
        print(f"         Junction score statistics in final selection:")
        print(f"           Mean: {selected_scores.mean():.3f}")
        print(f"           Median: {selected_scores.median():.3f}")
        print(f"           Min: {selected_scores.min():.3f}")
        print(f"           Max: {selected_scores.max():.3f}")
        
        # Return the mask and summary stats
        selection_stats = {
            'n_junctions_targeted': actual_target,
            'n_junctions_final': final_junction_count,
            'n_atses_selected': len(top_atses),
            'score_mean': float(selected_scores.mean()),
            'score_median': float(selected_scores.median()),
            'score_min': float(selected_scores.min()),
            'score_max': float(selected_scores.max())
        }
        
        return selected_junctions_mask, selection_stats
        
    except Exception as e:
        print(f"         ❌ Error computing junction scores: {str(e)}")
        traceback.print_exc()
        raise

def get_junc_ratio(splice_adata_subset):
    """Enhanced version with error checking for cell type subsets"""
    if "junc_ratio" not in splice_adata_subset.layers:
        print("         Computing sparse centered PSI values...")
        try:
            # Update junction_counts and cluster_counts
            junction_counts = splice_adata_subset.layers["cell_by_junction_matrix"]
            cluster_counts = splice_adata_subset.layers["cell_by_cluster_matrix"]
            
            # Convert to COO if needed for the wayp function
            if not isinstance(junction_counts, coo_matrix):
                junction_counts = junction_counts.tocoo()
            if not isinstance(cluster_counts, coo_matrix):
                cluster_counts = cluster_counts.tocoo()
            
            # Get sparse centered PSI values  
            junc_ratio = wayp.calculate_centered_psi(junction_counts, cluster_counts)
            
            # Convert result to CSR format immediately
            if isinstance(junc_ratio, coo_matrix):
                splice_adata_subset.layers["junc_ratio"] = junc_ratio.tocsr()
            else:
                splice_adata_subset.layers["junc_ratio"] = junc_ratio
                
            print(f"         ✓ Successfully computed junc_ratio layer (CSR format)")
            
        except Exception as e:
            print(f"         Error computing junc_ratio: {str(e)}")
            # Fallback: create dummy ratios if PSI calculation fails
            splice_adata_subset.layers["junc_ratio"] = csr_matrix(splice_adata_subset.X.shape)
            print("         Warning: Using dummy junc_ratio values")
    else:
        print("         junc_ratio layer already exists")
        # Ensure existing junc_ratio is also CSR
        if isinstance(splice_adata_subset.layers["junc_ratio"], coo_matrix):
            splice_adata_subset.layers["junc_ratio"] = splice_adata_subset.layers["junc_ratio"].tocsr()
            print("         Converted existing junc_ratio from COO to CSR")
    
    return splice_adata_subset

def filter_junctions_by_scores(splice_adata_subset, junction_mask):
    """Filter junctions by junction-based scores"""
    print("      ⚙️ Filtering junctions by junction scores...")
    
    try:
        # Filter junctions using the mask
        splice_adata_filtered = splice_adata_subset[:, junction_mask].copy()
        
        print(f"      ✓ Filtered to {splice_adata_filtered.shape[1]} junctions")
        
        # Reset junction indices
        splice_adata_filtered.var.reset_index(drop=True, inplace=True)
        if 'junction_id_index' in splice_adata_filtered.var.columns:
            splice_adata_filtered.var.rename(columns={'junction_id_index': 'old_junction_id_index'}, inplace=True)
        splice_adata_filtered.var['junction_id_index'] = splice_adata_filtered.var.index
        
        return splice_adata_filtered
        
    except Exception as e:
        print(f"      ❌ Error filtering junctions: {str(e)}")
        traceback.print_exc()
        raise

def process_cell_type(splice_adata, cell_type):
    """Process a single cell type with junction-based filtering"""
    print(f"\n>> Processing cell type: {cell_type}")
    
    try:
        # Subset data for this cell type
        cell_mask = splice_adata.obs['broad_cell_type'] == cell_type
        splice_adata_subset = splice_adata[cell_mask, :].copy()
        
        # Reset cell indices
        splice_adata_subset.obs.reset_index(drop=True, inplace=True)
        splice_adata_subset.obs["cell_id_index"] = splice_adata_subset.obs.index
        
        print(f"   ✓ Subset contains {splice_adata_subset.shape[0]} cells and {splice_adata_subset.shape[1]} junctions")
        
        # Skip if too few cells
        if splice_adata_subset.shape[0] < 100:
            print(f"   ⚠️ Skipping {cell_type}: insufficient cells ({splice_adata_subset.shape[0]})")
            return None
        
        # Calculate target junctions based on cell count (scale with cell type size)
        target_junctions = min(5000, max(5000, splice_adata_subset.shape[0] * 2))
        print(f"   ✓ Target junctions for this cell type: {target_junctions}")
        
        # Compute cell-type-specific junction scores
        junction_mask, selection_stats = compute_junction_scores_for_celltype(splice_adata_subset, target_junctions)
        
        # Filter junctions by scores
        splice_adata_subset = filter_junctions_by_scores(splice_adata_subset, junction_mask)
        
        # Skip if too few junctions after filtering
        if splice_adata_subset.shape[1] < 100:
            print(f"   ⚠️ Skipping {cell_type}: insufficient junctions after filtering ({splice_adata_subset.shape[1]})")
            return None
        
        # Store junction filtering stats
        splice_adata_subset.uns['junction_filter_stats'] = selection_stats
        
        # Compute dimensionality reduction
        splice_adata_subset = compute_dimensionality_reduction(splice_adata_subset, cell_type)
        
        # Identify waypoints and create metacells
        waypoints_dict, metacell_dicts = identify_waypoints_and_metacells(splice_adata_subset)
        
        # Generate and store initializations
        splice_adata_subset = generate_and_store_initializations(splice_adata_subset, waypoints_dict, metacell_dicts)
        
        return splice_adata_subset
        
    except Exception as e:
        print(f"   ❌ Error processing cell type {cell_type}: {str(e)}")
        traceback.print_exc()
        return None

# Main execution
print("\n========================================")
print("LeafletFA Input Preparation - Mouse Splicing Foundation (Per Cell Type)")
print("With Cell-Type-Specific ATSE Filtering")
print("========================================\n")

# Load data
splice_adata = load_data()
splice_adata = splice_adata[:, splice_adata.var["num_junctions"] <=5]

# Add this after loading the data in main execution
print("\n>> Converting COO matrices to CSR format...")
for layer_name, layer_data in splice_adata.layers.items():
    if isinstance(layer_data, coo_matrix):
        splice_adata.layers[layer_name] = layer_data.tocsr()
        print(f"   Converted {layer_name} from COO to CSR")

# Get valid cell types
valid_cell_types = get_valid_cell_types(splice_adata)

# Process each cell type
processed_cell_types = []
output_paths = []
stats_list = []

for cell_type in tqdm(valid_cell_types, desc="Processing cell types"):
    splice_adata_subset = process_cell_type(splice_adata, cell_type)
    
    if splice_adata_subset is not None:
        output_path = save_cell_type_anndata(splice_adata_subset, cell_type)
        processed_cell_types.append(cell_type)
        output_paths.append(output_path)
        
        # Collect statistics
        stats = {
            'n_cells': splice_adata_subset.shape[0],
            'n_junctions': splice_adata_subset.shape[1],
            'n_atses': splice_adata_subset.var['event_id'].nunique(),
            'junction_filter_stats': splice_adata_subset.uns.get('junction_filter_stats', {})
        }
        stats_list.append(stats)
    else:
        processed_cell_types.append(cell_type)
        output_paths.append(None)
        stats_list.append(None)
    
    # Clean up memory
    del splice_adata_subset
    gc.collect()

# Create summary file
create_summary_file(processed_cell_types, output_paths, stats_list)

print("\n========================================")
print("LeafletFA input preparation complete!")
print(f"Results saved to: {OUTPUT_DIR}")
print(f"Successfully processed {sum(1 for path in output_paths if path is not None)} out of {len(valid_cell_types)} cell types")
print("========================================\n")

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/062025
# sbatch --mem=250G -p cpu,dev,bigmem -J "prep_initialized_AnnData_per_celltype" --wrap="python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/LeafletFA_analysis/02_generate_cell_type_specific_splicing_inputs.py"