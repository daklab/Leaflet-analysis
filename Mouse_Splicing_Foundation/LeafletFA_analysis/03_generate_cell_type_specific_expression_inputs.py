#!/usr/bin/env python
"""
Gene Expression Input Preparation - Mouse Splicing Foundation (Per Cell Type)

This script:
1. Loads gene expression data from aligned AnnData
2. For each cell type:
   - Creates a subset of data
   - Computes dimensionality reduction (PCA) on log-normalized expression
   - Performs UMAP embedding
   - Performs Leiden clustering
   - Saves UMAP plots
   - Saves prepared AnnData with all computed embeddings and clusters
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
import scanpy as sc
import torch
from tqdm import tqdm

# Configure scanpy
sc.settings.verbosity = 1  # verbosity level
sc.settings.set_figure_params(dpi=80, facecolor='white')

# Configuration
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
BASE_OUTPUT_DIR = "/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/CELL_TYPE_SPECIFIC/gene_expression"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"per_celltype_GE_{timestamp}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}", flush=True)

# Input file paths
GE_INPUT = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/aligned_gene_expression_data_20250704_231739.h5ad"
print(f"The input file is: {GE_INPUT}")

MIN_CELLS_PER_TYPE = 1000  # Minimum number of cells required for a cell type to be processed

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)

def load_data():
    """Load gene expression data"""
    print("\n>> Loading datasets...")
    
    try:
        # Load gene expression data
        print(f"   ⚙️ Loading gene expression AnnData from {GE_INPUT}")
        ge_adata = ad.read_h5ad(GE_INPUT)
        ge_adata.obs.reset_index(drop=True, inplace=True)
        ge_adata.obs["cell_id_index"] = ge_adata.obs.index 
        print(f"   ✓ Loaded gene expression data with {ge_adata.shape[0]} cells and {ge_adata.shape[1]} genes")
                
        # Print summary of cell types
        print(f"   ✓ Data contains {ge_adata.obs['broad_cell_type'].nunique()} standardized cell types")
        print(f"   ✓ Cell type counts: {dict(ge_adata.obs['broad_cell_type'].value_counts())}")
        
        return ge_adata
        
    except Exception as e:
        print(f"   ❌ Error loading datasets: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def get_valid_cell_types(ge_adata):
    """Get cell types that have sufficient cells for processing"""
    print("\n>> Identifying valid cell types...")
    
    try:
        cell_type_counts = ge_adata.obs['broad_cell_type'].value_counts()
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

def compute_pca_umap_leiden(adata_subset, n_comps=50, n_neighbors=15, min_dist=0.5, resolution=0.5):
    """Compute PCA, UMAP, and Leiden clustering"""
    print(f"   ⚙️ Computing PCA, UMAP, and Leiden clustering...")
    
    try:
        # Ensure we have the log_norm layer
        if 'log_norm' not in adata_subset.layers:
            print(f"   ⚠️ log_norm layer not found, using X matrix")
            # Copy X to log_norm layer for consistency
            adata_subset.layers['log_norm'] = adata_subset.X.copy()
        
        # Use log_norm layer for dimensionality reduction
        adata_subset.X = adata_subset.layers['log_norm'].copy()
        
        # Compute PCA
        print(f"      - Computing PCA with {n_comps} components...")
        sc.tl.pca(adata_subset, n_comps=n_comps, svd_solver='arpack')
        
        # Compute neighborhood graph
        print(f"      - Computing neighborhood graph with {n_neighbors} neighbors...")
        sc.pp.neighbors(adata_subset, n_neighbors=n_neighbors, n_pcs=n_comps)
        
        # Compute UMAP
        print(f"      - Computing UMAP with min_dist={min_dist}...")
        sc.tl.umap(adata_subset, min_dist=min_dist)
        
        # Compute Leiden clustering
        print(f"      - Computing Leiden clustering with resolution={resolution}...")
        sc.tl.leiden(adata_subset, resolution=resolution, key_added='leiden')
        
        print(f"   ✓ Computed PCA ({adata_subset.obsm['X_pca'].shape[1]} components), UMAP, and Leiden clustering")
        print(f"      - Found {len(adata_subset.obs['leiden'].unique())} Leiden clusters")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error computing PCA/UMAP/Leiden: {str(e)}")
        traceback.print_exc()
        return False

def plot_umap_results(adata_subset, cell_type, output_dir):
    """Generate and save UMAP plots"""
    print(f"   ⚙️ Generating UMAP plots...")
    
    try:
        # Create plots directory
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Set up the plotting parameters
        plt.rcParams['figure.figsize'] = (8, 6)
        plt.rcParams['font.size'] = 12
        
        # Plot 1: UMAP colored by Leiden clusters
        fig, ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(adata_subset, color='leiden', ax=ax, show=False, frameon=False, 
                   title=f'{cell_type} - Leiden Clusters')
        plt.tight_layout()
        leiden_plot_path = os.path.join(plots_dir, f"{cell_type.replace(' ', '_').replace('/', '_')}_umap_leiden.png")
        plt.savefig(leiden_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: UMAP colored by any available metadata (if exists)
        if 'sample_id' in adata_subset.obs.columns:
            fig, ax = plt.subplots(figsize=(10, 8))
            sc.pl.umap(adata_subset, color='sample_id', ax=ax, show=False, frameon=False,
                       title=f'{cell_type} - Sample ID')
            plt.tight_layout()
            sample_plot_path = os.path.join(plots_dir, f"{cell_type.replace(' ', '_').replace('/', '_')}_umap_sample.png")
            plt.savefig(sample_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot 3: UMAP colored by total gene expression (if available)
        if 'total_counts' in adata_subset.obs.columns:
            fig, ax = plt.subplots(figsize=(10, 8))
            sc.pl.umap(adata_subset, color='total_counts', ax=ax, show=False, frameon=False,
                       title=f'{cell_type} - Total Gene Expression')
            plt.tight_layout()
            counts_plot_path = os.path.join(plots_dir, f"{cell_type.replace(' ', '_').replace('/', '_')}_umap_total_counts.png")
            plt.savefig(counts_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot 4: Summary plot with multiple panels
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Leiden clusters
        sc.pl.umap(adata_subset, color='leiden', ax=axes[0], show=False, frameon=False,
                   title='Leiden Clusters')
        
        # Total counts or first available numeric column
        color_col = 'total_counts' if 'total_counts' in adata_subset.obs.columns else None
        if color_col is None:
            # Find first numeric column
            numeric_cols = adata_subset.obs.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                color_col = numeric_cols[0]
        
        if color_col is not None:
            sc.pl.umap(adata_subset, color=color_col, ax=axes[1], show=False, frameon=False,
                       title=f'{color_col}')
        else:
            axes[1].text(0.5, 0.5, 'No numeric metadata\navailable', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Metadata')
        
        plt.suptitle(f'{cell_type} - UMAP Analysis', fontsize=16)
        plt.tight_layout()
        summary_plot_path = os.path.join(plots_dir, f"{cell_type.replace(' ', '_').replace('/', '_')}_umap_summary.png")
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ UMAP plots saved to {plots_dir}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error generating UMAP plots: {str(e)}")
        traceback.print_exc()
        return False

def process_cell_type(ge_adata, cell_type):
    """Process a single cell type"""
    print(f"\n>> Processing cell type: {cell_type}")
    
    try:
        # Subset data for this cell type
        cell_mask = ge_adata.obs['broad_cell_type'] == cell_type
        ge_adata_subset = ge_adata[cell_mask, :].copy()
        
        # Reset cell indices
        ge_adata_subset.obs.reset_index(drop=True, inplace=True)
        ge_adata_subset.obs["cell_id_index"] = ge_adata_subset.obs.index
        
        print(f"   ✓ Subset contains {ge_adata_subset.shape[0]} cells and {ge_adata_subset.shape[1]} genes")

        # Reset gene indices
        ge_adata_subset.var.reset_index(drop=True, inplace=True)
        if 'old_gene_id_index' in ge_adata_subset.var.columns:
            ge_adata_subset.var.drop(columns=['old_gene_id_index'], inplace=True)
        ge_adata_subset.var['gene_id_index'] = ge_adata_subset.var.index
                
        # Skip if too few cells or genes
        if ge_adata_subset.shape[0] < 100 or ge_adata_subset.shape[1] < 100:
            print(f"   ⚠️ Skipping {cell_type}: insufficient cells ({ge_adata_subset.shape[0]}) or genes ({ge_adata_subset.shape[1]})")
            return None
        
        # Create cell type specific directory
        cell_type_dir = os.path.join(OUTPUT_DIR, cell_type.replace(" ", "_").replace("/", "_"))
        os.makedirs(cell_type_dir, exist_ok=True)
        
        # Compute PCA, UMAP, and Leiden clustering using log_norm layer
        success = compute_pca_umap_leiden(ge_adata_subset)
        if not success:
            print(f"   ⚠️ Failed to compute embeddings for {cell_type}")
            return None
        
        # Generate and save UMAP plots
        plot_success = plot_umap_results(ge_adata_subset, cell_type, cell_type_dir)
        if not plot_success:
            print(f"   ⚠️ Failed to generate plots for {cell_type}")
        
        return ge_adata_subset
        
    except Exception as e:
        print(f"   ❌ Error processing cell type {cell_type}: {str(e)}")
        traceback.print_exc()
        return None

def save_cell_type_anndata(ge_adata_subset, cell_type):
    """Save the prepared AnnData object for a specific cell type"""
    print(f"   ⚙️ Saving AnnData for {cell_type}...")
    
    try:
        # Create cell type specific directory
        cell_type_dir = os.path.join(OUTPUT_DIR, cell_type.replace(" ", "_").replace("/", "_"))
        os.makedirs(cell_type_dir, exist_ok=True)
        
        # Define output filename
        output_filename = f"MOUSE_GE_FOUNDATION_{cell_type.replace(' ', '_').replace('/', '_')}_with_embeddings_{timestamp}.h5ad"
        output_path = os.path.join(cell_type_dir, output_filename)
        
        # Save AnnData object
        ge_adata_subset.write_h5ad(output_path, compression='gzip')
        
        print(f"   ✓ Successfully saved AnnData for {cell_type} to {output_path}")
        
        # Print summary of what's included
        print(f"      - Shape: {ge_adata_subset.shape}")
        print(f"      - PCA components: {ge_adata_subset.obsm['X_pca'].shape[1]}")
        print(f"      - UMAP coordinates: {ge_adata_subset.obsm['X_umap'].shape}")
        print(f"      - Leiden clusters: {len(ge_adata_subset.obs['leiden'].unique())}")
        print(f"      - Layers: {list(ge_adata_subset.layers.keys())}")
        
        return output_path
        
    except Exception as e:
        print(f"   ❌ Error saving AnnData for {cell_type}: {str(e)}")
        traceback.print_exc()
        return None

def create_summary_file(processed_cell_types, output_paths, ge_adata):
    """Create a summary file with information about processed cell types"""
    try:
        summary_data = []
        for cell_type, output_path in zip(processed_cell_types, output_paths):
            if output_path is not None:
                # Get cell count for this cell type
                cell_count = (ge_adata.obs['broad_cell_type'] == cell_type).sum()
                summary_data.append({
                    'cell_type': cell_type,
                    'cell_count': cell_count,
                    'output_path': output_path,
                    'status': 'success'
                })
            else:
                cell_count = (ge_adata.obs['broad_cell_type'] == cell_type).sum()
                summary_data.append({
                    'cell_type': cell_type,
                    'cell_count': cell_count,
                    'output_path': None,
                    'status': 'failed'
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(OUTPUT_DIR, f"processing_summary_{timestamp}.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n✓ Summary saved to: {summary_path}")
        print(f"✓ Successful processing: {summary_df['status'].value_counts().get('success', 0)} cell types")
        print(f"✓ Failed processing: {summary_df['status'].value_counts().get('failed', 0)} cell types")
        
    except Exception as e:
        print(f"\n❌ Error creating summary file: {str(e)}")

# Main execution
print("\n========================================")
print("Gene Expression Analysis - Mouse Splicing Foundation (Per Cell Type)")
print("========================================\n")

# Load data
ge_adata = load_data()

# Get valid cell types
valid_cell_types = get_valid_cell_types(ge_adata)

# Process each cell type
processed_cell_types = []
output_paths = []

for cell_type in tqdm(valid_cell_types, desc="Processing cell types"):
    ge_adata_subset = process_cell_type(ge_adata, cell_type)
    
    if ge_adata_subset is not None:
        output_path = save_cell_type_anndata(ge_adata_subset, cell_type)
        processed_cell_types.append(cell_type)
        output_paths.append(output_path)
    else:
        processed_cell_types.append(cell_type)
        output_paths.append(None)

# Create summary file
create_summary_file(processed_cell_types, output_paths, ge_adata)

print("\n========================================")
print("Gene Expression analysis complete!")
print(f"Results saved to: {OUTPUT_DIR}")
print(f"Successfully processed {sum(1 for path in output_paths if path is not None)} out of {len(valid_cell_types)} cell types")
print("========================================\n")

# Usage:
# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/062025
# sbatch --mem=300G -p cpu,bigmem -J "prep_GE_per_celltype" --wrap="python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/LeafletFA_analysis/03_generate_cell_type_specific_expression_inputs.py"