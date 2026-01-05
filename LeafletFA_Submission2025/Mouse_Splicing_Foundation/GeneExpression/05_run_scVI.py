#!/usr/bin/env python
"""
scVI Model Training - Mouse Splicing Foundation

This script:
1. Loads gene expression data aligned with splicing data
2. Applies both linear and standard scVI dimensionality reduction
3. Saves both latent representations in a single AnnData object
4. Generates training metric plots for both models
5. Saves the updated AnnData object for downstream clustering and visualization
"""

import os
import sys
import pandas as pd
import numpy as np
import anndata as ad
import datetime
import traceback
import scanpy as sc
import scvi
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.sparse import csr_matrix
import openpyxl 

# Import LeafletFA differential splicing code
src_path = "/gpfs/commons/home/kisaev/Leaflet-private/src/"
utils_path = "/gpfs/commons/home/kisaev/Leaflet-analysis/Multi_Species_Splicing_Foundation/shared_utils/"

if src_path not in sys.path:
    sys.path.append(src_path)
if utils_path not in sys.path:
    sys.path.append(utils_path)

# Import custom modules
from utils import *

# Configuration
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/scVI"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}", flush=True)

# Input file path
GE_INPUT = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/aligned_gene_expression_data_20251003_152725.h5ad"

# Model configuration
LINEAR_LATENT = 20
LINEAR_EPOCHS = 200

# Gene selection configuration
N_TOP_GENES = 20000  # Number of highly variable genes to select

# Clustering and visualization configuration
SCVI_LATENT_KEY = "X_scVI_linear"
SCVI_CLUSTERS_KEY = "leiden_scVI"
N_NEIGHBORS = 10
UMAP_MIN_DIST = 0.3
UMAP_SPREAD = 1.0
LEIDEN_RESOLUTION = 0.8

# Reference files
AGING_GENES_PATH = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/27857814"
RBP_FILE_PATH = "/gpfs/commons/groups/knowles_lab/Karin/VanNostrand_2020_supptable1_41586_2020_2077_MOESM3_ESM.xlsx"

def load_reference_genes(aging_genes_path, rbp_file_path):
    """Load aging genes and RBP genes."""
    aging_genes_mouse, aging_genes_human = load_aging_genes(aging_genes_path)
    rbps = load_rbp_genes(rbp_file_path)
    
    # Process gene names
    rbps["mouse_gene_name"] = rbps["mouse_gene_name"].str.upper()
    aging_genes_mouse = [g.upper() for g in aging_genes_mouse]
    
    return aging_genes_mouse, rbps

def annotate_genes(ge_adata, aging_genes_mouse, rbps):
    """Annotate genes with aging and RBP information."""
    print("\n>> Annotating genes with reference information...")
    
    try:
        # Check if gene_name column exists
        if "gene_name" not in ge_adata.var.columns:
            print("   ⚠️ 'gene_name' column not found in var, using var_names")
            ge_adata.var["gene_name"] = ge_adata.var_names.str.upper()
        else:
            # Ensure gene names are uppercase for matching
            ge_adata.var["gene_name"] = ge_adata.var["gene_name"].str.upper()
        
        # Annotate RBP genes
        if not rbps.empty and "mouse_gene_name" in rbps.columns:
            ge_adata.var["RBP_gene"] = ge_adata.var["gene_name"].isin(rbps["mouse_gene_name"])
            n_rbp_found = ge_adata.var["RBP_gene"].sum()
            print(f"   ✓ Found {n_rbp_found} RBP genes in dataset")
        else:
            print("   ⚠️ No RBP gene data available, setting all to False")
            ge_adata.var["RBP_gene"] = False
            n_rbp_found = 0
        
        # Annotate aging genes
        if aging_genes_mouse:
            ge_adata.var["Aging_gene"] = ge_adata.var["gene_name"].isin(aging_genes_mouse)
            n_aging_found = ge_adata.var["Aging_gene"].sum()
            print(f"   ✓ Found {n_aging_found} aging genes in dataset")
        else:
            print("   ⚠️ No aging gene data available, setting all to False")
            ge_adata.var["Aging_gene"] = False
            n_aging_found = 0
        
        return ge_adata, n_rbp_found, n_aging_found
        
    except Exception as e:
        print(f"   ❌ Error annotating genes: {str(e)}")
        traceback.print_exc()
        # Set default annotations
        ge_adata.var["RBP_gene"] = False
        ge_adata.var["Aging_gene"] = False
        return ge_adata, 0, 0

def load_data():
    """Load aligned gene expression data"""
    print("\n>> Loading gene expression data...")
    
    try:
        print(f"   ⚙️ Reading gene expression AnnData from {GE_INPUT}")
        ge_adata = ad.read_h5ad(GE_INPUT)
        print(f"   ✓ Loaded data with {ge_adata.n_obs} cells and {ge_adata.n_vars} genes")
        
        return ge_adata
    
    except Exception as e:
        print(f"   ❌ Error loading gene expression data: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def check_data_quality(ge_adata, layer_name="length_norm"):
    """Check data quality in specific layer for NaN, Inf, and negative values"""
    print(f"\n>> Checking data quality in '{layer_name}' layer...")
    
    try:
        # Get data from specified layer
        if layer_name in ge_adata.layers:
            data = ge_adata.layers[layer_name]
            print(f"   ⚙️ Analyzing {type(data)} in layer '{layer_name}'")
            
            # Convert to dense if sparse
            if isinstance(data, csr_matrix) or hasattr(data, "toarray"):
                print("   ⚙️ Converting sparse matrix to dense for analysis...")
                # For large matrices, analyze a subset
                if data.shape[0] * data.shape[1] > 10**8:  # If matrix is very large
                    print("   ⚙️ Matrix is large, sampling a subset for analysis...")
                    sample_cells = min(1000, data.shape[0])
                    sample_genes = min(1000, data.shape[1])
                    indices_cells = np.random.choice(data.shape[0], sample_cells, replace=False)
                    indices_genes = np.random.choice(data.shape[1], sample_genes, replace=False)
                    data_dense = data[indices_cells, :][:, indices_genes].toarray()
                    print(f"   ⚙️ Analyzing subset of {sample_cells} cells × {sample_genes} genes")
                else:
                    data_dense = data.toarray()
            else:
                data_dense = data
            
            # Check for NaN values
            nan_count = np.isnan(data_dense).sum()
            nan_percent = 100 * nan_count / data_dense.size
            print(f"   ✓ NaN values: {nan_count} ({nan_percent:.4f}% of total)")
            
            # Check for infinity values
            inf_count = np.isinf(data_dense).sum()
            inf_percent = 100 * inf_count / data_dense.size
            print(f"   ✓ Infinity values: {inf_count} ({inf_percent:.4f}% of total)")
            
            # Check for negative values
            neg_count = (data_dense < 0).sum()
            neg_percent = 100 * neg_count / data_dense.size
            print(f"   ✓ Negative values: {neg_count} ({neg_percent:.4f}% of total)")
            
            # Basic statistics
            print(f"   ✓ Min value: {np.min(data_dense)}")
            print(f"   ✓ Max value: {np.max(data_dense)}")
            print(f"   ✓ Mean value: {np.mean(data_dense)}")
            print(f"   ✓ Median value: {np.median(data_dense)}")
            
            # Check per gene
            if nan_count > 0 or inf_count > 0:
                print("\n   ⚙️ Analyzing problematic genes...")
                
                # If working with full matrix
                if data_dense.shape[1] == ge_adata.n_vars:
                    gene_nan_counts = np.isnan(data_dense).sum(axis=0)
                    gene_inf_counts = np.isinf(data_dense).sum(axis=0)
                    
                    # Find genes with NaN or Inf values
                    problem_genes_nan = np.where(gene_nan_counts > 0)[0]
                    problem_genes_inf = np.where(gene_inf_counts > 0)[0]
                    
                    if len(problem_genes_nan) > 0:
                        print(f"   ✓ Found {len(problem_genes_nan)} genes with NaN values")
                        # Get gene names for top 10 problematic genes
                        top_nan_genes = problem_genes_nan[np.argsort(gene_nan_counts[problem_genes_nan])[-10:]]
                        print("   ✓ Top genes with NaN values:")
                        for idx in top_nan_genes:
                            gene_name = ge_adata.var_names[idx]
                            print(f"      - {gene_name}: {gene_nan_counts[idx]} NaN values")
                    
                    if len(problem_genes_inf) > 0:
                        print(f"   ✓ Found {len(problem_genes_inf)} genes with Inf values")
                        # Get gene names for top 10 problematic genes
                        top_inf_genes = problem_genes_inf[np.argsort(gene_inf_counts[problem_genes_inf])[-10:]]
                        print("   ✓ Top genes with Inf values:")
                        for idx in top_inf_genes:
                            gene_name = ge_adata.var_names[idx]
                            print(f"      - {gene_name}: {gene_inf_counts[idx]} Inf values")
                else:
                    print("   ⚙️ Working with subset of genes, skipping per-gene analysis")
            
            # Overall assessment
            if nan_count > 0 or inf_count > 0 or neg_count > 0:
                print("\n   ⚠️ Potential issues found in the data that may affect scVI training")
                if nan_count > 0:
                    print("      - NaN values could cause training problems")
                if inf_count > 0:
                    print("      - Infinity values could cause numerical instability")
                if neg_count > 0 and neg_percent > 1.0:
                    print("      - Significant negative values may not be suitable for default scVI distributions")
            else:
                print("\n   ✓ No major issues found in the data")
                
        else:
            print(f"   ❌ Layer '{layer_name}' not found in AnnData object")
            print(f"   ✓ Available layers: {list(ge_adata.layers.keys())}")
            
    except Exception as e:
        print(f"   ❌ Error checking data quality: {str(e)}")
        traceback.print_exc()
        
def select_genes_for_training(ge_adata, n_top_genes=N_TOP_GENES):
    """
    Select genes for scVI training: top N highly variable genes + RBP genes + aging genes
    Uses the pre-annotated RBP_gene and Aging_gene columns
    """
    print(f"\n>> Selecting genes for scVI training...")
    print(f"   ⚙️ Target: {n_top_genes} HVGs + RBP genes + aging genes")
    
    try:
        # Make a copy to avoid modifying the original
        adata_copy = ge_adata.copy()
        
        # Step 1: Calculate highly variable genes
        print("   ⚙️ Calculating highly variable genes...")
        sc.pp.highly_variable_genes(
            adata_copy, 
            layer="length_norm",
            n_top_genes=n_top_genes,
            flavor='seurat_v3', 
            batch_key="dataset"
        )
        
        # Get HVG gene names
        hvg_genes = adata_copy.var_names[adata_copy.var.highly_variable].tolist()
        print(f"   ✓ Found {len(hvg_genes)} highly variable genes")
        
        # Step 2: Get RBP genes from annotations
        if "RBP_gene" in ge_adata.var.columns:
            rbp_genes = ge_adata.var_names[ge_adata.var.RBP_gene].tolist()
            print(f"   ✓ Found {len(rbp_genes)} RBP genes in dataset")
        else:
            print("   ⚠️ RBP_gene annotation not found")
            rbp_genes = []
        
        # Step 3: Get aging genes from annotations
        if "Aging_gene" in ge_adata.var.columns:
            aging_genes = ge_adata.var_names[ge_adata.var.Aging_gene].tolist()
            print(f"   ✓ Found {len(aging_genes)} aging genes in dataset")
        else:
            print("   ⚠️ Aging_gene annotation not found")
            aging_genes = []
        
        # Step 4: Combine all selected genes (remove duplicates)
        selected_genes = list(set(hvg_genes + rbp_genes + aging_genes))
        print(f"   ✓ Total selected genes: {len(selected_genes)}")
        
        # Step 5: Create subset AnnData with selected genes
        print("   ⚙️ Creating subset with selected genes...")
        ge_adata_subset = ge_adata[:, selected_genes].copy()
        
        # Step 6: Add gene selection metadata
        ge_adata_subset.var['is_hvg'] = ge_adata_subset.var_names.isin(hvg_genes)
        ge_adata_subset.var['is_rbp'] = ge_adata_subset.var.get('RBP_gene', False)
        ge_adata_subset.var['is_aging'] = ge_adata_subset.var.get('Aging_gene', False)
        
        # Count overlaps
        hvg_rbp_overlap = len(set(hvg_genes) & set(rbp_genes))
        hvg_aging_overlap = len(set(hvg_genes) & set(aging_genes))
        rbp_aging_overlap = len(set(rbp_genes) & set(aging_genes))
        
        print(f"   ✓ Gene category breakdown:")
        print(f"      - HVGs only: {len(hvg_genes) - hvg_rbp_overlap - hvg_aging_overlap}")
        print(f"      - RBP genes only: {len(rbp_genes) - hvg_rbp_overlap - rbp_aging_overlap}")
        print(f"      - Aging genes only: {len(aging_genes) - hvg_aging_overlap - rbp_aging_overlap}")
        print(f"      - HVG + RBP overlap: {hvg_rbp_overlap}")
        print(f"      - HVG + Aging overlap: {hvg_aging_overlap}")
        print(f"      - RBP + Aging overlap: {rbp_aging_overlap}")
        
        # Add selection metadata to uns
        ge_adata_subset.uns['gene_selection'] = {
            'n_top_hvgs_requested': n_top_genes,
            'n_hvgs_found': len(hvg_genes),
            'n_rbp_genes_found': len(rbp_genes),
            'n_aging_genes_found': len(aging_genes),
            'total_selected_genes': len(selected_genes),
            'selection_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'hvg_rbp_overlap': hvg_rbp_overlap,
            'hvg_aging_overlap': hvg_aging_overlap,
            'rbp_aging_overlap': rbp_aging_overlap
        }
        
        print(f"   ✓ Gene selection complete: {ge_adata_subset.n_vars} genes selected for training")
        
        return ge_adata_subset
        
    except Exception as e:
        print(f"   ❌ Error during gene selection: {str(e)}")
        traceback.print_exc()
        raise

def train_linear_scvi(ge_adata, batch_key=None):
    """Train LinearSCVI model for gene expression data using ALL genes"""
    print("\n>> Training LinearSCVI model...")
    
    try:
        # Step 1: Setup AnnData for model (using ALL genes)
        print("   ⚙️ Setting up AnnData for LinearSCVI using ALL genes...")
        if batch_key is None:
            scvi.model.LinearSCVI.setup_anndata(ge_adata, layer="length_norm")
        else:
            scvi.model.LinearSCVI.setup_anndata(ge_adata, layer="length_norm", batch_key=batch_key)

        # Step 2: Train model
        model = scvi.model.LinearSCVI(ge_adata, n_latent=LINEAR_LATENT)
        print(f"   ⚙️ Training model for {LINEAR_EPOCHS} epochs...")
        model.train(max_epochs=LINEAR_EPOCHS, check_val_every_n_epoch=10)

        # Step 3: Save latent representation and normalized expression
        print("   ⚙️ Extracting latent representation...")
        ge_adata.obsm["X_scVI_linear"] = model.get_latent_representation()
        ge_adata.obsm["X_normalized_scVI_linear"] = model.get_normalized_expression()

        # Step 4: Save gene loadings for ALL genes
        print("   ⚙️ Extracting gene loadings for ALL genes...")
        loadings = model.get_loadings()  # shape: (n_genes, n_latent)
        ge_adata.varm["scVI_linear_gene_loadings"] = loadings
        
        # Step 5: Metadata
        ge_adata.uns["scvi_linear"] = {
            "model_type": "LinearSCVI",
            "n_latent": LINEAR_LATENT,
            "training_date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "trained_on_all_genes": True,
            "n_genes_used": ge_adata.n_vars
        }

        print("   ✓ LinearSCVI training complete.")
        return ge_adata, model
                
    except Exception as e:
        print(f"   ❌ Error during LinearSCVI training: {str(e)}")
        traceback.print_exc()
        raise

def perform_clustering_and_visualization(
        ge_adata,
        latent_key=SCVI_LATENT_KEY,
        clusters_key=SCVI_CLUSTERS_KEY,
        neighbors=N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        spread=UMAP_SPREAD,
        leiden_res=LEIDEN_RESOLUTION,
        plot=True,
):
    """
    Compute neighbors → UMAP → Leiden on an existing latent space and
    (optionally) save a UMAP PNG coloured by clusters.
    """
    if latent_key not in ge_adata.obsm:
        raise KeyError(f"Latent key '{latent_key}' not found in .obsm")

    # Nearest-neighbour graph on latent space
    sc.pp.neighbors(ge_adata, use_rep=latent_key, n_neighbors=neighbors)

    # 2-D UMAP embedding
    sc.tl.umap(ge_adata, min_dist=min_dist, spread=spread)

    # Leiden clustering
    sc.tl.leiden(ge_adata, key_added=clusters_key, resolution=leiden_res)

    # Metadata for reproducibility
    ge_adata.uns[f"{clusters_key}_params"] = dict(
        latent_key_used=latent_key,
        n_neighbors=neighbors,
        umap_min_dist=min_dist,
        umap_spread=spread,
        leiden_resolution=leiden_res,
        date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    # Optional PNG
    if plot:
        fig_path = os.path.join(
            OUTPUT_DIR, f"umap_linear_scvi_{timestamp}.png"
        )
        sc.pl.umap(
            ge_adata,
            color=[clusters_key, "broad_cell_type"],
            # display in one column two rows 
            ncols=1,
            size=10,
            frameon=False,
            show=False,
        )
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   ✓ UMAP figure saved to {fig_path}")

    return ge_adata

def save_results(ge_adata):
    """Save updated AnnData with both scVI results"""
    print("\n>> Saving results...")
    
    try:
        # Define output filename
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        output_file = os.path.join(OUTPUT_DIR, f"ge_adata_with_scvi_model_latent_{LINEAR_LATENT}_{N_TOP_GENES}_{today}.h5ad")

        # Save file
        print(ge_adata)
        print(f"   ⚙️ Saving updated AnnData to {output_file}...")
        ge_adata.write_h5ad(output_file, compression="gzip")
        
        print(f"   ✓ Results successfully saved to {output_file}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error saving results: {str(e)}")
        traceback.print_exc()
        return False

print("\n========================================")
print("LinearSCVI – Mouse Splicing Foundation")
print("========================================\n")

# 1. Load & QC
ge_adata = load_data()
check_data_quality(ge_adata, "length_norm")
print(f"Number of latent dimensions: {LINEAR_LATENT}")

# Load reference genes
aging_genes_mouse, rbps = load_reference_genes(AGING_GENES_PATH, RBP_FILE_PATH)
# Annotate genes with reference information
ge_adata, n_rbp_found, n_aging_found = annotate_genes(
    ge_adata, aging_genes_mouse, rbps
)

# 3. Get ge_adata subset for training 
ge_adata_subset = select_genes_for_training(ge_adata)

# 2. Train LinearSCVI
ge_adata_subset, linear_model = train_linear_scvi(ge_adata_subset)

# 3. UMAP + Leiden on latent space
ge_adata_subset = perform_clustering_and_visualization(
    ge_adata_subset,
    latent_key="X_scVI_linear",
    clusters_key="leiden_scVI",
)

# 5. Save updated AnnData
save_results(ge_adata_subset)

print("\n========================================")
print("LinearSCVI training + UMAP complete!")
print(f"Outputs written to: {OUTPUT_DIR}")
print("========================================\n")

# conda activate scvi-env
# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/scVI
# sbatch --mem=100G -p gpu -J "scVI_GE_MOUSE" --wrap "python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/GeneExpression/05_run_scVI.py"
# sbatch --mem=300G -p cpu,bigmem,dev -J "scVI_GE_MOUSE" --wrap "python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/GeneExpression/05_run_scVI.py"