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

# Configuration
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/scVI"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}", flush=True)

# Input file path
GE_INPUT = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/062025/aligned_gene_expression_data_20250614_124502.h5ad"

# Model configuration
LINEAR_LATENT = 30
STANDARD_LATENT = 30
LINEAR_EPOCHS = 200
STANDARD_EPOCHS = 200

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
        
def train_linear_scvi(ge_adata):
    """Train LinearSCVI model for gene expression data"""
    print("\n>> Training LinearSCVI model...")
    
    try:
        # Setup for model
        print("   ⚙️ Setting up AnnData for LinearSCVI...")
        scvi.model.LinearSCVI.setup_anndata(ge_adata, layer="length_norm", batch_key="dataset")
        
        # Initialize and train model
        print(f"   ⚙️ Initializing LinearSCVI with {LINEAR_LATENT} latent dimensions...")
        model = scvi.model.LinearSCVI(ge_adata, n_latent=LINEAR_LATENT)
        
        print(f"   ⚙️ Training model for {LINEAR_EPOCHS} epochs...")
        model.train(max_epochs=LINEAR_EPOCHS, check_val_every_n_epoch=10)
        
        # Extract results and save to obsm
        print("   ⚙️ Extracting latent representation...")
        Z_hat = model.get_latent_representation()
        ge_adata.obsm["X_scVI_linear"] = Z_hat
        
        # Get loadings
        loadings = model.get_loadings()
        print("   ✓ Top genes with highest loadings in LinearSCVI:")
        print(loadings.head())
        
        # Add model metadata to uns
        ge_adata.uns["scvi_linear"] = {
            "model_type": "LinearSCVI",
            "n_latent": LINEAR_LATENT,
            "training_date": datetime.datetime.now().strftime("%Y-%m-%d")
        }
        
        print("   ✓ LinearSCVI model training complete")
        
        return ge_adata, model
        
    except Exception as e:
        print(f"   ❌ Error training LinearSCVI model: {str(e)}")
        traceback.print_exc()
        return ge_adata, None

def train_standard_scvi(ge_adata):
    """Train standard SCVI model for gene expression data"""
    print("\n>> Training standard SCVI model...")
    
    try:
        # Setup for model
        print("   ⚙️ Setting up AnnData for standard SCVI...")
        scvi.model.SCVI.setup_anndata(ge_adata, layer="length_norm", batch_key="dataset")
        
        # Initialize and train model
        print(f"   ⚙️ Initializing SCVI with {STANDARD_LATENT} latent dimensions...")
        model = scvi.model.SCVI(ge_adata, n_latent=STANDARD_LATENT)
        
        print(f"   ⚙️ Training model for {STANDARD_EPOCHS} epochs...")
        model.train(max_epochs=STANDARD_EPOCHS, check_val_every_n_epoch=10)
        
        # Extract results and save to obsm
        print("   ⚙️ Extracting latent representation...")
        Z_hat = model.get_latent_representation()
        ge_adata.obsm["X_scVI_standard"] = Z_hat
        
        # Add model metadata to uns
        ge_adata.uns["scvi_standard"] = {
            "model_type": "SCVI",
            "n_latent": STANDARD_LATENT,
            "training_date": datetime.datetime.now().strftime("%Y-%m-%d")
        }
        
        print("   ✓ Standard SCVI model training complete")
        
        return ge_adata, model
        
    except Exception as e:
        print(f"   ❌ Error training standard SCVI model: {str(e)}")
        traceback.print_exc()
        return ge_adata, None

def plot_training_metrics(model, model_type):
    """Plot training metrics for the scVI model"""
    print(f"\n>> Generating training metrics plot for {model_type}...")
    
    try:
        # Extract ELBO loss curves
        train_elbo = model.history["elbo_train"][1:]
        test_elbo = model.history["elbo_validation"]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        ax = train_elbo.plot(label="Training ELBO")
        test_elbo.plot(ax=ax, label="Validation ELBO")
        plt.title(f"{model_type} Training Metrics")
        plt.legend()
        
        # Save plot
        plot_file = os.path.join(OUTPUT_DIR, f"scvi_{model_type.lower()}_training_metrics_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"   ✓ Training metrics plot saved to {plot_file}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error plotting training metrics: {str(e)}")
        traceback.print_exc()
        return False

def save_results(ge_adata):
    """Save updated AnnData with both scVI results"""
    print("\n>> Saving results...")
    
    try:
        # Define output filename
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        output_file = os.path.join(OUTPUT_DIR, f"ge_adata_with_both_scvi_models_{today}.h5ad")
        
        # Save file
        print(f"   ⚙️ Saving updated AnnData to {output_file}...")
        ge_adata.write_h5ad(output_file, compression="lzf")
        
        print(f"   ✓ Results successfully saved to {output_file}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error saving results: {str(e)}")
        traceback.print_exc()
        return False

# Main execution
print("\n========================================")
print("scVI Model Training - Mouse Splicing Foundation")
print("Running both LinearSCVI and standard SCVI models")
print("========================================\n")

# Load data
ge_adata = load_data()
check_data_quality(ge_adata, "length_norm")

# Train LinearSCVI model
ge_adata, linear_model = train_linear_scvi(ge_adata)

# Plot training metrics for LinearSCVI
if linear_model is not None:
    plot_training_metrics(linear_model, "LinearSCVI")

# Train standard SCVI model
ge_adata, standard_model = train_standard_scvi(ge_adata)

# Plot training metrics for standard SCVI
if standard_model is not None:
    plot_training_metrics(standard_model, "StandardSCVI")

# Save combined results to single AnnData object
save_results(ge_adata)

print("\n========================================")
print("Both scVI models training complete!")
print("Low-dimensional representations saved for downstream analysis")
print(f"Results saved to: {OUTPUT_DIR}")
print("========================================\n")

# conda activate scvi-env
# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/scVI
# sbatch --mem=250G -p gpu --gres=gpu:1 --wrap "python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/GeneExpression/08_run_scVI.py"