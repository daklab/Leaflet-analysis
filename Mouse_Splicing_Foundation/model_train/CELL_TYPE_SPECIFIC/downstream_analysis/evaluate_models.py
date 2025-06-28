#!/usr/bin/env python
"""
Simplified LeafletFA model evaluation and UMAP generation script

This script:
1. Analyzes all runs for a specified cell type
2. Creates factor analysis plots
3. Generates UMAP visualizations for ALL runs (not just the best one)
4. Creates comparison tables
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pickle
import gzip
import torch
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')

# Set scanpy settings
sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=80, facecolor='white')

def load_model(model_file):
    """Load LeafletFA model from file"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def device_load(*args, **kwargs):
        kwargs.setdefault("map_location", device)
        return original_torch_load(*args, **kwargs)
    
    original_torch_load = torch.load
    torch.load = device_load
    
    try:
        model = {}
        with gzip.open(model_file, "rb") as f:
            while True:
                try:
                    attr_dict = pickle.load(f)
                    model.update(attr_dict)
                except EOFError:
                    break
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    finally:
        torch.load = original_torch_load

def get_valid_runs(celltype_dir):
    """Get all runs with valid model and summary files"""
    if not os.path.exists(celltype_dir):
        return []
    
    runs = [d for d in os.listdir(celltype_dir) if d.startswith('run_')]
    valid_runs = []
    
    for run in runs:
        run_path = os.path.join(celltype_dir, run)
        model_file = os.path.join(run_path, 'leafletfa_model.pkl.xz')
        summary_file = os.path.join(run_path, 'run_summary.csv')
        
        if os.path.exists(model_file) and os.path.exists(summary_file):
            valid_runs.append(run)
    
    return sorted(valid_runs)

def extract_phi_from_model(model):
    """Extract PHI matrix from model"""
    possible_keys = ['assign_post', 'PHI', 'cell_factors', 'factors', 'latent_factors']
    
    for key in possible_keys:
        if key in model:
            phi = model[key]
            if hasattr(phi, 'detach'):
                return phi.detach().cpu().numpy()
            elif hasattr(phi, 'numpy'):
                return phi.numpy()
            else:
                return phi
    return None

def create_parameter_label(summary_row):
    """Create parameter label from summary data"""
    junc_prior = summary_row.get('junc_specific_prior', 'NA')
    dir_conc = summary_row.get('dir_conc', 'NA')
    return f"jp_{junc_prior}_dc_{dir_conc}"

def analyze_single_celltype(results_dir, cell_type):
    """Analyze all runs for a single cell type"""
    print(f"\nAnalyzing {cell_type}...")
    
    celltype_dir = os.path.join(results_dir, cell_type)
    valid_runs = get_valid_runs(celltype_dir)
    
    if not valid_runs:
        print(f"No valid runs found for {cell_type}")
        return pd.DataFrame()
    
    print(f"Found {len(valid_runs)} valid runs")
    
    summaries = []
    
    for run in valid_runs:
        print(f"  Processing {run}...")
        run_dir = os.path.join(celltype_dir, run)
        
        # Load summary
        summary_file = os.path.join(run_dir, 'run_summary.csv')
        summary = pd.read_csv(summary_file)
        summary['cell_type'] = cell_type
        summary['run'] = run
        summary['param_label'] = create_parameter_label(summary.iloc[0])
        
        # Load model and extract info
        model_file = os.path.join(run_dir, 'leafletfa_model.pkl.xz')
        model = load_model(model_file)
        
        if model:
            # Get K information
            initial_K = summary['K'].iloc[0] if 'K' in summary.columns else 30
            pruned_K = summary['pruned_K'].iloc[0] if 'pruned_K' in summary.columns else None
            
            # Try to get K from model if not in summary
            if pruned_K is None:
                phi = extract_phi_from_model(model)
                if phi is not None:
                    pruned_K = phi.shape[1] if len(phi.shape) > 1 else phi.shape[0]
            
            summary['initial_K'] = initial_K
            summary['final_K'] = pruned_K
            summary['K_reduction'] = initial_K - pruned_K if pruned_K else None
            summary['K_retention_rate'] = pruned_K / initial_K if pruned_K else None
            
            print(f"    {summary['param_label'].iloc[0]}: K {initial_K} -> {pruned_K}")
        
        summaries.append(summary)
    
    return pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()

def create_factor_plots(summary_df, output_dir, cell_type):
    """Create factor analysis plots"""
    print(f"Creating factor analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'LeafletFA Factor Analysis - {cell_type}', fontsize=16)
    
    # Final K by parameters
    if 'final_K' in summary_df.columns:
        data = summary_df.sort_values('final_K')
        axes[0, 0].barh(data['param_label'], data['final_K'])
        axes[0, 0].set_xlabel('Final K')
        axes[0, 0].set_title('Final K After Pruning')
        axes[0, 0].grid(axis='x', alpha=0.3)
    
    # K retention rate
    if 'K_retention_rate' in summary_df.columns:
        data = summary_df.sort_values('K_retention_rate')
        axes[0, 1].barh(data['param_label'], data['K_retention_rate'])
        axes[0, 1].set_xlabel('K Retention Rate')
        axes[0, 1].set_title('Factor Retention Rate')
        axes[0, 1].axvline(x=0.5, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].grid(axis='x', alpha=0.3)
    
    # ELBO vs Final K
    if 'final_K' in summary_df.columns and 'best_elbo' in summary_df.columns:
        axes[1, 0].scatter(summary_df['final_K'], summary_df['best_elbo'], alpha=0.7, s=80)
        axes[1, 0].set_xlabel('Final K')
        axes[1, 0].set_ylabel('Best ELBO')
        axes[1, 0].set_title('ELBO vs Final K')
        axes[1, 0].grid(alpha=0.3)
    
    # Best ELBO by parameters
    if 'best_elbo' in summary_df.columns:
        data = summary_df.sort_values('best_elbo')
        bars = axes[1, 1].barh(data['param_label'], data['best_elbo'])
        axes[1, 1].set_xlabel('Best ELBO')
        axes[1, 1].set_title('Best ELBO by Parameters')
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        # Highlight best
        best_idx = summary_df['best_elbo'].idxmin()
        best_params = summary_df.loc[best_idx, 'param_label']
        for i, bar in enumerate(bars):
            if data.iloc[i]['param_label'] == best_params:
                bar.set_color('red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{cell_type}_factor_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_umap_for_run(cell_type, run, param_label, results_dir, anndata_path, output_dir):
    """Create UMAP visualization for a specific run"""
    print(f"  Creating UMAP for {param_label}...")
    
    try:
        # Load model
        model_file = os.path.join(results_dir, cell_type, run, 'leafletfa_model.pkl.xz')
        model = load_model(model_file)
        
        if not model:
            print(f"    Could not load model for {run}")
            return
        
        # Extract PHI
        phi = extract_phi_from_model(model)
        if phi is None:
            print(f"    No PHI found for {run}")
            return
        
        # Load anndata
        adata = sc.read_h5ad(anndata_path)
        
        # Match dimensions
        if phi.shape[0] != adata.n_obs:
            if phi.shape[0] > adata.n_obs:
                phi = phi[:adata.n_obs, :]
            else:
                print(f"    PHI too small for {run}")
                return
        
        # Add PHI and compute UMAP
        adata.obsm['X_phi'] = phi
        sc.pp.neighbors(adata, use_rep='X_phi', n_neighbors=15)
        sc.tl.umap(adata, min_dist=0.5, spread=1.0)
        
        # Clustering
        sc.tl.leiden(adata, resolution=0.5, key_added='leiden_phi')
        
        # Create plots
        metadata_cols = ['leiden_phi']  # Start with clustering
        for col in ['age', 'sex', 'subtissue', 'tissue', 'dataset', 'AgingScore_unweighted']:
            if col in adata.obs.columns:
                metadata_cols.append(col)
        
        # Calculate plot grid
        n_plots = len(metadata_cols)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1 and n_cols > 1:
            axes = axes.flatten()
        elif n_rows > 1:
            axes = axes.flatten()
        
        fig.suptitle(f'{cell_type} - {param_label}', fontsize=16)
        
        for i, col in enumerate(metadata_cols):
            ax = axes[i] if n_plots > 1 else axes
            
            if col == 'leiden_phi':
                sc.pl.umap(adata, color=col, ax=ax, show=False, frameon=False,
                          legend_loc='on data', legend_fontsize=8)
                ax.set_title('PHI-based Clusters')
            else:
                sc.pl.umap(adata, color=col, ax=ax, show=False, frameon=False)
                ax.set_title(f'Colored by {col}')
        
        # Hide empty subplots
        for i in range(n_plots, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save with safe filename
        safe_label = param_label.replace("/", "_").replace(" ", "_")
        output_file = os.path.join(output_dir, f'{cell_type}_{safe_label}_UMAP.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    UMAP saved: {safe_label}")
        
    except Exception as e:
        print(f"    Error creating UMAP for {run}: {e}")

def create_all_umaps(summary_df, cell_type, results_dir, anndata_path, output_dir):
    """Create UMAP visualizations for ALL runs"""
    print(f"\nCreating UMAP visualizations for all runs...")
    
    if not os.path.exists(anndata_path):
        print(f"Anndata file not found: {anndata_path}")
        return
    
    for _, row in summary_df.iterrows():
        create_umap_for_run(
            cell_type=cell_type,
            run=row['run'],
            param_label=row['param_label'],
            results_dir=results_dir,
            anndata_path=anndata_path,
            output_dir=output_dir
        )

def main():
    """Main function"""
    
    # Accept cell type from command line or use default
    if len(sys.argv) > 1:
        cell_type = sys.argv[1]
    else:
        cell_type = "Inhibitory_Neurons"  # Default for testing
    
    print("LeafletFA Analysis Script")
    print("=" * 50)
    model_date = "2025-06-28"
    
    # Paths
    results_dir = f"/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel_celltype/{model_date}/results"
    anndatas_dir = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/062025/per_celltype_20250625_215409"
    output_dir = f"/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/CELL_TYPE_SPECIFIC/results/{model_date}_simplified_{cell_type}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Target cell type: {cell_type}")
    print(f"Output directory: {output_dir}")
    
    # 1. Analyze cell type
    summary_df = analyze_single_celltype(results_dir, cell_type)
    
    if summary_df.empty:
        print(f"No data found for {cell_type}")
        return
    
    # Save summary
    summary_output = os.path.join(output_dir, f'{cell_type}_summary.csv')
    summary_df.to_csv(summary_output, index=False)
    print(f"\nSummary saved: {summary_output}")
    
    # Print statistics
    print(f"\nSummary Statistics:")
    print(f"  Number of runs: {len(summary_df)}")
    if 'final_K' in summary_df.columns:
        print(f"  Final K range: {summary_df['final_K'].min()} - {summary_df['final_K'].max()}")
        print(f"  Average final K: {summary_df['final_K'].mean():.1f}")
    if 'best_elbo' in summary_df.columns:
        best_run = summary_df.loc[summary_df['best_elbo'].idxmin()]
        print(f"  Best run: {best_run['param_label']} (ELBO: {best_run['best_elbo']:.2f})")
    
    # 2. Create factor analysis plots
    create_factor_plots(summary_df, output_dir, cell_type)
    
    # 3. Find anndata file
    anndata_files = glob.glob(os.path.join(anndatas_dir, cell_type, f"*{cell_type}*.h5ad"))
    
    if not anndata_files:
        print(f"\nNo anndata file found for {cell_type}")
        print(f"Searched in: {os.path.join(anndatas_dir, cell_type)}")
    else:
        anndata_path = anndata_files[0]
        print(f"\nUsing anndata: {anndata_path}")
        
        # 4. Create UMAP plots for ALL runs
        create_all_umaps(summary_df, cell_type, results_dir, anndata_path, output_dir)
    
    # 5. Create comparison table
    if not summary_df.empty:
        comparison_cols = ['param_label', 'best_elbo', 'final_K', 'K_retention_rate']
        if 'junc_specific_prior' in summary_df.columns:
            comparison_cols.append('junc_specific_prior')
        if 'dir_conc' in summary_df.columns:
            comparison_cols.append('dir_conc')
        
        comparison = summary_df[comparison_cols].sort_values('best_elbo')
        comparison.to_csv(os.path.join(output_dir, f'{cell_type}_comparison.csv'), index=False)
        print(f"\nComparison table saved")
    
    print(f"\nAnalysis complete! Results in: {output_dir}")


if __name__ == "__main__":
    main()

