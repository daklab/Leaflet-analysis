#!/usr/bin/env python
"""
Consolidated LeafletFA Model Analysis Script
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from scipy import stats
from scipy.sparse import csr_matrix
import gzip
import pickle
import torch
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr

# Configure plotting
sns.set_theme()
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

def load_model(model_path):
    """Load a compressed LeafletFA model"""
    print(f"Loading model from {model_path}")
    
    model_data = {}
    with gzip.open(model_path, 'rb') as f:
        # Load metadata
        metadata = pickle.load(f)
        print(f"Model metadata: {metadata['_meta']}")
        
        # Load all other attributes
        try:
            while True:
                data = pickle.load(f)
                model_data.update(data)
        except EOFError:
            pass
    
    return model_data, metadata['_meta']

def calculate_perplexity(PHI):
    """Calculate cell perplexity from PHI matrix"""
    PHI_safe = np.clip(PHI, 1e-10, 1)
    entropy = -np.sum(PHI_safe * np.log(PHI_safe), axis=1)
    perplexity = np.exp(entropy)
    return perplexity, entropy

def check_psi_correlation(splice_adata, PSI_CELLS, n_samples=10, sample_size=5000):
    """
    Check correlation between imputed and observed PSI values
    This is a critical sanity check for the model
    """
    print("\n=== PSI Correlation Check ===")
    
    # Get sparse matrices
    junction_counts = splice_adata.layers["cell_by_junction_matrix"]
    cluster_counts = splice_adata.layers["cell_by_cluster_matrix"]
    
    # Find valid (cell, junction) pairs where we have observed data
    valid_cells, valid_junctions = cluster_counts.nonzero()
    n_valid = len(valid_cells)
    
    correlations = []
    
    for i in tqdm(range(n_samples), desc="Sampling PSI correlations"):
        # Sample pairs
        n_to_sample = min(sample_size, n_valid)
        sample_idx = np.random.choice(n_valid, size=n_to_sample, replace=False)
        
        sampled_cells = valid_cells[sample_idx]
        sampled_junctions = valid_junctions[sample_idx]
        
        # Calculate observed PSI
        numerators = junction_counts[sampled_cells, sampled_junctions].A1
        denominators = cluster_counts[sampled_cells, sampled_junctions].A1
        psi_observed = numerators / denominators
        
        # Get imputed PSI
        psi_imputed = PSI_CELLS[sampled_cells, sampled_junctions]
        
        # Calculate correlation
        if len(psi_observed) > 2 and np.std(psi_observed) > 1e-9:
            corr, _ = pearsonr(psi_observed, psi_imputed)
            correlations.append(corr)
    
    mean_corr = np.mean(correlations)
    std_corr = np.std(correlations)
    
    print(f"PSI Correlation: {mean_corr:.3f} ± {std_corr:.3f}")
    print(f"Min: {np.min(correlations):.3f}, Max: {np.max(correlations):.3f}")
    return correlations, mean_corr

def analyze_factor_characteristics(PHI, PI, splice_adata, output_dir):
    """Analyze basic factor characteristics"""
    print("\n=== Factor Characteristics ===")
    
    K = PHI.shape[1]
    
    # 1. Factor usage (PI distribution)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # PI distribution
    axes[0].bar(range(K), PI)
    axes[0].set_xlabel('Factor Index')
    axes[0].set_ylabel('PI (Global Assignment Probability)')
    axes[0].set_title(f'Factor Usage (K={K})')
    axes[0].axhline(1/K, color='r', linestyle='--', alpha=0.5, label='Uniform')
    
    # Cumulative PI
    cumulative_pi = np.cumsum(np.sort(PI)[::-1])
    axes[1].plot(cumulative_pi, marker='o')
    axes[1].axhline(0.9, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Number of Factors')
    axes[1].set_ylabel('Cumulative PI')
    axes[1].set_title('Cumulative Factor Usage')
    axes[1].grid(True, alpha=0.3)
    
    # Factor sparsity in cells
    factor_activity = (PHI > 0.1).sum(axis=0)  # Count cells with >10% factor activity
    axes[2].bar(range(K), factor_activity)
    axes[2].set_xlabel('Factor Index')
    axes[2].set_ylabel('Number of Cells (>10% activity)')
    axes[2].set_title('Factor Sparsity')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'factor_characteristics.png'))
    plt.close()
    
    # Report statistics
    n_effective = (cumulative_pi < 0.9).sum() + 1
    print(f"Number of factors: {K}")
    print(f"Effective factors (90% cumulative PI): {n_effective}")
    print(f"Top 5 factor PIs: {PI[:5]}")
    print(f"Factor entropy: {stats.entropy(PI):.3f}")
    
    return n_effective

def analyze_batch_effects(splice_adata, batch_column, cell_type_column, PHI, output_dir):
    """Check for batch effects in the learned representation"""
    print("\n=== Batch Effect Analysis ===")
    
    if batch_column not in splice_adata.obs.columns:
        print("No dataset/batch information available")
        return
    
    # Compute UMAP on PHI
    print("Computing UMAP...")
    splice_adata.obsm['X_PHI'] = PHI
    sc.pp.neighbors(splice_adata, use_rep='X_PHI', n_neighbors=15)
    sc.tl.umap(splice_adata)
    
    # Create visualization - single plot with clear legend
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # UMAP by dataset
    sc.pl.umap(splice_adata, color=batch_column, ax=ax, show=False, 
               frameon=True, title=f'UMAP by {batch_column}')
    
    # Improve legend if it exists
    legend = ax.get_legend()
    if legend is not None:
        # Move legend to the right side with better formatting
        legend.set_bbox_to_anchor((1.05, 1))
        legend.set_loc('upper left')
        legend.set_title(batch_column, prop={'weight':'bold', 'size': 12})
        # Make legend text larger and clearer
        for text in legend.get_texts():
            text.set_fontsize(11)
        # Add a frame around the legend
        legend.set_frame_on(True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(0.5)
    
    # Improve plot aesthetics
    ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
    ax.set_title(f'UMAP Colored by {batch_column}', fontsize=14, fontweight='bold', pad=20)
    
    # Save with proper layout
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'batch_effects_umap.png'), 
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    # Quantify variance explained by batch
    from sklearn.linear_model import LinearRegression
    
    # Use all PHI dimensions
    X = PHI
    
    if batch_column in splice_adata.obs.columns:
        # One-hot encode dataset
        datasets = pd.get_dummies(splice_adata.obs[batch_column])
    else:
        datasets = None
    
    # Calculate R² for each PHI dimension
    r2_scores = []
    for i in range(X.shape[1]):
        model = LinearRegression()
        model.fit(datasets, X[:, i])
        y_pred = model.predict(datasets)
        r2 = r2_score(X[:, i], y_pred)
        r2_scores.append(r2)
    
    mean_r2 = np.mean(r2_scores)
    print(f"Mean R² of batch on all factors: {mean_r2:.3f}")
    print(f"Max R² on any factor: {np.max(r2_scores):.3f}")
    return mean_r2

def analyze_training_convergence(model_meta, output_dir):
    """Analyze training convergence from multi-pass training"""
    print("\n=== Training Convergence Analysis ===")
    
    # Check for pi evolution file
    pi_evolution_file = os.path.join(os.path.dirname(output_dir), 
                                     f"run_{model_meta.get('param_id', 'unknown')}", 
                                     "pi_evolution.csv")
    
    if os.path.exists(pi_evolution_file):
        pi_df = pd.read_csv(pi_evolution_file)
        
        # Plot PI evolution
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Top 10 factors evolution
        top_factors = pi_df.groupby('factor')['pi_value'].max().nlargest(10).index
        for factor in top_factors:
            factor_data = pi_df[pi_df['factor'] == factor]
            axes[0].plot(factor_data['pass'], factor_data['pi_value'], 
                        marker='o', label=f'Factor {factor}')
        
        axes[0].set_xlabel('Pass Number')
        axes[0].set_ylabel('PI Value')
        axes[0].set_title('Top 10 Factors PI Evolution')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # ELBO evolution
        elbo_by_pass = pi_df.groupby('pass')['avg_elbo'].first()
        axes[1].plot(elbo_by_pass.index, elbo_by_pass.values, marker='o', color='red')
        axes[1].set_xlabel('Pass Number')
        axes[1].set_ylabel('Average ELBO')
        axes[1].set_title('ELBO Convergence')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_convergence.png'))
        plt.close()
        
        # Check convergence
        if len(elbo_by_pass) > 1:
            elbo_change = (elbo_by_pass.iloc[-1] - elbo_by_pass.iloc[-2]) / abs(elbo_by_pass.iloc[-2])
            print(f"ELBO change in last pass: {elbo_change:.4%}")
    else:
        print("PI evolution file not found")
    
    print(f"Training mode: {model_meta.get('training_mode', 'unknown')}")
    print(f"Number of passes: {model_meta.get('num_passes', 'unknown')}")
    print(f"Batch size: {model_meta.get('batch_size', 'unknown')}")
    print(f"Final K: {model_meta.get('pruned_K', 'unknown')} (from {model_meta.get('original_K', 'unknown')})")

def generate_summary_report(results, output_dir):
    """Generate a summary report of all analyses"""
    
    report = []
    report.append("="*60)
    report.append("LEAFLETFA MODEL ANALYSIS SUMMARY")
    report.append("="*60)
    
    # Model info
    report.append(f"\nModel: {results['model_path']}")
    report.append(f"Param ID: {results['param_id']}")
    report.append(f"Training Date: {results.get('training_date', 'Unknown')}")
    
    # Key metrics
    report.append(f"\n--- KEY METRICS ---")
    report.append(f"Number of factors (K): {results['K']}")
    report.append(f"Effective factors: {results.get('n_effective', 'N/A')}")
    report.append(f"Best ELBO: {results.get('best_elbo', 'N/A'):.2e}")
    
    # PSI correlation (most important sanity check)
    report.append(f"\n--- PSI CORRELATION CHECK ---")
    report.append(f"Mean correlation (imputed vs observed): {results.get('psi_correlation', 'N/A'):.3f}")
    if results.get('psi_correlation', 0) < 0.5:
        report.append("⚠️ WARNING: Low PSI correlation detected!")
    else:
        report.append("✓ PSI correlation looks good")
    
    # Perplexity
    report.append(f"\n--- CELL PERPLEXITY ---")
    report.append(f"Median perplexity: {results.get('median_perplexity', 'N/A'):.2f}")
    report.append(f"Mean perplexity: {results.get('mean_perplexity', 'N/A'):.2f}")
    
    # Batch effects
    report.append(f"\n--- BATCH EFFECTS ---")
    report.append(f"Batch variance explained (R²): {results.get('batch_r2', 'N/A'):.3f}")
    if results.get('batch_r2', 1.0) > 0.3:
        report.append("⚠️ WARNING: Strong batch effects detected!")
    else:
        report.append("✓ Batch effects appear minimal")
    
    # Training
    report.append(f"\n--- TRAINING INFO ---")
    report.append(f"Training mode: {results.get('training_mode', 'Unknown')}")
    report.append(f"Number of passes: {results.get('num_passes', 'Unknown')}")
    report.append(f"Total batches: {results.get('total_batches', 'Unknown')}")
    
    # Save report
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w') as f:
        f.write(report_text)
    
    # Also save as CSV for easy parsing
    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(output_dir, 'analysis_metrics.csv'), index=False)

def main():
    """Main analysis pipeline"""
    
    # Parse arguments
    if len(sys.argv) < 4:
        print("Usage: python analyze_leafletfa.py <param_id> <model_dir> <adata_path> <batch_column> <cell_type_column> [output_dir]")
        sys.exit(1)
    
    param_id = sys.argv[1]
    model_dir = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 4 else f"analysis_run_{param_id}"
    batch_column = sys.argv[4]
    cell_type_column = sys.argv[5]

    # For given param load all the parameters that were used --> lr, gamma, initial_K, from corresponding json file
    param_file = os.path.join(model_dir, "parameter_combinations.csv")
    param_df = pd.read_csv(param_file)

    # Initialize results dictionary
    results = {
        'param_id': param_id,
        'model_path': os.path.join(model_dir, f"run_{param_id}", "leafletfa_model.pkl.gz"),
        'batch_column': batch_column,
        'cell_type_column': cell_type_column
    }

    # ensure param_id is an integer
    param_id = int(param_id)
    
    results['lr'] = param_df.iloc[param_id]['lr']
    results['gamma'] = param_df.iloc[param_id]['gamma']
    results['initial_K'] = param_df.iloc[param_id]['K']
    results['num_passes'] = param_df.iloc[param_id]['num_passes']
    results['batch_size'] = param_df.iloc[param_id]['batch_size']
    results['num_epochs_first'] = param_df.iloc[param_id]['num_epochs_first']
    results['num_epochs_later'] = param_df.iloc[param_id]['num_epochs_later']
    results['ELBO_num_particles'] = param_df.iloc[param_id]['ELBO_num_particles']
    results['junc_specific_prior'] = param_df.iloc[param_id]['junc_specific_prior']
    results["waypoints_use"] = param_df.iloc[param_id]['waypoints_use']
    results["max_junctions"] = param_df.iloc[param_id]['max_junctions']
    results["n_waypoints"] = param_df.iloc[param_id]['n_waypoints']
    results["adata_path"] = param_df.iloc[param_id]['anndata_file']

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    print("\n" + "="*60)
    print(f"LEAFLETFA MODEL ANALYSIS - Param ID: {param_id}")
    print("="*60)
    
    # 1. Load model
    print("\n>>> Loading model...")
    model_path = results['model_path']
    if not os.path.exists(model_path):
        # Try .xz extension
        model_path = model_path.replace('.gz', '.xz')
        results['model_path'] = model_path
    
    model_data, model_meta = load_model(model_path)
    results.update(model_meta)

    # load pruned_K and original_K from model metadata
    results['pruned_K'] = model_meta.get('pruned_K', 'Unknown')
    results['original_K'] = model_meta.get('original_K', 'Unknown')
    print(f"Pruned K: {results['pruned_K']}, Original K: {results['original_K']}")

    # Extract key components
    PHI = model_data['assign_post']
    PSI = model_data['psi']
    print(f"The shape of the PSI matrix is: {PSI.shape}")
    print(f"The shape of the PHI matrix is: {PHI.shape}")

    PI = model_data['pi']
    K = model_data['K']
    results['K'] = K
    results['best_elbo'] = model_data.get('best_elbo', np.nan)
    results['batch_column'] = batch_column
    results['cell_type_column'] = cell_type_column

    print(f"Model loaded: K={K}, PHI shape={PHI.shape}")
    
    # 2. Load data
    print("\n>>> Loading AnnData...")
    splice_adata = ad.read_h5ad(results["adata_path"])
    print(f"Data shape: {splice_adata.shape}")
    
    # Filter ATSEs by Junction Count (if specified)
    MAX_JUNCTIONS = results["max_junctions"] # make sure MAX_JUNCTIONS is an integer
    MAX_JUNCTIONS = int(MAX_JUNCTIONS)
    if "num_junctions" in splice_adata.var.columns:
        splice_adata = splice_adata[:, splice_adata.var["num_junctions"] <= MAX_JUNCTIONS].copy()
        splice_adata.var["junction_id_index"] = np.arange(splice_adata.shape[1])
    
    print(f"The number of junctions in the adata object is: {splice_adata.shape[1]}")
    
    # 3. Core sanity checks
    print("\n>>> Running core sanity checks...")
    
    # Calculate imputed PSI
    PSI_CELLS = np.dot(PHI, PSI)
    
    # Check PSI correlation (most important check)
    if 'cell_by_junction_matrix' in splice_adata.layers:
        correlations, mean_corr = check_psi_correlation(splice_adata, PSI_CELLS)
        results['psi_correlation'] = mean_corr
        results['psi_correlations'] = correlations
        
        # Save correlation plot
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.hist(correlations, bins=20, edgecolor='black', alpha=0.7)
        plt.axvline(mean_corr, color='red', linestyle='--', label=f'Mean: {mean_corr:.3f}')
        plt.xlabel('Pearson Correlation')
        plt.ylabel('Count')
        plt.title('PSI Correlation Distribution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.boxplot(correlations)
        plt.ylabel('Pearson Correlation')
        plt.title('PSI Correlation Boxplot')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'psi_correlation_check.png'))
        plt.close()
    else:
        print("Warning: Cannot check PSI correlation - missing junction matrices")
    
    # 4. Factor analysis
    print("\n>>> Analyzing factors...")
    n_effective = analyze_factor_characteristics(PHI, PI, splice_adata, output_dir)
    results['n_effective'] = n_effective

    # 5. Perplexity analysis
    print("\n>>> Computing cell perplexity...")
    perplexity, entropy = calculate_perplexity(PHI)
    results['median_perplexity'] = np.median(perplexity)
    results['mean_perplexity'] = np.mean(perplexity)
    
    # Plot perplexity distribution
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(perplexity, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(np.median(perplexity), color='red', linestyle='--', 
                label=f'Median: {np.median(perplexity):.2f}')
    plt.xlabel('Perplexity (Effective # of Factors)')
    plt.ylabel('Number of Cells')
    plt.title('Cell Perplexity Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(perplexity[:1000])), np.sort(perplexity)[:1000], 
                alpha=0.5, s=1)
    plt.xlabel('Cell Rank')
    plt.ylabel('Perplexity')
    plt.title('Sorted Perplexity (first 1000 cells)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'perplexity_distribution.png'))
    plt.close()
    
    # Batch effects (if applicable)
    print("\n>>> Checking for batch effects...")
    batch_r2 = analyze_batch_effects(splice_adata, batch_column, cell_type_column, PHI, output_dir)
    results['batch_r2'] = batch_r2 if batch_r2 is not None else np.nan
    
    # Training convergence
    print("\n>>> Analyzing training convergence...")
    analyze_training_convergence(model_meta, output_dir)
    
    # Generate summary report
    print("\n>>> Generating summary report...")
    generate_summary_report(results, output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()

