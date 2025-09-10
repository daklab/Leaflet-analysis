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

def analyze_age_regression(splice_adata, PHI, output_dir):
    """
    Perform age regression analysis using splicing factors.
    Tests both globally and within cell types/tissues.
    """
    print("\n=== Age Regression Analysis ===")
    
    # Ensure age_numeric exists and is properly formatted
    if 'age_numeric' not in splice_adata.obs.columns:
        if 'age' in splice_adata.obs.columns:
            age_str = splice_adata.obs['age'].astype(str)
            if age_str.str.contains('m').any():
                # Mouse data with months
                splice_adata.obs['age_numeric'] = pd.to_numeric(
                    age_str.str.replace('m', '', regex=False)
                )
                age_unit = 'months'
            else:
                # Human data or already numeric
                splice_adata.obs['age_numeric'] = pd.to_numeric(age_str)
                age_unit = 'years'
        else:
            print("Warning: No age information found")
            return None
    
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    results = {}
    
    # 1. Global age regression
    print("\n--- Global Age Regression ---")
    X = PHI
    y = splice_adata.obs['age_numeric'].values
    
    # Remove any NaN values
    valid_idx = ~np.isnan(y)
    X_clean = X[valid_idx]
    y_clean = y[valid_idx]
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Ridge regression
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    
    print(f"Global R² (train): {r2_train:.3f}")
    print(f"Global R² (test): {r2_test:.3f}")
    print(f"Global MSE (test): {mse_test:.2f}")
    
    results['global'] = {
        'r2_train': r2_train,
        'r2_test': r2_test,
        'mse_test': mse_test,
        'n_samples': len(y_clean)
    }
    
    # 2. Within cell type regression
    if 'broad_cell_type' in splice_adata.obs.columns:
        print("\n--- Age Regression by Cell Type ---")
        cell_type_results = {}
        
        # Get top cell types by frequency
        top_cell_types = splice_adata.obs['broad_cell_type'].value_counts().head(10).index
        
        for cell_type in top_cell_types:
            mask = (splice_adata.obs['broad_cell_type'] == cell_type) & valid_idx
            
            if mask.sum() < 50:  # Need minimum samples
                continue
            
            X_ct = X[mask]
            y_ct = y[mask]
            
            # Split data
            if len(y_ct) < 100:
                test_size = 0.3
            else:
                test_size = 0.2
                
            try:
                X_train_ct, X_test_ct, y_train_ct, y_test_ct = train_test_split(
                    X_ct, y_ct, test_size=test_size, random_state=42
                )
                
                # Scale and train
                scaler_ct = StandardScaler()
                X_train_ct_scaled = scaler_ct.fit_transform(X_train_ct)
                X_test_ct_scaled = scaler_ct.transform(X_test_ct)
                
                model_ct = Ridge(alpha=1.0)
                model_ct.fit(X_train_ct_scaled, y_train_ct)
                
                # Evaluate
                y_pred_ct = model_ct.predict(X_test_ct_scaled)
                r2_ct = r2_score(y_test_ct, y_pred_ct)
                
                cell_type_results[cell_type] = {
                    'r2': r2_ct,
                    'n_samples': len(y_ct),
                    'age_range': (y_ct.min(), y_ct.max())
                }
                
                print(f"  {cell_type}: R²={r2_ct:.3f}, n={len(y_ct)}")
                
            except Exception as e:
                print(f"  {cell_type}: Failed - {str(e)}")
                continue
        
        results['by_cell_type'] = cell_type_results
    
    # 3. Within tissue regression
    if 'tissue' in splice_adata.obs.columns:
        print("\n--- Age Regression by Tissue ---")
        tissue_results = {}
        
        # Get top tissues by frequency
        top_tissues = splice_adata.obs['tissue'].value_counts().head(10).index
        
        for tissue in top_tissues:
            mask = (splice_adata.obs['tissue'] == tissue) & valid_idx
            
            if mask.sum() < 50:  # Need minimum samples
                continue
            
            X_tissue = X[mask]
            y_tissue = y[mask]
            
            # Split data
            if len(y_tissue) < 100:
                test_size = 0.3
            else:
                test_size = 0.2
                
            try:
                X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
                    X_tissue, y_tissue, test_size=test_size, random_state=42
                )
                
                # Scale and train
                scaler_t = StandardScaler()
                X_train_t_scaled = scaler_t.fit_transform(X_train_t)
                X_test_t_scaled = scaler_t.transform(X_test_t)
                
                model_t = Ridge(alpha=1.0)
                model_t.fit(X_train_t_scaled, y_train_t)
                
                # Evaluate
                y_pred_t = model_t.predict(X_test_t_scaled)
                r2_t = r2_score(y_test_t, y_pred_t)
                
                tissue_results[tissue] = {
                    'r2': r2_t,
                    'n_samples': len(y_tissue),
                    'age_range': (y_tissue.min(), y_tissue.max())
                }
                
                print(f"  {tissue}: R²={r2_t:.3f}, n={len(y_tissue)}")
                
            except Exception as e:
                print(f"  {tissue}: Failed - {str(e)}")
                continue
        
        results['by_tissue'] = tissue_results
    
    # 4. Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Global prediction scatter
    ax = axes[0, 0]
    ax.scatter(y_test, y_pred_test, alpha=0.5, s=10)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('True Age')
    ax.set_ylabel('Predicted Age')
    ax.set_title(f'Global Age Prediction (R²={r2_test:.3f})')
    ax.grid(True, alpha=0.3)
    
    # Cell type R² comparison
    if 'by_cell_type' in results and results['by_cell_type']:
        ax = axes[0, 1]
        ct_df = pd.DataFrame(results['by_cell_type']).T
        ct_df = ct_df.sort_values('r2', ascending=False)
        ax.barh(range(len(ct_df)), ct_df['r2'])
        ax.set_yticks(range(len(ct_df)))
        ax.set_yticklabels(ct_df.index, fontsize=8)
        ax.set_xlabel('R² Score')
        ax.set_title('Age Prediction by Cell Type')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add sample size annotations
        for i, (idx, row) in enumerate(ct_df.iterrows()):
            ax.text(row['r2'] + 0.01, i, f"n={int(row['n_samples'])}", 
                   va='center', fontsize=8)
    
    # Tissue R² comparison
    if 'by_tissue' in results and results['by_tissue']:
        ax = axes[1, 0]
        tissue_df = pd.DataFrame(results['by_tissue']).T
        tissue_df = tissue_df.sort_values('r2', ascending=False)
        ax.barh(range(len(tissue_df)), tissue_df['r2'])
        ax.set_yticks(range(len(tissue_df)))
        ax.set_yticklabels(tissue_df.index, fontsize=8)
        ax.set_xlabel('R² Score')
        ax.set_title('Age Prediction by Tissue')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add sample size annotations
        for i, (idx, row) in enumerate(tissue_df.iterrows()):
            ax.text(row['r2'] + 0.01, i, f"n={int(row['n_samples'])}", 
                   va='center', fontsize=8)
    
    # Summary statistics
    ax = axes[1, 1]
    summary_text = []
    summary_text.append(f"Global Age Regression:")
    summary_text.append(f"  R² (test): {r2_test:.3f}")
    summary_text.append(f"  MSE: {mse_test:.2f}")
    summary_text.append(f"  N samples: {len(y_clean)}")
    summary_text.append(f"\nAge range: {y_clean.min():.1f} - {y_clean.max():.1f}")
    
    if 'by_cell_type' in results:
        r2_values = [v['r2'] for v in results['by_cell_type'].values()]
        summary_text.append(f"\nCell Type R² range:")
        summary_text.append(f"  Min: {min(r2_values):.3f}")
        summary_text.append(f"  Max: {max(r2_values):.3f}")
        summary_text.append(f"  Mean: {np.mean(r2_values):.3f}")
    
    if 'by_tissue' in results:
        r2_values = [v['r2'] for v in results['by_tissue'].values()]
        summary_text.append(f"\nTissue R² range:")
        summary_text.append(f"  Min: {min(r2_values):.3f}")
        summary_text.append(f"  Max: {max(r2_values):.3f}")
        summary_text.append(f"  Mean: {np.mean(r2_values):.3f}")
    
    ax.text(0.1, 0.9, '\n'.join(summary_text), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'age_regression_analysis.png'), dpi=300)
    plt.close()
    
    # Save results to CSV
    results_df = pd.DataFrame([results['global']])
    results_df.to_csv(os.path.join(output_dir, 'age_regression_global.csv'), index=False)
    
    if 'by_cell_type' in results:
        ct_results_df = pd.DataFrame(results['by_cell_type']).T
        ct_results_df.to_csv(os.path.join(output_dir, 'age_regression_by_cell_type.csv'))
    
    if 'by_tissue' in results:
        tissue_results_df = pd.DataFrame(results['by_tissue']).T
        tissue_results_df.to_csv(os.path.join(output_dir, 'age_regression_by_tissue.csv'))
    
    return results

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

def analyze_batch_effects(splice_adata, PHI, output_dir):
    """Check for batch effects in the learned representation"""
    print("\n=== Batch Effect Analysis ===")
    
    if 'dataset' not in splice_adata.obs.columns:
        print("No dataset/batch information available")
        return
    
    # Compute UMAP on PHI
    print("Computing UMAP...")
    splice_adata.obsm['X_PHI'] = PHI
    sc.pp.neighbors(splice_adata, use_rep='X_PHI', n_neighbors=15)
    sc.tl.umap(splice_adata)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # UMAP by dataset
    sc.pl.umap(splice_adata, color='dataset', ax=axes[0], show=False, 
               frameon=True, title='UMAP by Dataset')
    
    # UMAP by cell type (if available)
    if 'broad_cell_type' in splice_adata.obs.columns:
        # Show top 10 cell types
        top_types = splice_adata.obs['broad_cell_type'].value_counts().head(10).index
        splice_adata.obs['cell_type_display'] = splice_adata.obs['broad_cell_type'].apply(
            lambda x: x if x in top_types else 'Other'
        )
        sc.pl.umap(splice_adata, color='cell_type_display', ax=axes[1], show=False,
                  frameon=True, title='UMAP by Cell Type (Top 10)')
    
    # UMAP by perplexity
    splice_adata.obs['perplexity'] = calculate_perplexity(PHI)[0]
    sc.pl.umap(splice_adata, color='perplexity', ax=axes[2], show=False,
              frameon=True, title='UMAP by Perplexity', cmap='viridis')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'batch_effects_umap.png'))
    plt.close()
    
    # Quantify variance explained by batch
    from sklearn.linear_model import LinearRegression
    
    # Use first 10 PHI dimensions
    X = PHI[:, :min(10, PHI.shape[1])]
    
    # One-hot encode dataset
    datasets = pd.get_dummies(splice_adata.obs['dataset'])
    
    # Calculate R² for each PHI dimension
    r2_scores = []
    for i in range(X.shape[1]):
        model = LinearRegression()
        model.fit(datasets, X[:, i])
        y_pred = model.predict(datasets)
        r2 = r2_score(X[:, i], y_pred)
        r2_scores.append(r2)
    
    mean_r2 = np.mean(r2_scores)
    print(f"Mean R² of batch on first 10 factors: {mean_r2:.3f}")
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
        print("Usage: python analyze_leafletfa.py <param_id> <model_dir> <adata_path> [output_dir]")
        sys.exit(1)
    
    param_id = sys.argv[1]
    model_dir = sys.argv[2]
    adata_path = sys.argv[3]
    output_dir = sys.argv[4] if len(sys.argv) > 4 else f"analysis_run_{param_id}"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Initialize results dictionary
    results = {
        'param_id': param_id,
        'model_path': os.path.join(model_dir, f"run_{param_id}", "leafletfa_model.pkl.gz")
    }
    
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
    
    # Extract key components
    PHI = model_data['assign_post']
    PSI = model_data['psi']
    print(f"The shape of the PSI matrix is: {PSI.shape}")
    print(f"The shape of the PHI matrix is: {PHI.shape}")

    PI = model_data['pi']
    K = model_data['K']
    results['K'] = K
    results['best_elbo'] = model_data.get('best_elbo', np.nan)
    
    print(f"Model loaded: K={K}, PHI shape={PHI.shape}")
    
    # 2. Load data
    print("\n>>> Loading AnnData...")
    splice_adata = ad.read_h5ad(adata_path)
    print(f"Data shape: {splice_adata.shape}")
    
    # Filter ATSEs by Junction Count (if specified)
    MAX_JUNCTIONS = 4 
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
    
    # 5.5 Age regression analysis
    print("\n>>> Performing age regression analysis...")
    age_results = analyze_age_regression(splice_adata, PHI, output_dir)
    if age_results:
        results['age_regression_global_r2'] = age_results['global']['r2_test']

    # 6. Batch effects (if applicable)
    print("\n>>> Checking for batch effects...")
    batch_r2 = analyze_batch_effects(splice_adata, PHI, output_dir)
    results['batch_r2'] = batch_r2 if batch_r2 is not None else np.nan
    
    # 7. Training convergence
    print("\n>>> Analyzing training convergence...")
    analyze_training_convergence(model_meta, output_dir)
    
    # 8. Generate summary report
    print("\n>>> Generating summary report...")
    generate_summary_report(results, output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()

