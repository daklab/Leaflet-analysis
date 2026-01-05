#!/usr/bin/env python
"""
Comprehensive Parameter Analysis for LeafletFA Model Results

This script analyzes the effects of model parameters on key performance metrics:
- gamma, lr, batch_size, and junc_specific_prior effects on age regression R² and cell perplexity
- PSI correlation analysis and its relationship to model performance
- Factor usage and pruning analysis
- Gene expression R² analysis (when available)

Generates multiple output files:
1. parameter_effects.png - Main parameter effects grid
2. correlation_analysis.png - PSI correlation vs performance metrics
3. factor_analysis.png - Factor usage and pruning analysis
4. parameter_summary.csv - Summary statistics for each parameter
5. simple_results.csv - Raw data for further analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

def load_results(base_dir):
    """Load all results and parameters"""
    results = []
    
    print("Loading analysis results...")
    
    # Load each analysis result
    analysis_dirs = list(Path(base_dir).glob("analysis_run_*"))
    print(f"Found {len(analysis_dirs)} analysis directories")
    
    for analysis_dir in analysis_dirs:
        try:
            run_id = int(analysis_dir.name.split("_")[-1])
            
            # Load metrics
            metrics_file = analysis_dir / "analysis_metrics.csv"
            if not metrics_file.exists():
                print(f"  Warning: No metrics file for run {run_id}")
                continue
                
            metrics = pd.read_csv(metrics_file).iloc[0].to_dict()
            metrics['run_id'] = run_id
            
            # Load age regression results
            age_file = analysis_dir / "age_regression_global.csv"
            if age_file.exists():
                age_df = pd.read_csv(age_file)
                if not age_df.empty:
                    metrics['age_r2_test'] = age_df['r2_test'].iloc[0]
                    metrics['age_r2_train'] = age_df['r2_train'].iloc[0]
                    metrics['age_mse_test'] = age_df['mse_test'].iloc[0]
                    metrics['age_n_samples'] = age_df['n_samples'].iloc[0]
            
            # Look for gene expression regression results (if available)
            ge_file = analysis_dir / "gene_expression_regression_global.csv"
            if ge_file.exists():
                ge_df = pd.read_csv(ge_file)
                if not ge_df.empty:
                    metrics['ge_r2_test'] = ge_df['r2_test'].iloc[0]
                    metrics['ge_r2_train'] = ge_df['r2_train'].iloc[0]
            
            results.append(metrics)
            
        except Exception as e:
            print(f"  Error loading run {analysis_dir.name}: {e}")
            continue
    
    if not results:
        raise ValueError("No valid results found!")
    
    df = pd.DataFrame(results)
    print(f"Loaded results for {len(df)} runs")
    
    # Load parameters
    param_file = Path(base_dir) / "parameter_combinations.csv"
    if param_file.exists():
        print("Loading parameter combinations...")
        params = pd.read_csv(param_file)
        
        # Handle different indexing schemes
        if 'run_id' not in params.columns:
            params['run_id'] = params.index
        
        # Merge with results
        df = pd.merge(df, params, on='run_id', how='left')
        print(f"Merged parameters for {len(df)} runs")
    else:
        print("Warning: No parameter_combinations.csv found")
    
    return df

def analyze_parameter_effects(df, output_dir):
    """Analyze effects of key parameters on performance metrics"""
    print("\nAnalyzing parameter effects...")
    
    # Key metrics to analyze
    metrics = ['age_r2_test', 'mean_perplexity', 'psi_correlation']
    
    # Key parameters from your grid
    parameters = ['gamma', 'lr', 'batch_size', 'junc_specific_prior']
    
    # Filter to parameters that exist and have variation
    available_params = []
    for param in parameters:
        if param in df.columns and df[param].nunique() > 1:
            available_params.append(param)
    
    if not available_params:
        print("Warning: No varying parameters found for analysis")
        return
    
    print(f"Analyzing parameters: {available_params}")
    
    # Create parameter effects plot
    n_params = len(available_params)
    n_metrics = len([m for m in metrics if m in df.columns])
    
    fig, axes = plt.subplots(n_metrics, n_params, figsize=(4*n_params, 4*n_metrics))
    if n_metrics == 1:
        axes = axes.reshape(1, -1)
    if n_params == 1:
        axes = axes.reshape(-1, 1)
    
    for i, metric in enumerate(metrics):
        if metric not in df.columns:
            continue
            
        for j, param in enumerate(available_params):
            ax = axes[i, j] if n_metrics > 1 else axes[j]
            
            # Handle different parameter types
            if df[param].dtype in ['object', 'bool']:
                # Categorical parameter
                sns.boxplot(data=df, x=param, y=metric, ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            else:
                # Numerical parameter
                if df[param].nunique() <= 10:
                    # Few unique values - treat as categorical
                    sns.boxplot(data=df, x=param, y=metric, ax=ax)
                else:
                    # Many values - treat as continuous
                    sns.scatterplot(data=df, x=param, y=metric, ax=ax)
            
            ax.set_title(f'{metric} vs {param}')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_effects.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate correlation matrix for numerical parameters
    numerical_params = [p for p in available_params if df[p].dtype in ['int64', 'float64']]
    numerical_metrics = [m for m in metrics if m in df.columns and df[m].dtype in ['int64', 'float64']]
    
    if numerical_params and numerical_metrics:
        correlation_data = df[numerical_params + numerical_metrics].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_data, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.3f')
        plt.title('Parameter-Metric Correlation Matrix')
        plt.tight_layout()
        plt.savefig(output_dir / 'parameter_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

def analyze_psi_correlation_effects(df, output_dir):
    """Analyze PSI correlation vs other metrics"""
    print("\nAnalyzing PSI correlation effects...")
    
    if 'psi_correlation' not in df.columns:
        print("Warning: PSI correlation data not available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # PSI correlation vs Age R²
    if 'age_r2_test' in df.columns:
        axes[0, 0].scatter(df['psi_correlation'], df['age_r2_test'], alpha=0.7)
        if df['psi_correlation'].notna().sum() > 2:
            corr, p = pearsonr(df['psi_correlation'].dropna(), 
                             df['age_r2_test'][df['psi_correlation'].notna()])
            axes[0, 0].set_title(f'PSI Correlation vs Age R² (r={corr:.3f}, p={p:.3f})')
        axes[0, 0].set_xlabel('PSI Correlation')
        axes[0, 0].set_ylabel('Age R² (test)')
        axes[0, 0].grid(True, alpha=0.3)
    
    # PSI correlation vs Perplexity
    if 'mean_perplexity' in df.columns:
        axes[0, 1].scatter(df['psi_correlation'], df['mean_perplexity'], alpha=0.7)
        if df['psi_correlation'].notna().sum() > 2:
            corr, p = pearsonr(df['psi_correlation'].dropna(), 
                             df['mean_perplexity'][df['psi_correlation'].notna()])
            axes[0, 1].set_title(f'PSI Correlation vs Perplexity (r={corr:.3f}, p={p:.3f})')
        axes[0, 1].set_xlabel('PSI Correlation')
        axes[0, 1].set_ylabel('Mean Perplexity')
        axes[0, 1].grid(True, alpha=0.3)
    
    # PSI correlation distribution
    axes[1, 0].hist(df['psi_correlation'].dropna(), bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(df['psi_correlation'].median(), color='red', linestyle='--', 
                      label=f'Median: {df["psi_correlation"].median():.3f}')
    axes[1, 0].set_xlabel('PSI Correlation')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('PSI Correlation Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Batch effects vs PSI correlation
    if 'batch_r2' in df.columns:
        axes[1, 1].scatter(df['psi_correlation'], df['batch_r2'], alpha=0.7)
        if df['psi_correlation'].notna().sum() > 2:
            corr, p = pearsonr(df['psi_correlation'].dropna(), 
                             df['batch_r2'][df['psi_correlation'].notna()])
            axes[1, 1].set_title(f'PSI Correlation vs Batch Effects (r={corr:.3f}, p={p:.3f})')
        axes[1, 1].set_xlabel('PSI Correlation')
        axes[1, 1].set_ylabel('Batch R²')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'psi_correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_factor_effects(df, output_dir):
    """Analyze factor-related metrics"""
    print("\nAnalyzing factor effects...")
    
    factor_metrics = ['K', 'n_effective', 'mean_perplexity', 'median_perplexity']
    available_metrics = [m for m in factor_metrics if m in df.columns]
    
    if len(available_metrics) < 2:
        print("Warning: Insufficient factor metrics for analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # K vs effective factors
    if 'K' in df.columns and 'n_effective' in df.columns:
        axes[0, 0].scatter(df['K'], df['n_effective'], alpha=0.7)
        axes[0, 0].plot([df['K'].min(), df['K'].max()], 
                       [df['K'].min(), df['K'].max()], 'r--', alpha=0.5)
        axes[0, 0].set_xlabel('Original K')
        axes[0, 0].set_ylabel('Effective Factors')
        axes[0, 0].set_title('Factor Usage Efficiency')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Effective factors vs Age R²
    if 'n_effective' in df.columns and 'age_r2_test' in df.columns:
        axes[0, 1].scatter(df['n_effective'], df['age_r2_test'], alpha=0.7)
        if df['n_effective'].notna().sum() > 2:
            corr, p = pearsonr(df['n_effective'].dropna(), 
                             df['age_r2_test'][df['n_effective'].notna()])
            axes[0, 1].set_title(f'Effective Factors vs Age R² (r={corr:.3f}, p={p:.3f})')
        axes[0, 1].set_xlabel('Effective Factors')
        axes[0, 1].set_ylabel('Age R² (test)')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Perplexity distribution
    if 'mean_perplexity' in df.columns:
        axes[1, 0].hist(df['mean_perplexity'].dropna(), bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(df['mean_perplexity'].median(), color='red', linestyle='--',
                          label=f'Median: {df["mean_perplexity"].median():.2f}')
        axes[1, 0].set_xlabel('Mean Perplexity')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Cell Perplexity Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Perplexity vs Age R²
    if 'mean_perplexity' in df.columns and 'age_r2_test' in df.columns:
        axes[1, 1].scatter(df['mean_perplexity'], df['age_r2_test'], alpha=0.7)
        if df['mean_perplexity'].notna().sum() > 2:
            corr, p = pearsonr(df['mean_perplexity'].dropna(), 
                             df['age_r2_test'][df['mean_perplexity'].notna()])
            axes[1, 1].set_title(f'Perplexity vs Age R² (r={corr:.3f}, p={p:.3f})')
        axes[1, 1].set_xlabel('Mean Perplexity')
        axes[1, 1].set_ylabel('Age R² (test)')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'factor_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_statistics(df, output_dir):
    """Generate summary statistics for parameters and metrics"""
    print("\nGenerating summary statistics...")
    
    # Parameter summary
    param_cols = ['gamma', 'lr', 'batch_size', 'junc_specific_prior', 'K']
    available_params = [col for col in param_cols if col in df.columns]
    
    if available_params:
        param_summary = []
        
        for param in available_params:
            if df[param].dtype in ['object', 'bool']:
                # Categorical parameter
                value_counts = df[param].value_counts()
                for value, count in value_counts.items():
                    param_summary.append({
                        'parameter': param,
                        'value': str(value),
                        'count': count,
                        'percentage': count / len(df) * 100
                    })
            else:
                # Numerical parameter
                param_summary.append({
                    'parameter': param,
                    'value': 'mean',
                    'count': df[param].mean(),
                    'percentage': np.nan
                })
                param_summary.append({
                    'parameter': param,
                    'value': 'std',
                    'count': df[param].std(),
                    'percentage': np.nan
                })
        
        param_df = pd.DataFrame(param_summary)
        param_df.to_csv(output_dir / 'parameter_summary.csv', index=False)
    
    # Metric summary
    metric_cols = ['age_r2_test', 'mean_perplexity', 'psi_correlation', 'batch_r2', 'n_effective']
    available_metrics = [col for col in metric_cols if col in df.columns]
    
    if available_metrics:
        metric_summary = df[available_metrics].describe()
        metric_summary.to_csv(output_dir / 'metric_summary.csv')
    
    # Best performing models
    if 'age_r2_test' in df.columns:
        best_models = df.nlargest(5, 'age_r2_test')[['run_id'] + available_params + available_metrics]
        best_models.to_csv(output_dir / 'best_models_age_r2.csv', index=False)
    
    # Save all results
    df.to_csv(output_dir / 'simple_results.csv', index=False)

def print_key_findings(df):
    """Print key findings from the analysis"""
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    print(f"\nDataset Overview:")
    print(f"  Total models analyzed: {len(df)}")
    
    # Parameter ranges
    param_cols = ['gamma', 'lr', 'batch_size', 'junc_specific_prior', 'K']
    available_params = [col for col in param_cols if col in df.columns]
    
    print(f"\nParameter Ranges:")
    for param in available_params:
        if df[param].dtype in ['object', 'bool']:
            unique_vals = df[param].value_counts()
            print(f"  {param}: {dict(unique_vals)}")
        else:
            print(f"  {param}: {df[param].min():.4f} - {df[param].max():.4f}")
    
    # Performance metrics
    if 'age_r2_test' in df.columns:
        print(f"\nAge Regression Performance:")
        print(f"  Best Age R²: {df['age_r2_test'].max():.4f}")
        print(f"  Mean Age R²: {df['age_r2_test'].mean():.4f}")
        print(f"  Std Age R²: {df['age_r2_test'].std():.4f}")
        
        # Best model
        best_idx = df['age_r2_test'].idxmax()
        print(f"\nBest Model (Run {df.loc[best_idx, 'run_id']}):")
        for param in available_params:
            if param in df.columns:
                print(f"  {param}: {df.loc[best_idx, param]}")
    
    if 'psi_correlation' in df.columns:
        print(f"\nPSI Correlation:")
        print(f"  Mean: {df['psi_correlation'].mean():.4f}")
        print(f"  Range: {df['psi_correlation'].min():.4f} - {df['psi_correlation'].max():.4f}")
    
    # Parameter correlations with performance
    if 'age_r2_test' in df.columns:
        print(f"\nParameter Correlations with Age R²:")
        for param in available_params:
            if param in df.columns and df[param].dtype in ['int64', 'float64']:
                if df[param].nunique() > 1:
                    corr, p = pearsonr(df[param].dropna(), 
                                     df['age_r2_test'][df[param].notna()])
                    print(f"  {param}: r={corr:.4f}, p={p:.4f}")

def main():
    """Main analysis pipeline"""
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_parameters.py <model_output_directory>")
        sys.exit(1)
    
    model_dir = Path(sys.argv[1])
    output_dir = model_dir / "parameter_analysis"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Analyzing results from: {model_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load results
    df = load_results(model_dir)
    
    if len(df) == 0:
        print("Error: No valid results found!")
        sys.exit(1)
    
    print(f"Loaded {len(df)} model results")
    print(f"Available columns: {list(df.columns)}")
    
    # Run analyses
    analyze_parameter_effects(df, output_dir)
    analyze_psi_correlation_effects(df, output_dir)
    analyze_factor_effects(df, output_dir)
    generate_summary_statistics(df, output_dir)
    
    # Print findings
    print_key_findings(df)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()

#script=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/full_workflow/05_review_results.py
#model_dir=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel/2025-09-07
#python $script $model_dir
