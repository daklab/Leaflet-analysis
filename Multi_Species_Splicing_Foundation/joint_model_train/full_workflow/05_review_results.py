#!/usr/bin/env python
"""
Simple analysis: How do gamma and lr affect r2 and perplexity?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

def load_results(base_dir):
    """Load all results and parameters"""
    results = []
    
    # Load each analysis result
    for analysis_dir in Path(base_dir).glob("analysis_run_*"):
        run_id = int(analysis_dir.name.split("_")[-1])
        
        # Load metrics
        metrics_file = analysis_dir / "analysis_metrics.csv"
        if metrics_file.exists():
            try:
                metrics = pd.read_csv(metrics_file).iloc[0].to_dict()
                metrics['run_id'] = run_id
                
                # Load age regression results
                age_file = analysis_dir / "age_regression_global.csv"
                if age_file.exists():
                    age_df = pd.read_csv(age_file)
                    if not age_df.empty:
                        metrics['r2_test'] = age_df['r2_test'].iloc[0]
                        metrics['r2_train'] = age_df['r2_train'].iloc[0]
                
                results.append(metrics)
            except:
                continue
    
    df = pd.DataFrame(results)
    
    # Load parameters
    param_file = Path(base_dir) / "parameter_combinations.csv"
    if param_file.exists():
        params = pd.read_csv(param_file)
        params['run_id'] = params.index
        df = pd.merge(df, params[['run_id', 'gamma', 'lr', 'K']], on='run_id', how='left')
    
    return df

def plot_parameter_effects(df):
    """Create simple plots showing parameter effects"""
    
    # Setup figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Color palettes
    gamma_colors = {0.001: 'blue', 0.01: 'green', 0.1: 'red'}
    lr_colors = {0.2: 'purple', 0.5: 'orange', 0.8: 'brown'}
    
    # 1. Gamma vs R2
    ax = axes[0, 0]
    for gamma in sorted(df['gamma'].unique()):
        data = df[df['gamma'] == gamma]['r2_test']
        ax.scatter([gamma]*len(data), data, alpha=0.6, s=80, 
                  color=gamma_colors.get(gamma, 'gray'), label=f'γ={gamma}')
    
    # Add means
    means = df.groupby('gamma')['r2_test'].mean()
    ax.plot(means.index, means.values, 'k--', linewidth=2, label='Mean')
    
    ax.set_xscale('log')
    ax.set_xlabel('Gamma (log scale)')
    ax.set_ylabel('Age R² (test)')
    ax.set_title('How Gamma Affects Age Prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. LR vs R2
    ax = axes[0, 1]
    for lr in sorted(df['lr'].unique()):
        data = df[df['lr'] == lr]['r2_test']
        ax.scatter([lr]*len(data), data, alpha=0.6, s=80,
                  color=lr_colors.get(lr, 'gray'), label=f'lr={lr}')
    
    # Add means
    means = df.groupby('lr')['r2_test'].mean()
    ax.plot(means.index, means.values, 'k--', linewidth=2, label='Mean')
    
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Age R² (test)')
    ax.set_title('How Learning Rate Affects Age Prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Interaction: Gamma x LR on R2
    ax = axes[0, 2]
    pivot = df.pivot_table(values='r2_test', index='gamma', columns='lr', aggfunc='mean')
    im = ax.imshow(pivot, cmap='viridis', aspect='auto')
    
    # Add text annotations
    for i, gamma in enumerate(pivot.index):
        for j, lr in enumerate(pivot.columns):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', color='white')
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{x:.1f}' for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{x:.3f}' for x in pivot.index])
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Gamma')
    ax.set_title('Mean R² by Gamma × LR')
    plt.colorbar(im, ax=ax)
    
    # 4. Gamma vs Perplexity
    ax = axes[1, 0]
    for gamma in sorted(df['gamma'].unique()):
        data = df[df['gamma'] == gamma]['median_perplexity']
        ax.scatter([gamma]*len(data), data, alpha=0.6, s=80,
                  color=gamma_colors.get(gamma, 'gray'), label=f'γ={gamma}')
    
    # Add means
    means = df.groupby('gamma')['median_perplexity'].mean()
    ax.plot(means.index, means.values, 'k--', linewidth=2, label='Mean')
    
    ax.set_xscale('log')
    ax.set_xlabel('Gamma (log scale)')
    ax.set_ylabel('Median Perplexity')
    ax.set_title('How Gamma Affects Cell Perplexity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. LR vs Perplexity
    ax = axes[1, 1]
    for lr in sorted(df['lr'].unique()):
        data = df[df['lr'] == lr]['median_perplexity']
        ax.scatter([lr]*len(data), data, alpha=0.6, s=80,
                  color=lr_colors.get(lr, 'gray'), label=f'lr={lr}')
    
    # Add means
    means = df.groupby('lr')['median_perplexity'].mean()
    ax.plot(means.index, means.values, 'k--', linewidth=2, label='Mean')
    
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Median Perplexity')
    ax.set_title('How Learning Rate Affects Cell Perplexity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Interaction: Gamma x LR on Perplexity
    ax = axes[1, 2]
    pivot = df.pivot_table(values='median_perplexity', index='gamma', columns='lr', aggfunc='mean')
    im = ax.imshow(pivot, cmap='plasma', aspect='auto')
    
    # Add text annotations
    for i, gamma in enumerate(pivot.index):
        for j, lr in enumerate(pivot.columns):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', color='white')
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{x:.1f}' for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{x:.3f}' for x in pivot.index])
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Gamma')
    ax.set_title('Mean Perplexity by Gamma × LR')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle('Parameter Effects on Key Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

    
def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_analysis.py <model_dir>")
        sys.exit(1)
    
    base_dir = Path(sys.argv[1])
    
    print("Loading data...")
    df = load_results(base_dir)
    
    if df.empty or 'r2_test' not in df.columns:
        print("No valid results found!")
        sys.exit(1)
    
    print(f"Loaded {len(df)} runs")
    
    # Create plots
    fig = plot_parameter_effects(df)
    output_file = base_dir / "parameter_effects.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to: {output_file}")
    
    # Save data for further analysis
    df.to_csv(base_dir / "simple_results.csv", index=False)
    print(f"\nData saved to: {base_dir}/simple_results.csv")

if __name__ == "__main__":
    main()

#script=/gpfs/commons/home/kisaev/Leaflet-analysis/Multi_Species_Splicing_Foundation/joint_model_train/full_workflow/05_review_results.py
#model_dir=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/CROSS_SPECIES_AGING/Leaflet/model_combo_train/2025-09-04
#python $script $model_dir
