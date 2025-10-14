#!/usr/bin/env python
"""
Comprehensive LeafletFA Model Comparison and Best Model Nomination

This script analyzes all trained models and compares how different hyperparameters
affect key performance metrics to nominate the best models.

Key metrics analyzed:
- ELBO (evidence lower bound) - model fit quality (lower is better)
- PSI correlation - how well imputed PSI matches observed (higher is better)
- Final K (number of factors) - model complexity
- Batch R² - batch effect strength (lower is better)
- Perplexity - cell representation quality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import seaborn for better plots, fall back to matplotlib if not available
try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_theme(style="whitegrid")
except ImportError:
    HAS_SEABORN = False
    plt.style.use('default')

# Configure plotting
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

def load_all_results(base_dir):
    """
    Load all analysis results and parameters into a comprehensive dataframe
    """
    print("Loading all model results...")
    results = []
    
    # Load each analysis result
    for analysis_dir in Path(base_dir).glob("analysis_run_*"):
        
        run_id = int(analysis_dir.name.split("_")[-1])
        print(f"Loading results for {analysis_dir}")
        
        # Load metrics
        metrics_file = analysis_dir / "analysis_metrics.csv"
        if metrics_file.exists():
                metrics = pd.read_csv(metrics_file).iloc[0].to_dict()
                metrics['run_id'] = run_id
                metrics['analysis_dir'] = str(analysis_dir)
                
                # Load age regression results if available
                age_file = analysis_dir / "age_regression_global.csv"
                if age_file.exists():
                    age_df = pd.read_csv(age_file)
                    if not age_df.empty:
                        metrics['r2_test'] = age_df['r2_test'].iloc[0]
                        metrics['r2_train'] = age_df['r2_train'].iloc[0]
                
                results.append(metrics)
    
    if not results:
        raise ValueError("No valid results found!")
    
    df = pd.DataFrame(results)
    
    # Load parameters
    param_file = Path(base_dir) / "parameter_combinations.csv"
    if param_file.exists():
        params = pd.read_csv(param_file)
        params['run_id'] = params.index
        
        # Merge with results
        df = pd.merge(df, params, on='run_id', how='left', suffixes=('', '_param'))
        
        # Remove duplicate columns (prefer from analysis results)
        for col in params.columns:
            if col + '_param' in df.columns and col in df.columns and col != 'run_id':
                df = df.drop(columns=[col + '_param'])
    
    print(f"Loaded {len(df)} model results")
    return df

def clean_and_validate_data(df):
    """
    Clean data and validate key metrics are present
    """
    print("Cleaning and validating data...")
    
    # Define expected columns and their importance based on actual output
    key_metrics = {
        'best_elbo': 'ELBO',
        'psi_correlation': 'PSI Correlation', 
        'K': 'Final K',
        'pruned_K': 'Pruned K',
        'original_K': 'Original K',
        'batch_r2': 'Batch R²',
        'median_perplexity': 'Median Perplexity',
        'mean_perplexity': 'Mean Perplexity',
        'n_effective': 'Effective Factors',
        'total_batches': 'Total Batches'
    }
    
    key_params = {
        'lr': 'Learning Rate',
        'gamma': 'Gamma',
        'batch_size': 'Batch Size',
        'initial_K': 'Initial K',
        'num_passes': 'Number of Passes',
        'num_epochs_first': 'Epochs (First Pass)',
        'num_epochs_later': 'Epochs (Later Passes)',
        'ELBO_num_particles': 'ELBO Particles',
        'junc_specific_prior': 'Junction-Specific Prior',
        'waypoints_use': 'Waypoints Used',
        'max_junctions': 'Max Junctions',
        'n_waypoints': 'Number of Waypoints'
    }
    
    # Check which metrics are available
    available_metrics = {}
    for col, name in key_metrics.items():
        if col in df.columns:
            valid_count = df[col].notna().sum()
            available_metrics[col] = f"{name} ({valid_count}/{len(df)} valid)"
            print(f"  ✓ {name}: {valid_count}/{len(df)} valid values")
        else:
            print(f"  ⚠ {name}: Column '{col}' not found")
    
    # Check parameters
    available_params = {}
    for col, name in key_params.items():
        if col in df.columns:
            unique_count = df[col].nunique()
            available_params[col] = f"{name} ({unique_count} unique values)"
            print(f"  ✓ {name}: {unique_count} unique values")
        else:
            print(f"  ⚠ {name}: Column '{col}' not found")
    
    # Convert numeric columns
    numeric_cols = ['best_elbo', 'psi_correlation', 'K', 'pruned_K', 'original_K', 'batch_r2', 
                   'median_perplexity', 'mean_perplexity', 'n_effective', 'total_batches',
                   'lr', 'gamma', 'batch_size', 'initial_K', 'num_passes', 'num_epochs_first', 
                   'num_epochs_later', 'ELBO_num_particles', 'max_junctions', 'n_waypoints']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle psi_correlations list column if present
    if 'psi_correlations' in df.columns:
        def parse_correlation_list(corr_str):
            if pd.isna(corr_str) or corr_str == '':
                return []
            try:
                # Remove brackets and split by comma
                corr_str = str(corr_str).strip('[]')
                return [float(x.strip()) for x in corr_str.split(',') if x.strip()]
            except:
                return []
        
        df['psi_correlations_parsed'] = df['psi_correlations'].apply(parse_correlation_list)
        # Calculate additional statistics from correlation lists
        df['psi_correlation_std'] = df['psi_correlations_parsed'].apply(lambda x: np.std(x) if len(x) > 1 else np.nan)
        df['psi_correlation_min'] = df['psi_correlations_parsed'].apply(lambda x: np.min(x) if len(x) > 0 else np.nan)
        df['psi_correlation_max'] = df['psi_correlations_parsed'].apply(lambda x: np.max(x) if len(x) > 0 else np.nan)
    
    # Remove rows with all NaN key metrics
    before_filter = len(df)
    key_available = [col for col in key_metrics.keys() if col in df.columns]
    df = df.dropna(subset=key_available, how='all')
    after_filter = len(df)
    
    if after_filter < before_filter:
        print(f"Filtered out {before_filter - after_filter} rows with missing key metrics")
    
    return df, available_metrics, available_params

def calculate_model_scores(df):
    """
    Calculate composite scores for model ranking
    """
    print("Calculating model scores...")
    
    # Define scoring criteria (higher is better for all)
    scoring_metrics = {}
    
    # ELBO - lower is better (invert for scoring)
    if 'best_elbo' in df.columns and df['best_elbo'].notna().sum() > 0:
        scoring_metrics['elbo_score'] = -df['best_elbo']  # Invert so lower ELBO = higher score
    
    # PSI Correlation - higher is better (already positive direction)
    if 'psi_correlation' in df.columns and df['psi_correlation'].notna().sum() > 0:
        # Base PSI correlation score
        psi_base_score = df['psi_correlation']
        
        # Stability bonus: penalize high variance in PSI correlations
        if 'psi_correlation_std' in df.columns and df['psi_correlation_std'].notna().sum() > 0:
            # Lower std is better - add stability bonus
            max_std = df['psi_correlation_std'].max()
            stability_bonus = 1 - (df['psi_correlation_std'] / max_std)
            scoring_metrics['psi_score'] = 0.8 * psi_base_score + 0.2 * stability_bonus
        else:
            scoring_metrics['psi_score'] = psi_base_score
    
    # Batch R² - lower is better (invert)
    if 'batch_r2' in df.columns and df['batch_r2'].notna().sum() > 0:
        scoring_metrics['batch_score'] = 1 - df['batch_r2']  # Invert so lower batch effects = higher score
    
    # Model complexity penalty - prefer models that aren't too complex or too simple
    if 'K' in df.columns and df['K'].notna().sum() > 0:
        # Penalize very high or very low K values relative to initial K
        if 'initial_K' in df.columns and df['initial_K'].notna().sum() > 0:
            # Prefer models that maintain reasonable factor count relative to initial
            k_retention = df['K'] / df['initial_K']
            # Ideal retention between 0.5 and 1.0 (some pruning is good, too much is bad)
            complexity_values = np.where(
                k_retention <= 1.0, 
                k_retention,  # Reward retention up to 1.0
                1.0 / k_retention  # Penalize expansion
            )
            # Convert back to pandas Series to maintain compatibility
            scoring_metrics['complexity_score'] = pd.Series(complexity_values, index=df.index)
        else:
            # Fallback to median-based scoring
            median_K = df['K'].median()
            scoring_metrics['complexity_score'] = 1 - np.abs(df['K'] - median_K) / median_K
    
    # Perplexity - moderate values preferred (not too low, not too high)
    if 'median_perplexity' in df.columns and df['median_perplexity'].notna().sum() > 0:
        # Target perplexity around 5-20 (reasonable for biological data)
        target_perplexity = 10
        perp_penalty = np.abs(df['median_perplexity'] - target_perplexity) / target_perplexity
        scoring_metrics['perplexity_score'] = 1 / (1 + perp_penalty)
    
    # Normalize all scores to 0-1
    normalized_scores = {}
    for score_name, values in scoring_metrics.items():
        if values.notna().sum() > 0:
            normalized = (values - values.min()) / (values.max() - values.min())
            normalized_scores[score_name] = normalized
            df[score_name] = normalized
    
    # Calculate composite score (weighted average)
    weights = {
        'elbo_score': 0.3,      # Most important - model fit (inverted: lower ELBO = higher score)
        'psi_score': 0.3,       # Critical - prediction accuracy  
        'batch_score': 0.2,     # Important - no batch effects (inverted: lower R² = higher score)
        'complexity_score': 0.1, # Moderate importance
        'perplexity_score': 0.1  # Moderate importance
    }
    
    # Only use available scores
    available_scores = [score for score in weights.keys() if score in df.columns]
    
    if available_scores:
        # Renormalize weights for available scores
        total_weight = sum(weights[score] for score in available_scores)
        normalized_weights = {score: weights[score]/total_weight for score in available_scores}
        
        df['composite_score'] = 0
        for score, weight in normalized_weights.items():
            df['composite_score'] += weight * df[score].fillna(0)
        
        print(f"Composite score calculated using: {', '.join(available_scores)}")
        print(f"Weights: {normalized_weights}")
    else:
        print("Warning: No metrics available for composite scoring")
        df['composite_score'] = 0
    
    return df

def plot_parameter_effects(df, output_dir):
    """
    Create comprehensive parameter effect visualizations
    """
    print("Creating parameter effect plots...")
    
    # Define key parameters and metrics to plot (updated for actual available columns)
    params = ['lr', 'gamma', 'batch_size', 'initial_K', 'num_passes', 'num_epochs_first', 
              'num_epochs_later', 'ELBO_num_particles', 'max_junctions', 'n_waypoints']
    metrics = ['best_elbo', 'psi_correlation', 'K', 'batch_r2', 'median_perplexity', 
               'n_effective', 'pruned_K', 'original_K']
    
    # Filter to available columns
    available_params = [p for p in params if p in df.columns and df[p].notna().sum() > 1]
    available_metrics = [m for m in metrics if m in df.columns and df[m].notna().sum() > 1]
    
    if not available_params or not available_metrics:
        print("Warning: Insufficient data for parameter effect plots")
        return
    
    # 1. Parameter vs Metric Correlation Heatmap
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Correlation matrix
    plot_df = df[available_params + available_metrics].select_dtypes(include=[np.number])
    corr_matrix = plot_df.corr()
    
    # Plot correlation heatmap
    if HAS_SEABORN:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    square=True, linewidths=0.5, ax=axes[0,0])
    else:
        # Fallback to matplotlib
        im = axes[0,0].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0,0].set_xticks(range(len(corr_matrix.columns)))
        axes[0,0].set_yticks(range(len(corr_matrix.columns)))
        axes[0,0].set_xticklabels(corr_matrix.columns, rotation=45)
        axes[0,0].set_yticklabels(corr_matrix.columns)
        plt.colorbar(im, ax=axes[0,0])
    axes[0,0].set_title('Parameter-Metric Correlations')
    
    # 2. Best models by composite score
    if 'composite_score' in df.columns:
        top_models = df.nlargest(10, 'composite_score')
        
        # Plot top model scores
        bars = axes[0,1].bar(range(len(top_models)), top_models['composite_score'])
        axes[0,1].set_xlabel('Model Rank')
        axes[0,1].set_ylabel('Composite Score')
        axes[0,1].set_title('Top 10 Models by Composite Score')
        axes[0,1].set_xticks(range(len(top_models)))
        axes[0,1].set_xticklabels([f"Run {int(run_id)}" for run_id in top_models['run_id']], rotation=45)
    
    # 3. Parameter distribution for top models
    if 'composite_score' in df.columns and len(available_params) > 0:
        top_10_pct = df['composite_score'].quantile(0.9)
        top_models = df[df['composite_score'] >= top_10_pct]
        
        if len(top_models) > 1:
            # Box plot of parameter values for top models
            param_data = []
            for param in available_params[:4]:  # Show top 4 parameters
                values = top_models[param].dropna()
                if len(values) > 0:
                    param_data.extend([(param, val) for val in values])
            
            if param_data:
                param_df = pd.DataFrame(param_data, columns=['Parameter', 'Value'])
                if HAS_SEABORN:
                    sns.boxplot(data=param_df, x='Parameter', y='Value', ax=axes[1,0])
                else:
                    # Fallback to matplotlib boxplot
                    param_groups = [param_df[param_df['Parameter'] == p]['Value'].values 
                                  for p in param_df['Parameter'].unique()]
                    axes[1,0].boxplot(param_groups, labels=param_df['Parameter'].unique())
                axes[1,0].set_title('Parameter Distributions in Top 10% Models')
                axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Metric distributions
    if len(available_metrics) > 0:
        metric_data = []
        for metric in available_metrics[:4]:  # Show top 4 metrics
            values = df[metric].dropna()
            if len(values) > 0:
                metric_data.extend([(metric, val) for val in values])
        
        if metric_data:
            metric_df = pd.DataFrame(metric_data, columns=['Metric', 'Value'])
            if HAS_SEABORN:
                sns.boxplot(data=metric_df, x='Metric', y='Value', ax=axes[1,1])
            else:
                # Fallback to matplotlib boxplot
                metric_groups = [metric_df[metric_df['Metric'] == m]['Value'].values 
                               for m in metric_df['Metric'].unique()]
                axes[1,1].boxplot(metric_groups, labels=metric_df['Metric'].unique())
            axes[1,1].set_title('Overall Metric Distributions')
            axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_effects_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Detailed parameter sweep plots
    if len(available_params) >= 2:
        n_params = len(available_params)
        n_metrics = len(available_metrics)
        
        fig, axes = plt.subplots(n_metrics, n_params, figsize=(4*n_params, 3*n_metrics))
        if n_metrics == 1:
            axes = axes.reshape(1, -1)
        if n_params == 1:
            axes = axes.reshape(-1, 1)
        
        for i, metric in enumerate(available_metrics):
            for j, param in enumerate(available_params):
                ax = axes[i, j] if n_metrics > 1 else axes[j]
                
                # Create scatter plot with trend line
                valid_data = df[[param, metric]].dropna()
                if len(valid_data) > 2:
                    x = valid_data[param]
                    y = valid_data[metric]
                    
                    ax.scatter(x, y, alpha=0.6, s=30)
                    
                    # Add trend line
                    if len(x.unique()) > 1:
                        z = np.polyfit(x, y, 1)
                        p = np.poly1d(z)
                        ax.plot(x, p(x), "r--", alpha=0.8)
                        
                        # Add correlation coefficient
                        corr, p_val = stats.pearsonr(x, y)
                        ax.text(0.05, 0.95, f'r={corr:.3f}', transform=ax.transAxes,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                
                ax.set_xlabel(param)
                ax.set_ylabel(metric)
                ax.set_title(f'{metric} vs {param}')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'parameter_metric_relationships.png', dpi=300, bbox_inches='tight')
        plt.close()

def nominate_best_models(df, output_dir, top_n=5):
    """
    Nominate the best models based on different criteria
    """
    print(f"\nNominating top {top_n} models...")
    
    nominations = {}
    
    # 1. Best by composite score
    if 'composite_score' in df.columns:
        best_composite = df.nlargest(top_n, 'composite_score')
        nominations['composite_score'] = best_composite
        print(f"\nBest by Composite Score:")
        for i, (_, row) in enumerate(best_composite.iterrows(), 1):
            print(f"  {i}. Run {int(row['run_id'])}: Score={row['composite_score']:.3f}")
            print(f"     lr={row.get('lr', 'N/A')}, gamma={row.get('gamma', 'N/A')}, "
                  f"batch_size={row.get('batch_size', 'N/A')}")
    
    # 2. Best by individual metrics
    individual_metrics = {
        'best_elbo': 'Best ELBO (Lowest)',
        'psi_correlation': 'Best PSI Correlation', 
        'batch_r2': 'Lowest Batch Effects'
    }
    
    for metric, name in individual_metrics.items():
        if metric in df.columns and df[metric].notna().sum() > 0:
            if metric in ['batch_r2', 'best_elbo']:
                # Lower is better for batch R² and ELBO
                best_metric = df.nsmallest(top_n, metric)
            else:
                # Higher is better for others
                best_metric = df.nlargest(top_n, metric)
            
            nominations[metric] = best_metric
            print(f"\n{name}:")
            for i, (_, row) in enumerate(best_metric.iterrows(), 1):
                if metric == 'best_elbo':
                    print(f"  {i}. Run {int(row['run_id'])}: {metric}={row[metric]:.2e}")
                else:
                    print(f"  {i}. Run {int(row['run_id'])}: {metric}={row[metric]:.3f}")
    
    # 3. Find consensus models (appearing in multiple top lists)
    all_top_runs = set()
    for criterion, models in nominations.items():
        all_top_runs.update(models['run_id'])
    
    consensus_scores = {}
    for run_id in all_top_runs:
        score = 0
        appearances = []
        for criterion, models in nominations.items():
            if run_id in models['run_id'].values:
                rank = list(models['run_id']).index(run_id) + 1
                score += (top_n + 1 - rank)  # Higher score for better rank
                appearances.append(f"{criterion}(#{rank})")
        consensus_scores[run_id] = (score, appearances)
    
    # Sort by consensus score
    consensus_ranking = sorted(consensus_scores.items(), key=lambda x: x[1][0], reverse=True)
    
    print(f"\nConsensus Ranking (appearing in multiple top lists):")
    consensus_models = []
    for i, (run_id, (score, appearances)) in enumerate(consensus_ranking[:top_n], 1):
        model_row = df[df['run_id'] == run_id].iloc[0]
        consensus_models.append(model_row)
        print(f"  {i}. Run {int(run_id)}: Consensus Score={score}")
        print(f"     Appears in: {', '.join(appearances)}")
        print(f"     lr={model_row.get('lr', 'N/A')}, gamma={model_row.get('gamma', 'N/A')}, "
              f"batch_size={model_row.get('batch_size', 'N/A')}")
    
    # Save nominations
    nominations_summary = []
    for criterion, models in nominations.items():
        for rank, (_, row) in enumerate(models.iterrows(), 1):
            nominations_summary.append({
                'criterion': criterion,
                'rank': rank,
                'run_id': int(row['run_id']),
                'metric_value': row.get(criterion, np.nan),
                'composite_score': row.get('composite_score', np.nan),
                'lr': row.get('lr', np.nan),
                'gamma': row.get('gamma', np.nan),
                'batch_size': row.get('batch_size', np.nan),
                'analysis_dir': row.get('analysis_dir', '')
            })
    
    nominations_df = pd.DataFrame(nominations_summary)
    nominations_df.to_csv(output_dir / 'model_nominations.csv', index=False)
    
    # Save consensus models
    if consensus_models:
        consensus_df = pd.DataFrame(consensus_models)
        consensus_df.to_csv(output_dir / 'consensus_best_models.csv', index=False)
    
    return nominations

def generate_summary_report(df, nominations, output_dir):
    """
    Generate a comprehensive summary report
    """
    report = []
    report.append("="*80)
    report.append("LEAFLETFA MODEL COMPARISON SUMMARY")
    report.append("="*80)
    
    # Dataset overview
    report.append(f"\nDATASET OVERVIEW:")
    report.append(f"Total models analyzed: {len(df)}")
    report.append(f"Successful models: {df['run_id'].notna().sum()}")
    
    # Parameter ranges
    param_cols = ['lr', 'gamma', 'batch_size', 'initial_K', 'num_passes', 'num_epochs_first', 
                  'num_epochs_later', 'ELBO_num_particles', 'max_junctions', 'n_waypoints']
    available_params = [col for col in param_cols if col in df.columns]
    
    if available_params:
        report.append(f"\nPARAMETER RANGES:")
        for param in available_params:
            if df[param].notna().sum() > 0:
                min_val = df[param].min()
                max_val = df[param].max()
                unique_count = df[param].nunique()
                report.append(f"{param}: {min_val} - {max_val} ({unique_count} unique values)")
    
    # Metric statistics
    metric_cols = ['best_elbo', 'psi_correlation', 'K', 'pruned_K', 'original_K', 'batch_r2', 
                   'median_perplexity', 'mean_perplexity', 'n_effective', 'total_batches']
    available_metrics = [col for col in metric_cols if col in df.columns]
    
    if available_metrics:
        report.append(f"\nMETRIC STATISTICS:")
        for metric in available_metrics:
            if df[metric].notna().sum() > 0:
                mean_val = df[metric].mean()
                std_val = df[metric].std()
                min_val = df[metric].min()
                max_val = df[metric].max()
                report.append(f"{metric}: {mean_val:.3f} ± {std_val:.3f} (range: {min_val:.3f} - {max_val:.3f})")
    
    # Best model summary
    if 'composite_score' in df.columns:
        best_model = df.loc[df['composite_score'].idxmax()]
        report.append(f"\nBEST MODEL (by composite score):")
        report.append(f"Run ID: {int(best_model['run_id'])}")
        report.append(f"Composite Score: {best_model['composite_score']:.3f}")
        if 'lr' in best_model and pd.notna(best_model['lr']):
            report.append(f"Learning Rate: {best_model['lr']}")
        if 'gamma' in best_model and pd.notna(best_model['gamma']):
            report.append(f"Gamma: {best_model['gamma']}")
        if 'batch_size' in best_model and pd.notna(best_model['batch_size']):
            report.append(f"Batch Size: {best_model['batch_size']}")
        if 'initial_K' in best_model and pd.notna(best_model['initial_K']):
            report.append(f"Initial K: {best_model['initial_K']}")
        if 'num_passes' in best_model and pd.notna(best_model['num_passes']):
            report.append(f"Number of Passes: {best_model['num_passes']}")
        
        for metric in available_metrics:
            if metric in best_model and pd.notna(best_model[metric]):
                if metric == 'best_elbo':
                    report.append(f"{metric}: {best_model[metric]:.2e}")
                else:
                    report.append(f"{metric}: {best_model[metric]:.3f}")
    
    # Key findings
    report.append(f"\nKEY FINDINGS:")
    
    # Parameter correlations with performance
    if 'composite_score' in df.columns and available_params:
        report.append("Parameter correlations with composite score:")
        for param in available_params:
            if df[param].notna().sum() > 2:
                corr, p_val = stats.pearsonr(df[param].dropna(), 
                                           df.loc[df[param].notna(), 'composite_score'])
                significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                report.append(f"  {param}: r={corr:.3f} {significance}")
    
    # Quality checks
    if 'psi_correlation' in df.columns:
        low_psi_count = (df['psi_correlation'] < 0.5).sum()
        if low_psi_count > 0:
            report.append(f"⚠️ WARNING: {low_psi_count} models have PSI correlation < 0.5")
    
    if 'batch_r2' in df.columns:
        high_batch_count = (df['batch_r2'] > 0.3).sum()
        if high_batch_count > 0:
            report.append(f"⚠️ WARNING: {high_batch_count} models have strong batch effects (R² > 0.3)")
    
    # Save report
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    with open(output_dir / 'comparison_summary.txt', 'w') as f:
        f.write(report_text)
    
def main():
    """Main analysis pipeline"""
    
    if len(sys.argv) < 2:
        print("Usage: python 05_review_results.py <model_dir> [output_dir]")
        print("\nThis script analyzes all LeafletFA model results and nominates the best models")
        print("based on comprehensive performance metrics.")
        sys.exit(1)
    
    base_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else base_dir / "model_comparison"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    print("\n" + "="*80)
    print("LEAFLETFA MODEL COMPARISON AND NOMINATION")
    print("="*80)
    
    try:
        # 1. Load all results
        df = load_all_results(base_dir)
        
        # 2. Clean and validate data
        df, available_metrics, available_params = clean_and_validate_data(df)
        
        if len(df) == 0:
            print("Error: No valid model results found!")
            sys.exit(1)
        
        # 3. Calculate model scores
        df = calculate_model_scores(df)
        
        # 4. Create visualizations
        plot_parameter_effects(df, output_dir)
        
        # 5. Nominate best models
        nominations = nominate_best_models(df, output_dir)
        
        # 6. Generate summary report
        generate_summary_report(df, nominations, output_dir)
        
        # 7. Save complete results
        df.to_csv(output_dir / 'all_model_results.csv', index=False)
        
        print(f"\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print(f"Results saved to: {output_dir}")
        print("="*80)
        
        # Print quick summary
        if 'composite_score' in df.columns:
            best_run = df.loc[df['composite_score'].idxmax(), 'run_id']
            print(f"\nRECOMMENDED MODEL: Run {int(best_run)}")
            print(f"Check detailed results in: {output_dir}")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
if __name__ == "__main__":
    main()

# to run:
# script=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/full_workflow/05_review_results.py
# model_dir=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel/2025-09-22
# model_output=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/output
# python $script $model_dir $model_output

# best_model=2
# scp -r $model_dir/analysis_run_$best_model $model_output