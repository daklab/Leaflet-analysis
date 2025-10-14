"""
ATSE Coordination Analysis: QC and Biological Assessment
Focus on ATSEs with 2-3 junctions for interpretability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# %%
# =============================================================================
# STEP 1: QUALITY CONTROL - CELLS FIRST, THEN ATSES
# =============================================================================

def perform_cell_qc(adata, min_junctions_per_cell=100, min_reads_per_cell=1000):
    """
    Quality control for cells:
    1. Check number of junctions detected per cell
    2. Check total read counts per cell
    3. Flag low-quality cells
    """
    
    print("="*60)
    print("STEP 1A: CELL QUALITY CONTROL")
    print("="*60)
    
    # Calculate cell-level metrics without converting to dense
    psi_matrix = adata.layers["psi"]
    
    # Count junctions detected per cell (non-zero PSI values)
    junctions_per_cell = (psi_matrix > 0).sum(axis=1).A1  # .A1 to get 1D array from sparse
    
    # Get total read counts per cell if available
    if 'total_counts' in adata.obs.columns:
        reads_per_cell = adata.obs['total_counts'].values
    else:
        # Estimate from junction coverage if read counts not available
        reads_per_cell = psi_matrix.sum(axis=1).A1 * 100  # Rough estimate
    
    # Apply QC filters
    cell_pass_qc = (
        (junctions_per_cell >= min_junctions_per_cell) & 
        (reads_per_cell >= min_reads_per_cell)
    )
    
    print(f"Cell QC Results:")
    print(f"  Total cells: {len(cell_pass_qc):,}")
    print(f"  Cells passing QC: {cell_pass_qc.sum():,}")
    print(f"  Cells filtered: {(~cell_pass_qc).sum():,}")
    print(f"  Mean junctions per cell: {junctions_per_cell.mean():.1f}")
    print(f"  Mean reads per cell: {reads_per_cell.mean():.1f}")
    
    # Store QC results in adata
    adata.obs['junctions_per_cell'] = junctions_per_cell
    adata.obs['reads_per_cell'] = reads_per_cell
    adata.obs['cell_pass_qc'] = cell_pass_qc
    
    return adata[cell_pass_qc].copy()  # Return filtered adata

def perform_atse_qc(adata, min_cells=50):
    """
    Quality control for ATSEs:
    1. Check coverage (number of cells with data)
    2. Focus on ATSEs with 2-3 junctions for interpretability
    """
    
    print("\n" + "="*60)
    print("STEP 1B: ATSE QUALITY CONTROL")
    print("="*60)
    
    # Get ATSEs with 2-3 junctions only
    atse_counts = adata.var.groupby("event_id").size()
    simple_atses = atse_counts[(atse_counts >= 2) & (atse_counts <= 3)].index
    print(f"Analyzing {len(simple_atses):,} ATSEs with 2-3 junctions")
    
    # Initialize QC tracking
    atse_qc_results = []
    
    # Work with sparse matrix directly
    
    psi_matrix = adata.layers["psi"]
    
    # Check each ATSE
    for atse_id in tqdm(simple_atses, desc="ATSE QC check"):
        # Get junctions in this ATSE
        junc_mask = adata.var["event_id"] == atse_id
        junc_indices = np.where(junc_mask)[0]
        
        if len(junc_indices) < 2:
            continue
            
        # Get PSI values for these junctions (sparse matrix slice)
        atse_psi = psi_matrix[:, junc_indices]
        
        # Count cells with any data for this ATSE (any non-zero PSI)
        cells_with_data = (atse_psi > 0).sum(axis=1).A1 > 0
        n_cells_detected = cells_with_data.sum()
        
        # Calculate mean PSI values across cells with data
        if n_cells_detected > 0:
            # Get mean PSI for each junction
            mean_psi_per_junction = np.array([atse_psi[:, j].data.mean() if atse_psi[:, j].nnz > 0 else 0 
                                            for j in range(atse_psi.shape[1])])
            
            atse_qc_results.append({
                'atse_id': atse_id,
                'n_junctions': len(junc_indices),
                'n_cells_detected': n_cells_detected,
                'mean_psi_per_junction': mean_psi_per_junction.tolist(),
                'max_mean_psi': mean_psi_per_junction.max(),
                'coverage_rate': n_cells_detected / adata.n_obs
            })
    
    qc_df = pd.DataFrame(atse_qc_results)
    
    # Classify ATSEs based on coverage only
    qc_df['qc_status'] = 'good'
    qc_df.loc[qc_df['n_cells_detected'] < min_cells, 'qc_status'] = 'low_coverage'
    qc_df.loc[qc_df['coverage_rate'] < 0.01, 'qc_status'] = 'very_low_coverage'  # <1% of cells
    
    # Summary statistics
    print(f"\nATSE QC Summary:")
    print(f"  Good coverage ATSEs: {(qc_df['qc_status'] == 'good').sum():,}")
    print(f"  Low coverage ATSEs: {(qc_df['qc_status'] == 'low_coverage').sum():,}")
    print(f"  Very low coverage ATSEs: {(qc_df['qc_status'] == 'very_low_coverage').sum():,}")
    print(f"  Mean cells per ATSE: {qc_df['n_cells_detected'].mean():.1f}")
    
    # Plot QC results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Distribution of cell coverage
    axes[0].hist(qc_df['n_cells_detected'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(min_cells, color='red', linestyle='--', label=f'Min threshold ({min_cells})')
    axes[0].set_xlabel('Number of Cells with Data')
    axes[0].set_ylabel('Number of ATSEs')
    axes[0].set_title('ATSE Cell Coverage Distribution')
    axes[0].set_xscale('log')
    axes[0].legend()
    
    # Coverage rate distribution
    axes[1].hist(qc_df['coverage_rate'], bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(0.01, color='orange', linestyle='--', label='1% threshold')
    axes[1].set_xlabel('Coverage Rate (fraction of total cells)')
    axes[1].set_ylabel('Number of ATSEs')
    axes[1].set_title('ATSE Coverage Rate Distribution')
    axes[1].legend()
    
    # Max mean PSI distribution
    axes[2].hist(qc_df['max_mean_psi'], bins=30, edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Maximum Mean PSI Across Junctions')
    axes[2].set_ylabel('Number of ATSEs')
    axes[2].set_title('ATSE Expression Level Distribution')
    
    plt.tight_layout()
    plt.suptitle('ATSE Quality Control', y=1.02, fontsize=14)
    plt.show()
    
    return qc_df

# %%
# =============================================================================
# STEP 2: GENERATE CELL TYPE LEVEL PSI VALUES
# =============================================================================

def generate_celltype_psi_table(adata, qc_df, cell_type_level='broad_cell_type'):
    """
    Generate cell type level PSI by aggregating junction reads across cells in same cell type.
    This creates a table with ATSE-cell type columns for downstream analysis.
    """
    
    print("\n" + "="*60)
    print("STEP 2: GENERATE CELL TYPE LEVEL PSI")
    print("="*60)
    
    # Filter for good quality ATSEs
    good_atses = qc_df[qc_df['qc_status'] == 'good']['atse_id'].values
    print(f"Processing {len(good_atses):,} good quality ATSEs")
    
    # Get cell types and age groups
    cell_types = adata.obs[cell_type_level].unique()
    age_groups = adata.obs['age_group'].unique()
    print(f"Cell types: {len(cell_types)}, Age groups: {age_groups}")
    
    # Initialize results storage
    celltype_psi_results = []
    
    # Work with sparse matrix directly
    psi_matrix = adata.layers["psi"]
    
    for ct in tqdm(cell_types, desc="Processing cell types"):
        # Get cells for this cell type
        ct_mask = adata.obs[cell_type_level] == ct
        ct_indices = np.where(ct_mask)[0]
        
        if len(ct_indices) < 10:  # Skip cell types with too few cells
            continue
        
        # Separate by age group
        for age in age_groups:
            age_ct_mask = (adata.obs[cell_type_level] == ct) & (adata.obs['age_group'] == age)
            age_ct_indices = np.where(age_ct_mask)[0]
            
            if len(age_ct_indices) < 5:  # Need minimum cells per age-celltype combo
                continue
            
            # Process each good ATSE
            for atse_id in good_atses:
                # Get junctions in this ATSE
                junc_mask = adata.var["event_id"] == atse_id
                junc_indices = np.where(junc_mask)[0]
                
                if len(junc_indices) < 2:
                    continue
                
                # Get PSI values for this cell type + age combination
                # Extract submatrix for this age-celltype combo and these junctions
                atse_psi = psi_matrix[age_ct_indices, :][:, junc_indices]
                
                # Calculate cell-type level PSI by taking mean across cells
                # Only consider cells with actual data (non-zero PSI)
                celltype_psi_values = []
                for j in range(len(junc_indices)):
                    junction_data = atse_psi[:, j]
                    if junction_data.nnz > 0:  # Has data
                        mean_psi = junction_data.data.mean()
                        n_cells_with_data = junction_data.nnz
                    else:
                        mean_psi = 0.0
                        n_cells_with_data = 0
                    
                    celltype_psi_values.append({
                        'junction_idx': junc_indices[j],
                        'junction_id': adata.var.iloc[junc_indices[j]]['junction_id'],
                        'mean_psi': mean_psi,
                        'n_cells_with_data': n_cells_with_data
                    })
                
                # Store results
                celltype_psi_results.append({
                    'cell_type': ct,
                    'age_group': age,
                    'atse_id': atse_id,
                    'n_junctions': len(junc_indices),
                    'total_cells': len(age_ct_indices),
                    'psi_values': celltype_psi_values,
                    'psi_sum': sum([pv['mean_psi'] for pv in celltype_psi_values])
                })
    
    celltype_psi_df = pd.DataFrame(celltype_psi_results)
    print(f"Generated {len(celltype_psi_df):,} cell type - ATSE combinations")
    
    return celltype_psi_df

# =============================================================================
# STEP 3: DELTA PSI ANALYSIS FOR DOWNSTREAM STORAGE
# =============================================================================

def analyze_delta_psi_patterns(celltype_psi_df):
    """
    Generate delta PSI analysis focusing on young vs old changes.
    Store results in format suitable for downstream analysis.
    """
    
    print("\n" + "="*60)
    print("STEP 3: DELTA PSI ANALYSIS")
    print("="*60)
    
    # Pivot to get young and old side by side for comparison
    delta_psi_results = []
    
    # Group by cell type and ATSE to compare young vs old
    for (ct, atse_id), group in celltype_psi_df.groupby(['cell_type', 'atse_id']):
        # Check if we have both young and old data
        ages_present = group['age_group'].unique()
        if len(ages_present) < 2:
            continue
        
        young_data = group[group['age_group'] == 'young']
        old_data = group[group['age_group'] == 'old']
        
        if len(young_data) == 0 or len(old_data) == 0:
            continue
        
        # Get PSI values for young and old
        young_psi_values = young_data.iloc[0]['psi_values']
        old_psi_values = old_data.iloc[0]['psi_values']
        
        # Calculate delta PSI for each junction
        delta_psi_per_junction = []
        junction_info = []
        
        for i, (young_psi, old_psi) in enumerate(zip(young_psi_values, old_psi_values)):
            if young_psi['n_cells_with_data'] > 0 and old_psi['n_cells_with_data'] > 0:
                delta = old_psi['mean_psi'] - young_psi['mean_psi']
                delta_psi_per_junction.append(delta)
                junction_info.append({
                    'junction_idx': young_psi['junction_idx'],
                    'junction_id': young_psi['junction_id'],
                    'young_psi': young_psi['mean_psi'],
                    'old_psi': old_psi['mean_psi'],
                    'delta_psi': delta,
                    'young_n_cells': young_psi['n_cells_with_data'],
                    'old_n_cells': old_psi['n_cells_with_data']
                })
        
        if len(delta_psi_per_junction) < 2:
            continue
        
        # Analyze pattern and coordination
        delta_psi_array = np.array(delta_psi_per_junction)
        max_abs_delta = np.max(np.abs(delta_psi_array))
        
        # Skip if no meaningful change
        if max_abs_delta < 0.01:
            continue
        
        # Determine pattern and coordination
        n_junctions = len(delta_psi_per_junction)
        
        if n_junctions == 2:
            # For 2 junctions: should be anticorrelated
            correlation = np.corrcoef(delta_psi_array[:2])[0, 1] if len(delta_psi_array) >= 2 else 0
            pattern = "two_junction"
            coordinated = correlation < -0.3  # Less strict threshold
            
        elif n_junctions == 3:
            # For cassette exon: identify skip junction (longest span)
            spans = []
            for jinfo in junction_info:
                jid = jinfo['junction_id']
                parts = jid.split("_")
                if len(parts) >= 3:
                    try:
                        span = int(parts[2]) - int(parts[1])
                        spans.append(span)
                    except:
                        spans.append(0)
                else:
                    spans.append(0)
            
            if len(spans) == 3:
                skip_idx = np.argmax(spans)
                incl_indices = [i for i in range(3) if i != skip_idx]
                
                if len(incl_indices) == 2:
                    incl_corr = np.corrcoef([delta_psi_array[incl_indices[0]], delta_psi_array[incl_indices[1]]])[0, 1]
                    skip_vs_incl_corr = np.corrcoef(delta_psi_array[skip_idx], np.mean(delta_psi_array[incl_indices]))[0, 1]
                    
                    pattern = "cassette_exon"
                    coordinated = (incl_corr > 0.3) and (skip_vs_incl_corr < -0.3)
                    correlation = skip_vs_incl_corr
                else:
                    correlation = 0
                    pattern = "three_junction"
                    coordinated = False
            else:
                correlation = 0
                pattern = "three_junction"
                coordinated = False
        else:
            correlation = 0
            pattern = "multi_junction"
            coordinated = False
        
        # Calculate PSI sum changes
        young_sum = young_data.iloc[0]['psi_sum']
        old_sum = old_data.iloc[0]['psi_sum']
        sum_change = abs(old_sum - young_sum)
        
        # Store detailed results
        delta_psi_results.append({
            'cell_type': ct,
            'atse_id': atse_id,
            'n_junctions': n_junctions,
            'pattern': pattern,
            'coordinated': coordinated,
            'correlation': correlation,
            'max_abs_delta_psi': max_abs_delta,
            'young_psi_sum': young_sum,
            'old_psi_sum': old_sum,
            'sum_change': sum_change,
            'junction_details': junction_info,
            'young_total_cells': young_data.iloc[0]['total_cells'],
            'old_total_cells': old_data.iloc[0]['total_cells']
        })
    
    delta_psi_df = pd.DataFrame(delta_psi_results)
    
    print(f"Delta PSI Analysis Results:")
    print(f"  Total ATSE-celltype combinations: {len(delta_psi_df):,}")
    print(f"  Two-junction patterns: {(delta_psi_df['pattern'] == 'two_junction').sum():,}")
    print(f"  Cassette exon patterns: {(delta_psi_df['pattern'] == 'cassette_exon').sum():,}")
    print(f"  Coordinated changes: {delta_psi_df['coordinated'].sum():,}")
    print(f"  Mean |delta PSI|: {delta_psi_df['max_abs_delta_psi'].mean():.3f}")
    
    return delta_psi_df

# %%
# =============================================================================
# STEP 3: VISUALIZATION OF COORDINATION PATTERNS
# =============================================================================

def plot_delta_psi_results(delta_psi_df, output_dir, today):
    """
    Create visualizations for delta PSI analysis results
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Pattern distribution
    pattern_counts = delta_psi_df['pattern'].value_counts()
    colors = {'two_junction': 'blue', 'cassette_exon': 'green', 
              'three_junction': 'orange', 'multi_junction': 'gray'}
    axes[0, 0].bar(pattern_counts.index, pattern_counts.values, 
                   color=[colors.get(x, 'gray') for x in pattern_counts.index])
    axes[0, 0].set_xlabel('ATSE Pattern')
    axes[0, 0].set_ylabel('Number of ATSE-CellType Combinations')
    axes[0, 0].set_title('ATSE Pattern Distribution')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Correlation distribution by pattern
    for pattern in delta_psi_df['pattern'].unique():
        pattern_data = delta_psi_df[delta_psi_df['pattern'] == pattern]['correlation'].dropna()
        axes[0, 1].hist(pattern_data, alpha=0.5, label=pattern, bins=20, edgecolor='black')
    axes[0, 1].axvline(0, color='black', linestyle='--')
    axes[0, 1].set_xlabel('Junction Correlation')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Junction Correlation by Pattern')
    axes[0, 1].legend()
    
    # 3. Delta PSI magnitude distribution
    axes[0, 2].hist(delta_psi_df['max_abs_delta_psi'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 2].axvline(0.1, color='orange', linestyle='--', label='10% threshold')
    axes[0, 2].axvline(0.2, color='red', linestyle='--', label='20% threshold')
    axes[0, 2].set_xlabel('Max |ΔPSI| (old - young)')
    axes[0, 2].set_ylabel('Number of ATSE-CellType Combinations')
    axes[0, 2].set_title('Delta PSI Magnitude Distribution')
    axes[0, 2].legend()
    
    # 4. Coordinated vs uncoordinated by cell type
    coord_by_ct = delta_psi_df.groupby(['cell_type', 'coordinated']).size().unstack(fill_value=0)
    if coord_by_ct.shape[1] >= 2:
        coord_by_ct_norm = coord_by_ct.div(coord_by_ct.sum(axis=1), axis=0)
        coord_by_ct_norm.plot(kind='barh', stacked=True, ax=axes[1, 0], 
                             color=['red', 'green'])
        axes[1, 0].set_xlabel('Proportion')
        axes[1, 0].set_ylabel('Cell Type')
        axes[1, 0].set_title('Coordination Rate by Cell Type')
        axes[1, 0].legend(['Uncoordinated', 'Coordinated'])
    
    # 5. Scatter: max delta PSI vs correlation
    for pattern in delta_psi_df['pattern'].unique():
        pattern_data = delta_psi_df[delta_psi_df['pattern'] == pattern]
        axes[1, 1].scatter(pattern_data['max_abs_delta_psi'], 
                          pattern_data['correlation'],
                          alpha=0.6, label=pattern, s=20,
                          color=colors.get(pattern, 'gray'))
    axes[1, 1].set_xlabel('Max |ΔPSI|')
    axes[1, 1].set_ylabel('Junction Correlation')
    axes[1, 1].set_title('Effect Size vs Coordination')
    axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[1, 1].legend()
    
    # 6. Summary statistics table
    summary_text = "Summary Statistics:\n\n"
    summary_text += f"Total combinations: {len(delta_psi_df):,}\n"
    summary_text += f"Coordinated: {delta_psi_df['coordinated'].sum():,} ({delta_psi_df['coordinated'].mean()*100:.1f}%)\n"
    summary_text += f"Mean |ΔPSI|: {delta_psi_df['max_abs_delta_psi'].mean():.3f}\n"
    summary_text += f"Median |ΔPSI|: {delta_psi_df['max_abs_delta_psi'].median():.3f}\n\n"
    
    summary_text += "Top changing ATSEs:\n"
    top_changes = delta_psi_df.nlargest(3, 'max_abs_delta_psi')
    for _, row in top_changes.iterrows():
        summary_text += f"  {row['atse_id'][:15]}... ({row['cell_type'][:10]})\n"
        summary_text += f"    ΔPSI: {row['max_abs_delta_psi']:.3f}\n"
    
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Summary Statistics')
    
    plt.tight_layout()
    plt.suptitle('Delta PSI Analysis Results', y=1.02, fontsize=14)
    plt.savefig(f"{output_dir}/delta_psi_analysis_{today}.pdf", bbox_inches='tight')
    plt.show()
    
    return fig

# %%
# =============================================================================
# DECIDE: CELL TYPE VS TISSUE-CELL TYPE ANALYSIS
# =============================================================================

def create_tissue_celltype_column(adata):
    """
    Create tissue-cell type combinations for more granular analysis
    """
    # Assuming you have a tissue column, otherwise derive from cell type
    if 'tissue' in adata.obs.columns:
        adata.obs['tissue_celltype'] = (adata.obs['tissue'].astype(str) + '_' + 
                                         adata.obs['broad_cell_type'].astype(str))
    else:
        # Derive tissue from cell type (simplified mapping)
        tissue_map = {
            'Neuron': 'Brain',
            'Glial cell': 'Brain',
            'Cardiac cell': 'Heart',
            'Liver cell': 'Liver',
            'Kidney cell': 'Kidney',
            'Immune cell': 'Blood',
            'Muscle cell': 'Muscle',
            'Epithelial cell': 'Epithelium',
            'Stromal cell': 'Stroma',
            'Vascular cell': 'Vasculature',
            # Add more mappings as needed
        }
        adata.obs['tissue'] = adata.obs['broad_cell_type'].map(tissue_map).fillna('Other')
        adata.obs['tissue_celltype'] = (adata.obs['tissue'].astype(str) + '_' + 
                                        adata.obs['broad_cell_type'].astype(str))
    
    return adata

# %%
# =============================================================================
# MAIN EXECUTION WRAPPER
# =============================================================================

def run_full_delta_psi_analysis(adata, output_dir, use_tissue_celltype=False):
    """
    Run complete delta PSI analysis with cell and ATSE QC
    
    Parameters:
    -----------
    adata : AnnData object with PSI values calculated
    output_dir : where to save figures and results
    use_tissue_celltype : if True, analyze at tissue-cell type level
                         if False, analyze at broad cell type level
    """
    
    from datetime import datetime
    today = datetime.now().strftime("%Y%m%d")
    
    print(f"Starting Delta PSI Analysis with QC")
    print(f"Analysis level: {'Tissue-Cell Type' if use_tissue_celltype else 'Broad Cell Type'}")
    print("="*60)
    
    # Step 1A: Cell Quality Control (First!)
    print("Filtering low-quality cells first...")
    adata_qc = perform_cell_qc(adata)
    print(f"Retained {adata_qc.n_obs:,} cells after QC (from {adata.n_obs:,})")
    
    # Step 1B: ATSE Quality Control
    atse_qc_df = perform_atse_qc(adata_qc)
    atse_qc_df.to_csv(f"{output_dir}/atse_qc_results_{today}.csv", index=False)
    print(f"ATSE QC results saved to {output_dir}/atse_qc_results_{today}.csv")
    
    # Decide on analysis level
    if use_tissue_celltype:
        adata_qc = create_tissue_celltype_column(adata_qc)
        analysis_column = 'tissue_celltype'
    else:
        analysis_column = 'broad_cell_type'
    
    # Step 2: Generate Cell Type Level PSI
    celltype_psi_df = generate_celltype_psi_table(adata_qc, atse_qc_df, analysis_column)
    celltype_psi_df.to_csv(f"{output_dir}/celltype_psi_table_{today}.csv", index=False)
    print(f"Cell type PSI table saved to {output_dir}/celltype_psi_table_{today}.csv")
    
    # Step 3: Delta PSI Analysis
    delta_psi_df = analyze_delta_psi_patterns(celltype_psi_df)
    delta_psi_df.to_csv(f"{output_dir}/delta_psi_analysis_{today}.csv", index=False)
    print(f"Delta PSI analysis saved to {output_dir}/delta_psi_analysis_{today}.csv")
    
    # Step 4: Visualization
    fig = plot_delta_psi_results(delta_psi_df, output_dir, today)
    
    # Summary statistics
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    summary = {
        'Original cells': adata.n_obs,
        'Cells after QC': adata_qc.n_obs,
        'Total ATSEs analyzed': len(atse_qc_df),
        'Good coverage ATSEs': (atse_qc_df['qc_status'] == 'good').sum(),
        'Cell type combinations': len(celltype_psi_df),
        'Delta PSI comparisons': len(delta_psi_df),
        'Coordinated changes': delta_psi_df['coordinated'].sum(),
        'Cell types analyzed': delta_psi_df['cell_type'].nunique(),
        'Mean |delta PSI|': delta_psi_df['max_abs_delta_psi'].mean()
    }
    
    for key, value in summary.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value:.3f}")
    
    return adata_qc, atse_qc_df, celltype_psi_df, delta_psi_df

# Example usage:
# adata_qc, atse_qc_df, celltype_psi_df, delta_psi_df = run_full_delta_psi_analysis(splice_adata, output_dir, use_tissue_celltype=False)