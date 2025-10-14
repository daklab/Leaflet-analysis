#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import anndata as ad
import torch
import gseapy as gp
from collections import defaultdict
from tqdm import tqdm

# Import utility functions
sys.path.append('/gpfs/commons/home/kisaev/Leaflet-analysis/Multi_Species_Splicing_Foundation/shared_utils/')
from utils import *

# Check if using CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get today's date
from datetime import datetime
today = datetime.now().strftime("%Y%m%d")
print(f"Today's date: {today}")

# %%
BASE_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION"

# Output directory 
OUTPUT_DIR = "/gpfs/commons/home/kisaev/Leaflet-analysis/Multi_Species_Splicing_Foundation/GE_PLOTS/HUMAN"
OUTPUT_DIR = os.path.join(OUTPUT_DIR, today)

# If doesn't exist, make dir
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

mouse = False 
GE_ANNDATA_scVI_PATH = f"{BASE_DIR}/scVI/ge_adata_with_scvi_model_latent_20_2025-10-03.h5ad"

AGING_GENES_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/27857814"
RBP_FILE_PATH="/gpfs/commons/groups/knowles_lab/Karin/VanNostrand_2020_supptable1_41586_2020_2077_MOESM3_ESM.xlsx"

# %%
print("Loading data...")
ge_adata = ad.read_h5ad(GE_ANNDATA_scVI_PATH)
print(f"Done reading scVI anndata from {GE_ANNDATA_scVI_PATH}")

# %%
print("Setting up data...")
# If ge_adata.obs doesn't have cell_id make it from cell_id_clean
if "cell_id" not in ge_adata.obs.columns:
    ge_adata.obs["cell_id"] = ge_adata.obs["cell_id_clean"]

# Load aging gene lists
aging_genes_mouse, aging_genes_human = load_aging_genes(AGING_GENES_PATH)

# Load RBP genes
rbps = load_rbp_genes(RBP_FILE_PATH)

# If ge_adata.var["gene_name"] is not in ge_adata.var_names, then add it
if "gene_name" not in ge_adata.var.columns:
    ge_adata.var["gene_name"] = ge_adata.var_names

rbps["mouse_gene_name"] = rbps["mouse_gene_name"].str.upper()
aging_genes_mouse = [g.upper() for g in aging_genes_mouse]

if mouse:
    print(f"Processing mouse data...")
    ge_adata.var["RBP_gene"] = ge_adata.var["gene_name"].isin(rbps["mouse_gene_name"])
    ge_adata.var["Aging_gene"] = ge_adata.var["gene_name"].isin(aging_genes_mouse)
    rbps = rbps["mouse_gene_name"]
else:
    ge_adata.var["RBP_gene"] = ge_adata.var["gene_name"].isin(rbps["gene_name"])
    ge_adata.var["Aging_gene"] = ge_adata.var["gene_name"].isin(aging_genes_human)
    rbps = rbps["gene_name"]

print(f"RBP genes found: {ge_adata.var.RBP_gene.sum()}")
print(f"Aging genes found: {ge_adata.var.Aging_gene.sum()}")

# %%
print("Building factor loading matrices...")
gene_names = ge_adata.var["gene_name"].values
print(f"Total genes: {len(gene_names)}")

print("Building factor loading matrices...")
gene_names = ge_adata.var["gene_name"].values
print(f"Total genes: {len(gene_names)}")

# Use .values to get the raw numpy array
scvi_loadings = pd.DataFrame(
    ge_adata.varm["scVI_linear_gene_loadings"].values,  # <-- Add .values here!
    index=gene_names,
    columns=[f"Z_{i}" for i in range(ge_adata.varm["scVI_linear_gene_loadings"].shape[1])]
)

print(scvi_loadings.head())
print(f"scVI factors: {scvi_loadings.shape[1]}")

# %%
def plot_factor_heatmap(adata, factor_key, method_label="NMF", celltype_col="broad_cell_type", 
                       output_dir=None, filename=None, max_factors=20):
    """Plot heatmap of mean latent factor activity per cell type."""
    
    # Get factor values and cell types
    factors = pd.DataFrame(adata.obsm[factor_key], index=adata.obs.index)
    factors[celltype_col] = adata.obs[celltype_col]
    
    # Compute mean factor value per cell type
    mean_matrix = factors.groupby(celltype_col).mean()
    
    # Restrict to top variable factors
    if max_factors and mean_matrix.shape[1] > max_factors:
        top_factors = mean_matrix.var(axis=0).sort_values(ascending=False).head(max_factors).index
        mean_matrix = mean_matrix[top_factors]
    
    # Create heatmap
    g = sns.clustermap(mean_matrix, cmap="PRGn", center=0, figsize=(12, 8), 
                      annot=False, linewidths=0.3, linecolor="gray")
    g.fig.suptitle(f"Mean {method_label} factor activity per {celltype_col}", fontsize=14)
    
    # Save if needed
    if output_dir and filename:
        path = os.path.join(output_dir, filename)
        g.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")
    
    plt.show()
    return g

print("Plotting factor-celltype associations...")

plot_factor_heatmap(ge_adata, factor_key="X_scVI_linear", method_label="scVI",
                   output_dir=OUTPUT_DIR, filename="scvi_factor_celltype_heatmap.pdf")

# %%
def run_focused_gsea(loadings_df, organism="Mouse", output_dir=None):
    """
    Run GSEA on factor loadings using focused, high-quality gene set databases.
    """
    
    # Focused gene set databases - only essential ones
    gene_set_databases = {
        # Core biological processes
        'GO_Biological_Process_2023': 'GO Biological Process (2023)',
        'GO_Molecular_Function_2023': 'GO Molecular Function (2023)', 
        'GO_Cellular_Component_2023': 'GO Cellular Component (2023)',
        
        # High-quality pathway database
        'Reactome_2022': 'Reactome Pathways',
        'PanglaoDB_Augmented_2021': 'Human Cell Type Markers',  # replace if available
    }
    
    all_results = {}
    
    for factor in tqdm(loadings_df.columns, desc="Processing factors"):
        print(f"\nProcessing {factor}...")
        factor_results = {}
        
        # Get ranked gene list (sort by loading values)
        ranked_genes = loadings_df[factor].sort_values(ascending=False)
        
        for db_key, db_name in gene_set_databases.items():
            try:
                print(f"  Running GSEA on {db_name}...")
                
                # Adjust parameters for better specificity
                res = gp.prerank(
                    rnk=ranked_genes,
                    gene_sets=db_key,
                    organism=organism,
                    permutation_num=100, 
                    outdir=None,
                    threads=16,
                    min_size=10,    # Smaller min size for more specific pathways
                    max_size=300,   # Smaller max size to avoid overly broad terms
                    ascending=False,
                    seed=42  # Reproducibility
                )
                
                if not res.res2d.empty:
                    # Add database info
                    res_df = res.res2d.copy()
                    res_df['Database'] = db_name
                    res_df['Factor'] = factor
                    factor_results[db_key] = res_df
                else:
                    factor_results[db_key] = None
                    
            except Exception as e:
                print(f"    Error with {db_name}: {str(e)}")
                factor_results[db_key] = None
        
        all_results[factor] = factor_results
    
    return all_results

def filter_and_rank_gsea_results(gsea_results, fdr_threshold=0.05, min_abs_nes=1.5, 
                                top_n_per_factor=8, remove_redundant=True):
    """Filter GSEA results to get the most specific and significant pathways."""
    
    filtered_results = {}
    
    for factor, db_results in gsea_results.items():
        factor_filtered = {}
        
        for db_key, df in db_results.items():
            if df is None:
                continue
                
            # Apply filters
            filtered_df = df[
                (df['FDR q-val'] <= fdr_threshold) &
                (df['NES'].abs() >= min_abs_nes)
            ].copy()
            
            if filtered_df.empty:
                continue
            
            # Sort by absolute NES (effect size)
            filtered_df = filtered_df.reindex(
                filtered_df['NES'].abs().sort_values(ascending=False).index
            )
            
            # Take top N
            filtered_df = filtered_df.head(top_n_per_factor)
            
            # Remove redundant terms (optional)
            if remove_redundant:
                filtered_df = remove_redundant_terms(filtered_df)
            
            factor_filtered[db_key] = filtered_df
        
        filtered_results[factor] = factor_filtered
    
    return filtered_results

def remove_redundant_terms(df, similarity_threshold=0.7):
    """Remove redundant pathway terms based on word overlap."""
    if len(df) <= 1:
        return df
    
    # Simple approach: remove terms with high word overlap in names
    terms = df['Term'].tolist()
    keep_indices = []
    
    for i, term1 in enumerate(terms):
        is_redundant = False
        term1_words = set(term1.lower().split())
        
        for j in keep_indices:
            term2_words = set(terms[j].lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(term1_words & term2_words)
            union = len(term1_words | term2_words)
            
            if union > 0 and intersection / union > similarity_threshold:
                is_redundant = True
                break
        
        if not is_redundant:
            keep_indices.append(i)
    
    return df.iloc[keep_indices]

def create_focused_pathway_summary(filtered_results, output_dir=None):
    """Create a focused summary of the most important pathways per factor."""
    
    # Flatten results with better organization
    summary_rows = []
    
    for factor, db_results in filtered_results.items():
        for db_key, df in db_results.items():
            if df is None or df.empty:
                continue
                
            for _, row in df.iterrows():
                summary_rows.append({
                    'Factor': factor,
                    'Database': row['Database'],
                    'Term': row['Term'],
                    'NES': row['NES'],
                    'FDR_qval': row['FDR q-val'],
                    'Leading_edge_num': row['Tag %'],
                    'Gene_set_size': len(str(row['Lead_genes']).split(';')) if pd.notna(row['Lead_genes']) else 0
                })
    
    summary_df = pd.DataFrame(summary_rows)
    
    if output_dir:
        summary_df.to_csv(
            os.path.join(output_dir, "focused_pathway_summary.tsv.gz"),
            sep='\t', index=False, compression='gzip'
        )
    
    return summary_df

def plot_top_pathways_per_factor(summary_df, method_name="NMF", top_n=5, output_dir=None):
    """Create a focused heatmap showing only the top pathways per factor."""
    
    if summary_df.empty:
        print(f"No pathways found for {method_name}")
        return
    
    # Get top pathways per factor
    top_pathways = []
    
    for factor in summary_df['Factor'].unique():
        factor_data = summary_df[summary_df['Factor'] == factor]
        
        # Sort by absolute NES and take top N
        top_factor = factor_data.reindex(
            factor_data['NES'].abs().sort_values(ascending=False).index
        ).head(top_n)
        
        top_pathways.append(top_factor)
    
    if not top_pathways:
        print("No pathways found for plotting")
        return
    
    top_df = pd.concat(top_pathways, ignore_index=True)
    
    # Create matrix
    pathway_matrix = top_df.pivot_table(
        index='Term', 
        columns='Factor', 
        values='NES', 
        fill_value=0
    )
    
    # Plot
    g = sns.clustermap(
        pathway_matrix,
        cmap="RdBu_r",
        center=0,
        figsize=(14, max(8, len(pathway_matrix) * 0.4)),
        col_cluster=True,
        row_cluster=True,
        xticklabels=True,
        yticklabels=True,
        dendrogram_ratio=0.1,
        linewidths=0.1,
        cbar_kws={'label': 'Normalized Enrichment Score (NES)'}
    )
    
    # Formatting
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=8)
    
    g.fig.suptitle(f'Top {top_n} Pathways per {method_name} Factor', 
                   fontsize=14, y=0.98)
    
    if output_dir:
        save_path = os.path.join(output_dir, f"{method_name.lower()}_top_pathways_heatmap.pdf")
        g.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return g

def build_factor_term_matrix(gsea_results, score_type='NES', qval_threshold=0.05, max_terms=8):
    """
    Build matrix of terms vs factors using NES scores.
    """
    term_dict = defaultdict(dict)
    for factor, df in gsea_results.items():
        if df is not None:
            # Filter for significant terms first
            sig_df = df[df['FDR q-val'] <= qval_threshold]
            
            # If more than max_terms significant, keep top max_terms
            if len(sig_df) > max_terms:
                sig_df = sig_df.head(max_terms)
            
            # Add terms to matrix
            for _, row in sig_df.iterrows():
                if score_type == 'NES':
                    term_dict[row['Term']][factor] = row['NES']
                elif score_type == 'ES':
                    term_dict[row['Term']][factor] = row['ES']  
                elif score_type == 'qval':
                    term_dict[row['Term']][factor] = -np.log10(row['FDR q-val'] + 1e-10)
    
    return pd.DataFrame(term_dict).T.fillna(0)

def plot_gsea_heatmap(term_matrix, title, filename, output_dir=None, score_type='NES'):
    """Plot GSEA results heatmap."""
    
    if term_matrix.empty:
        print(f"No data to plot for {title}")
        return
        
    # Create the clustermap 
    g = sns.clustermap(
        term_matrix,
        cmap="PRGn",
        center=0,
        figsize=(10, max(8, len(term_matrix) * 0.3)),
        col_cluster=True,
        row_cluster=True,
        xticklabels=True,
        yticklabels=True, 
        dendrogram_ratio=0.1, 
        linewidths=0.1,
        linecolor='gray',
        cbar_kws={'label': f'{score_type} Score'}
    )
    
    # Formatting
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=8)
    g.fig.suptitle(title, fontsize=14)
    
    # Save the figure
    if output_dir:
        save_path = os.path.join(output_dir, filename)
        g.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return g

# %%
print("=== RUNNING FOCUSED GSEA ANALYSIS ===")

# Run focused GSEA
focused_gsea_scvi = run_focused_gsea(scvi_loadings, organism="Mouse", output_dir=OUTPUT_DIR)  
print("=== FILTERING RESULTS FOR SPECIFICITY ===")

# Filter for high-quality, specific results
filtered_scvi = filter_and_rank_gsea_results(
    focused_gsea_scvi,
    fdr_threshold=0.05,
    min_abs_nes=1.5,
    top_n_per_factor=8
)

print("=== CREATING FOCUSED SUMMARIES ===")

# Create summaries
scvi_summary = create_focused_pathway_summary(filtered_scvi, output_dir=OUTPUT_DIR)

print("=== PLOTTING TOP PATHWAYS ===")
# Plot focused heatmaps
plot_top_pathways_per_factor(scvi_summary, method_name="scVI", output_dir=OUTPUT_DIR)

# %%
print("=== SAVING RESULTS ===")

# Save loadings
scvi_loadings.to_csv(os.path.join(OUTPUT_DIR, "scvi_loadings.tsv.gz"), sep="\t", compression="gzip")
print(f"Saved loadings to {OUTPUT_DIR}")

# Flatten and save comprehensive results
def flatten_comprehensive_results(gsea_dict, label):
    """Convert comprehensive GSEA results to flat format."""
    all_rows = []
    for factor, db_results in gsea_dict.items():
        for db_key, df in db_results.items():
            if df is not None:
                temp = df.copy()
                temp["Factor"] = factor
                temp["Method"] = label
                temp["Database_key"] = db_key
                all_rows.append(temp)
    return pd.concat(all_rows, axis=0) if all_rows else pd.DataFrame()

# Save comprehensive results
flat_scvi_comprehensive = flatten_comprehensive_results(focused_gsea_scvi, label="scVI")
flat_scvi_comprehensive.to_csv(os.path.join(OUTPUT_DIR, "gsea_results_comprehensive_scvi.tsv.gz"), 
                              sep="\t", index=False, compression="gzip")
print("✅ Saved comprehensive GSEA result tables")

# %%  
print("=== CREATING FACTOR-TERM MATRICES AND HEATMAPS ===")

# Build simplified factor-term matrices from filtered results
def build_simple_matrix(filtered_results):
    """Build factor x term matrix from filtered results."""
    term_dict = defaultdict(dict)
    
    for factor, db_results in filtered_results.items():
        for db_key, df in db_results.items():
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                term_dict[row['Term']][factor] = row['NES']
    
    return pd.DataFrame(term_dict).T.fillna(0)

# Build matrices
scvi_matrix = build_simple_matrix(filtered_scvi)  

plot_gsea_heatmap(
    scvi_matrix,
    title="scVI Factors – Top GSEA Terms", 
    filename="scvi_gsea_focused_heatmap.pdf",
    output_dir=OUTPUT_DIR,
    score_type='NES'
)

print("✅ Focused GSEA analysis complete!")

# Summary statistics
print(f"\n=== SUMMARY STATISTICS ===")
print(f"scVI: {len(scvi_summary)} pathway annotations across {scvi_loadings.shape[1]} factors")  
print(f"Results saved to: {OUTPUT_DIR}")

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION
# script=/gpfs/commons/home/kisaev/Leaflet-analysis/Human_Splicing_Foundation/GeneExpression/09_GE_vs_GSEA.py
# sbatch --mem=300G -p dev,cpu,bigmem -J "HUM_GE_vs_AGING" --wrap="python $script"