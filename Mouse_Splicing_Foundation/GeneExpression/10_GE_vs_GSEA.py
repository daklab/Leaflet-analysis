# %%
#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as stats
import scanpy as sc
from collections import defaultdict
import torch  # ← this is needed
import gseapy as gp

# Import utility functions - simple direct import
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
BASE_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"

# Output directory 
OUTPUT_DIR = "/gpfs/commons/home/kisaev/Leaflet-analysis/Multi_Species_Splicing_Foundation/GE_PLOTS/MOUSE"
OUTPUT_DIR = os.path.join(OUTPUT_DIR, today)

# If doesn't exist, make dir
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

make_plot = False # already made plots
mouse = True 

GE_ANNDATA_scVI_PATH = f"{BASE_DIR}/scVI/ge_adata_with_both_scvi_models_2025-07-30.h5ad"
GE_ANNDATA_NMF_PATH = f"{BASE_DIR}/NMF/ge_adata_with_NMF_standard_50_1024_2025-07-30.h5ad"

AGING_GENES_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/27857814"
RBP_FILE_PATH="/gpfs/commons/groups/knowles_lab/Karin/VanNostrand_2020_supptable1_41586_2020_2077_MOESM3_ESM.xlsx"

# %%
ge_adata = ad.read_h5ad(GE_ANNDATA_scVI_PATH)
print(f"Done reading scVI anndata from {GE_ANNDATA_scVI_PATH}")
ge_adata_nmf = ad.read_h5ad(GE_ANNDATA_NMF_PATH)
print(f"Done reading NMF anndata from {GE_ANNDATA_NMF_PATH}")

# %%
# If ge_adata.obs doesn't have cell_id make it from cell_id_clean
if "cell_id" not in ge_adata.obs.columns:
    ge_adata.obs["cell_id"] = ge_adata.obs["cell_id_clean"]
    ge_adata_nmf.obs["cell_id"] = ge_adata_nmf.obs["cell_id_clean"]

# Load aging gene lists
aging_genes_mouse, aging_genes_human = load_aging_genes(AGING_GENES_PATH)

# Load RBP genes
rbps = load_rbp_genes(RBP_FILE_PATH)

# If ge_adata.var["gene_name"] is not in ge_adata.var_names, then add it
if "gene_name" not in ge_adata.var.columns:
    ge_adata.var["gene_name"] = ge_adata.var_names
rbps["mouse_gene_name"] = rbps["mouse_gene_name"].str.upper()
aging_genes_mouse = [g.upper() for g in aging_genes_mouse]

# if "mouse.id" is in splice_adata.obs rename it to donor_id 
if mouse:
    print(f"Renaming mouse.id to donor_id in splice_adata.obs")
    ge_adata.var["RBP_gene"] = ge_adata.var["gene_name"].isin(rbps["mouse_gene_name"])
    ge_adata.var["Aging_gene"] = ge_adata.var["gene_name"].isin(aging_genes_mouse)
    rbps = rbps["mouse_gene_name"]

else:
    ge_adata.var["RBP_gene"] = ge_adata.var["gene_name"].isin(rbps["gene_name"])
    ge_adata.var["Aging_gene"] = ge_adata.var["gene_name"].isin(aging_genes_human)
    rbps = rbps["gene_name"]

print(ge_adata.var.RBP_gene.value_counts())
assert np.all(ge_adata.obs_names == ge_adata_nmf.obs_names), "Cell IDs in ge_adata and ge_adata_nmf do not match or are not in the same order."
assert np.all(ge_adata.var_names == ge_adata_nmf.var_names), "Gene names in ge_adata and ge_adata_nmf do not match or are not in the same order."

ge_adata.obsm["X_nmf_standard_mb"] = ge_adata_nmf.obsm["X_nmf_standard_mb"]
ge_adata.varm["nmf_standard_mb_components"] = ge_adata_nmf.varm["nmf_standard_mb_components"]

# %%
# from scVI 
#  This mapping goes through intermediate values rho_gn which is output of get_normalized_expression from scVI
#, which provide a batch-corrected, normalized estimate of the percentage of transcripts in each cell n that originate from each gene g. We used these estimates for differential expression analysis and its scaled version (multiplying 
# by the estimated library size from the model L_n
#) for imputation.

# %% [markdown]
# ### Do some sanity checks of expected marker genes expression across cell types and also previously defined "aging" genes 

# %%
# BROAD MARKERS: For use with mapping_level="broad" (matches broad_cell_type_mappings output)
markers_broad = {
    # Broad categories from your broad mapping
    'Neuron': ['SLC17A7', 'CAMK2A', 'RBFOX3', 'NEUROD6', 'GAD1', 'GAD2', 'SLC32A1', 'DLX1'],
    'Glial cell': ['CX3CR1', 'TMEM119', 'P2RY12', 'AIF1', 'GFAP', 'AQP4', 'ALDH1L1', 'SLC1A2', 'MBP', 'MOG', 'PLP1', 'OLIG2'],
    'Immune cell': ['CD79A', 'MS4A1', 'PAX5', 'CD19', 'CD3E', 'CD3G', 'LCK', 'CD8A', 'CD68', 'ADGRE1', 'CSF1R', 'FCGR1A'],
    'Vascular cell': ['PECAM1', 'CDH5', 'FLT1', 'VWF'],
    'Stromal cell': ['COL1A1', 'COL3A1', 'PDGFRA', 'DCN'],
    'Epithelial cell': ['EPCAM', 'KRT18', 'KRT19', 'CDH1', 'KRT14', 'KRT5', 'KRT10', 'IVL'],
    'Stem cell': ['SOX2', 'NANOG', 'POU5F1', 'KIT'],
    'Muscle cell': ['MYOD1', 'MYOG', 'MYH1', 'ACTA1', 'ACTA2', 'MYH11', 'TAGLN', 'CNN1'],
    'Pancreatic cell': ['INS', 'GCG', 'SST', 'PPY'],
    'Liver cell': ['ALB', 'AFP', 'APOB', 'CYP3A4'],
    'Lung cell': ['SFTPA1', 'SFTPB', 'SFTPC', 'SFTPD', 'SCGB1A1'],
    'Kidney cell': ['AQP1', 'AQP2', 'SLC12A3', 'UMOD'],
    'Intestinal cell': ['LGR5', 'OLFM4', 'MUC2', 'CHGA'],
    
    # Aging and senescence markers (keeping these as functional categories)
    'Senescence cell cycle': ['CDKN1A', 'CDKN2A', 'TP53', 'RB1'],
    'SASP cytokines': ['IL6', 'IL1A', 'IL1B', 'CXCL8', 'CXCL1', 'CXCL12', 'CCL2', 'MMP3', 'MMP10', 'SERPINE1'],
    'Senescence structural': ['LMNB1', 'GLB1', 'HMGA2', 'TRF1', 'EZH2'],
    'Apoptosis core': ['CASP3', 'CASP8', 'CASP9', 'BAX', 'BAK1', 'BCL2', 'BCL2L1', 'CYCS', 'FAS', 'FASLG'],
    'Mitochondrial stress': ['SOD2', 'GPX1', 'PRDX1', 'UCP2', 'UCP3', 'NFE2L2'],
    'DNA damage response': ['H2AFX', 'ATM', 'ATR', 'TP53BP1', 'RAD51'],
    'Autophagy/proteostasis': ['SQSTM1', 'MAP1LC3B', 'ATG5', 'ATG7', 'HSPA1A', 'HSP90AA1'],
    'Age-related inflammation': ['TLR2', 'TLR4', 'NLRP3', 'CXCL10', 'TNF', 'IL18'],
    'Stemness decline': ['SOX2', 'POU5F1', 'NANOG', 'KLF4'],
    'Aging markers': ['CDKN1A', 'CDKN2A', 'TP53', 'LMNB1', 'IL6', 'TGFB1', 'CXCL12', 'GDF15']
}

# %%
def plot_marker_violins(adata, markers, expression_source="X", cell_type_column="broad_cell_type", title_suffix="", dir_save=None):
    """
    Plot marker gene violins using different expression sources.
    - Plots every marker gene for each group.
    - Saves plots into subdirectories named after marker groups.
    
    Parameters:
        adata: AnnData object
        expression_source: "X", "log_norm", "scvi_linear", or "scvi_standard"
        title_suffix: string appended to plot title and filename
        dir_save: output directory for saving plots (plots saved only if not None)
    """
    # Plot each group of marker genes
    for group, gene_list in markers.items():
        # Filter marker genes that exist in adata
        valid_genes = [g for g in gene_list if g in adata.var["gene_name"].values]
        
        if not valid_genes:
            print(f"[Warning] No valid genes found for group {group}")
            continue
        
        print(f"[{group}] Valid marker genes: {valid_genes}")
        
        # Make subdirectory if saving
        subdir = None
        if dir_save:
            subdir = os.path.join(dir_save, group.replace(" ", "_").upper())
            os.makedirs(subdir, exist_ok=True)
        
        # Plot each gene
        for gene in valid_genes:
            # Get expression
            if expression_source == "log_norm":
                expr = adata[:, gene].layers['log_norm'].toarray().flatten()
            elif expression_source == "scvi_linear":
                expr = adata.obsm['X_normalized_scVI_linear'][gene]
            elif expression_source == "scvi_standard":
                expr = adata.obsm['X_normalized_scVI_standard'][gene]
            else:  # default to .X
                expr = adata[:, gene].X.toarray().flatten()
            
            df = pd.DataFrame({
                'expression': expr,
                'cell_type': adata.obs[cell_type_column]
            })

            # Filter: non-zero expression
            df = df[df["expression"] != 0]

            # Filter: cell types with sufficient cells
            cell_types_keep = df["cell_type"].value_counts()[df["cell_type"].value_counts() > 10].index.tolist()
            df = df[df["cell_type"].isin(cell_types_keep)]

            if df.empty:
                print(f"[{gene}] Skipped: no valid expression data")
                continue

            # Order cell types by median expression
            median_expr = df.groupby('cell_type', observed=True)['expression'].median().sort_values(ascending=False)
            cell_type_order = median_expr.index.tolist()

            # Plot
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            sns.violinplot(data=df, x='cell_type', y='expression', ax=ax, order=cell_type_order)
            ax.set_title(f'{gene} ({group}) - {expression_source}{title_suffix}')
            ax.tick_params(axis='x', rotation=90)
            ax.set_xlabel('')
            plt.tight_layout()

            # Save if needed
            if subdir:
                filename = f"{gene}_{expression_source}{title_suffix.replace(' ', '_')}.png"
                filepath = os.path.join(subdir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()

# %%
if make_plot:
    print("=== MARKER GENE EXPRESSION COMPARISON ===")
    print("First plot very broad cell types")
    print("1. Using log_norm layer (input to NMF):")
    plot_marker_violins(ge_adata, markers_broad, expression_source="log_norm", cell_type_column="broad_cell_type", title_suffix=" - log_norm layer", dir_save=OUTPUT_DIR)
    print("2. Using linear scVI normalized expression:")
    plot_marker_violins(ge_adata, markers_broad, expression_source = "scvi_linear", cell_type_column="broad_cell_type", title_suffix = " - scVI linear normalized", dir_save=OUTPUT_DIR)

# %%
# 1. via NMF loadings or linear scVI loadings, add top GSEA label or labels for every factor, K by P 
# 2. Associate NMF or scVI factors with cell types... similar to how we look at splicing factors

# Get gene names
gene_names = ge_adata.var["gene_name"].values
print(len(gene_names))

# Directly build the DataFrame
nmf_loadings = pd.DataFrame(
    ge_adata.varm["nmf_standard_mb_components"],  # shape: (genes, components)
    index=gene_names,
    columns=[f"NMF_{i}" for i in range(ge_adata.varm["nmf_standard_mb_components"].shape[1])]
)

# scVI linear loadings are already in gene × factor format
scvi_loadings = pd.DataFrame(
    ge_adata.varm["scVI_linear_gene_loadings"],  # also shape (genes, components)
    index=gene_names,
    columns=[f"Z_{i}" for i in range(ge_adata.varm["scVI_linear_gene_loadings"].shape[1])]
)

# ### Get associations of NMF / scVI loadings with cell types
def plot_factor_heatmap(
    adata,
    factor_key,
    method_label="NMF",
    celltype_col="broad_cell_type",
    output_dir=None,
    filename=None,
    max_factors=None,
    order_factors_by="variance"  # or "mean_abs"
):
    """
    Plot heatmap of mean latent factor activity per cell type.
    Optionally saves the plot to a file if output_dir is provided.
    """
    # Get factor values and cell types
    factors = pd.DataFrame(adata.obsm[factor_key], index=adata.obs.index)
    factors[celltype_col] = adata.obs[celltype_col]
    
    # Compute mean factor value per cell type
    mean_matrix = factors.groupby(celltype_col).mean()
    
    # Optionally restrict to top-k variable factors
    if max_factors:
        if order_factors_by == "variance":
            top_factors = mean_matrix.var(axis=0).sort_values(ascending=False).head(max_factors).index
        elif order_factors_by == "mean_abs":
            top_factors = mean_matrix.abs().mean(axis=0).sort_values(ascending=False).head(max_factors).index
        else:
            raise ValueError("order_factors_by must be 'variance' or 'mean_abs'")
        mean_matrix = mean_matrix[top_factors]
    
    # Create heatmap
    plt.figure(figsize=(10, 6))
    sns.clustermap(mean_matrix, cmap="PRGn", center=0, annot=False, linewidths=0.3, linecolor="gray")
    plt.title(f"Mean {method_label} factor activity per {celltype_col}")
    plt.xlabel("Latent Factors")
    plt.ylabel(celltype_col)
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    # Save if needed
    if output_dir and filename:
        path = os.path.join(output_dir, filename)
        plt.savefig(path, dpi=300)
        print(f"Saved: {path}")
    plt.show()

plot_factor_heatmap(
    ge_adata,
    factor_key="X_nmf_standard_mb",
    method_label="NMF",
    output_dir=OUTPUT_DIR,
    filename="nmf_factor_celltype_heatmap.pdf"
)

plot_factor_heatmap(
    ge_adata,
    factor_key="X_scVI_linear",
    method_label="scVI",
    output_dir=OUTPUT_DIR,
    filename="scvi_factor_celltype_heatmap.pdf"
)

def run_gsea_on_loadings(loadings_df, gene_set="GO_Biological_Process_2021", organism="Mouse"):
    annotations = {}
    for factor in tqdm(loadings_df.columns, desc="Processing factors"):
        try:
            res = gp.prerank(rnk=loadings_df[factor].sort_values(ascending=False), 
                           gene_sets=gene_set, organism=organism, permutation_num=100, 
                           outdir=None, threads=16, min_size=15, max_size=500)
            annotations[factor] = res.res2d[["Term", "ES", "NES", "NOM p-val", "FDR q-val", "Lead_genes", "Tag %", "Gene %"]] if not res.res2d.empty else None
            print(annotations[factor])
        except:
            annotations[factor] = None
    return annotations

# Usage
gsea_nmf = run_gsea_on_loadings(nmf_loadings)
print(f"Done with GSEA on NMF!")
gsea_scvi = run_gsea_on_loadings(scvi_loadings)
print(f"Done with GSEA on scVI!")

# Save full dataframes nmf_loadings, scvi_loadings, gsea_nmf, gsea_scvi
# Save to compressed tsv files 
nmf_loadings.to_csv(os.path.join(OUTPUT_DIR, "nmf_loadings.tsv.gz"), sep="\t", compression="gzip")
scvi_loadings.to_csv(os.path.join(OUTPUT_DIR, "scvi_loadings.tsv.gz"), sep="\t", compression="gzip")

print(f"Done saving GSEA results to {OUTPUT_DIR}")

def build_factor_term_matrix(gsea_results, score_type='NES', qval_threshold=0.05, max_terms=5):
    """
    Build matrix of terms vs factors using ES, NES, or -log10(q-value)
    
    Parameters:
    -----------
    score_type : str
        'ES' = Enrichment Score
        'NES' = Normalized Enrichment Score (recommended)  
        'qval' = -log10(FDR q-value)
    qval_threshold : float
        FDR q-value threshold for significance (default: 0.1)
    max_terms : int
        Maximum number of significant terms per factor (default: 5)
    """
    term_dict = defaultdict(dict)
    for factor, df in gsea_results.items():
        if df is not None:
            # Filter for significant terms first
            sig_df = df[df['FDR q-val'] < qval_threshold]
            
            # If more than max_terms significant, keep top max_terms
            if len(sig_df) > max_terms:
                sig_df = sig_df.head(max_terms)
            
            # Add terms to matrix
            for _, row in sig_df.iterrows():
                if score_type == 'ES':
                    term_dict[row['Term']][factor] = row['ES']
                elif score_type == 'NES':
                    term_dict[row['Term']][factor] = row['NES']
                elif score_type == 'qval':
                    term_dict[row['Term']][factor] = -np.log10(row['FDR q-val'] + 1e-10)
    
    return pd.DataFrame(term_dict).T.fillna(0)

# Build matrices using different scores
nmf_nes_matrix = build_factor_term_matrix(gsea_nmf, score_type='NES')
scvi_nes_matrix = build_factor_term_matrix(gsea_scvi, score_type='NES')

nmf_es_matrix = build_factor_term_matrix(gsea_nmf, score_type='ES')
scvi_es_matrix = build_factor_term_matrix(gsea_scvi, score_type='ES')

# Save all matrices to output directory compressed tsv files 
nmf_nes_matrix.to_csv(os.path.join(OUTPUT_DIR, "nmf_gsea_nes_matrix.tsv"), sep="\t")
scvi_nes_matrix.to_csv(os.path.join(OUTPUT_DIR, "scvi_gsea_nes_matrix.tsv"), sep="\t")

nmf_es_matrix.to_csv(os.path.join(OUTPUT_DIR, "nmf_gsea_es_matrix.tsv"), sep="\t")
scvi_es_matrix.to_csv(os.path.join(OUTPUT_DIR, "scvi_gsea_es_matrix.tsv"), sep="\t")

# %%
def plot_gsea_heatmap(term_matrix, title, filename, output_dir=None, score_type='NES'):
    """
    Plot GSEA results heatmap with proper title based on score type
    """
    # Update title based on score type
    if score_type == 'NES':
        title = title.replace('-log10 adj p', 'NES Scores')
    elif score_type == 'ES':
        title = title.replace('-log10 adj p', 'ES Scores')
    elif score_type == 'qval':
        title = title.replace('NES Scores', '-log10 FDR q-val')
    
    # Create the clustermap without dendrograms
    g = sns.clustermap(
        term_matrix,
        cmap="PRGn",  # Purple-Green colormap
        center=0,     # Center colormap at 0 for NES
        figsize=(7, 12),
        col_cluster=True,
        row_cluster=True,
        xticklabels=True,
        yticklabels=True, 
        dendrogram_ratio=0.1, 
        linewidths=0.1,
        linecolor='gray' 
    )
    
    # Hide the dendrograms completely
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    
    # Rotate x-axis labels
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    
    # Make y-axis labels smaller font size
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=6)
    
    # Save the figure (remove bbox_inches='tight' to avoid the error)
    if output_dir:
        save_path = os.path.join(output_dir, filename)
        g.savefig(save_path, dpi=300)  # Removed bbox_inches='tight'
        print(f"Saved: {save_path}")
    
    plt.show()
    return g

# %%
# Updated plotting calls
plot_gsea_heatmap(
    nmf_nes_matrix,
    title="NMF Factors – GSEA Terms",
    filename="nmf_gsea_nes_clustermap.pdf",
    output_dir=OUTPUT_DIR,
    score_type='NES'
)

plot_gsea_heatmap(
    scvi_nes_matrix,
    title="scVI Factors – GSEA Terms", 
    filename="scvi_gsea_nes_clustermap.pdf",
    output_dir=OUTPUT_DIR,
    score_type='NES'
)

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION
# script=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/GeneExpression/10_GE_vs_GSEA.py
# sbatch --mem=350G -p cpu,bigmem -J "MUS_GE_vs_AGING" --wrap="python $script"