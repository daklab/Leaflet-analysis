#!/usr/bin/env python
"""
LeafletFA Advanced Analysis - Mouse Splicing Foundation

This script performs extended analysis on LeafletFA model results:
1. Regression analysis of factors vs aging, cell type, and tissue
2. Correlation structure analysis across factors
3. Integration with gene expression data (NMF/scVI features)
4. Variance explained by different variables (ANOVA-based analysis)
5. RBP gene expression correlation with splicing factors
"""

import os
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
from sklearn.decomposition import PCA, NMF
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from mord import OrdinalRidge
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tqdm import tqdm
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# import LeafletFA differential splicing code
# Define module paths
src_path = "/gpfs/commons/home/kisaev/Leaflet-private/src/"

# Add to sys.path if not already present
if src_path not in sys.path:
    sys.path.append(src_path)

# Import custom modules
import BetaDirichletFactor.differential_splicing as ds

# Import utility functions - simple direct import
sys.path.append('/gpfs/commons/home/kisaev/Leaflet-analysis/Multi_Species_Splicing_Foundation/shared_utils/')
from utils import *

# Check if using CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#############################
#### Functions Section ####
#############################

def plot_individual_factor_junction_analysis(final_df, results_dir, top_n=30):
    """
    Create individual plots for each factor showing their effect on top junctions.
    
    Args:
        final_df: DataFrame with columns ['junction_id_index', 'gene_name', 'junction_label', 'factor_idx', 'effect_size']
        results_dir: Base results directory
        top_n: Number of top junctions to show per factor
    """
    # Create DS_FACTORS directory
    ds_factors_dir = os.path.join(results_dir, "DS_FACTORS")
    os.makedirs(ds_factors_dir, exist_ok=True)
    print(f"Creating individual factor plots in: {ds_factors_dir}")
    
    # Junction type color mapping
    label_colors_map = {"Aging": "gold", "RBP": "lightgreen", "Aging+RBP": "orange", "Other": "lightgray"}
    
    # Get unique factors
    unique_factors = sorted(final_df["factor_idx"].unique())
    print(f"Found {len(unique_factors)} factors to analyze")
    
    for factor_idx in unique_factors:
        
        print(f"  Processing factor_{factor_idx}...")
        
        # Filter data for this factor
        factor_df = final_df[final_df["factor_idx"] == factor_idx].copy()
        
        if factor_df.empty:
            print(f"    No data for factor_{factor_idx}, skipping...")
            continue
        
        # Get top N junctions by absolute effect size for this factor
        factor_df["abs_effect_size"] = factor_df["effect_size"].abs()
        top_junctions = factor_df.nlargest(top_n, "abs_effect_size")
        
        if top_junctions.empty:
            print(f"    No significant junctions for factor_{factor_idx}, skipping...")
            continue
        
        # Create publication-ready plot with 3 subplots
        # Calculate height based on number of junctions (min 6", max 12")
        fig_height = max(6, min(12, 4 + top_n * 0.25))
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, fig_height), 
                                            gridspec_kw={'width_ratios': [3.5, 1.25, 1.25]})
        
        # Set publication-ready style
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14
        })
        
        # ---- Left plot: Horizontal bar chart of effect sizes ----
        
        # Prepare data for plotting
        plot_data = top_junctions.copy()
        plot_data["display_label"] = plot_data.apply(
            lambda row: f"{row['gene_name']} (J{row['junction_id_index']})", axis=1
        )
        
        # Sort by effect size for better visualization
        plot_data = plot_data.sort_values("effect_size", ascending=True)
        
        # Create color array based on junction labels
        colors = [label_colors_map.get(label, "lightgray") for label in plot_data["junction_label"]]
        
        # Create horizontal bar plot with publication styling
        bars = ax1.barh(range(len(plot_data)), plot_data["effect_size"], 
                       color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)
        
        # Customize left plot with publication standards
        ax1.set_yticks(range(len(plot_data)))
        ax1.set_yticklabels(plot_data["display_label"], fontsize=8)
        ax1.set_xlabel("Effect Size", fontsize=11, fontweight='bold')
        ax1.set_title(f"Factor {factor_idx}: Top {len(plot_data)} Affected Junctions", 
                     fontsize=12, fontweight='bold', pad=15)
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=1.0)
        ax1.grid(axis='x', alpha=0.4, linewidth=0.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_linewidth(0.8)
        ax1.spines['bottom'].set_linewidth(0.8)
        
        # Add effect size values on bars with better formatting
        for i, (bar, effect) in enumerate(zip(bars, plot_data["effect_size"])):
            width = bar.get_width()
            x_pos = width + (0.02 if width >= 0 else -0.02)
            ha = 'left' if width >= 0 else 'right'
            ax1.text(x_pos, bar.get_y() + bar.get_height()/2, f'{effect:.3f}', 
                    ha=ha, va='center', fontsize=7, fontweight='normal')
        
        # ---- Right plot 1: Junction type distribution ----
        
        # Count junction types
        type_counts = plot_data["junction_label"].value_counts()
        
        # Create publication-ready pie chart
        colors_pie = [label_colors_map.get(label, "lightgray") for label in type_counts.index]
        wedges, texts, autotexts = ax2.pie(type_counts.values, labels=type_counts.index, 
                                          colors=colors_pie, autopct='%1.1f%%', 
                                          startangle=90, textprops={'fontsize': 8},
                                          wedgeprops={'linewidth': 0.8, 'edgecolor': 'white'})
        
        # Style the percentage labels
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(8)
        
        ax2.set_title(f"Junction Type\n(n={len(plot_data)})", 
                     fontsize=10, fontweight='bold', pad=10)
        
        # ---- Right plot 2: Junction annotation distribution ----
        
        # Count junction annotations (if the column exists)
        if 'junction_annotation' in plot_data.columns:
            annotation_counts = plot_data["junction_annotation"].value_counts()
            
            # Create publication-ready color palette for annotations
            n_annotations = len(annotation_counts)
            annotation_colors = plt.cm.Set3(np.linspace(0, 1, n_annotations))
            
            # Create pie chart for annotations with publication styling
            wedges3, texts3, autotexts3 = ax3.pie(annotation_counts.values, 
                                                  labels=annotation_counts.index, 
                                                  colors=annotation_colors, 
                                                  autopct='%1.1f%%', 
                                                  startangle=90, 
                                                  textprops={'fontsize': 7},
                                                  wedgeprops={'linewidth': 0.8, 'edgecolor': 'white'})
            
            # Style the percentage labels
            for autotext in autotexts3:
                autotext.set_color('black')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(7)
            
            ax3.set_title(f"Junction Annotation\n(n={len(plot_data)})", 
                         fontsize=10, fontweight='bold', pad=10)
        else:
            # If junction_annotation column doesn't exist, show a clean message
            ax3.text(0.5, 0.5, 'Junction annotation\ndata not available', 
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', 
                             alpha=0.3, edgecolor='gray', linewidth=0.8))
            ax3.set_title("Junction Annotation", fontsize=10, fontweight='bold', pad=10)
            ax3.set_xlim(-1, 1)
            ax3.set_ylim(-1, 1)
            ax3.axis('off')
                
        # ---- Create publication-ready legend ----
        
        # Create legend for junction types
        legend_handles = [plt.Rectangle((0,0),1,1, color=color, label=label, 
                                       edgecolor='black', linewidth=0.5) 
                         for label, color in label_colors_map.items() 
                         if label in plot_data["junction_label"].values]
        
        if legend_handles:  # Only create legend if there are items
            fig.legend(handles=legend_handles, 
                      title="Junction Type", 
                      loc="lower center", 
                      bbox_to_anchor=(0.5, -0.02),
                      ncol=min(4, len(legend_handles)),
                      frameon=True, 
                      fontsize=9, 
                      title_fontsize=10,
                      edgecolor='black',
                      fancybox=True,
                      shadow=True)
        
        # ---- Save publication-ready plot ----
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.97])  # Better spacing for publication
        
        output_path = os.path.join(ds_factors_dir, f"factor_{factor_idx}_junction_analysis.pdf")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"    ✓ Saved: factor_{factor_idx}_junction_analysis.pdf")
            
        # ---- Make junction-specific clustermap ----
            
        print(f"    Creating junction-specific clustermap for factor_{factor_idx}...")
            
        # Get the top junction IDs for this factor
        top_junction_ids = top_junctions["junction_id_index"].unique()
            
        # Filter final_df to get ALL factor effects on these specific junctions
        junction_specific_df = final_df[final_df["junction_id_index"].isin(top_junction_ids)].copy()
            
        if not junction_specific_df.empty and len(junction_specific_df["factor_idx"].unique()) > 1:
            # Create pivot table: rows=junctions, columns=factors, values=effect_size
            clustermap_data = junction_specific_df.pivot_table(
                index=["junction_id_index", "gene_name", "junction_label"],
                columns="factor_idx",
                values="effect_size",
                fill_value=0
            )
            
            # Rename columns to include "factor_" prefix
            clustermap_data.columns = [f"factor_{col}" for col in clustermap_data.columns]
            
            # Prepare row colors based on junction_label
            junction_types_for_colors = clustermap_data.index.get_level_values("junction_label")
            row_color_series = junction_types_for_colors.map(label_colors_map).rename("Junction Type")
            row_colors_df = row_color_series.to_frame()
            
            # Prepare display labels for y-axis
            display_row_labels = [f"{gene} (J{jid})" for jid, gene, _ in clustermap_data.index]
            clustermap_data_display = clustermap_data.copy()
            clustermap_data_display.index = display_row_labels
            row_colors_df.index = display_row_labels
            
            # Create the clustermap
            fig_height = max(8, min(16, 4 + len(top_junction_ids) * 0.4))
            fig_width = max(10, min(20, 6 + len(clustermap_data.columns) * 0.3))
            
            g = sns.clustermap(
                clustermap_data_display,
                row_colors=row_colors_df,
                cmap="RdBu_r",
                center=0,
                annot=False,
                linewidths=0.3,
                figsize=(fig_width, fig_height),
                cbar_kws={'label': 'Effect Size'},
                xticklabels=True,
                yticklabels=True
            )
            
            # Highlight the current factor column
            current_factor_col = f"factor_{factor_idx}"
            if current_factor_col in clustermap_data_display.columns:
                # Find the position of the current factor in the reordered columns
                col_order = g.dendrogram_col.reordered_ind
                reordered_cols = clustermap_data_display.columns[col_order]
                if current_factor_col in reordered_cols:
                    highlight_pos = list(reordered_cols).index(current_factor_col)
                    
                    # Add a thick border around the highlighted column
                    ax_heatmap = g.ax_heatmap
                    ax_heatmap.axvline(x=highlight_pos, color='yellow', linewidth=3, alpha=0.8)
                    ax_heatmap.axvline(x=highlight_pos + 1, color='yellow', linewidth=3, alpha=0.8)
            
            # Customize the plot
            plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=9)
            plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
            
            # Add title
            g.fig.suptitle(f'All Factor Effects on Top {len(top_junction_ids)} Junctions from Factor {factor_idx}\\n(Yellow highlight = Factor {factor_idx})', 
                          fontsize=12, fontweight='bold', y=0.98)
            
            # Create legend for junction types
            legend_handles = [plt.Rectangle((0,0),1,1, color=color, label=label) 
                             for label, color in label_colors_map.items() 
                             if label in junction_types_for_colors.values]
            
            g.fig.legend(handles=legend_handles, 
                        title="Junction Type", 
                        loc="upper left", 
                        bbox_to_anchor=(0.02, 0.95),
                        frameon=True, 
                        fontsize=9, 
                        title_fontsize=10)
            
            # Save the clustermap
            clustermap_path = os.path.join(ds_factors_dir, f"factor_{factor_idx}_junction_clustermap.pdf")
            g.savefig(clustermap_path, dpi=300, bbox_inches='tight')
            plt.close(g.fig)
            
            print(f"    ✓ Saved junction clustermap: factor_{factor_idx}_junction_clustermap.pdf")
        else:
            print(f"    ⚠️ Insufficient data for clustermap (factor_{factor_idx})")
        
        print(f"    ✓ Completed analysis for factor_{factor_idx}")
        print()
        
    return ds_factors_dir

###########################
### Main Analysis Script ##
###########################

# Get param_id and MODEL_OUTPUTS_DIR from command line if provided
if len(sys.argv) > 1:
    param_id = sys.argv[1]
    MODEL_OUTPUTS_DIR = sys.argv[2]
    ATSE_ANNDATA_PATH = sys.argv[3]
    ATSE_FILE_PATH = sys.argv[4]
    GE_ANNDATA_scVI_PATH = sys.argv[5]
    GE_ANNDATA_NMF_PATH = sys.argv[6]
    AGING_GENES_PATH = sys.argv[7]
    RBP_FILE_PATH = sys.argv[8]
    OUTPUT_DIR = sys.argv[9]
    print(f"Using specified param_id: {param_id}")
    print(f"Using specified MODEL_OUTPUTS_DIR: {MODEL_OUTPUTS_DIR}")
    print(f"Using specified ATSE_ANNDATA_PATH: {ATSE_ANNDATA_PATH}")
    print(f"Using specified ATSE_FILE_PATH: {ATSE_FILE_PATH}")
    print(f"Using specified GE_ANNDATA_scVI_PATH: {GE_ANNDATA_scVI_PATH}")
    print(f"Using specified GE_ANNDATA_NMF_PATH: {GE_ANNDATA_NMF_PATH}")
    print(f"Using specified AGING_GENES_PATH: {AGING_GENES_PATH}")
    print(f"Using specified RBP_FILE_PATH: {RBP_FILE_PATH}")
    print(f"Using specified OUTPUT_DIR: {OUTPUT_DIR}")
def main():
    print("\n========================================")
    print("LeafletFA Model Analysis - Mouse Splicing Foundation")
    print("========================================\n")
    
    ############################
    # 1. Load Data and Model
    ############################
    print("\n>> Loading data and model...")
    
    splice_adata = ad.read_h5ad(ATSE_ANNDATA_PATH)
    ge_adata = ad.read_h5ad(GE_ANNDATA_scVI_PATH)
    ge_adata_nmf = ad.read_h5ad(GE_ANNDATA_NMF_PATH)

    # Load aging gene lists
    aging_genes_mouse, aging_genes_human = load_aging_genes(AGING_GENES_PATH)
    
    # Load RBP genes
    rbps = load_rbp_genes(RBP_FILE_PATH)

    # if "mouse.id" is in splice_adata.obs rename it to donor_id 
    if "mouse.id" in splice_adata.obs.columns:
        print(f"Renaming mouse.id to donor_id in splice_adata.obs")
        splice_adata.obs.rename(columns={"mouse.id": "donor_id"}, inplace=True)
        # Update gene annotations
        splice_adata.var["RBP_gene"] = splice_adata.var["gene_name"].isin(rbps["mouse_gene_name"])
        splice_adata.var["Aging_gene"] = splice_adata.var["gene_name"].isin(aging_genes_mouse)
        ge_adata.var["RBP_gene"] = ge_adata.var["gene_name"].isin(rbps["mouse_gene_name"])
        ge_adata.var["Aging_gene"] = ge_adata.var["gene_name"].isin(aging_genes_mouse)
        aging_genes = aging_genes_mouse
        rbps = rbps["mouse_gene_name"]
    
    else:
        splice_adata.var["RBP_gene"] = splice_adata.var["gene_name"].isin(rbps["gene_name"]) # when running with Human data... 
        splice_adata.var["Aging_gene"] = splice_adata.var["gene_name"].isin(aging_genes_human)
        ge_adata.var["RBP_gene"] = ge_adata.var["gene_name"].isin(rbps["gene_name"])
        ge_adata.var["Aging_gene"] = ge_adata.var["gene_name"].isin(aging_genes_human)
        aging_genes = aging_genes_human
        rbps = rbps["gene_name"]
    
    splice_adata.var["gene_id"] = splice_adata.var["gene_id"].str.split(".").str[0]

    assert np.all(ge_adata.obs_names == ge_adata_nmf.obs_names), "Cell IDs in ge_adata and ge_adata_nmf do not match or are not in the same order."
    assert np.all(ge_adata.var_names == ge_adata_nmf.var_names), "Gene names in ge_adata and ge_adata_nmf do not match or are not in the same order."
    ge_adata.obsm["X_nmf_standard_mb"] = ge_adata_nmf.obsm["X_nmf_standard_mb"]
    ge_adata.varm["nmf_standard_mb_components"] = ge_adata_nmf.varm["nmf_standard_mb_components"]

    # Load ATSE file
    atse_df = pd.read_csv(ATSE_FILE_PATH, sep="\t")
    atse_df = atse_df[["junction_id", "perfect_match_5_prime", "perfect_match_3_prime"]]
    atse_df["junction_annotation"] = "Novel_SS" 
    # If perfect_match_5_prime is nonempty then set junction_annotation to "5_prime_annotated"
    atse_df.loc[atse_df["perfect_match_5_prime"].notna(), "junction_annotation"] = "5_prime_annotated"
    # If perfect_match_3_prime is nonempty then set junction_annotation to "3_prime_annotated"
    atse_df.loc[atse_df["perfect_match_3_prime"].notna(), "junction_annotation"] = "3_prime_annotated"
    # If perfect_match_5_prime and perfect_match_3_prime are both nonempty then set junction_annotation to "Both_SS_annotated"
    atse_df.loc[atse_df["perfect_match_5_prime"].notna() & atse_df["perfect_match_3_prime"].notna(), "junction_annotation"] = "Both_SS_annotated"
    
    model_path = os.path.join(MODEL_OUTPUTS_DIR, f"run_{param_id}", "leafletfa_model.pkl.xz")
    print(f"Using specified model: run_{param_id} with model path {model_path}")
    
    # Create output directory
    from datetime import datetime
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
    DATA_DIR = os.path.join(OUTPUT_DIR, "data")
    
    os.makedirs(PLOTS_DIR, exist_ok=True); os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load the model
    leaflet_model = load_model(model_path)

    # Update gene annotations
    splice_adata.var["gene_id"] = splice_adata.var["gene_id"].str.split(".").str[0]
    print(f"   ✓ Data loaded successfully")
    
    ############################
    # 2. Extract Model Parameters
    ############################

    print("\n>> Extracting model parameters...")
    PHI = leaflet_model["assign_post"]
    K_factors_model = leaflet_model["K"]
    print(f"   ✓ Extracted {K_factors_model} factors from the model")

    splice_adata.obsm["X_PHI"] = PHI
    PSI_CELLS = np.dot(PHI, leaflet_model["psi_learned"])
    splice_adata.layers["PSI_CELLS"] = PSI_CELLS
    psi_samples = leaflet_model["psi_samples"]

    ############################
    # 3. Run analysis functions 
    ############################

    print("\n>> Running analysis functions...")

    results = ds.analyze_all_factors_psi(psi_samples, top_junctions=splice_adata.var["junction_id_index"].values, min_effect_size=0.2)
    all_results = []

    for factor_idx, (effect_size_list, significance_df) in results.items():
        # Append to list
        all_results.append(significance_df)

    # Concatenate all results into a single DataFrame
    final_df = pd.concat(all_results, ignore_index=True)
    final_df["junction_id_index"] = final_df["junction_idx"]

    # convert splice_adata.var["junction_id_index"] to dtype int64
    splice_adata.var["junction_id_index"] = splice_adata.var["junction_id_index"].astype("int64")
    # Merge with adata.var using junction_id_index
    final_df = final_df.merge(splice_adata.var, on="junction_id_index")
    # filter just significant junctions
    final_df = final_df[final_df["significant"]]

    # Create boolean masks for gene types
    is_aging_gene = final_df["gene_name"].isin(aging_genes)
    is_rbp_gene = final_df["gene_name"].isin(rbps)

    # Initialize junction_label with 'Other'
    final_df["junction_label"] = "Other"
    final_df.loc[is_rbp_gene, "junction_label"] = "RBP"
    final_df.loc[is_aging_gene, "junction_label"] = "Aging"
    # Set 'Aging+RBP' if it's both (this will overwrite 'Aging' or 'RBP' if applicable)
    final_df.loc[is_aging_gene & is_rbp_gene, "junction_label"] = "Aging+RBP"
    final_df = final_df.merge(atse_df, on="junction_id")
    print(final_df.junction_label.value_counts())
    # save final_df
    final_df.to_csv(os.path.join(DATA_DIR, "differential_splicing_results.csv"), index=False)

    ############################
    # 4. Visualization Section
    ############################

    print("\n>> Running visualization functions...")

    # VISUALIZATION 1: Barplot of significant junctions per factor
    def plot_factor_junction_counts(final_df, output_path):
        """Create an improved barplot showing number of significant junctions per factor"""
        # Get counts of junctions/factor
        counts = final_df.groupby("factor_idx")["junction_id_index"].nunique()
        counts_df = pd.DataFrame({"factor_idx": counts.index, "num_sig_junctions": counts.values})
        counts_df = counts_df.sort_values(by="num_sig_junctions", ascending=False)

        # Add "factor_" prefix
        counts_df["factor_name"] = "factor_" + counts_df["factor_idx"].astype(str)

        # Create figure with appropriate size
        plt.figure(figsize=(12, 6))

        # Create barplot with appealing color palette
        ax = sns.barplot(x="factor_name", y="num_sig_junctions", data=counts_df, palette="viridis")

        # Add count labels on top of bars
        for i, v in enumerate(counts_df["num_sig_junctions"]):
            ax.text(i, v + 0.5, str(v), ha='center', fontweight='bold')

        # Add title and labels with better formatting
        plt.title("Number of Significant Junctions per Factor", fontsize=14, fontweight='bold')
        plt.xlabel("Factor", fontsize=12)
        plt.ylabel("Number of significant junctions", fontsize=12)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        # Adjust layout
        plt.tight_layout()

        # Save figure with higher DPI
        plt.savefig(output_path, dpi=300)
        plt.close()

    def plot_factor_junction_annotation_counts(final_df, output_path):
        """
        Create a stacked barplot showing the number of significant junctions per factor,
        colored by junction annotation category.
        """

        # Count unique junctions by (factor, annotation)
        counts = (
            final_df.groupby(["factor_idx", "junction_annotation"])["junction_id_index"]
            .nunique()
            .reset_index(name="count")
        )

        # Add factor name column
        counts["factor_name"] = "factor_" + counts["factor_idx"].astype(str)

        # Pivot to wide format: rows=factor_name, columns=annotation, values=count
        pivot_df = counts.pivot(index="factor_name", columns="junction_annotation", values="count").fillna(0)

        # print table
        print(pivot_df)

        # Sort by total number of junctions
        pivot_df["total"] = pivot_df.sum(axis=1)
        pivot_df = pivot_df.sort_values("total", ascending=False)
        pivot_df = pivot_df.drop(columns="total")

        # Plot
        plt.figure(figsize=(14, 6))
        pivot_df.plot(
            kind="bar",
            stacked=True,
            colormap="viridis",  # Or use sns.color_palette("Set2", n_colors=...)
            edgecolor="black"
        )

        plt.title("Number of Significant Junctions per Factor by Annotation", fontsize=14, fontweight="bold")
        plt.xlabel("Factor", fontsize=12)
        plt.ylabel("Number of Significant Junctions", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title="Junction Annotation", fontsize=9, title_fontsize=10, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()

        # Save
        plt.savefig(output_path, dpi=300)
        plt.close()


    # VISUALIZATION 2: Distribution of factors per junction
    def plot_junction_factor_distribution(final_df, output_path):
        """Create an improved histogram showing distribution of factors per junction"""
        # Get counts of factors/junction
        junc_counts = final_df.groupby("junction_id_index")["factor_idx"].nunique()
        junc_counts_df = pd.DataFrame({"junction_id_index": junc_counts.index, "num_factors": junc_counts.values})

        # Create figure with appropriate size
        plt.figure(figsize=(10, 6))

        # Create histogram with better styling
        ax = sns.histplot(junc_counts_df["num_factors"], bins=range(1, junc_counts_df["num_factors"].max() + 2), 
                         kde=True, color='steelblue', edgecolor='darkblue', alpha=0.7)

        # Add mean line
        mean_factors = junc_counts_df["num_factors"].mean()
        plt.axvline(mean_factors, color='red', linestyle='--', linewidth=2)
        plt.text(mean_factors + 0.2, plt.ylim()[1]*0.9, f'Mean: {mean_factors:.2f}', 
                 color='red', fontweight='bold')

        # Add title and labels with better formatting
        plt.title("Distribution of Factors per Junction (Significant Only)", fontsize=14, fontweight='bold')
        plt.xlabel("Number of factors affecting junction", fontsize=12)
        plt.ylabel("Count of junctions", fontsize=12)

        # Add grid for better readability
        plt.grid(axis='y', alpha=0.3)

        # Adjust layout
        plt.tight_layout()

        # Save figure with higher DPI
        plt.savefig(output_path, dpi=300)
        plt.close()

    # VISUALIZATION 3: Heatmap of factor effects on top N junctions
    def plot_factor_junction_heatmap(final_df, output_path, top_n=50):
        """Create a clustermap showing the effect sizes of factors on top junctions, with junction type colors."""
        # Get junctions that are affected by the most factors
        top_junction_ids = final_df.groupby("junction_id_index")["factor_idx"].nunique().sort_values(ascending=False).head(top_n).index

        # Filter dataframe to only include top junctions
        heatmap_df = final_df[final_df["junction_id_index"].isin(top_junction_ids)]

        if heatmap_df.empty:
            print(f"No data to plot for clustermap for top {top_n} junctions. Skipping clustermap.")
            plt.figure(figsize=(10,4))
            plt.text(0.5, 0.5, f"No significant junctions found for top {top_n} to display in clustermap.",
                     ha='center', va='center', fontsize=12)
            plt.axis('off')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return

        # Create a pivot table for the heatmap data
        # Index: junction_id_index, gene_name, junction_label
        # Columns: factor_idx
        # Values: effect_size
        data_matrix = heatmap_df.pivot_table(
            index=["junction_id_index", "gene_name", "junction_label"],
            columns="factor_idx",
            values="effect_size",
            fill_value=0
        )
        data_matrix.columns = [f"factor_{col}" for col in data_matrix.columns] # Rename factor columns

        # Prepare row colors based on junction_label
        label_colors_map = {"Aging": "gold", "RBP": "lightgreen", "Aging+RBP": "orange", "Other": "lightgray"}
        junction_types_for_colors = data_matrix.index.get_level_values("junction_label")
        row_color_series = junction_types_for_colors.map(label_colors_map).rename("Junction Type")
        # Ensure row_color_series is a DataFrame for clustermap (even if one column)
        row_colors_df = row_color_series.to_frame()

        # Prepare display labels for y-axis (rows) of the heatmap
        display_row_labels = [f"{gene} (J{jid})" for jid, gene, _ in data_matrix.index]
        
        # Create a new data matrix with these display labels as index for clustermap
        data_matrix_for_display = data_matrix.copy()
        data_matrix_for_display.index = display_row_labels
        
        # Align row_colors_df index with data_matrix_for_display.index
        row_colors_df.index = display_row_labels

        # Determine appropriate figure height based on number of junctions
        # Min height 8, add 0.35 inch per junction beyond a small number
        fig_height = max(8, 4 + top_n * 0.35)
        fig_width = 12 # Keep width somewhat constant or adjust based on num factors

        # Create clustermap
        g = sns.clustermap(
            data_matrix_for_display,
            row_colors=row_colors_df,
            cmap="RdBu_r",  # Red-Blue diverging colormap
            center=0,      # Center colormap at 0 for effect sizes
            annot=False,   # Annotations can be cluttered; keep False
            fmt=".2f",
            linewidths=0.5,
            figsize=(fig_width, fig_height),
            cbar_kws={'label': 'Effect Size', 'orientation': 'vertical'},
            metric='correlation', # Distance metric for clustering
            method='average',     # Linkage method for clustering
            xticklabels=True,
            yticklabels=True      # Ensure yticklabels are on
        )

        # Customize the plot
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
        # Y-tick labels are taken from data_matrix_for_display.index
        # g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8) # Adjust fontsize if needed

        # Create legend for junction types (row colors)
        handles = [plt.Rectangle((0,0),1,1, color=color, label=label) 
                   for label, color in label_colors_map.items()]
        
        # Place legend outside the main clustermap area
        # Adjust bbox_to_anchor: (x, y) specifies the location of 'loc'
        # (e.g., x=1 means right edge of the plot area before adjustment)
        legend_x_position = 1.01 # Start slightly to the right of the heatmap/dendrograms
        
        # Adjust legend position dynamically based on presence of col dendrogram
        if g.ax_col_dendrogram.get_visible():
             legend_x_position = g.ax_col_dendrogram.get_position().x1 + 0.02
        elif g.ax_heatmap.get_visible():
             legend_x_position = g.ax_heatmap.get_position().x1 + 0.02

        # If there are many factors, the factor labels might take up space.
        # The legend for row colors should ideally be to the right of the row dendrogram and row colors.
        # The position of g.ax_row_colors might be a good reference.
        # Bbox for legend: (x0, y0, width, height) in figure coordinates
        # Or using bbox_to_anchor with loc.
        # Let's try to anchor it relative to the figure.
        legend_pos_x = g.ax_row_dendrogram.get_position().x0 - 0.1 # Left of row dendrogram
        if legend_pos_x < 0.01 : legend_pos_x = 0.01 # clamp

        g.fig.legend(handles=handles, 
                     labels=label_colors_map.keys(),
                     title="Junction Type", 
                     loc="upper left", # Use a corner for loc
                     bbox_to_anchor=(legend_pos_x, g.ax_heatmap.get_position().y1), # Position it near top-left of plot elements
                     frameon=True, fontsize=10, title_fontsize=11)

        # Adjust layout to make space for legend and title
        g.fig.tight_layout(rect=[0, 0, 0.95, 0.96]) # rect=[left, bottom, right, top] to make space

        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(g.fig)


    # Execute all visualizations
    plot_factor_junction_counts(final_df, os.path.join(PLOTS_DIR, "differential_splicing_counts_barplot.pdf"))
    plot_factor_junction_annotation_counts(final_df, os.path.join(PLOTS_DIR, "differential_splicing_annotation_counts_barplot.pdf"))
    plot_junction_factor_distribution(final_df, os.path.join(PLOTS_DIR, "differential_splicing_junc_counts_distribution.pdf"))
    plot_factor_junction_heatmap(final_df, os.path.join(PLOTS_DIR, "differential_splicing_factor_junction_heatmap.pdf"))

    # Assuming your data is in final_df with required columns
    ds_factors_dir = plot_individual_factor_junction_analysis(
        final_df=final_df,
        results_dir=PLOTS_DIR,
        top_n=50  # or however many top junctions you want
    )

    print("\n========================================")
    print("LeafletFA Model Analysis Completed.")
    print("========================================\n")

if __name__ == "__main__":
    main()

# For sbatch execution:
# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025
# sbatch --mem=250G -p dev,cpu --wrap="python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/downstream_analysis/03_differential_splicing_analysis.py"