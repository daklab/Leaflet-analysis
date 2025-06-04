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
### Configuration Section ###
#############################

# Input/Output paths
BASE_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
ATSE_ANNDATA_PATH = f"{BASE_DIR}/MODEL_INPUT/052025/MOUSE_SPLICING_FOUNDATION_Anndata_ATSE_counts_with_waypoints_20250513_073829.h5ad"

# Reference gene lists
RBP_FILE_PATH = "/gpfs/commons/groups/knowles_lab/Karin/VanNostrand_2020_supptable1_41586_2020_2077_MOESM3_ESM.xlsx"
AGING_GENES_PATH = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/27857814"

###########################
### Main Analysis Script ##
###########################

# Get param_id and MODEL_OUTPUTS_DIR from command line if provided
if len(sys.argv) > 1:
    param_id = sys.argv[1]
    MODEL_OUTPUTS_DIR = sys.argv[2]
    print(f"Using specified param_id: {param_id}")
    print(f"Using specified MODEL_OUTPUTS_DIR: {MODEL_OUTPUTS_DIR}")

def main():
    print("\n========================================")
    print("LeafletFA Model Analysis - Mouse Splicing Foundation")
    print("========================================\n")
    
    ############################
    # 1. Load Data and Model
    ############################
    print("\n>> Loading data and model...")
    
    splice_adata = ad.read_h5ad(ATSE_ANNDATA_PATH)
    aging_genes = load_aging_genes(AGING_GENES_PATH)
    rbps_df = load_rbp_genes(RBP_FILE_PATH)
        
    model_path = os.path.join(MODEL_OUTPUTS_DIR, f"run_{param_id}", "leafletfa_model.pkl.xz")
    print(f"Using specified model: run_{param_id} with model path {model_path}")
    
    # Create output directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d")
    OUTPUT_DIR = f"/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/results/{timestamp}/param_id_{param_id}"
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
    DATA_DIR = os.path.join(OUTPUT_DIR, "data")
    
    os.makedirs(PLOTS_DIR, exist_ok=True); os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load the model
    leaflet_model = load_model(model_path)

    # Update gene annotations
    splice_adata.var["gene_id"] = splice_adata.var["gene_id"].str.split(".").str[0]
    splice_adata.var["RBP_gene"] = splice_adata.var["gene_name"].isin(rbps_df["mouse_gene_name"])
    splice_adata.var["Aging_gene"] = splice_adata.var["gene_name"].isin(aging_genes)
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

    results = ds.analyze_all_factors_psi(psi_samples, top_junctions=splice_adata.var["junction_id_index"].values, min_effect_size=0.25)
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
    # label junctions as "aging" if they appear in aging genes and RBP if they appear in RBP genes
    # final_df["junction_label"] = final_df["gene_name"].isin(aging_genes) | final_df["gene_name"].isin(rbps_df["mouse_gene_name"])  
    # # Handle cases where a gene is both Aging and RBP
    # final_df.loc[final_df["gene_name"].isin(aging_genes) & final_df["gene_name"].isin(rbps_df["mouse_gene_name"]), "junction_label"] = "Aging+RBP"

    # Create boolean masks for gene types
    is_aging_gene = final_df["gene_name"].isin(aging_genes)
    is_rbp_gene = final_df["gene_name"].isin(rbps_df["mouse_gene_name"])

    # Initialize junction_label with 'Other'
    final_df["junction_label"] = "Other"
    # Set 'RBP' if it's an RBP gene
    final_df.loc[is_rbp_gene, "junction_label"] = "RBP"
    # Set 'Aging' if it's an Aging gene
    final_df.loc[is_aging_gene, "junction_label"] = "Aging"
    # Set 'Aging+RBP' if it's both (this will overwrite 'Aging' or 'RBP' if applicable)
    final_df.loc[is_aging_gene & is_rbp_gene, "junction_label"] = "Aging+RBP"

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

        g.fig.suptitle("Clustermap of Factor Effect Sizes on Top Junctions", fontsize=16, fontweight='bold', y=1.00) # Adjust y for title position

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
    plot_factor_junction_counts(final_df, os.path.join(PLOTS_DIR, "differential_splicing_counts_barplot.png"))
    plot_junction_factor_distribution(final_df, os.path.join(PLOTS_DIR, "differential_splicing_junc_counts_distribution.png"))
    plot_factor_junction_heatmap(final_df, os.path.join(PLOTS_DIR, "differential_splicing_factor_junction_heatmap.png"))

    print("\n========================================")
    print("LeafletFA Model Analysis Completed.")
    print("========================================\n")

if __name__ == "__main__":
    main()

# For sbatch execution:
# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025
# sbatch --mem=250G -p dev,cpu --wrap="python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/downstream_analysis/03_differential_splicing_analysis.py"