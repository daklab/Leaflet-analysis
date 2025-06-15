import scanpy as sc
import anndata as ad
import matplotlib.cm as cm
import os
import numpy as np
import matplotlib.pyplot as plt

# Input/Output paths
BASE_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"

# Gene expression data
GE_ANNDATA_scVI_PATH = f"{BASE_DIR}/scVI/ge_adata_with_both_scvi_models_2025-05-13.h5ad"
GE_ANNDATA_NMF_PATH = f"{BASE_DIR}/NMF/ge_adata_with_NMF_models_2025-05-16.h5ad"
PLOTS_DIR = "/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/results/gene_expression/plots"
# if doesn't exist, create it
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

# Load data
print(f"Loading scVI data from {GE_ANNDATA_scVI_PATH}")
ge_adata = ad.read_h5ad(GE_ANNDATA_scVI_PATH)
print(f"Loading NMF data from {GE_ANNDATA_NMF_PATH}")
ge_adata_nmf = ad.read_h5ad(GE_ANNDATA_NMF_PATH)

assert np.all(ge_adata.obs_names == ge_adata_nmf.obs_names), "Cell IDs in ge_adata and ge_adata_nmf do not match or are not in the same order."
assert np.all(ge_adata.var_names == ge_adata_nmf.var_names), "Gene names in ge_adata and ge_adata_nmf do not match or are not in the same order."
ge_adata.obsm["X_nmf_standard_mb"] = ge_adata_nmf.obsm["X_nmf_standard_mb"]
ge_adata.varm["nmf_standard_mb_components"] = ge_adata_nmf.varm["nmf_standard_mb_components"]
print(f"Obtained one Gene Expression Object with both NMF and scVI models")
print(ge_adata)

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scanpy as sc

def plot_umap(ge_adata, rep_input, variable_name, PLOTS_DIR, num_groups=None):
    print(f"Generating UMAPs for {rep_input} and {variable_name}...")

    # Identify top N groups by frequency
    if num_groups is not None:
        top_groups = ge_adata.obs[variable_name].value_counts().head(num_groups).index.tolist()
        print(f"The top {num_groups} groups are: {top_groups}")
    else:
        top_groups = ge_adata.obs[variable_name].unique().tolist()
        num_groups = len(top_groups)
        print(f"The unique groups are: {top_groups}")

    # Create a simplified group column
    ge_adata.obs['group_highlighted'] = 'Other'
    ge_adata.obs.loc[ge_adata.obs[variable_name].isin(top_groups), 'group_highlighted'] = \
        ge_adata.obs[variable_name]

    # Assign colors using tab20
    cmap = cm.get_cmap('tab20', len(top_groups))
    colors = [cmap(i) for i in range(len(top_groups))]
    color_dict = {group: colors[i] for i, group in enumerate(top_groups)}
    color_dict['Other'] = [0.9, 0.9, 0.9, 1.0]  # very light gray for background

    # Plot UMAP
    plt.figure(figsize=(8, 5))
    sc.pl.umap(
        ge_adata,
        color='group_highlighted',
        palette=color_dict,
        show=False,
        frameon=True,
        legend_fontsize=10,
        legend_loc='right margin'
    )
    plt.title(f'UMAP by {variable_name} (Top {num_groups} Highlighted)')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(
        os.path.join(PLOTS_DIR, f"umap_group_{variable_name}_{rep_input}_top{num_groups}.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

# NMF
# Generate UMAP from provided latent space
print("Generating UMAP from NMF standard...")
sc.pp.neighbors(ge_adata, use_rep="X_nmf_standard_mb", n_neighbors=8)
sc.tl.umap(ge_adata)
plot_umap(ge_adata, "X_nmf_standard_mb", "broad_cell_type", PLOTS_DIR, num_groups=10)
plot_umap(ge_adata, "X_nmf_standard_mb", "dataset", PLOTS_DIR)

# scVI
print("Generating UMAP from scVI linear...")
sc.pp.neighbors(ge_adata, use_rep="X_scVI_linear", n_neighbors=8)
sc.tl.umap(ge_adata)
plot_umap(ge_adata, "X_scVI_linear", "broad_cell_type", PLOTS_DIR, num_groups=10)
plot_umap(ge_adata, "X_scVI_linear", "dataset", PLOTS_DIR)

# scVI standard
print("Generating UMAP from scVI standard...")
sc.pp.neighbors(ge_adata, use_rep="X_scVI_standard", n_neighbors=8)
sc.tl.umap(ge_adata)
plot_umap(ge_adata, "X_scVI_standard", "broad_cell_type", PLOTS_DIR, num_groups=10)
plot_umap(ge_adata, "X_scVI_standard", "dataset", PLOTS_DIR)

# to submit 
# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION
# script=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/GeneExpression/08_visualize_scVI_NMF.py
# sbatch --mem=350G -p dev,cpu,bigmem -J "MF_GE_UMAPs" --wrap="python $script"
