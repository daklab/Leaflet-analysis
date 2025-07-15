import scanpy as sc
import anndata as ad
import matplotlib.cm as cm
import os
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scanpy as sc

# Input/Output paths
BASE_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"

# Gene expression data
GE_ANNDATA_scVI_PATH = f"{BASE_DIR}/scVI/ge_adata_with_both_scvi_models_2025-07-06.h5ad"
GE_ANNDATA_NMF_PATH = f"{BASE_DIR}/NMF/ge_adata_with_NMF_standard_30_1024_2025-07-06.h5ad"
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

def plot_umap(ge_adata, rep_input, variable_name, PLOTS_DIR, num_groups=None, highlight_values=None):
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D

    print(f"Generating UMAPs for {rep_input} and {variable_name}...")

    # Determine groups to highlight
    if highlight_values is not None:
        top_groups = highlight_values
        num_groups = len(top_groups)
        print(f"Highlighting only specified groups: {top_groups}")
    elif num_groups is not None:
        top_groups = ge_adata.obs[variable_name].value_counts().head(num_groups).index.tolist()
        print(f"Top {num_groups} groups: {top_groups}")
    else:
        top_groups = ge_adata.obs[variable_name].unique().tolist()
        num_groups = len(top_groups)
        print(f"Highlighting all groups: {top_groups}")

    # Create 'group_highlighted' column
    ge_adata.obs['group_highlighted'] = 'Other'
    ge_adata.obs.loc[ge_adata.obs[variable_name].isin(top_groups), 'group_highlighted'] = \
        ge_adata.obs[variable_name]

    # Create color palette
    cmap = cm.get_cmap('tab20', len(top_groups))
    colors = [cmap(i) for i in range(len(top_groups))]
    color_dict = {group: colors[i] for i, group in enumerate(top_groups)}
    color_dict['Other'] = [0.9, 0.9, 0.9, 1.0]  # light gray

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    sc.pl.umap(
        ge_adata,
        color='group_highlighted',
        palette=color_dict,
        ax=ax,
        show=False,
        frameon=True,
        legend_loc=None  # we'll add our own legend
    )

    # Create custom compact legend
    handles = [
        Line2D([0], [0], marker='o', color='w', label=grp,
               markerfacecolor=color_dict[grp], markersize=5)
        for grp in top_groups
    ]
    if 'Other' in ge_adata.obs['group_highlighted'].unique():
        handles.append(
            Line2D([0], [0], marker='o', color='w', label='Other',
                   markerfacecolor=color_dict['Other'], markersize=4)
        )

    ax.legend(
        handles=handles,
        loc='upper right',
        fontsize=6,
        frameon=False,
        ncol=1,
        handletextpad=0.4,
        columnspacing=0.8,
        borderaxespad=0.2
    )

    # Save
    outname = f"umap_group_{variable_name}_{rep_input}"
    if highlight_values is not None:
        outname += "_shared"
    elif num_groups is not None:
        outname += f"_top{num_groups}"

    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, f"{outname}.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    print(f"✓ Saved UMAP to {outname}.png")

# === Identify shared cell types across all datasets ===
celltype_by_dataset = (
    ge_adata.obs.groupby(["dataset", "broad_cell_type"])
    .size()
    .unstack(fill_value=0)
)

# Get only the cell types that exist in *all* datasets
shared_celltypes = celltype_by_dataset.columns[
    (celltype_by_dataset > 0).all(axis=0)
].tolist()

print(f"✓ Found {len(shared_celltypes)} shared cell types:")

# Generate UMAP from provided latent space
print("Generating UMAP from NMF standard...")
sc.pp.neighbors(ge_adata, use_rep="X_nmf_standard_mb", n_neighbors=8)
sc.tl.umap(ge_adata)
plot_umap(ge_adata, "X_nmf_standard_mb", "broad_cell_type", PLOTS_DIR, num_groups=10)
plot_umap(ge_adata, "X_nmf_standard_mb", "tissue", PLOTS_DIR, num_groups=20)
plot_umap(ge_adata, "X_nmf_standard_mb", "dataset", PLOTS_DIR)
plot_umap(ge_adata, "X_nmf_standard_mb", "broad_cell_type", PLOTS_DIR, highlight_values=shared_celltypes)

# scVI
print("Generating UMAP from scVI linear...")
sc.pp.neighbors(ge_adata, use_rep="X_scVI_linear", n_neighbors=8)
sc.tl.umap(ge_adata)
plot_umap(ge_adata, "X_scVI_linear", "broad_cell_type", PLOTS_DIR, num_groups=10)
plot_umap(ge_adata, "X_scVI_linear", "tissue", PLOTS_DIR, num_groups=20)
plot_umap(ge_adata, "X_scVI_linear", "dataset", PLOTS_DIR)
plot_umap(ge_adata, "X_scVI_linear", "broad_cell_type", PLOTS_DIR, highlight_values=shared_celltypes)

# scVI standard
print("Generating UMAP from scVI standard...")
sc.pp.neighbors(ge_adata, use_rep="X_scVI_standard", n_neighbors=8)
sc.tl.umap(ge_adata)
plot_umap(ge_adata, "X_scVI_standard", "broad_cell_type", PLOTS_DIR, num_groups=10)
plot_umap(ge_adata, "X_scVI_standard", "tissue", PLOTS_DIR, num_groups=20)
plot_umap(ge_adata, "X_scVI_standard", "dataset", PLOTS_DIR)
plot_umap(ge_adata, "X_scVI_standard", "broad_cell_type", PLOTS_DIR, highlight_values=shared_celltypes)

# to submit 
# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION
# script=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/GeneExpression/09_visualize_scVI_NMF.py
# sbatch --mem=350G -p cpu,bigmem -J "MUS_GE_UMAPs" --wrap="python $script"
