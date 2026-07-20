#!/usr/bin/env python
"""
plot_factor_space.py — UMAP of LeafletFA cell factor weights (assign_post).

Usage:
    PYTHON=/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python
    SPLICE=/gpfs/commons/groups/knowles_lab/Karin/.../splice_adata_aligned_filtered_*.h5ad
    MODEL_RUN=/gpfs/commons/groups/knowles_lab/Karin/.../LEAFLETFA_MODELS/2026-06-16/run_0

    $PYTHON tests/plot_factor_space.py --model-run $MODEL_RUN --splice $SPLICE
"""

import argparse
import gzip
import pickle
from pathlib import Path

import anndata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import scanpy as sc


def load_assign_post(model_run_dir):
    pkl = Path(model_run_dir) / "leafletfa_model.pkl.gz"
    with gzip.open(pkl, "rb") as f:
        while True:
            try:
                obj = pickle.load(f)
                if isinstance(obj, dict) and "assign_post" in obj:
                    return obj["assign_post"]
            except EOFError:
                break
    raise ValueError("assign_post not found in pkl")


def top_colors(labels, n=12):
    top = pd.Series(labels).value_counts().index[:n].tolist()
    palette = cm.get_cmap("tab20", len(top))
    color_map = {lab: palette(i) for i, lab in enumerate(top)}
    color_map["Other"] = (0.8, 0.8, 0.8, 1.0)
    return color_map, top


def plot_umap(coords, labels, color_map, top_labels, title, out_path, alpha=0.3, s=3):
    fig, ax = plt.subplots(figsize=(9, 7))
    # plot "Other" first (background)
    mask_other = ~pd.Series(labels).isin(top_labels)
    ax.scatter(coords[mask_other, 0], coords[mask_other, 1],
               c=[color_map["Other"]], s=s, alpha=alpha, linewidths=0)
    for lab in top_labels:
        mask = np.array(labels) == lab
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[color_map[lab]], s=s, alpha=alpha, label=lab, linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=4, fontsize=7, loc="upper right",
              bbox_to_anchor=(1.35, 1), frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-run", required=True, help="Path to run_N/ dir")
    parser.add_argument("--splice", required=True, help="Filtered splice h5ad (for obs metadata)")
    parser.add_argument("--out", default=None, help="Output dir (default: tests/)")
    args = parser.parse_args()

    run_dir = Path(args.model_run)
    out_dir = Path(args.out) if args.out else Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    run_name = run_dir.name  # e.g. run_0

    print("Loading assign_post...")
    phi = load_assign_post(run_dir)  # (N, K)
    print(f"  assign_post: {phi.shape}")

    print("Loading splice obs metadata...")
    adata = anndata.read_h5ad(args.splice, backed="r")
    obs = adata.obs[["broad_cell_type", "tissue_cell_type", "tissue", "dataset"]].copy()
    adata.file.close()

    assert phi.shape[0] == len(obs), f"Cell count mismatch: {phi.shape[0]} vs {len(obs)}"

    print("Running UMAP on assign_post (K dims)...")
    phi_adata = anndata.AnnData(X=phi, obs=obs)
    # phi is already low-dim (K=10), skip PCA — run UMAP directly
    sc.pp.neighbors(phi_adata, use_rep="X", n_neighbors=15)
    sc.tl.umap(phi_adata)
    coords = phi_adata.obsm["X_umap"]

    tag = run_dir.parent.name + f"_{run_name}"  # e.g. 2026-06-16_run_0

    # ── broad cell type ────────────────────────────────────────────────────────
    labels = obs["broad_cell_type"].astype(str).replace("nan", "Unknown").tolist()
    color_map, top = top_colors(labels, n=12)
    plot_umap(coords, labels, color_map, top,
              f"UMAP φ — broad_cell_type | {tag}",
              out_dir / f"umap_phi_broad_cell_type_{tag}.png")

    # ── tissue × cell type ────────────────────────────────────────────────────
    labels2 = obs["tissue_cell_type"].astype(str).replace("nan", "Unknown").tolist()
    color_map2, top2 = top_colors(labels2, n=15)
    plot_umap(coords, labels2, color_map2, top2,
              f"UMAP φ — tissue_cell_type | {tag}",
              out_dir / f"umap_phi_tissue_cell_type_{tag}.png")

    # ── dataset (TMS vs Allen Brain) ──────────────────────────────────────────
    labels3 = obs["dataset"].astype(str).replace("nan", "Unknown").tolist()
    color_map3, top3 = top_colors(labels3, n=5)
    plot_umap(coords, labels3, color_map3, top3,
              f"UMAP φ — dataset | {tag}",
              out_dir / f"umap_phi_dataset_{tag}.png")

    # ── factor heatmap: mean φ per broad cell type ────────────────────────────
    phi_df = pd.DataFrame(phi, columns=[f"F{i}" for i in range(phi.shape[1])])
    phi_df["broad_cell_type"] = obs["broad_cell_type"].astype(str).replace("nan", "Unknown").values
    mean_phi = phi_df.groupby("broad_cell_type").mean()

    fig, ax = plt.subplots(figsize=(max(8, phi.shape[1]), max(5, len(mean_phi) * 0.4)))
    im = ax.imshow(mean_phi.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(phi.shape[1])); ax.set_xticklabels(mean_phi.columns, fontsize=9)
    ax.set_yticks(range(len(mean_phi))); ax.set_yticklabels(mean_phi.index, fontsize=8)
    ax.set_title(f"Mean φ per broad cell type | {tag}", fontsize=10)
    plt.colorbar(im, ax=ax, label="Mean factor weight")
    plt.tight_layout()
    heatmap_path = out_dir / f"phi_heatmap_broad_cell_type_{tag}.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {heatmap_path}")


if __name__ == "__main__":
    main()
