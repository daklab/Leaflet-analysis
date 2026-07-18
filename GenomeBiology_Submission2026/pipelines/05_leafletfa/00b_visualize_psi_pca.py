#!/usr/bin/env python
"""
00b_visualize_psi_pca.py — QC visualization of centered PSI before LeafletFA training.

Run after 00_extract_train_splice.py. Verifies that beta-binomial weighted centered
PSI captures biologically meaningful variation, before spending GPU time on training.

Steps:
  1. Load junc_ratio_centered (sparse) from train h5ad via h5py
  2. TruncatedSVD (PCA on sparse centered matrix)
  3. UMAP on top PCs
  4. PCA and UMAP plots colored by:
       - broad_cell_type:  top 10 → colors, rest → grey
       - tissue_cell_type: top 15 → colors, rest → grey

Usage:
    python 00b_visualize_psi_pca.py <train_h5ad> <output_dir> [--n-pcs N] [--skip-umap]
"""

import sys
import os
import argparse
import h5py
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


# ── helpers ───────────────────────────────────────────────────────────────────

def read_sparse_layer_h5(h5ad_path, layer_name):
    """Read a CSR sparse layer from an h5ad file without loading the full adata."""
    with h5py.File(h5ad_path, "r") as f:
        grp = f[f"layers/{layer_name}"]
        data    = grp["data"][:]
        indices = grp["indices"][:]
        indptr  = grp["indptr"][:]
        if "shape" in grp.attrs:
            shape = tuple(int(x) for x in grp.attrs["shape"])
        else:
            shape = (len(indptr) - 1, int(indices.max()) + 1 if len(indices) > 0 else 0)
    return csr_matrix((data, indices, indptr), shape=shape)


def _palette(n):
    """Return n visually distinct colors."""
    if n <= 10:
        return list(plt.cm.tab10.colors[:n])
    elif n <= 20:
        return list(plt.cm.tab20.colors[:n])
    else:
        return [plt.cm.hsv(i / n) for i in range(n)]


def scatter_colored(ax, xy, labels, top_n, title, xlabel, ylabel, var_label=""):
    """
    Scatter plot colored by top_n most frequent labels; rest are grey.
    Grey cells plotted first so colored cells appear on top.
    """
    counts   = pd.Series(labels).value_counts()
    top_cats = list(counts.index[:top_n])
    colors   = _palette(top_n)
    cat_color = {cat: colors[i] for i, cat in enumerate(top_cats)}

    mask_other = ~np.isin(labels, top_cats)
    if mask_other.any():
        ax.scatter(xy[mask_other, 0], xy[mask_other, 1],
                   c="#cccccc", s=1, alpha=0.2, linewidths=0, rasterized=True,
                   label=f"Other ({mask_other.sum():,})")

    for cat in top_cats:
        mask = labels == cat
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   c=[cat_color[cat]], s=1.5, alpha=0.6, linewidths=0,
                   rasterized=True, label=f"{cat} ({mask.sum():,})")

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10, pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    return cat_color


def save_plot(xy, labels, top_n, title, xlabel, ylabel, out_path, figsize=(11, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    scatter_colored(ax, xy, labels, top_n, title, xlabel, ylabel)
    handles, leg_labels = ax.get_legend_handles_labels()
    ax.legend(handles, leg_labels,
              markerscale=6, fontsize=7, frameon=False,
              loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_h5ad")
    ap.add_argument("output_dir")
    ap.add_argument("--n-pcs",    type=int, default=50, help="PCs for SVD (default 50)")
    ap.add_argument("--skip-umap", action="store_true",
                    help="Only generate PCA plots (faster)")
    ap.add_argument("--umap-seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load obs metadata (backed — only reads obs, not layers) ──────────────
    print("Loading obs metadata...")
    adata = ad.read_h5ad(args.train_h5ad, backed="r")
    obs   = adata.obs.copy()
    adata.file.close()
    n_cells = len(obs)
    print(f"  {n_cells:,} cells")
    for col in ["broad_cell_type", "tissue_cell_type", "dataset", "age"]:
        if col in obs.columns:
            vc = obs[col].value_counts()
            print(f"  {col}: {len(vc)} unique | top 3: {dict(vc.head(3))}")

    # ── Load junc_ratio_centered sparse matrix ────────────────────────────────
    print("\nLoading junc_ratio_centered (sparse)...")
    X = read_sparse_layer_h5(args.train_h5ad, "junc_ratio_centered").astype(np.float32)
    print(f"  Shape: {X.shape}  nnz={X.nnz:,}  sparsity={1 - X.nnz/(X.shape[0]*X.shape[1]):.1%}")

    # ── TruncatedSVD ─────────────────────────────────────────────────────────
    n_pcs = min(args.n_pcs, min(X.shape) - 1)
    print(f"\nTruncatedSVD: {n_cells:,} cells × {X.shape[1]:,} junctions → {n_pcs} PCs...")
    svd   = TruncatedSVD(n_components=n_pcs, random_state=42, n_iter=5)
    X_pca = svd.fit_transform(X).astype(np.float32)  # (N, n_pcs)
    var   = svd.explained_variance_ratio_

    print(f"  PC1={var[0]:.3f}  PC2={var[1]:.3f}  PC3={var[2]:.3f}  "
          f"cumulative({n_pcs})={var.sum():.3f}")

    # Save variance explained
    pd.DataFrame({
        "PC": np.arange(1, n_pcs + 1),
        "var_explained": var,
        "cumulative_var": np.cumsum(var),
    }).to_csv(os.path.join(args.output_dir, "pca_variance_explained.csv"), index=False)

    # ── Variance explained elbow plot ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, n_pcs + 1), np.cumsum(var) * 100, marker="o", ms=3, lw=1.5)
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Cumulative variance explained (%)")
    ax.set_title("TruncatedSVD on junc_ratio_centered")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "pca_variance_explained.png"), dpi=150)
    plt.close()

    # ── PCA scatter plots ─────────────────────────────────────────────────────
    print("\nGenerating PCA plots (PC1 vs PC2)...")
    for col, top_n in [("broad_cell_type", 10), ("tissue_cell_type", 15)]:
        if col not in obs.columns:
            print(f"  Skipping {col} (not in obs)")
            continue
        labels = obs[col].astype(str).values
        save_plot(
            X_pca[:, :2], labels, top_n,
            title  = f"PCA — centered PSI — {col}",
            xlabel = f"PC1 ({var[0]:.1%})",
            ylabel = f"PC2 ({var[1]:.1%})",
            out_path = os.path.join(args.output_dir, f"pca_{col}.png"),
        )

    # PC1 vs PC3 for broad_cell_type as a secondary check
    if "broad_cell_type" in obs.columns:
        labels = obs["broad_cell_type"].astype(str).values
        save_plot(
            np.column_stack([X_pca[:, 0], X_pca[:, 2]]), labels, 10,
            title    = "PCA — centered PSI — broad_cell_type (PC1 vs PC3)",
            xlabel   = f"PC1 ({var[0]:.1%})",
            ylabel   = f"PC3 ({var[2]:.1%})",
            out_path = os.path.join(args.output_dir, "pca_broad_cell_type_pc1_pc3.png"),
        )

    if args.skip_umap:
        print("\n--skip-umap set. PCA done.")
        print(f"Outputs in {args.output_dir}")
        return

    # ── UMAP ──────────────────────────────────────────────────────────────────
    try:
        from umap import UMAP
    except ImportError:
        print("\numap-learn not installed. Skipping UMAP.")
        print("  Install: pip install umap-learn")
        return

    print(f"\nRunning UMAP on {n_cells:,} cells × {n_pcs} PCs (this takes ~5-15 min)...")
    reducer  = UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                    random_state=args.umap_seed, verbose=False, low_memory=True)
    X_umap   = reducer.fit_transform(X_pca).astype(np.float32)

    # Save embedding + metadata for downstream reuse
    np.save(os.path.join(args.output_dir, "umap_embedding.npy"), X_umap)
    obs_out          = obs.copy()
    obs_out["UMAP1"] = X_umap[:, 0]
    obs_out["UMAP2"] = X_umap[:, 1]
    obs_out.to_csv(os.path.join(args.output_dir, "umap_metadata.csv"))
    print("  UMAP done. Embedding saved.")

    # ── UMAP plots ────────────────────────────────────────────────────────────
    print("\nGenerating UMAP plots...")
    for col, top_n in [("broad_cell_type", 10), ("tissue_cell_type", 15)]:
        if col not in obs.columns:
            continue
        labels = obs[col].astype(str).values
        save_plot(
            X_umap, labels, top_n,
            title    = f"UMAP — centered PSI (β-binomial) — {col}",
            xlabel   = "UMAP 1",
            ylabel   = "UMAP 2",
            out_path = os.path.join(args.output_dir, f"umap_{col}.png"),
        )

    # Extra: color by dataset and age for batch/covariate check
    for col, top_n in [("dataset", 5), ("age", 6), ("sex", 2)]:
        if col not in obs.columns:
            continue
        labels = obs[col].astype(str).values
        save_plot(
            X_umap, labels, top_n,
            title    = f"UMAP — centered PSI — {col}",
            xlabel   = "UMAP 1",
            ylabel   = "UMAP 2",
            out_path = os.path.join(args.output_dir, f"umap_{col}.png"),
            figsize  = (9, 7),
        )

    print(f"\nAll outputs in {args.output_dir}")
    print("Key files:")
    print(f"  umap_broad_cell_type.png  — main biological check")
    print(f"  umap_tissue_cell_type.png — tissue × cell type detail")
    print(f"  umap_dataset.png          — batch check (should mix, not separate)")
    print(f"  umap_embedding.npy        — (N, 2) array for reuse")


if __name__ == "__main__":
    main()
