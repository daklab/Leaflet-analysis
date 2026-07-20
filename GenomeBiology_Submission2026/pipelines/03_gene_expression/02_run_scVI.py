#!/usr/bin/env python
"""
Step 8e: Train LinearSCVI on gene expression data.

Selects HVGs + RBP genes + aging-related genes, trains a LinearSCVI model,
computes UMAP + Leiden clustering, and saves the AnnData with latent embedding.

Usage:
    python pipelines/03_gene_expression/05_run_scVI.py config/mouse_smartseq2.yaml \
        --ge <ge_adata_corrected.h5ad>

SLURM: --mem=100G --gres=gpu:1 -p gpu
"""

import sys
import os
import glob
import datetime
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scvi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from scipy.sparse import csr_matrix


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def latest_file(pattern: str) -> str:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching: {pattern}")
    return files[-1]


def select_genes(adata: ad.AnnData, n_top_hvg: int = 20000) -> ad.AnnData:
    """Top N HVGs (batch-aware) + any RBP / aging genes already annotated in .var."""
    print(f"Selecting up to {n_top_hvg} HVGs...")
    tmp = adata.copy()
    sc.pp.highly_variable_genes(tmp, layer="log_norm", n_top_genes=n_top_hvg,
                                 flavor="seurat_v3", batch_key="dataset")
    hvg = set(tmp.var_names[tmp.var.highly_variable])

    rbp    = set(adata.var_names[adata.var.get("RBP_gene", False)]) if "RBP_gene" in adata.var else set()
    aging  = set(adata.var_names[adata.var.get("Aging_gene", False)]) if "Aging_gene" in adata.var else set()

    selected = sorted(hvg | rbp | aging)
    print(f"  HVG: {len(hvg)}  RBP: {len(rbp)}  Aging: {len(aging)}  Total: {len(selected)}")
    return adata[:, selected].copy()


def train_linear_scvi(adata: ad.AnnData, n_latent: int = 20, max_epochs: int = 200,
                      batch_key: str = "dataset") -> ad.AnnData:
    print(f"Setting up LinearSCVI (n_latent={n_latent}, batch_key={batch_key})...")
    setup_kwargs = dict(layer="log_norm")
    if batch_key is not None:
        setup_kwargs["batch_key"] = batch_key
    scvi.model.LinearSCVI.setup_anndata(adata, **setup_kwargs)
    model = scvi.model.LinearSCVI(adata, n_latent=n_latent)
    print(f"Training for up to {max_epochs} epochs...")
    model.train(max_epochs=max_epochs, check_val_every_n_epoch=10)

    adata.obsm["X_scVI_linear"] = model.get_latent_representation()
    adata.obsm["X_normalized_scVI_linear"] = model.get_normalized_expression()
    adata.varm["scVI_linear_gene_loadings"] = model.get_loadings()
    adata.uns["scvi_linear"] = {
        "n_latent":     n_latent,
        "max_epochs":   max_epochs,
        "batch_key":    batch_key,
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d"),
    }
    return adata, model


def cluster_and_umap(adata: ad.AnnData, latent_key: str = "X_scVI_linear",
                     n_neighbors: int = 10, resolution: float = 0.8,
                     out_dir: str = ".") -> ad.AnnData:
    sc.pp.neighbors(adata, use_rep=latent_key, n_neighbors=n_neighbors)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, key_added="leiden_scVI", resolution=resolution)

    fig_path = os.path.join(out_dir, "umap_scVI.png")
    color_keys = ["leiden_scVI"] + (["broad_cell_type"] if "broad_cell_type" in adata.obs else [])
    sc.pl.umap(adata, color=color_keys,
               ncols=1, size=10, frameon=False, show=False)
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"UMAP saved → {fig_path}")
    return adata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--ge", required=True, help="Path to combined GE AnnData (tms_ab_exons_combo_ge_adata_*.h5ad)")
    parser.add_argument("--batch-key", default="dataset",
                        help="obs column to use as scVI batch key, or 'none' to disable batch correction (default: dataset)")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    ge_cfg  = cfg["gene_expression"]
    batch_key = None if args.batch_key.lower() == "none" else args.batch_key
    run_label = "no_batch" if batch_key is None else f"batch_{batch_key}"
    out_dir = os.path.join(cfg["wd"], "scVI", run_label)
    os.makedirs(out_dir, exist_ok=True)
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    n_latent   = ge_cfg.get("scvi_n_latent",   20)
    max_epochs = ge_cfg.get("scvi_max_epochs",  200)
    n_top_hvg  = ge_cfg.get("scvi_n_top_genes", 20000)

    print(f"Loading GE AnnData from {args.ge}")
    adata = ad.read_h5ad(args.ge)
    print(f"  {adata.n_obs} cells × {adata.n_vars} genes")

    adata_sub = select_genes(adata, n_top_hvg=n_top_hvg)
    print(f"Batch key: {batch_key!r}")
    adata_sub, model = train_linear_scvi(adata_sub, n_latent=n_latent, max_epochs=max_epochs,
                                         batch_key=batch_key)
    adata_sub = cluster_and_umap(adata_sub, out_dir=out_dir)

    out_path = os.path.join(out_dir, f"ge_adata_scVI_{today}.h5ad")
    adata_sub.write_h5ad(out_path, compression="gzip")
    print(f"Saved scVI AnnData ({adata_sub.shape}) → {out_path}")


if __name__ == "__main__":
    main()
