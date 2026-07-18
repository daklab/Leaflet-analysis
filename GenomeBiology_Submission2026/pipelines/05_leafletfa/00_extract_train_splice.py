#!/usr/bin/env python
"""
Extract splicing data and save as a model-ready h5ad with junc_ratio layers.

Accepts either:
  - an h5mu file (reads mod/splicing group)
  - a splice h5ad file (reads top-level obs/var/layers directly)

Recomputes junc_ratio and junc_ratio_centered from the sparse
cell_by_junction_matrix / cell_by_cluster_matrix layers.

junc_ratio_centered uses the beta-binomial weighted mean from LeafletFA:
    w = AT / (1 + (AT - 1) * rho)     [precision of PSI estimate]
    centered = PSI - weighted_mean_per_junction
This downweights low-depth junctions where PSI estimates are noisy.

Usage:
    python 00_extract_train_splice.py <input.h5mu|input.h5ad> <output.h5ad>

Memory: ~20-30 GB peak (safe on a 50 GB cpu node).
"""

import sys
import os
import datetime
import h5py
import numpy as np
import anndata as ad
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix, vstack
from leafletfa.waypoints import calculate_centered_psi


# ── helpers ──────────────────────────────────────────────────────────────────

def read_sparse_layer(grp):
    """Read a CSR sparse matrix stored as data/indices/indptr in an HDF5 group."""
    data    = grp["data"][:]
    indices = grp["indices"][:]
    indptr  = grp["indptr"][:]
    shape   = tuple(grp.attrs.get("shape", (len(indptr) - 1, indices.max() + 1)))
    return csr_matrix((data, indices, indptr), shape=shape)


def read_obs(grp):
    """Reconstruct obs DataFrame from HDF5 group (handles categoricals)."""
    index_key = "_index"
    idx = grp[index_key][:].astype(str)
    cols = {}
    for key in grp.keys():
        if key == index_key:
            continue
        item = grp[key]
        if isinstance(item, h5py.Dataset):
            vals = item[:]
            cols[key] = vals.astype(str) if vals.dtype.kind in ("S", "O") else vals
        elif isinstance(item, h5py.Group) and "categories" in item:
            cats = item["categories"][:].astype(str)
            codes = item["codes"][:]
            cols[key] = pd.Categorical.from_codes(codes, categories=cats)
    df = pd.DataFrame(cols, index=idx)
    df.index.name = None
    return df


def read_var(grp):
    """Reconstruct var DataFrame from HDF5 group."""
    # var index = junction IDs (the _index field if present, else junction_id)
    if "_index" in grp:
        idx = grp["_index"][:].astype(str)
    elif "junction_id" in grp:
        idx = grp["junction_id"][:].astype(str)
    else:
        idx = np.arange(grp["junction_id"].shape[0]).astype(str)

    cols = {}
    for key in grp.keys():
        if key == "_index":
            continue
        item = grp[key]
        if isinstance(item, h5py.Dataset):
            vals = item[:]
            cols[key] = vals.astype(str) if vals.dtype.kind in ("S", "O") else vals
        elif isinstance(item, h5py.Group) and "categories" in item:
            cats = item["categories"][:].astype(str)
            codes = item["codes"][:]
            cols[key] = pd.Categorical.from_codes(codes, categories=cats)
    df = pd.DataFrame(cols, index=idx)
    df.index.name = None
    return df


def compute_psi_layers(jc, at, rho=0.1):
    """
    Compute junc_ratio and junc_ratio_centered from sparse jc/at matrices.

    junc_ratio: PSI = jc / at wherever at > 0 (sparse CSR, float32).
    junc_ratio_centered: PSI minus per-junction beta-binomial weighted mean.
        Uses calculate_centered_psi() from leafletfa.waypoints, which weights
        each observation by w = AT / (1 + (AT-1)*rho) so that high-coverage
        cells contribute more to the per-junction mean.

    jc and at must share the same sparsity structure (at's positions).
    We enforce this by extracting jc values at all at-nonzero positions.
    """
    # Build COO of at (defines observed positions)
    at_coo = at.tocoo().astype(np.float32)
    rows, cols = at_coo.row, at_coo.col

    # Extract jc values at exactly at's non-zero (row, col) positions
    # (handles case where jc doesn't store explicit 0s where at > 0)
    jc_csr = jc.tocsr()
    jc_vals = np.array(jc_csr[rows, cols], dtype=np.float32).flatten()
    jc_coo_matched = coo_matrix((jc_vals, (rows, cols)), shape=jc.shape)

    # junc_ratio: PSI = jc / at at observed positions
    psi_vals = jc_vals / at_coo.data
    junc_ratio = coo_matrix((psi_vals, (rows, cols)), shape=jc.shape).tocsr()

    # junc_ratio_centered: beta-binomial weighted centering via LeafletFA
    print(f"  Calling calculate_centered_psi (rho={rho})...")
    Y_sparse = calculate_centered_psi(jc_coo_matched, at_coo, rho=rho)
    junc_ratio_centered = Y_sparse.tocsr().astype(np.float32)

    return junc_ratio.astype(np.float32), junc_ratio_centered


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage: python 00_extract_train_splice.py <input.h5mu|input.h5ad> <output.h5ad>")
        sys.exit(1)

    in_path  = sys.argv[1]
    out_path = sys.argv[2]

    is_h5ad = in_path.endswith(".h5ad")
    print(f"Opening {'h5ad' if is_h5ad else 'h5mu'}: {in_path}")

    with h5py.File(in_path, "r") as f:
        # Splice group is top-level for h5ad, nested under mod/splicing for h5mu
        splice_grp = f if is_h5ad else f["mod/splicing"]

        print("Reading obs...")
        obs = read_obs(splice_grp["obs"])

        print("Reading var...")
        var = read_var(splice_grp["var"])

        print(f"  {len(obs)} cells × {len(var)} junctions")

        print("Loading cell_by_junction_matrix (sparse)...")
        jc = read_sparse_layer(splice_grp["layers/cell_by_junction_matrix"])
        print(f"  jc: {jc.shape}, nnz={jc.nnz:,}")

        print("Loading cell_by_cluster_matrix (sparse)...")
        at = read_sparse_layer(splice_grp["layers/cell_by_cluster_matrix"])
        print(f"  at: {at.shape}, nnz={at.nnz:,}")

        if "layers/psi_mask" in splice_grp:
            print("Loading psi_mask (sparse)...")
            psi_mask = read_sparse_layer(splice_grp["layers/psi_mask"])
            print(f"  psi_mask: {psi_mask.shape}, nnz={psi_mask.nnz:,}")
        else:
            # Build psi_mask from ATSE coverage: 1 wherever cell_by_cluster_matrix > 0
            # (i.e. the junction's ATSE had detectable expression in that cell)
            psi_mask = None  # computed after at is loaded (needs at)

    # Build psi_mask from ATSE coverage if not loaded from file
    if psi_mask is None:
        print("Building psi_mask from ATSE coverage (1 where cell_by_cluster_matrix > 0)...")
        psi_mask = (at > 0).astype(np.float32)
        print(f"  psi_mask: {psi_mask.shape}, nnz={psi_mask.nnz:,}")

    print("\nComputing junc_ratio and junc_ratio_centered (beta-binomial weighted centering)...")
    junc_ratio, junc_ratio_centered = compute_psi_layers(jc, at)

    print("\nBuilding AnnData...")
    adata = ad.AnnData(
        X   = jc,
        obs = obs,
        var = var,
    )
    adata.layers["cell_by_junction_matrix"]  = jc
    adata.layers["cell_by_cluster_matrix"]   = at
    adata.layers["junc_ratio"]               = junc_ratio
    adata.layers["junc_ratio_centered"]      = junc_ratio_centered
    if psi_mask is not None:
        adata.layers["psi_mask"]             = psi_mask

    print(f"\nSaving → {out_path}")
    adata.write_h5ad(out_path, compression="gzip")
    print(f"Done. Shape: {adata.shape}")
    print(f"Layers: {list(adata.layers.keys())}")
    print(f"Obs cols: {list(adata.obs.columns)}")
    print(f"Var cols: {list(adata.var.columns)}")


if __name__ == "__main__":
    main()
