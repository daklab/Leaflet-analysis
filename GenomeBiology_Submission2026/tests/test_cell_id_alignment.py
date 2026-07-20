#!/usr/bin/env python
"""
test_cell_id_alignment.py — verify splice and GE AnnDatas have the same cells
in the same order after align_splice_ge.py runs.

This catches the most common alignment bug: both AnnDatas have the same SET
of cell IDs but in different orders, causing silently wrong downstream analysis.

Usage:
    python test_cell_id_alignment.py <splice_adata.h5ad> <ge_adata.h5ad>
"""

import argparse
import sys

import anndata as ad
import numpy as np


def run(splice_path: str, ge_path: str) -> bool:
    print("Loading AnnDatas...")
    splice = ad.read_h5ad(splice_path)
    ge     = ad.read_h5ad(ge_path)

    print(f"  Splice: {splice.shape[0]:,} cells × {splice.shape[1]:,} features")
    print(f"  GE:     {ge.shape[0]:,} cells × {ge.shape[1]:,} features")

    ok = True

    # 1. Same number of cells
    if splice.n_obs != ge.n_obs:
        print(f"FAIL: cell count mismatch — splice={splice.n_obs} ge={ge.n_obs}")
        ok = False
    else:
        print(f"✓ Cell count matches: {splice.n_obs:,}")

    # 2. Exact same obs_names in the same order
    if not (splice.obs_names == ge.obs_names).all():
        mismatch_idx = np.where(splice.obs_names != ge.obs_names)[0]
        print(f"FAIL: obs_names mismatch at {len(mismatch_idx)} positions")
        for i in mismatch_idx[:5]:
            print(f"  pos {i}: splice='{splice.obs_names[i]}' ge='{ge.obs_names[i]}'")
        ok = False
    else:
        print(f"✓ obs_names are identical and in the same order")

    # 3. No duplicate obs_names in either
    for name, adata in [("splice", splice), ("ge", ge)]:
        n_dup = adata.obs_names.duplicated().sum()
        if n_dup:
            print(f"FAIL: {n_dup} duplicate obs_names in {name} AnnData")
            ok = False
        else:
            print(f"✓ {name}: no duplicate obs_names")

    # 4. Check a shared metadata column if available (e.g. cell_type)
    shared_cols = set(splice.obs.columns) & set(ge.obs.columns)
    check_col = next((c for c in ["cell_type", "cell_ontology_class", "broad_cell_type"]
                      if c in shared_cols), None)
    if check_col:
        if (splice.obs[check_col].values == ge.obs[check_col].values).all():
            print(f"✓ Shared column '{check_col}' matches across both AnnDatas")
        else:
            n_mismatch = (splice.obs[check_col].values != ge.obs[check_col].values).sum()
            print(f"FAIL: '{check_col}' mismatch in {n_mismatch} cells")
            ok = False

    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("splice_adata")
    parser.add_argument("ge_adata")
    args = parser.parse_args()

    ok = run(args.splice_adata, args.ge_adata)
    if ok:
        print("\nAll alignment checks passed.")
        sys.exit(0)
    else:
        print("\nAlignment checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
