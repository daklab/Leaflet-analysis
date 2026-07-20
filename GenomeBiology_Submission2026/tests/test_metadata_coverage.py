#!/usr/bin/env python
"""
test_metadata_coverage.py — check that every cell in the merged AnnData
has a valid cell_id match in the metadata CSV.

Cells with no metadata are silently retained in the AnnData with NaN metadata
columns, which corrupts downstream cell-type-stratified analysis.

Usage:
    python test_metadata_coverage.py <merged_anndata.h5ad> <metadata.csv>
"""

import argparse
import sys

import anndata as ad
import pandas as pd


REQUIRED_COLUMNS = ["cell_id", "age", "cell_ontology_class", "tissue"]


def run(adata_path: str, metadata_path: str) -> bool:
    print("Loading inputs...")
    adata    = ad.read_h5ad(adata_path)
    metadata = pd.read_csv(metadata_path, low_memory=False)

    print(f"  AnnData:  {adata.n_obs:,} cells")
    print(f"  Metadata: {len(metadata):,} rows")

    ok = True

    # 1. Required columns present in metadata
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in metadata.columns]
    if missing_cols:
        print(f"FAIL: metadata is missing columns: {missing_cols}")
        ok = False
    else:
        print(f"✓ Required metadata columns present: {REQUIRED_COLUMNS}")

    # 2. Every AnnData cell_id has a metadata row
    if "cell_id" in adata.obs.columns:
        adata_cell_ids = set(adata.obs["cell_id"])
    else:
        adata_cell_ids = set(adata.obs_names)

    meta_cell_ids  = set(metadata["cell_id"])
    unmatched = adata_cell_ids - meta_cell_ids
    pct_unmatched = 100 * len(unmatched) / max(len(adata_cell_ids), 1)

    if unmatched:
        print(f"FAIL: {len(unmatched):,} cells ({pct_unmatched:.1f}%) "
              f"have no metadata row")
        print(f"  Examples: {list(unmatched)[:5]}")
        ok = False
    else:
        print(f"✓ All {len(adata_cell_ids):,} AnnData cells have metadata")

    # 3. Check for NaN in critical metadata columns (in AnnData obs)
    for col in REQUIRED_COLUMNS:
        if col in adata.obs.columns:
            n_nan = adata.obs[col].isna().sum()
            if n_nan:
                print(f"WARNING: {n_nan} NaN values in obs['{col}']")
            else:
                print(f"✓ obs['{col}']: no NaN values")

    # 4. Duplicate cell_ids in metadata
    n_dup_meta = metadata["cell_id"].duplicated().sum()
    if n_dup_meta:
        print(f"WARNING: {n_dup_meta} duplicate cell_ids in metadata CSV")
    else:
        print(f"✓ Metadata: no duplicate cell_ids")

    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("merged_anndata")
    parser.add_argument("metadata_csv")
    args = parser.parse_args()

    ok = run(args.merged_anndata, args.metadata_csv)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
