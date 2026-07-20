#!/usr/bin/env python
"""
09c_subset_dataset.py — subset aligned splice + GE h5ads to specific dataset(s).

Runs before 09b_filter_aligned.py when you want to train on a subset of sources
(e.g. allen_brain only, not combined allen_brain + tabula_sapiens).

Usage:
    python pipelines/04_align/09c_subset_dataset.py \\
        --splice model_ready_aligned_splicing.h5ad \\
        --ge     model_ready_gene_expression.h5ad \\
        --datasets allen_brain \\
        --output-dir /path/to/MODEL_INPUT/

    # Multiple datasets:
    --datasets allen_brain tabula_sapiens
"""

import argparse
import os
from datetime import datetime

import anndata as ad
import scipy.sparse as sp


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splice",      required=True)
    parser.add_argument("--ge",          required=True)
    parser.add_argument("--datasets",    nargs="+", required=True,
                        help="dataset values to keep (from obs['dataset'])")
    parser.add_argument("--output-dir",  required=True)
    args = parser.parse_args()

    print(f"Loading splice: {args.splice}")
    splice = ad.read_h5ad(args.splice)
    if not sp.issparse(splice.X):
        splice.X = sp.csr_matrix(splice.X)
    print(f"  shape={splice.shape}  datasets: {dict(splice.obs['dataset'].value_counts())}")

    print(f"Loading GE: {args.ge}")
    ge = ad.read_h5ad(args.ge)
    print(f"  shape={ge.shape}")

    mask = splice.obs["dataset"].isin(args.datasets)
    print(f"\nSubsetting to datasets: {args.datasets}")
    print(f"  {splice.n_obs:,} → {mask.sum():,} cells retained")

    splice = splice[mask].copy()
    ge     = ge[ge.obs_names.isin(splice.obs_names)].copy()
    # reorder GE to match splice cell order
    ge = ge[splice.obs_names].copy()

    print(f"  Splice: {splice.shape}")
    print(f"  GE:     {ge.shape}")
    print(f"\n  broad_cell_type distribution:")
    print(splice.obs["broad_cell_type"].value_counts().to_string())

    today = datetime.now().strftime("%Y-%m-%d")
    tag   = "_".join(args.datasets) + f"_{today}"
    os.makedirs(args.output_dir, exist_ok=True)

    splice_out = os.path.join(args.output_dir, f"splice_adata_{tag}.h5ad")
    ge_out     = os.path.join(args.output_dir, f"ge_adata_{tag}.h5ad")

    print(f"\nWriting splice → {splice_out}")
    splice.write_h5ad(splice_out)
    print(f"Writing GE     → {ge_out}")
    ge.write_h5ad(ge_out)
    print("Done.")


if __name__ == "__main__":
    main()
