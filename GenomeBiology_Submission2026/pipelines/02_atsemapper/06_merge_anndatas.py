#!/usr/bin/env python
"""
06_merge_anndatas.py — merge per-chunk AnnDatas into a single dataset.

Usage:
    python pipelines/02_atsemapper/06_merge_anndatas.py \
        --input-dir $ANNDATA_DIR \
        --output-file $ANNDATA_DIR/merged_anndata.h5ad
"""

import argparse
import gc
from pathlib import Path

import anndata as ad
import scipy.sparse as sp


def merge_anndata_files(input_dir: str, output_file: str) -> ad.AnnData:
    files = sorted(Path(input_dir).glob("chunk_*_anndata.h5ad"))
    if not files:
        raise FileNotFoundError(f"No chunk_*_anndata.h5ad files found in {input_dir}")
    print(f"Found {len(files)} chunk files to merge")

    adatas = []
    for i, file in enumerate(files):
        print(f"  Loading {i+1}/{len(files)}: {file.name}", flush=True)
        adata = ad.read_h5ad(file)
        adatas.append(adata)

    print(f"\nConcatenating {len(adatas)} chunks...")
    # join="inner" keeps only junctions present in all chunks (same junction universe expected)
    # layers are stacked row-wise automatically
    combined = ad.concat(adatas, join="inner", merge="first")
    del adatas
    gc.collect()

    # Verify no duplicate cell IDs
    n_unique = combined.obs_names.nunique()
    n_total = len(combined.obs_names)
    if n_unique != n_total:
        raise ValueError(f"{n_total - n_unique} duplicate cell IDs after merge")

    print(f"\nMerge complete: {combined.shape[0]:,} cells × {combined.shape[1]:,} junctions")
    for layer_name in ["cell_by_junction_matrix", "cell_by_cluster_matrix"]:
        if layer_name in combined.layers:
            print(f"  layer '{layer_name}': {combined.layers[layer_name].shape}")
        else:
            print(f"  WARNING: layer '{layer_name}' missing from merged AnnData")
    print(f"Saving to {output_file}...")
    combined.write_h5ad(output_file, compression="gzip")
    print("Done.")
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",   required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()
    merge_anndata_files(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()
