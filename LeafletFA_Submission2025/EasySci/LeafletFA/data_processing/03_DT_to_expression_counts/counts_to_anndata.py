#!/usr/bin/env python3
"""
Convert featureCounts output to AnnData with metacell names as cell (obs) IDs.
Usage:
  python counts_to_anndata.py --counts gene_counts.tsv --bam-list bam_list.txt \\
      --metacell-ids metacell_ids.txt --output DT_metacell_expression.h5ad
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix

try:
    import anndata as ad
except ImportError:
    raise ImportError("Please install anndata: pip install anndata")


def read_featurecounts(counts_path: str):
    """
    Read featureCounts main output (tab-delimited) in a memory-efficient way.
    First 6 columns: Geneid, Chr, Start, End, Strand, Length.
    Remaining columns: one per BAM, in input order.
    Returns (gene_annotation_df, count_matrix, n_samples).
    Uses pandas with explicit dtypes to avoid huge string arrays (long BAM paths in header).
    """
    n_anno = 6
    # Count comment lines and get column count from first non-comment line
    with open(counts_path) as f:
        skip = 0
        for line in f:
            if line.startswith("#"):
                skip += 1
            else:
                ncols = len(line.rstrip().split("\t"))
                break
        else:
            raise ValueError(f"No data lines in {counts_path}")

    if ncols <= n_anno:
        raise ValueError(f"Expected at least {n_anno + 1} columns in featureCounts output, got {ncols}")
    n_samples = ncols - n_anno

    # Read with explicit dtypes: no long path strings, count columns as float32
    col_names = ["Geneid", "Chr", "Start", "End", "Strand", "Length"] + [f"c{i}" for i in range(n_samples)]
    dtype = {c: str for c in ["Geneid", "Chr", "Start", "End", "Strand", "Length"]}
    for i in range(n_samples):
        dtype[f"c{i}"] = np.float32

    df = pd.read_csv(
        counts_path,
        sep="\t",
        skiprows=range(skip + 1),  # skip # lines and header
        names=col_names,
        dtype=dtype,
        low_memory=False,
    )

    # Start/End/Length can be semicolon-separated for multi-feature genes in featureCounts; keep as str
    gene_anno = pd.DataFrame(
        {
            "gene_id": df["Geneid"].values,
            "Chr": df["Chr"].values,
            "Start": df["Start"].astype(str).values,
            "End": df["End"].astype(str).values,
            "Strand": df["Strand"].values,
            "Length": df["Length"].astype(str).values,
        },
        index=df["Geneid"].values,
    )
    count_cols = df[[f"c{i}" for i in range(n_samples)]].values
    return gene_anno, count_cols, n_samples


def main():
    ap = argparse.ArgumentParser(description="Convert featureCounts output to AnnData (metacells as cells).")
    ap.add_argument("--counts", required=True, help="Path to featureCounts output .tsv")
    ap.add_argument("--bam-list", required=True, help="Path to file with one BAM path per line (same order as featureCounts)")
    ap.add_argument("--metacell-ids", required=True, help="Path to file with one metacell ID per line (same order as BAM list)")
    ap.add_argument("--output", required=True, help="Output .h5ad path")
    args = ap.parse_args()

    with open(args.metacell_ids) as f:
        metacell_ids = [line.strip() for line in f if line.strip()]
    with open(args.bam_list) as f:
        bam_paths = [line.strip() for line in f if line.strip()]

    if len(metacell_ids) != len(bam_paths):
        raise ValueError(
            f"metacell_ids and bam_list must have same length; got {len(metacell_ids)} and {len(bam_paths)}"
        )

    gene_anno, count_matrix, n_samples_fc = read_featurecounts(args.counts)
    if n_samples_fc != len(metacell_ids):
        raise ValueError(
            f"featureCounts has {n_samples_fc} sample columns but metacell_ids has {len(metacell_ids)}. "
            "Ensure bam_list and metacell_ids match the BAM order passed to featureCounts."
        )

    # AnnData expects X shape (n_obs, n_vars) = (samples, genes); featureCounts gives (genes, samples)
    X = count_matrix.T.astype(np.float32)  # (n_metacells, n_genes)
    # Sparse often saves space for count matrices
    if (X == 0).sum() / X.size > 0.5:
        X = csr_matrix(X)

    # Use metacell IDs as cell (obs) names
    obs = pd.DataFrame(index=metacell_ids)
    obs["metacell_id"] = metacell_ids
    obs["n_counts"] = np.array(count_matrix.sum(axis=0)).ravel()  # per-sample sums (axis=0 = columns = samples)

    var = gene_anno[["gene_id", "Chr", "Strand", "Length"]].copy()
    var.index = var["gene_id"]
    var.index.name = None

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obs_names = metacell_ids
    adata.var_names = gene_anno["gene_id"].values

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output, compression="gzip")
    print(f"Wrote AnnData: {args.output} (n_obs={adata.n_obs}, n_vars={adata.n_vars})")


if __name__ == "__main__":
    main()
