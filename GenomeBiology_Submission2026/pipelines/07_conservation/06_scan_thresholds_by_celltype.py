#!/usr/bin/env python
"""
06_scan_thresholds_by_celltype.py — pick D9 coverage thresholds SAFELY on the bulk atlas.

09b_filter_aligned.py --scan reports junction/cell retention POOLED over all cells. On
this atlas that's dangerous: immune cell types (CD14+ monocytes, NK, CD3+ T cells) have
low exonic coverage (Ribo-Zero total-RNA capturing pre-mRNA; WORKFLOW D8), so a cell filter
chosen from the pooled median can silently wipe out exactly the cell types we need for the
LeafCutter DS comparison.

This helper replicates 09b's filter semantics EXACTLY and stratifies by cell_type:
  - junction filter: keep junctions detected in >= t cells        (X.getnnz(axis=0) >= t)
  - cell filter:     keep cells with >= M junctions AFTER the junction filter
                     (recomputed on the surviving junction subset — not the pooled approx)

Outputs, per candidate junction threshold t:
  1. global junctions/ATSEs kept (matches 09b --scan)
  2. per cell_type: n cells, median junctions/cell after the filter, and the fraction of
     cells retained at each cell filter M — flagging cell types that collapse.

Usage:
    python 06_scan_thresholds_by_celltype.py <splice_h5ad>
        [--celltype-col cell_type]
        [--junction-thresholds 5,10,20,30]
        [--cell-thresholds 50,100,250,500]
        [--out-csv <path>]      # long-format per (t, M, cell_type) retention table
"""

import argparse
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def parse_int_list(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("splice_h5ad")
    ap.add_argument("--celltype-col", default="cell_type")
    ap.add_argument("--junction-thresholds", default="5,10,20,30", type=parse_int_list)
    ap.add_argument("--cell-thresholds", default="50,100,250,500", type=parse_int_list)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    print(f"Loading {args.splice_h5ad} ...")
    A = ad.read_h5ad(args.splice_h5ad)
    X = A.X.tocsc()                       # csc → fast column (junction) slicing
    n_cell, n_junc = A.shape
    print(f"  {n_cell:,} cells × {n_junc:,} junctions")

    if args.celltype_col not in A.obs.columns:
        sys.exit(f"ERROR: obs has no '{args.celltype_col}'. Columns: {list(A.obs.columns)}")
    ct = A.obs[args.celltype_col].astype(str).values
    ct_levels, ct_sizes = np.unique(ct, return_counts=True)
    print(f"  {len(ct_levels)} cell types | cells/type: "
          f"min={ct_sizes.min()} median={int(np.median(ct_sizes))} max={ct_sizes.max()}")

    has_event = "event_id" in A.var.columns
    event_ids = A.var["event_id"].values if has_event else None
    n_atse_total = len(set(event_ids)) if has_event else 0

    n_cells_per_junc = X.getnnz(axis=0)   # detection count per junction (matches 09b)

    rows = []
    for t in args.junction_thresholds:
        jmask = n_cells_per_junc >= t
        n_kept = int(jmask.sum())
        atse_kept = len(set(event_ids[jmask])) if has_event else 0
        print(f"\n{'='*78}\nJUNCTION FILTER t={t}: "
              f"{n_kept:,}/{n_junc:,} junctions kept ({100*n_kept/n_junc:.1f}%)"
              + (f" | {atse_kept:,}/{n_atse_total:,} ATSEs retain >=1 junction" if has_event else ""))

        # per-cell junction count AFTER the junction filter (the real cell-filter input)
        juncs_per_cell = np.asarray((X[:, jmask] != 0).sum(axis=1)).ravel()

        # header
        mcols = "  ".join(f"M>={M:>4}" for M in args.cell_thresholds)
        print(f"  {'cell_type':38s} {'n':>4} {'medJ/cell':>9}  {mcols}")
        # order cell types by worst retention at the largest M so problem types surface first
        Mmax = max(args.cell_thresholds)
        order = sorted(range(len(ct_levels)),
                       key=lambda i: (juncs_per_cell[ct == ct_levels[i]] >= Mmax).mean())
        for i in order:
            c = ct_levels[i]
            sel = ct == c
            jpc = juncs_per_cell[sel]
            med = int(np.median(jpc)) if jpc.size else 0
            retentions = [(jpc >= M).mean() for M in args.cell_thresholds]
            flag = " ⚠" if retentions[-1] < 0.5 else ""
            rcells = "  ".join(f"{100*r:>5.0f}%" for r in retentions)
            print(f"  {c[:38]:38s} {sel.sum():>4} {med:>9,}  {rcells}{flag}")
            for M, r in zip(args.cell_thresholds, retentions):
                rows.append(dict(junction_t=t, cell_M=M, cell_type=c, n_cells=int(sel.sum()),
                                 median_juncs_per_cell=med, frac_retained=float(r),
                                 n_retained=int((jpc >= M).sum())))
        # global cell retention for reference
        print(f"  {'— ALL (pooled) —':38s} {n_cell:>4} "
              f"{int(np.median(juncs_per_cell)):>9,}  " +
              "  ".join(f"{100*(juncs_per_cell>=M).mean():>5.0f}%" for M in args.cell_thresholds))

    if args.out_csv:
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print(f"\nLong-format retention table → {args.out_csv}")

    print("\nGuidance: pick (junction_t, cell_M) where NO wanted cell type drops below ~50-70%\n"
          "retention. ⚠ marks cell types losing >50% of cells at the largest cell filter.")


if __name__ == "__main__":
    main()
