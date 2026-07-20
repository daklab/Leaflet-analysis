#!/usr/bin/env python
"""
Step 08-00: Export junction → ATSE mapping for LeafCutter DS input preparation.

Background
----------
LeafCutter DS is a differential splicing tool that tests for changes in intron
usage between two conditions (e.g. young vs old) using a Dirichlet-Multinomial
model. It operates on "clusters" of junctions that share splice sites — analogous
to our ATSEs. The key input it expects is a gzip-compressed counts matrix where:

  - Each ROW is one junction, identified by:
        chr:start:end:CLUSTER_ID_strand
    e.g. chr10:81350189:81350272:ENSMUSG00000024012.18_atse_5_-

  - Each COLUMN is one sample (in our case: a pseudobulk of cells from one
    donor × cell-type combination).

  - Values are raw integer junction read counts.

LeafCutter infers which junctions belong to the same cluster by parsing the
CLUSTER_ID field from the junction row name. All junctions sharing the same
CLUSTER_ID are tested jointly. This is exactly equivalent to our ATSEs: all
junctions assigned to the same ATSE (event_id) are tested together.

Why we use our ATSE event_id as the CLUSTER_ID
-----------------------------------------------
LeafCutter's own clustering (leafcutter-cluster) would re-derive clusters from
scratch by finding junctions that share donor or acceptor splice sites. We have
already done this, more carefully, via ATSEmapper which groups junctions into
ATSEs. Rather than re-running clustering (which might produce slightly different
groupings), we embed our ATSE event_id directly in the junction row name so
LeafCutter DS uses our pre-defined groupings.

The strand is parsed by LeafCutter as the suffix after the last underscore in the
cluster field, so the format chr:start:end:ATSE_ID_strand is unambiguous.

What this script does
---------------------
Reads only the var (column metadata) from the splice h5ad — no matrix loading —
and writes a reference TSV mapping each junction to its LeafCutter row name. This
file is used in the next step (01_pseudobulk_counts.py) when building the actual
counts matrix to ensure consistent junction naming.

Output columns:
  junction_id        : internal format used in h5ad var (chr_start_end_strand)
  event_id           : ATSE ID from ATSEmapper (ENSMUSG..._atse_N)
  leafcutter_junc_id : reformatted junction ID for LeafCutter row names
                       (chr:start:end:ATSE_ID_strand)
  gene_id            : Ensembl gene ID
  gene_name          : gene symbol

Note: ATSEs with only 1 junction will appear in this file but LeafCutter DS will
automatically skip them (status = SingleJunc) since a single-junction cluster has
no within-cluster variation to test.

Usage:
    python 00_export_junction_atse_map.py <splice_h5ad> <output_tsv>
"""

import sys
import h5py
import numpy as np
import pandas as pd


def read_h5ad_var(path):
    with h5py.File(path, "r") as f:
        grp = f["var"]
        idx = grp["_index"][:].astype(str) if "_index" in grp else None
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


def junction_to_leafcutter(junction_id, event_id):
    """
    chr10_81350189_81350272_- + ENSMUSG00000024012.18_atse_5
    → chr10:81350189:81350272:ENSMUSG00000024012.18_atse_5_-
    """
    # strand is the last _X where X is + or -
    parts = junction_id.rsplit("_", 1)
    coords = parts[0].replace("_", ":", 2)   # chr:start:end
    strand = parts[1]                          # + or -
    return f"{coords}:{event_id}_{strand}"


def main():
    if len(sys.argv) < 3:
        print("Usage: python 00_export_junction_atse_map.py <splice_h5ad> <output_tsv>")
        sys.exit(1)

    splice_h5ad = sys.argv[1]
    output_tsv  = sys.argv[2]

    print(f"Reading var from {splice_h5ad} ...", flush=True)
    var = read_h5ad_var(splice_h5ad)
    print(f"  {len(var):,} junctions", flush=True)

    keep_cols = [c for c in ["junction_id", "event_id", "gene_id", "gene_name"] if c in var.columns]
    df = var[keep_cols].copy()

    # Build LeafCutter junction ID
    df["leafcutter_junc_id"] = [
        junction_to_leafcutter(jid, eid)
        for jid, eid in zip(df["junction_id"], df["event_id"])
    ]

    # Summary
    n_atses = df["event_id"].nunique()
    print(f"  {n_atses:,} unique ATSEs", flush=True)
    atse_sizes = df.groupby("event_id", observed=True).size()
    print(f"  Junctions per ATSE: min={atse_sizes.min()} "
          f"median={atse_sizes.median()} max={atse_sizes.max()}", flush=True)
    print(f"  ATSEs with 1 junction: {(atse_sizes == 1).sum():,} "
          f"(will be SingleJunc in LeafCutter — skipped automatically)", flush=True)

    # Example
    print(f"\nExample junction → LeafCutter ID mapping:")
    for _, row in df.head(3).iterrows():
        print(f"  {row['junction_id']}  →  {row['leafcutter_junc_id']}")

    # Save
    df.to_csv(output_tsv, sep="\t", index=False)
    print(f"\nSaved {len(df):,} rows → {output_tsv}", flush=True)


if __name__ == "__main__":
    main()
