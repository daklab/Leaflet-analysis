#!/usr/bin/env python
"""
Step 08-01: Export ATSE junctions to a BED file (coords + ATSE id).

Reads ONLY var from a splice h5ad (no matrix load) and writes a BED of every junction
with its ATSE grouping. Run on the FILTERED / ATSE-regrouped splice object (post-09b) so
the BED reflects the final ATSEs, not the pre-filter set.

Junction var_names are `chrom_start_end_strand` (chrom has no underscore for GRCh38
primary + the scaffold names here), so a right-split on '_' recovers the 4 fields.

Output: BED6 + 3 extra columns (headerless — standard BED; bedtools ignores extra cols):
    chrom  start  end  event_id  score  strand  junction_id  gene_id  gene_name
  - col4 (name) = event_id (ATSE id) so junctions sharing an ATSE share col4 → easy
    grouping / colouring in a genome browser.
  - col5 (score) = number of ATSE junctions in that event (0 if unknown), a rough
    "cluster size" that browsers can shade by.

⚠️ COORDINATE CONVENTION: coordinates are emitted verbatim as stored by the junction
extraction / ATSE mapping (regtools intron coordinates). They are NOT shifted to exon
boundaries or to strict BED 0-based half-open. If you intersect this against an
annotation-derived ATSE catalog, reconcile the ±1 intron-vs-exon offset first (WORKFLOW D6).

Usage:
    python 01_export_atse_junctions_bed.py <splice_h5ad> <output_bed>
        [--name-col event_id]      # what goes in BED col4 (default event_id)
"""

import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("splice_h5ad")
    ap.add_argument("output_bed")
    ap.add_argument("--name-col", default="event_id",
                    help="var column to place in BED col4 (default: event_id).")
    args = ap.parse_args()

    print(f"Reading var from {args.splice_h5ad} ...", flush=True)
    var = read_h5ad_var(args.splice_h5ad)
    print(f"  {len(var):,} junctions", flush=True)

    # junction id → chrom/start/end/strand. Prefer explicit junction_id col; else index.
    jids = var["junction_id"].astype(str).values if "junction_id" in var.columns \
        else var.index.astype(str).values
    split = pd.Series(jids).str.rsplit("_", n=3, expand=True)
    if split.shape[1] != 4:
        sys.exit(f"ERROR: could not parse chrom_start_end_strand from junction ids "
                 f"(got {split.shape[1]} fields). Example: {jids[0]}")
    chrom  = split[0].values
    start  = pd.to_numeric(split[1], errors="coerce")
    end    = pd.to_numeric(split[2], errors="coerce")
    strand = split[3].values
    if start.isna().any() or end.isna().any():
        bad = jids[np.where(start.isna() | end.isna())[0][:3]]
        sys.exit(f"ERROR: non-numeric start/end in junction ids, e.g. {list(bad)}")

    event = var[args.name_col].astype(str).values if args.name_col in var.columns \
        else np.array(["NA"] * len(var))

    # score = n junctions in this event (rough cluster size for browser shading)
    ev_size = pd.Series(event).map(pd.Series(event).value_counts()).clip(upper=1000).astype(int).values

    bed = pd.DataFrame({
        "chrom":  chrom,
        "start":  start.astype(int).values,
        "end":    end.astype(int).values,
        "name":   event,
        "score":  ev_size,
        "strand": strand,
        "junction_id": jids,
        "gene_id":   var["gene_id"].astype(str).values   if "gene_id"   in var.columns else "NA",
        "gene_name": var["gene_name"].astype(str).values if "gene_name" in var.columns else "NA",
    })
    bed = bed.sort_values(["chrom", "start", "end"]).reset_index(drop=True)

    n_atse = pd.Series(event).nunique()
    sizes = pd.Series(event).value_counts()
    print(f"  {n_atse:,} unique ATSEs (col4='{args.name_col}')", flush=True)
    print(f"  junctions/ATSE: min={sizes.min()} median={int(sizes.median())} max={sizes.max()}",
          flush=True)
    print(f"  singleton ATSEs (1 junction): {(sizes == 1).sum():,}", flush=True)

    bed.to_csv(args.output_bed, sep="\t", index=False, header=False)
    print(f"\nWrote {len(bed):,} junctions → {args.output_bed}", flush=True)
    print("Columns: chrom start end event_id score strand junction_id gene_id gene_name")


if __name__ == "__main__":
    main()
