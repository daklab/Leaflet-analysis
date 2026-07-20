#!/usr/bin/env python
"""
04_prepare_bam_list.py — BAM list for featureCounts, PRIMARY CELL TYPES only.

Filters fastqs/manifest.csv to sample_type == "cell type" (2,228 samples), maps each
SRR to its STAR BAM, and writes one absolute path per line.

Uses the PLAIN BAM (Aligned.sortedByCoord.out.bam), NOT the .CB.bam — the CB-tagged copy
exists only so ATSEmapper sees one pseudocell per sample; it adds nothing for gene quant.

Bulk mixed "tissue" samples (673) are excluded so GE matches the splice AnnData scope.
Cancer / normal cell lines were never downloaded.

Usage:
    python 04_prepare_bam_list.py [--wd <RNA_ATLAS dir>] [--out <bam_list.txt>]
"""

import argparse
import os
import sys
import pandas as pd

DEFAULT_WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/RNA_ATLAS/GSE138734"
BAM_NAME = "Aligned.sortedByCoord.out.bam"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wd", default=DEFAULT_WD)
    ap.add_argument("--out", default=None, help="Default: <wd>/bams_celltype_only.txt")
    args = ap.parse_args()

    out_path = args.out or os.path.join(args.wd, "bams_celltype_only.txt")
    manifest = os.path.join(args.wd, "fastqs", "manifest.csv")

    m = pd.read_csv(manifest)
    ct = m[m["sample_type"] == "cell type"]
    print(f"[bamlist] manifest rows: {len(m):,} | sample_type=='cell type': {len(ct):,}")

    paths, missing = [], []
    for srr in ct["srr"]:
        p = os.path.join(args.wd, "bams", str(srr), BAM_NAME)
        (paths if os.path.isfile(p) else missing).append(p)

    if missing:
        print(f"[bamlist] WARNING: {len(missing):,} BAMs missing, e.g.:", file=sys.stderr)
        for p in missing[:5]:
            print(f"    {p}", file=sys.stderr)

    with open(out_path, "w") as f:
        f.write("\n".join(paths) + "\n")

    batch_size = 50
    n_tasks = -(-len(paths) // batch_size)   # ceil
    print(f"[bamlist] {len(paths):,} BAMs → {out_path}")
    print(f"[bamlist] with BATCH_SIZE={batch_size}: submit --array=0-{n_tasks - 1}%15")


if __name__ == "__main__":
    main()
