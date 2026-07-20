#!/usr/bin/env python
"""
Step 08-03: Collect and summarize LeafCutter DS results across all completed runs.

Reads all ds_results_cluster_significance.txt + ds_results_effect_sizes.txt files
from both the age DS (by_celltype/) and cell-type marker DS (by_tissue/) outputs,
merges them, and writes a single ranked summary TSV for easy access to top hits.

Output columns
--------------
analysis       : "age_ds" or "celltype_ds"
condition      : tissue_cell_type (age_ds) or tissue (celltype_ds)
cluster        : ATSE cluster ID (e.g. chr11:ENSMUSG00000018381.15_atse_3_-)
gene_id        : parsed from cluster
atse_id        : parsed from cluster
chrom          : chromosome
strand         : + or -
status         : Success / SingleJunc / etc.
p              : raw LRT p-value
p_adjust       : BH-adjusted p-value
max_abs_dpsi   : max |deltaPSI| across all pairwise comparisons in this run
[dpsi cols]    : per-group deltaPSI (varies by run)

Usage:
    python 03_collect_ds_results.py \\
        --lc-dir    <LEAFCUTTER_DS base dir> \\
        --output    <output_summary.tsv> \\
        --padj-cutoff  0.05 \\
        --dpsi-cutoff  0.0

    --dpsi-cutoff: additional max|deltaPSI| filter (default 0 = no filter beyond p.adj)
"""

import argparse
import os
import glob
import re
import pandas as pd
import numpy as np


def parse_cluster_id(cluster):
    """
    chr11:ENSMUSG00000018381.15_atse_3_-
    → chrom=chr11, gene_id=ENSMUSG00000018381.15, atse_id=ENSMUSG00000018381.15_atse_3, strand=-
    """
    try:
        chrom, rest = cluster.split(":", 1)
        # strand is the last _+ or _-
        strand = rest[-1]
        atse_id = rest[:-2]  # strip _+ or _-
        # gene_id = everything before _atse_N
        gene_id = re.sub(r"_atse_\d+$", "", atse_id)
        return chrom, gene_id, atse_id, strand
    except Exception:
        return "", "", cluster, ""


def load_run(sig_path, ef_path, condition, analysis):
    """Load and merge significance + effect sizes for one run."""
    sig = pd.read_csv(sig_path, sep="\t")
    ef  = pd.read_csv(ef_path, sep="\t")

    # Build cluster key from intron ID: chr + ATSE_ID_strand (without coordinates)
    ef["cluster"] = ef["intron"].apply(
        lambda x: x.split(":")[0] + ":" + x.split(":")[3]
    )

    # All deltapsi columns
    dpsi_cols = [c for c in ef.columns if c.lower().startswith("deltapsi")]

    # Per-cluster max |deltaPSI| across all comparisons
    ef["abs_dpsi"] = ef[dpsi_cols].apply(pd.to_numeric, errors="coerce").abs().max(axis=1)
    ef_max = ef.groupby("cluster").agg(
        max_abs_dpsi=("abs_dpsi", "max"),
        **{f"mean_{c}": (c, lambda s: pd.to_numeric(s, errors="coerce").mean())
           for c in dpsi_cols}
    ).reset_index()

    # Filter significance to Success rows with valid p.adjust
    sig["p_adjust_num"] = pd.to_numeric(sig["p.adjust"], errors="coerce")
    sig["p_num"]        = pd.to_numeric(sig["p"],        errors="coerce")

    # Parse cluster fields
    sig[["chrom", "gene_id", "atse_id", "strand"]] = sig["cluster"].apply(
        lambda x: pd.Series(parse_cluster_id(x))
    )

    merged = sig.merge(ef_max, on="cluster", how="left")
    merged["analysis"]  = analysis
    merged["condition"] = condition

    # Rename for cleaner output
    merged = merged.rename(columns={"p": "p_raw", "p.adjust": "p_adjust",
                                     "p_num": "p_float", "p_adjust_num": "p_adjust_float"})

    keep = ["analysis", "condition", "cluster", "gene_id", "atse_id",
            "chrom", "strand", "status", "p_raw", "p_adjust",
            "max_abs_dpsi"] + [c for c in merged.columns if c.startswith("mean_deltapsi")]
    keep = [c for c in keep if c in merged.columns]
    return merged[keep], merged["p_adjust_float"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lc-dir",       required=True,
                    help="Base LEAFCUTTER_DS directory (contains by_celltype/ and by_tissue/)")
    ap.add_argument("--output",       required=True, help="Output TSV path")
    ap.add_argument("--padj-cutoff",  type=float, default=0.05)
    ap.add_argument("--dpsi-cutoff",  type=float, default=0.0,
                    help="Min max|deltaPSI| to include (default: 0 = all significant)")
    args = ap.parse_args()

    all_dfs = []
    total_sig = 0

    # ── Age DS (by_celltype) ──────────────────────────────────────────────────
    print("[age_ds] Scanning by_celltype/ ...")
    for sig_path in sorted(glob.glob(
            os.path.join(args.lc_dir, "by_celltype", "*", "ds_results_cluster_significance.txt"))):
        ef_path = sig_path.replace("cluster_significance", "effect_sizes")
        if not os.path.exists(ef_path):
            continue
        condition = os.path.basename(os.path.dirname(sig_path))
        df, padj = load_run(sig_path, ef_path, condition, "age_ds")
        n_sig = (padj < args.padj_cutoff).sum()
        n_dpsi = ((padj < args.padj_cutoff) & (df["max_abs_dpsi"].fillna(0) > args.dpsi_cutoff)).sum()
        print(f"  {condition:<45} sig={n_sig:>5}  |dPSI|>{args.dpsi_cutoff}={n_dpsi:>5}")
        total_sig += n_dpsi
        all_dfs.append(df[padj < args.padj_cutoff])

    # ── Cell-type DS (by_tissue) ──────────────────────────────────────────────
    print("\n[celltype_ds] Scanning by_tissue/ ...")
    for sig_path in sorted(glob.glob(
            os.path.join(args.lc_dir, "by_tissue", "*", "ds_results_cluster_significance.txt"))):
        ef_path = sig_path.replace("cluster_significance", "effect_sizes")
        if not os.path.exists(ef_path):
            continue
        condition = os.path.basename(os.path.dirname(sig_path))
        df, padj = load_run(sig_path, ef_path, condition, "celltype_ds")
        n_sig = (padj < args.padj_cutoff).sum()
        n_dpsi = ((padj < args.padj_cutoff) & (df["max_abs_dpsi"].fillna(0) > args.dpsi_cutoff)).sum()
        print(f"  {condition:<45} sig={n_sig:>5}  |dPSI|>{args.dpsi_cutoff}={n_dpsi:>5}")
        total_sig += n_dpsi
        all_dfs.append(df[padj < args.padj_cutoff])

    if not all_dfs:
        print("\nNo completed runs found.")
        return

    # ── Combine and rank ──────────────────────────────────────────────────────
    combined = pd.concat(all_dfs, ignore_index=True)
    combined["p_adjust_float"] = pd.to_numeric(combined["p_adjust"], errors="coerce")
    combined["max_abs_dpsi"]   = pd.to_numeric(combined["max_abs_dpsi"], errors="coerce")

    # Apply optional dpsi filter
    if args.dpsi_cutoff > 0:
        combined = combined[combined["max_abs_dpsi"] >= args.dpsi_cutoff]

    # Sort: by condition then p.adjust within each analysis type
    combined = combined.sort_values(["analysis", "condition", "p_adjust_float"])

    combined.to_csv(args.output, sep="\t", index=False)
    print(f"\n=== Summary ===")
    print(f"  Total significant ATSEs written: {len(combined):,}")
    print(f"  Analyses covered: {combined['analysis'].value_counts().to_dict()}")
    print(f"  Conditions covered: {combined['condition'].nunique()}")
    print(f"  Output → {args.output}")

    # Print top 5 hits overall by p.adjust
    print("\nTop 10 hits overall:")
    top = combined.nsmallest(10, "p_adjust_float")[
        ["analysis", "condition", "atse_id", "gene_id", "p_adjust", "max_abs_dpsi"]
    ]
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
