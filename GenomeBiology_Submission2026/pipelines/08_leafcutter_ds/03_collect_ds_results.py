#!/usr/bin/env python
"""
Step 08-03: Collect and summarize LeafCutter DS results across all completed runs.

Reads all ds_results_cluster_significance.txt + ds_results_effect_sizes.txt files,
merges them, and writes a single ranked summary TSV for easy access to top hits.

Scans three layouts:
  by_pair/<A>__vs__<B>/   pairwise cell-type DS from 02c   -> analysis "pairwise_ds"
  by_celltype/<CT>/       age DS from 02                   -> analysis "age_ds"
  by_tissue/<TISSUE>/     multi-group cell-type DS from 02b -> analysis "celltype_ds"

The by_celltype/ and by_tissue/ outputs from 2026-06-20 are CORRUPTED — see lc_utils
for the whitespace-parsing bug that merged distinct cell types and truncated sample
names. They are still scanned so a re-run of 02/02b can be collected, but anything
carrying the tell-tale signature (a bare "CELL" group, or a psi_Control column from
the nonexistent default baseline) is flagged in a `suspect` column rather than
silently ranked alongside good results.

Output columns
--------------
analysis       : "pairwise_ds", "age_ds" or "celltype_ds"
condition      : contrast name (pairwise_ds) / tissue_cell_type (age_ds) / tissue (celltype_ds)
group_1,group_2: the two contrasted groups (pairwise_ds only)
suspect        : non-empty if this run shows the 2026-06-20 corruption signature
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


def corruption_signature(ef_columns, groups_path=None):
    """Classify a run as corrupted, merely mislabelled, or clean.

    Returns (suspect, warning).

    `suspect` — the run's groups are wrong, so its RESULTS are wrong. Both fingerprints
    are direct consequences of LeafCutter's `sep='\\s+'` splitting labels that contain
    spaces (see lc_utils):
      * a bare `CELL` group — what "B CELL", "T CELL", "ENDOTHELIAL CELL" all collapse to
      * a groups file that parses ragged — covariates land in different columns per row

    `warning` — cosmetic only. A `psi_Control` column means `--baseline_group` was left
    at its default, so LeafCutter named the reference level "Control" even though it
    silently used the alphabetically-first real group. The test is valid; the column
    name and hence the SIGN convention of deltapsi is not self-describing. 02c always
    passes `-0` explicitly, so new runs never show this.
    """
    suspect, warning = [], []
    cols = set(ef_columns)

    bare = [c for c in cols if c.split("_", 1)[-1] == "CELL"]
    if bare:
        suspect.append(f"bare 'CELL' group ({', '.join(sorted(bare)[:3])})")

    if groups_path and os.path.exists(groups_path):
        parsed = pd.read_table(groups_path, header=None, sep=r"\s+")
        if parsed.isna().any().any():
            suspect.append("groups file parses ragged (covariates shifted)")

    if "psi_Control" in cols:
        warning.append("baseline left at default; deltapsi sign convention unlabelled")

    return "; ".join(suspect), "; ".join(warning)


def load_run(sig_path, ef_path, condition, analysis, group_1="", group_2=""):
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
    merged["group_1"]   = group_1
    merged["group_2"]   = group_2
    merged["suspect"], merged["warning"] = corruption_signature(
        ef.columns, os.path.join(os.path.dirname(sig_path), "groups.txt"))

    # Rename for cleaner output
    merged = merged.rename(columns={"p": "p_raw", "p.adjust": "p_adjust",
                                     "p_num": "p_float", "p_adjust_num": "p_adjust_float"})

    keep = ["analysis", "condition", "group_1", "group_2", "suspect", "warning",
            "cluster", "gene_id", "atse_id",
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
    n_suspect = 0

    def scan(subdir, analysis, split_pair=False):
        """Collect every completed run under one layout directory."""
        nonlocal total_sig, n_suspect
        pattern = os.path.join(args.lc_dir, subdir, "*",
                               "ds_results_cluster_significance.txt")
        found = sorted(glob.glob(pattern))
        print(f"\n[{analysis}] Scanning {subdir}/ ... ({len(found)} completed runs)")
        for sig_path in found:
            ef_path = sig_path.replace("cluster_significance", "effect_sizes")
            if not os.path.exists(ef_path):
                continue
            condition = os.path.basename(os.path.dirname(sig_path))
            g1, g2 = condition.split("__vs__") if split_pair and "__vs__" in condition \
                     else ("", "")
            df, padj = load_run(sig_path, ef_path, condition, analysis, g1, g2)
            n_sig = (padj < args.padj_cutoff).sum()
            n_dpsi = ((padj < args.padj_cutoff)
                      & (df["max_abs_dpsi"].fillna(0) > args.dpsi_cutoff)).sum()
            flag = df["suspect"].iloc[0] if len(df) else ""
            warn = df["warning"].iloc[0] if len(df) else ""
            if flag:
                n_suspect += 1
            print(f"  {condition:<55} sig={n_sig:>5}  "
                  f"|dPSI|>{args.dpsi_cutoff}={n_dpsi:>5}"
                  f"{'  ** SUSPECT' if flag else ''}")
            if flag:
                print(f"      CORRUPT: {flag}")
            if warn:
                print(f"      note:    {warn}")
            total_sig += n_dpsi
            all_dfs.append(df[padj < args.padj_cutoff])

    scan("by_pair",     "pairwise_ds", split_pair=True)   # 02c — the current analysis
    scan("by_celltype", "age_ds")                         # 02
    scan("by_tissue",   "celltype_ds")                    # 02b — superseded by 02c

    if not all_dfs:
        print("\nNo completed runs found.")
        return

    if n_suspect:
        print(f"\n  WARNING: {n_suspect} run(s) carry the 2026-06-20 corruption "
              f"signature and are flagged in the `suspect` column. Do not quote them.")

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
