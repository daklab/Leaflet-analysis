#!/usr/bin/env python
"""
Step 08-02b: Run LeafCutter DS for cell-type marker splicing events.

!! SUPERSEDED — DO NOT RE-RUN. Use 02c_run_leafcutter_ds_pairwise_celltypes.py. !!

Kept for provenance only. Two independent problems:

  1. It writes group labels straight from `broad_cell_type`, which contains spaces
     ("ENDOTHELIAL CELL", "B CELL"). LeafCutter parses the groups file
     whitespace-separated, so sample names truncated, "B CELL" and "T CELL" both
     became "CELL", and covariates shifted between columns. See lc_utils for the full
     account. Its 2026-06-20 output under $WD/LEAFCUTTER_DS/by_tissue/ is corrupted
     and 03_collect_ds_results.py flags it as `suspect`.
  2. It assumes LeafCutter handles >2 conditions natively. LeafCutter2's own help says
     "currently only two groups, i.e. pairwise, supported"; the multi-group form gives
     an omnibus p-value plus one-vs-baseline coefficients, which does not match
     LeafletFA's pairwise ΔPSI.

02c fixes both: sanitised labels validated against LeafCutter's own parser before
submission, and genuinely pairwise contrasts.

Comparison: cell type A vs cell type B within tissue, with age_group + sex
as covariates. This identifies junctions that are differentially spliced
between cell types, controlling for age — complementary to the age DS analysis
in 02_run_leafcutter_ds.py.

Strategy
--------
We group by BROAD tissue (e.g., Brain, Marrow, Heart) rather than fine tissue
(e.g., Brain_Isocortex_Visual) so that comparisons have more statistical power.
For example, EXCITATORY NEURON vs INHIBITORY NEURON is tested across all brain
regions simultaneously, with age_group + sex as covariates.

For each tissue with >=2 cell types (>=2 samples each):
  - Condition  = broad_cell_type (e.g., "MICROGLIA", "ENDOTHELIAL CELL")
  - Covariates = age_group + sex
  - LeafCutter DS runs a likelihood ratio test (full vs null model), which
    handles >2 conditions natively — no pairwise comparisons needed.

Groups file format (tab-separated, no header):
  sample_name  broad_cell_type  age_group  sex

Output per tissue (in output_dir/by_tissue/TISSUE/):
  counts_subset.tsv.gz
  groups.txt
  ds_results_cluster_significance.txt
  ds_results_effect_sizes.txt

Usage:
    python 02b_run_leafcutter_ds_celltype_markers.py \\
        --counts-file  <leafcutter_counts.tsv.gz> \\
        --metadata     <sample_metadata.csv> \\
        --output-dir   <output_directory> \\
        --min-samples-per-ct  2
"""

import argparse
import os
import re
import subprocess
import pandas as pd

LEAFCUTTER_DS = "/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/leafcutter-ds"


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_-]", "_", s)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts-file",         required=True)
    ap.add_argument("--metadata",            required=True)
    ap.add_argument("--output-dir",          required=True)
    ap.add_argument("--min-samples-per-ct",  type=int, default=2,
                    help="Min samples per broad_cell_type within tissue (default: 2)")
    ap.add_argument("--dry-run",             action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tissue_dir = os.path.join(args.output_dir, "by_tissue")
    os.makedirs(tissue_dir, exist_ok=True)

    # ── 1. Load metadata ──────────────────────────────────────────────────────
    print("[1] Loading sample metadata...", flush=True)
    meta = pd.read_csv(args.metadata)
    meta["age_group"] = meta["age"].map({"2m": "young", "3m": "young",
                                          "18m": "old",   "24m": "old"})

    # Broad tissue: first underscore-split token of tissue column
    # (groups Brain_Isocortex_Visual + Brain_Myeloid + Brain_Non-Myeloid → "Brain")
    meta["broad_tissue"] = meta["tissue"].str.split("_").str[0]

    # ── 2. Find testable tissues ──────────────────────────────────────────────
    print("\n[2] Identifying testable tissues...", flush=True)
    ct_counts = (
        meta.groupby(["broad_tissue", "broad_cell_type"])
        .size()
        .reset_index(name="n_samples")
    )
    # Keep cell types with enough samples
    ct_counts = ct_counts[ct_counts["n_samples"] >= args.min_samples_per_ct]
    # Keep tissues with >=2 cell types
    tissue_ct_n = ct_counts.groupby("broad_tissue")["broad_cell_type"].count()
    testable_tissues = tissue_ct_n[tissue_ct_n >= 2].index.tolist()

    print(f"  Testable tissues: {len(testable_tissues)}", flush=True)
    for t in sorted(testable_tissues):
        cts = ct_counts[ct_counts["broad_tissue"] == t]["broad_cell_type"].tolist()
        print(f"    {t}: {cts}", flush=True)

    # ── 3. Load counts matrix ─────────────────────────────────────────────────
    print(f"\n[3] Loading counts matrix...", flush=True)
    counts = pd.read_csv(args.counts_file, sep="\t", index_col=0, compression="gzip")
    print(f"  {counts.shape[0]:,} junctions × {counts.shape[1]} samples", flush=True)

    # ── 4. Per-tissue: write inputs and submit ────────────────────────────────
    print(f"\n[4] Writing per-tissue inputs and submitting jobs...", flush=True)
    submitted = 0
    skipped   = 0

    for tissue in sorted(testable_tissues):
        sname   = safe_name(tissue)
        out_t   = os.path.join(tissue_dir, sname)
        os.makedirs(out_t, exist_ok=True)

        sig_file = os.path.join(out_t, "ds_results_cluster_significance.txt")
        if os.path.exists(sig_file):
            print(f"  skip {tissue} — results already exist", flush=True)
            skipped += 1
            continue

        # Samples from this broad tissue with enough samples per cell type
        valid_cts = ct_counts[ct_counts["broad_tissue"] == tissue]["broad_cell_type"].tolist()
        t_meta = meta[
            (meta["broad_tissue"] == tissue) &
            (meta["broad_cell_type"].isin(valid_cts))
        ].copy()

        sample_names = [s for s in t_meta["sample_name"].tolist() if s in counts.columns]
        missing = len(t_meta) - len(sample_names)
        if missing:
            print(f"  WARNING: {missing} samples not in counts matrix for {tissue}", flush=True)

        counts_sub = counts[sample_names]
        active = counts_sub.sum(axis=1) > 0
        counts_sub = counts_sub[active]

        n_ct = t_meta["broad_cell_type"].nunique()
        print(f"\n  {tissue}: {len(sample_names)} samples, {n_ct} cell types, "
              f"{active.sum():,} active junctions", flush=True)

        # Write subset counts
        counts_path = os.path.join(out_t, "counts_subset.tsv.gz")
        counts_sub.to_csv(counts_path, sep="\t", compression="gzip", index_label="chrom")

        # Write groups file: condition=broad_cell_type, covariates=age_group + sex
        groups_path = os.path.join(out_t, "groups.txt")
        t_meta_ordered = t_meta.set_index("sample_name").loc[sample_names].reset_index()
        with open(groups_path, "w") as f:
            for _, row in t_meta_ordered.iterrows():
                f.write(f"{row['sample_name']}\t{row['broad_cell_type']}\t"
                        f"{row['age_group']}\t{row['sex']}\n")

        out_prefix = os.path.join(out_t, "ds_results")
        log_out    = os.path.join(out_t, "leafcutter_ds_%j.out")
        log_err    = os.path.join(out_t, "leafcutter_ds_%j.err")

        cmd = (
            f"export PYTHONUNBUFFERED=1; "
            f"{LEAFCUTTER_DS} "
            f"{counts_path} "
            f"{groups_path} "
            f"--output_prefix {out_prefix} "
            f"--num_threads 4 "
            f"--min_samples_per_intron 2 "
            f"--min_samples_per_group 2"
        )

        if args.dry_run:
            print(f"  [dry-run] {tissue}: {len(sample_names)} samples, "
                  f"{n_ct} conditions (cell types)", flush=True)
            print(f"    groups preview:", flush=True)
            ct_preview = t_meta_ordered.groupby("broad_cell_type").size()
            for ct, n in ct_preview.items():
                print(f"      {ct}: {n} samples", flush=True)
        else:
            result = subprocess.run([
                "sbatch",
                "--mem=16G",
                "--partition=cpu",
                "--cpus-per-task=4",
                f"--job-name=lc_ct_{sname[:20]}",
                f"--output={log_out}",
                f"--error={log_err}",
                "--wrap", cmd,
            ], capture_output=True, text=True)

            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                print(f"  submitted {tissue} → job {job_id}", flush=True)
                submitted += 1
            else:
                print(f"  ERROR {tissue}: {result.stderr.strip()}", flush=True)

    print(f"\n=== Done: {submitted} submitted, {skipped} skipped ===", flush=True)
    print(f"Results in: {tissue_dir}/", flush=True)


if __name__ == "__main__":
    main()
