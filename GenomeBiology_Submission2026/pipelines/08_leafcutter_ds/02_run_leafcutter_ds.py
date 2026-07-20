#!/usr/bin/env python
"""
Step 08-02: Run LeafCutter DS for each testable tissue_cell_type.

Strategy
--------
Comparison: young (2m + 3m) vs old (18m + 24m) within each tissue_cell_type.
Each tissue_cell_type is tested independently — samples from other cell types
are excluded so that the junction coverage profile is homogeneous within each run.

For each testable tissue_cell_type (>=2 young + >=2 old pseudobulk donors):
  1. Extract the relevant sample columns from the full counts matrix
  2. Write a per-cell-type subset counts file
  3. Write a groups file: sample_name  condition [sex covariate]
  4. Submit one SLURM CPU job running leafcutter-ds

Groups file format (tab-separated, no header):
  sample_name  condition  [covariate1 ...]
  Condition must be a string ("young"/"old").
  LeafCutter z-scores numeric covariates; one-hot encodes string covariates.
  We include sex as a covariate since it can affect splicing.

Output per cell type (in output_dir/by_celltype/CELLTYPE/):
  counts_subset.tsv.gz          subset of full counts matrix (this cell type only)
  groups.txt                    sample → condition + sex mapping
  ds_results_cluster_significance.txt
  ds_results_effect_sizes.txt

Usage:
    python 02_run_leafcutter_ds.py \\
        --counts-file  <leafcutter_counts.tsv.gz> \\
        --metadata     <sample_metadata.csv> \\
        --output-dir   <output_directory> \\
        --min-young    2 \\
        --min-old      2

Requirements:
    pip install -e /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/LeafCutter2
"""

import argparse
import os
import re
import subprocess
import gzip
import pandas as pd

LEAFCUTTER_DS = "/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/leafcutter-ds"


def safe_name(tissue_cell_type):
    """Convert tissue_cell_type to a filesystem-safe string."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", tissue_cell_type)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts-file", required=True,
                    help="leafcutter_counts.tsv.gz from 01_pseudobulk_counts.py")
    ap.add_argument("--metadata",    required=True,
                    help="sample_metadata.csv from 01_pseudobulk_counts.py")
    ap.add_argument("--output-dir",  required=True,
                    help="Base output directory; per-cell-type subdirs created here")
    ap.add_argument("--min-young",   type=int, default=2,
                    help="Minimum young (2m+3m) pseudobulk samples required (default: 2)")
    ap.add_argument("--min-old",     type=int, default=2,
                    help="Minimum old (18m+24m) pseudobulk samples required (default: 2)")
    ap.add_argument("--dry-run",     action="store_true",
                    help="Print what would be submitted without submitting")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ct_dir = os.path.join(args.output_dir, "by_celltype")
    os.makedirs(ct_dir, exist_ok=True)

    # ── 1. Load sample metadata ───────────────────────────────────────────────
    print("[1] Loading sample metadata...", flush=True)
    meta = pd.read_csv(args.metadata)
    meta["age_group"] = meta["age"].map({"2m": "young", "3m": "young",
                                          "18m": "old",   "24m": "old"})
    print(f"  {len(meta)} pseudobulk samples", flush=True)
    print(f"  Age groups: {meta['age_group'].value_counts().to_dict()}", flush=True)

    # ── 2. Identify testable tissue_cell_types ────────────────────────────────
    print("\n[2] Identifying testable tissue_cell_types...", flush=True)
    age_counts = (
        meta.groupby(["tissue_cell_type", "age_group"])
        .size()
        .unstack(fill_value=0)
    )
    age_counts.columns.name = None
    for col in ["young", "old"]:
        if col not in age_counts.columns:
            age_counts[col] = 0

    testable = age_counts[
        (age_counts["young"] >= args.min_young) &
        (age_counts["old"]   >= args.min_old)
    ].index.tolist()
    print(f"  Testable cell types: {len(testable)}", flush=True)
    for ct in testable:
        y = int(age_counts.loc[ct, "young"])
        o = int(age_counts.loc[ct, "old"])
        print(f"    {ct}: {y} young, {o} old", flush=True)

    # ── 3. Load full counts matrix ────────────────────────────────────────────
    # 24MB gzipped — loads fully into memory
    print(f"\n[3] Loading counts matrix from {args.counts_file} ...", flush=True)
    counts = pd.read_csv(args.counts_file, sep="\t", index_col=0, compression="gzip")
    print(f"  Shape: {counts.shape}  ({counts.shape[0]:,} junctions × {counts.shape[1]} samples)",
          flush=True)

    # ── 4. For each testable cell type: write inputs and submit ──────────────
    print(f"\n[4] Writing per-cell-type inputs and submitting jobs...", flush=True)

    submitted = 0
    skipped   = 0

    for ct in testable:
        sname = safe_name(ct)
        out_ct = os.path.join(ct_dir, sname)
        os.makedirs(out_ct, exist_ok=True)

        # Already done?
        sig_file = os.path.join(out_ct, "ds_results_cluster_significance.txt")
        if os.path.exists(sig_file):
            print(f"  skip {ct} — results already exist", flush=True)
            skipped += 1
            continue

        # Samples for this cell type
        ct_meta = meta[meta["tissue_cell_type"] == ct].copy()
        sample_names = ct_meta["sample_name"].tolist()

        # Subset counts matrix to these samples
        # Some sample names may have spaces — match by exact column name
        missing = [s for s in sample_names if s not in counts.columns]
        if missing:
            print(f"  WARNING: {len(missing)} samples not found in counts matrix for {ct}: {missing[:3]}")
            sample_names = [s for s in sample_names if s in counts.columns]

        counts_sub = counts[sample_names]

        # Drop junctions with zero reads in ALL samples for this cell type
        # (keeps the file small; LeafCutter would skip them anyway)
        active = counts_sub.sum(axis=1) > 0
        counts_sub = counts_sub[active]
        print(f"\n  {ct}: {len(sample_names)} samples, {active.sum():,} active junctions",
              flush=True)

        # Write subset counts file
        counts_path = os.path.join(out_ct, "counts_subset.tsv.gz")
        counts_sub.to_csv(counts_path, sep="\t", compression="gzip", index_label="chrom")

        # Write groups file: sample_name  condition  sex
        # No header — LeafCutter infers types (string→one-hot, numeric→z-score)
        groups_path = os.path.join(out_ct, "groups.txt")
        with open(groups_path, "w") as f:
            for _, row in ct_meta.iterrows():
                condition = row["age_group"]   # "young" or "old"
                sex       = row["sex"]          # "male" or "female"
                f.write(f"{row['sample_name']}\t{condition}\t{sex}\n")

        out_prefix = os.path.join(out_ct, "ds_results")
        log_out    = os.path.join(out_ct, "leafcutter_ds_%j.out")
        log_err    = os.path.join(out_ct, "leafcutter_ds_%j.err")

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
            print(f"  [dry-run] would submit: {ct}", flush=True)
            print(f"    counts: {counts_path}", flush=True)
            print(f"    groups: {groups_path}", flush=True)
        else:
            result = subprocess.run([
                "sbatch",
                "--mem=16G",
                "--partition=cpu",
                "--cpus-per-task=4",
                f"--job-name=lc_ds_{sname[:20]}",
                f"--output={log_out}",
                f"--error={log_err}",
                "--wrap", cmd,
            ], capture_output=True, text=True)

            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                print(f"  submitted {ct} → job {job_id}", flush=True)
                submitted += 1
            else:
                print(f"  ERROR {ct}: {result.stderr.strip()}", flush=True)

    print(f"\n=== Done: {submitted} submitted, {skipped} skipped ===", flush=True)
    print(f"Results in: {ct_dir}/", flush=True)


if __name__ == "__main__":
    main()
