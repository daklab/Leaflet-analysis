#!/usr/bin/env python
"""
Step 05b: Submit HVJ-filtered imputation evaluation jobs for all trained LeafletFA runs.

Calls 04b_evaluate_imputation_hvj.py per run. Results go to eval_output_dir.
Safe to re-run: existing CSVs are skipped.

Usage:
    python 05b_submit_evaluation_hvj.py <model_dir> <train_h5ad> <resampled_dir> <eval_output_dir>
        [n_hvj=10000] [min_atse=30]
"""

import sys
import os
import glob
import subprocess
import re

PYTHON = "/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python"
SCRIPT = os.path.join(
    "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026",
    "pipelines/05_leafletfa/04b_evaluate_imputation_hvj.py"
)


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)

    model_dir     = sys.argv[1]
    train_h5ad    = sys.argv[2]
    resampled_dir = sys.argv[3]
    eval_out_dir  = sys.argv[4]
    n_hvj         = int(sys.argv[5]) if len(sys.argv) > 5 else 10000
    min_atse      = int(sys.argv[6]) if len(sys.argv) > 6 else 30

    os.makedirs(eval_out_dir, exist_ok=True)

    completed = sorted(
        glob.glob(os.path.join(model_dir, "run_*", "leafletfa_model.pkl.gz")),
        key=lambda p: int(os.path.basename(os.path.dirname(p)).replace("run_", ""))
    )
    print(f"Trained runs found: {len(completed)}")

    submitted = 0
    skipped   = 0

    for pkl_path in completed:
        run_dir  = os.path.dirname(pkl_path)
        run_name = os.path.basename(run_dir)
        out_csv  = os.path.join(eval_out_dir, f"{run_name}_hvj{n_hvj}_atse{min_atse}_imputation_results.csv")

        if os.path.exists(out_csv):
            print(f"  skip {run_name} — CSV exists")
            skipped += 1
            continue

        cmd = (
            f"export PYTHONUNBUFFERED=1; "
            f"{PYTHON} {SCRIPT} "
            f"{run_dir} {train_h5ad} {resampled_dir} {out_csv} "
            f"2000 300 15 {n_hvj} {min_atse}"
        )

        result = subprocess.run([
            "sbatch",
            "--mem=200G",
            "--partition=cpu",
            "--cpus-per-task=4",
            "--time=24:00:00",
            f"--job-name=hvjeval_{run_name}",
            f"--output={eval_out_dir}/hvjeval_{run_name}_%j.out",
            f"--error={eval_out_dir}/hvjeval_{run_name}_%j.err",
            "--wrap", cmd,
        ], capture_output=True, text=True)

        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"  submitted {run_name} → job {job_id}")
            submitted += 1
        else:
            print(f"  ERROR {run_name}: {result.stderr.strip()}")

    print(f"\nDone: {submitted} submitted, {skipped} skipped")
    print(f"Results → {eval_out_dir}/")


if __name__ == "__main__":
    main()
