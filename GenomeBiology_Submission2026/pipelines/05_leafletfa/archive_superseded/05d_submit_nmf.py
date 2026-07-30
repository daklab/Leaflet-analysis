#!/usr/bin/env python
"""
Step 05d: Submit NMF imputation evaluation jobs (one per K value).

Usage:
    python 05d_submit_nmf.py <train_h5ad> <resampled_dir> <eval_output_dir>
        [n_hvj=10000] [min_atse=30]

Submits one SLURM job per K in {5, 10, 20, 50}.
Output CSVs: <eval_output_dir>/nmf_K{k}_perjunc_hvj{n_hvj}_atse{min_atse}_results.csv
"""

import sys
import os
import subprocess

PYTHON = "/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python"
SCRIPT = os.path.join(
    "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026",
    "pipelines/05_leafletfa/04d_evaluate_imputation_nmf.py"
)

K_VALUES = [5, 10, 20, 50]


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    train_h5ad    = sys.argv[1]
    resampled_dir = sys.argv[2]
    eval_out_dir  = sys.argv[3]
    n_hvj         = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
    min_atse      = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    os.makedirs(eval_out_dir, exist_ok=True)

    submitted = 0
    skipped   = 0

    for K in K_VALUES:
        out_csv = os.path.join(eval_out_dir,
                               f"nmf_K{K}_perjunc_hvj{n_hvj}_atse{min_atse}_results.csv")

        if os.path.exists(out_csv):
            print(f"  skip K={K} — CSV exists: {out_csv}")
            skipped += 1
            continue

        # args: train_h5ad resampled_dir output_csv K n_hvj min_atse min_cells_per_junc=20
        cmd = (
            f"export PYTHONUNBUFFERED=1; "
            f"{PYTHON} {SCRIPT} "
            f"{train_h5ad} {resampled_dir} {out_csv} "
            f"{K} {n_hvj} {min_atse} 20"
        )

        result = subprocess.run([
            "sbatch",
            "--mem=120G",
            "--partition=cpu",
            "--cpus-per-task=4",
            "--time=12:00:00",
            f"--job-name=nmf_K{K}",
            f"--output={eval_out_dir}/nmf_K{K}_%j.out",
            f"--error={eval_out_dir}/nmf_K{K}_%j.err",
            "--wrap", cmd,
        ], capture_output=True, text=True)

        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"  submitted K={K} → job {job_id}")
            submitted += 1
        else:
            print(f"  ERROR K={K}: {result.stderr.strip()}")

    print(f"\nDone: {submitted} submitted, {skipped} skipped")
    print(f"Results → {eval_out_dir}/")


if __name__ == "__main__":
    main()
