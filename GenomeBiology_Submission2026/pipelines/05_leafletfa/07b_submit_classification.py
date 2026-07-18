#!/usr/bin/env python
"""
Submit SLURM array job to run classifier evaluation across all LeafletFA runs.

Usage:
    python 07b_submit_classification.py <model_dir> <train_h5ad> <output_dir>
        [max_concurrent=10]

One job per run_N directory. Each job evaluates broad_cell_type, tissue_cell_type,
age (ridge regression), and dataset for that run vs. the permuted-label null.
The scVI baseline is decoupled — run it separately on cells matched to this training
split. Results are collected by 08b_collect_classification.py.

max_concurrent caps how many array tasks run at once (SLURM `--array=...%N`) so the
lab's cluster share is not overwhelmed. Default 10.
"""

import sys
import os
import glob
import subprocess

def main():
    if len(sys.argv) < 4:
        print("Usage: python 07b_submit_classification.py <model_dir> <train_h5ad> "
              "<output_dir> [max_concurrent=10]")
        sys.exit(1)

    model_dir      = sys.argv[1]
    train_h5ad     = sys.argv[2]
    out_dir        = sys.argv[3]
    max_concurrent = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    os.makedirs(out_dir, exist_ok=True)

    run_dirs = sorted(glob.glob(os.path.join(model_dir, "run_*")))
    n_runs   = len(run_dirs)
    print(f"Found {n_runs} run directories")

    PYTHON = "/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python"
    SCRIPT = os.path.join(
        "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026",
        "pipelines/05_leafletfa/07_classify_splicing_programs.py"
    )

    run_list_path = os.path.join(out_dir, "run_list.txt")
    with open(run_list_path, "w") as f:
        for d in run_dirs:
            f.write(d + "\n")
    print(f"Run list → {run_list_path}")

    array_script = os.path.join(out_dir, "classify_array_job.sh")
    with open(array_script, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name=lfa_clf
#SBATCH --array=0-{n_runs - 1}%{max_concurrent}
#SBATCH --mem=250G
#SBATCH --cpus-per-task=4
#SBATCH -p cpu
#SBATCH --output={out_dir}/clf_%A_%a.out
#SBATCH --error={out_dir}/clf_%A_%a.err

export PYTHONUNBUFFERED=1

RUN_DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {run_list_path})
echo "Evaluating $RUN_DIR"

{PYTHON} {SCRIPT} \\
    "{model_dir}" \\
    "{train_h5ad}" \\
    "{out_dir}" \\
    --run-dir "$RUN_DIR"
""")

    os.chmod(array_script, 0o755)
    print(f"Array script → {array_script}")
    print(f"\nSubmit command:\n  sbatch {array_script}")


if __name__ == "__main__":
    main()
