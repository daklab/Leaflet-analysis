#!/usr/bin/env python
"""
Step 05: Submit imputation evaluation jobs for completed LeafletFA runs.

Finds all run_* directories with a saved leafletfa_model.pkl.gz (training
complete) and submits one CPU SLURM job per run calling 04_evaluate_imputation.py.

Each job writes its own per-run CSV (e.g. run_3_imputation_results.csv) to
avoid concurrent write conflicts. Use 06_collect_evaluation.py to merge them.

Safe to re-run: runs whose output CSV already exists are skipped.

Usage:
    python 05_submit_evaluation.py <model_dir> <train_h5ad> <resampled_dir> <eval_output_dir>

    model_dir       LEAFLETFA_MODELS/YYYY-MM-DD/ containing run_* subdirs
    train_h5ad      train_splice_model_ready_70pct_2026-06-17.h5ad (junction order reference)
    resampled_dir   MODEL_INPUT/20260526/resampled/ (must contain *_2026-06-17.h5mu files)
    eval_output_dir directory to write per-run CSVs and SLURM logs
"""

import sys
import os
import glob
import subprocess

PYTHON = "/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python"
SCRIPT = os.path.join(
    "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026",
    "pipelines/05_leafletfa/04_evaluate_imputation.py"
)


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)

    model_dir     = sys.argv[1]
    train_h5ad    = sys.argv[2]
    resampled_dir = sys.argv[3]
    eval_out_dir  = sys.argv[4]

    os.makedirs(eval_out_dir, exist_ok=True)

    # Only submit runs that have a completed model
    completed = sorted(
        glob.glob(os.path.join(model_dir, "run_*", "leafletfa_model.pkl.gz")),
        key=lambda p: int(os.path.basename(os.path.dirname(p)).replace("run_", ""))
    )
    print(f"Completed runs (have leafletfa_model.pkl.gz): {len(completed)}")

    submitted = 0
    skipped   = 0

    for pkl_path in completed:
        run_dir  = os.path.dirname(pkl_path)
        run_name = os.path.basename(run_dir)
        out_csv  = os.path.join(eval_out_dir, f"{run_name}_imputation_results.csv")

        if os.path.exists(out_csv):
            print(f"  skip {run_name} — output CSV already exists")
            skipped += 1
            continue

        # chunk_size=2000, num_epochs=300, patience=15
        cmd = (
            f"export PYTHONUNBUFFERED=1; "
            f"{PYTHON} {SCRIPT} "
            f"{run_dir} "
            f"{train_h5ad} "
            f"{resampled_dir} "
            f"{out_csv} "
            f"2000 300 15"
        )

        result = subprocess.run([
            "sbatch",
            "--mem=200G",
            "--partition=cpu",
            "--cpus-per-task=4",
            "--time=24:00:00",
            f"--job-name=eval_{run_name}",
            f"--output={eval_out_dir}/eval_{run_name}_%j.out",
            f"--error={eval_out_dir}/eval_{run_name}_%j.err",
            "--wrap", cmd,
        ], capture_output=True, text=True)

        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"  submitted {run_name} → job {job_id}")
            submitted += 1
        else:
            print(f"  ERROR submitting {run_name}: {result.stderr.strip()}")

    print(f"\nDone: {submitted} submitted, {skipped} skipped (already evaluated)")
    print(f"Logs and results in: {eval_out_dir}/")
    print(f"After jobs complete, run 06_collect_evaluation.py to merge CSVs")


if __name__ == "__main__":
    main()
