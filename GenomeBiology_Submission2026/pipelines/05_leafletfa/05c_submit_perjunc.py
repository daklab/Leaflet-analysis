#!/usr/bin/env python
"""
Step 05c: Submit per-junction evaluation as a THROTTLED SLURM array job.

Usage:
    python 05c_submit_perjunc.py <model_dir> <train_h5ad> <resampled_dir> <eval_output_dir>
        [n_hvj=10000] [min_atse=0] [phi_init=both] [max_concurrent=10]

phi_init controls the E-step phi initialization for each test cell:
    pi     — diversified Dirichlet centered at the learned pi (honest out-of-box
             baseline; each cell a distinct simplex point, no reference atlas)
    atlas  — nearest-training-cell phi (reference-atlas-guided init)
    both   — one array task per (run, mode) for each of pi and atlas (default)

max_concurrent caps how many array tasks run at once (SLURM `--array=...%N`),
so the lab's cluster share is not overwhelmed. Default 10.

This builds a task list + array script and submits ONE array job. Tasks whose
output CSV already exists are excluded up front (so restarts skip finished work).
Output CSVs: <eval_output_dir>/{run}_perjunc_hvj{n_hvj}_atse{min_atse}_{mode}_results.csv
"""

import sys
import os
import glob
import subprocess

PYTHON = "/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python"
SCRIPT = os.path.join(
    "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026",
    "pipelines/05_leafletfa/04c_evaluate_imputation_perjunc.py"
)


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)

    model_dir      = sys.argv[1]
    train_h5ad     = sys.argv[2]
    resampled_dir  = sys.argv[3]
    eval_out_dir   = sys.argv[4]
    n_hvj          = int(sys.argv[5]) if len(sys.argv) > 5 else 10000
    min_atse       = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    phi_init       = sys.argv[7]      if len(sys.argv) > 7 else "both"
    max_concurrent = int(sys.argv[8]) if len(sys.argv) > 8 else 10

    if phi_init not in ("pi", "atlas", "both"):
        print(f"ERROR: phi_init must be 'pi', 'atlas', or 'both', got '{phi_init}'")
        sys.exit(1)
    modes = ["pi", "atlas"] if phi_init == "both" else [phi_init]

    os.makedirs(eval_out_dir, exist_ok=True)

    completed = sorted(
        glob.glob(os.path.join(model_dir, "run_*", "leafletfa_model.pkl.gz")),
        key=lambda p: int(os.path.basename(os.path.dirname(p)).replace("run_", ""))
    )
    print(f"Trained runs found: {len(completed)}")
    print(f"phi-init modes: {modes}   max_concurrent: {max_concurrent}")

    # Build the task list, skipping any (run, mode) whose CSV already exists
    tasks   = []   # each: (run_dir, run_name, mode, out_csv)
    skipped = 0
    for pkl_path in completed:
        run_dir  = os.path.dirname(pkl_path)
        run_name = os.path.basename(run_dir)
        for mode in modes:
            out_csv = os.path.join(
                eval_out_dir,
                f"{run_name}_perjunc_hvj{n_hvj}_atse{min_atse}_{mode}_results.csv"
            )
            if os.path.exists(out_csv):
                skipped += 1
                continue
            tasks.append((run_dir, run_name, mode, out_csv))

    if not tasks:
        print(f"Nothing to submit — all {skipped} outputs already exist.")
        return

    # Write task list: one line per array task → "run_dir<TAB>mode<TAB>out_csv"
    task_list_path = os.path.join(eval_out_dir, "perjunc_task_list.txt")
    with open(task_list_path, "w") as f:
        for run_dir, run_name, mode, out_csv in tasks:
            f.write(f"{run_dir}\t{mode}\t{out_csv}\n")
    print(f"Task list ({len(tasks)} tasks, {skipped} skipped) → {task_list_path}")

    n_tasks = len(tasks)
    array_script = os.path.join(eval_out_dir, "perjunc_array_job.sh")
    with open(array_script, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name=pjeval
#SBATCH --array=0-{n_tasks - 1}%{max_concurrent}
#SBATCH --mem=200G
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output={eval_out_dir}/pjeval_%A_%a.out
#SBATCH --error={eval_out_dir}/pjeval_%A_%a.err

export PYTHONUNBUFFERED=1

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "{task_list_path}")
RUN_DIR=$(echo "$LINE" | cut -f1)
MODE=$(echo "$LINE" | cut -f2)
OUT_CSV=$(echo "$LINE" | cut -f3)
echo "Task $SLURM_ARRAY_TASK_ID: run=$RUN_DIR mode=$MODE"

# 04c args: run_dir train resampled out_csv chunk epochs patience n_hvj min_atse min_cells phi_init
{PYTHON} {SCRIPT} \\
    "$RUN_DIR" \\
    "{train_h5ad}" \\
    "{resampled_dir}" \\
    "$OUT_CSV" \\
    2000 300 15 {n_hvj} {min_atse} 20 "$MODE"
""")
    os.chmod(array_script, 0o755)
    print(f"Array script → {array_script}")

    result = subprocess.run(["sbatch", array_script], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Submitted: {result.stdout.strip()}")
        print(f"  {n_tasks} tasks, ≤{max_concurrent} running at once")
    else:
        print(f"ERROR submitting array job: {result.stderr.strip()}")
        sys.exit(1)

    print(f"\nResults → {eval_out_dir}/")


if __name__ == "__main__":
    main()
