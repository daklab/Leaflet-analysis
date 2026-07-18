#!/usr/bin/env python
"""
Step 05e: Submit GPU imputation evaluation (04e) as a throttled SLURM array,
plus a dependent aggregation job that concatenates the per-run CSVs.

Usage:
    python 05e_submit_gpu.py <model_dir> <train_h5ad> <resampled_dir> <eval_output_dir>
        [n_hvj=10000] [min_atse=0] [phi_init=pi] [max_concurrent=5]

    # aggregation only (run manually if the dependency job did not fire):
    python 05e_submit_gpu.py --collect <eval_output_dir>

One array task per run_N. GPU directives mirror training's 02_submit_jobs.py:
partition=gpu, gres=gpu:1, exclude=ne1dg7-[001-011], %max_concurrent throttle
(default 5 — be fair to lab mates). Tasks whose CSV already exists are skipped.
Output CSVs: <eval_output_dir>/{run}_perjunc_hvj{n_hvj}_atse{min_atse}_gpu_results.csv
Aggregated:  <eval_output_dir>/all_gpu_imputation_results.csv
"""

import sys
import os
import glob
import subprocess

PYTHON = "/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python"
SCRIPT = os.path.join(
    "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026",
    "pipelines/05_leafletfa/04e_evaluate_imputation_gpu.py"
)
EXCLUDE_NODES = "ne1dg7-[001-011]"   # same special GPU nodes training excluded


def collect(eval_out_dir):
    """Concatenate every *_gpu_results.csv into one table."""
    import pandas as pd
    files = sorted(glob.glob(os.path.join(eval_out_dir, "*_gpu_results.csv")))
    if not files:
        print(f"No *_gpu_results.csv found in {eval_out_dir}")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out = os.path.join(eval_out_dir, "all_gpu_imputation_results.csv")
    df.to_csv(out, index=False)
    print(f"Aggregated {len(files)} run CSVs → {out}  ({len(df)} rows)")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--collect":
        collect(sys.argv[2])
        return

    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)

    model_dir      = sys.argv[1]
    train_h5ad     = sys.argv[2]
    resampled_dir  = sys.argv[3]
    eval_out_dir   = sys.argv[4]
    n_hvj          = int(sys.argv[5]) if len(sys.argv) > 5 else 10000
    min_atse       = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    phi_init       = sys.argv[7]      if len(sys.argv) > 7 else "pi"
    max_concurrent = int(sys.argv[8]) if len(sys.argv) > 8 else 5

    if phi_init not in ("pi", "atlas"):
        print(f"ERROR: phi_init must be 'pi' or 'atlas', got '{phi_init}'")
        sys.exit(1)

    os.makedirs(eval_out_dir, exist_ok=True)

    completed = sorted(
        glob.glob(os.path.join(model_dir, "run_*", "leafletfa_model.pkl.gz")),
        key=lambda p: int(os.path.basename(os.path.dirname(p)).replace("run_", ""))
    )
    print(f"Trained runs found: {len(completed)}   phi_init={phi_init}   "
          f"max_concurrent={max_concurrent}")

    # Build task list, skipping runs whose CSV already exists
    tasks, skipped = [], 0
    for pkl in completed:
        run_dir  = os.path.dirname(pkl)
        run_name = os.path.basename(run_dir)
        out_csv  = os.path.join(
            eval_out_dir,
            f"{run_name}_perjunc_hvj{n_hvj}_atse{min_atse}_gpu_results.csv"
        )
        if os.path.exists(out_csv):
            skipped += 1
            continue
        tasks.append((run_dir, out_csv))

    if not tasks:
        print(f"Nothing to submit — all {skipped} outputs exist. Collecting instead.")
        collect(eval_out_dir)
        return

    task_list_path = os.path.join(eval_out_dir, "gpu_task_list.txt")
    with open(task_list_path, "w") as f:
        for run_dir, out_csv in tasks:
            f.write(f"{run_dir}\t{out_csv}\n")
    print(f"Task list ({len(tasks)} tasks, {skipped} skipped) → {task_list_path}")

    n_tasks = len(tasks)
    array_script = os.path.join(eval_out_dir, "gpu_array_job.sh")
    with open(array_script, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name=gpu_imp
#SBATCH --array=0-{n_tasks - 1}%{max_concurrent}
#SBATCH --mem=150G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude={EXCLUDE_NODES}
#SBATCH --time=12:00:00
#SBATCH --output={eval_out_dir}/gpu_imp_%A_%a.out
#SBATCH --error={eval_out_dir}/gpu_imp_%A_%a.err

export PYTHONUNBUFFERED=1

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "{task_list_path}")
RUN_DIR=$(echo "$LINE" | cut -f1)
OUT_CSV=$(echo "$LINE" | cut -f2)
echo "Task $SLURM_ARRAY_TASK_ID: run=$RUN_DIR"

# 04e args: run_dir train resampled out_csv chunk epochs patience n_hvj min_atse min_cells phi_init
{PYTHON} -u {SCRIPT} \\
    "$RUN_DIR" \\
    "{train_h5ad}" \\
    "{resampled_dir}" \\
    "$OUT_CSV" \\
    4096 100 10 {n_hvj} {min_atse} 20 "{phi_init}"
""")
    os.chmod(array_script, 0o755)
    print(f"Array script → {array_script}")

    result = subprocess.run(["sbatch", array_script], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR submitting array job: {result.stderr.strip()}")
        sys.exit(1)
    array_job_id = result.stdout.strip().split()[-1]
    print(f"Submitted array job {array_job_id} ({n_tasks} tasks, ≤{max_concurrent} at once)")

    # Dependent aggregation job — runs once the whole array finishes (any exit state)
    agg_script = os.path.join(eval_out_dir, "gpu_collect_job.sh")
    with open(agg_script, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name=gpu_imp_collect
#SBATCH --mem=16G
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --dependency=afterany:{array_job_id}
#SBATCH --output={eval_out_dir}/gpu_collect_%j.out
#SBATCH --error={eval_out_dir}/gpu_collect_%j.err

export PYTHONUNBUFFERED=1
{PYTHON} -u {os.path.abspath(__file__)} --collect "{eval_out_dir}"
""")
    os.chmod(agg_script, 0o755)
    result = subprocess.run(["sbatch", agg_script], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Submitted collect job {result.stdout.strip().split()[-1]} "
              f"(runs after array {array_job_id})")
    else:
        print(f"WARNING: collect job not submitted: {result.stderr.strip()}")
        print(f"  run manually later: python {os.path.abspath(__file__)} "
              f"--collect {eval_out_dir}")

    print(f"\nResults → {eval_out_dir}/")


if __name__ == "__main__":
    main()
