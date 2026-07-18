#!/usr/bin/env python
"""
Step 05f: Submit the within-CT imputation eval (04f) for a chosen set of runs,
as a throttled GPU array + dependent collect job.

Usage:
    python 05f_submit_withinCT.py <model_dir> <train_h5ad> <resampled_dir> <eval_output_dir>
        <runs> [n_hvj=10000] [sets=hvj,all] [max_concurrent=2]

    runs: comma-separated run names (e.g. run_91,run_61,run_85,run_11) or "all".
          04f is CHARACTERIZATION of chosen models — default to the top classification
          candidates, not all 96.
    sets: hvj | all | hvj,all  (all-junctions is memory/time heavy → 250G).

GPU directives mirror training (exclude ne1dg7-[001-011]). NMF is a baseline INSIDE
04f, so no separate NMF submission is needed.
Output: <eval_output_dir>/{run}_withinCT_hvj{n_hvj}_results.csv
        + all_withinCT_results.csv (collect job)
"""

import sys
import os
import glob
import subprocess

PYTHON = "/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python"
SCRIPT = os.path.join(
    "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026",
    "pipelines/05_leafletfa/04f_evaluate_imputation_withinCT.py"
)
EXCLUDE_NODES = "ne1dg7-[001-011]"


def collect(eval_out_dir):
    import pandas as pd
    files = sorted(glob.glob(os.path.join(eval_out_dir, "*_withinCT_*_results.csv")))
    if not files:
        print(f"No *_withinCT_*_results.csv in {eval_out_dir}"); return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out = os.path.join(eval_out_dir, "all_withinCT_results.csv")
    df.to_csv(out, index=False)
    print(f"Aggregated {len(files)} → {out}  ({len(df)} rows)")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--collect":
        collect(sys.argv[2]); return
    if len(sys.argv) < 6:
        print(__doc__); sys.exit(1)

    model_dir      = sys.argv[1]
    train_h5ad     = sys.argv[2]
    resampled_dir  = sys.argv[3]
    eval_out_dir   = sys.argv[4]
    runs_arg       = sys.argv[5]
    n_hvj          = int(sys.argv[6]) if len(sys.argv) > 6 else 10000
    sets           = sys.argv[7]      if len(sys.argv) > 7 else "hvj,all"
    max_concurrent = int(sys.argv[8]) if len(sys.argv) > 8 else 2

    os.makedirs(eval_out_dir, exist_ok=True)

    if runs_arg == "all":
        run_dirs = sorted(glob.glob(os.path.join(model_dir, "run_*", "leafletfa_model.pkl.gz")))
        run_names = [os.path.basename(os.path.dirname(p)) for p in run_dirs]
    else:
        run_names = [r.strip() for r in runs_arg.split(",") if r.strip()]
    print(f"Runs: {run_names}   sets={sets}   max_concurrent={max_concurrent}")

    tasks, skipped = [], 0
    for rn in run_names:
        rd = os.path.join(model_dir, rn)
        if not os.path.exists(os.path.join(rd, "leafletfa_model.pkl.gz")):
            print(f"  skip {rn} — no model pkl"); continue
        out_csv = os.path.join(eval_out_dir, f"{rn}_withinCT_hvj{n_hvj}_results.csv")
        if os.path.exists(out_csv):
            skipped += 1; continue
        tasks.append((rd, out_csv))

    if not tasks:
        print(f"Nothing to submit ({skipped} exist). Collecting."); collect(eval_out_dir); return

    task_list = os.path.join(eval_out_dir, "withinCT_task_list.txt")
    with open(task_list, "w") as f:
        for rd, oc in tasks:
            f.write(f"{rd}\t{oc}\n")
    print(f"Task list ({len(tasks)} tasks, {skipped} skipped) → {task_list}")

    array_script = os.path.join(eval_out_dir, "withinCT_array_job.sh")
    with open(array_script, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name=wct_imp
#SBATCH --array=0-{len(tasks) - 1}%{max_concurrent}
#SBATCH --mem=250G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude={EXCLUDE_NODES}
#SBATCH --time=16:00:00
#SBATCH --output={eval_out_dir}/wct_%A_%a.out
#SBATCH --error={eval_out_dir}/wct_%A_%a.err

export PYTHONUNBUFFERED=1
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "{task_list}")
RUN_DIR=$(echo "$LINE" | cut -f1)
OUT_CSV=$(echo "$LINE" | cut -f2)
echo "Task $SLURM_ARRAY_TASK_ID: $RUN_DIR"

# 04f args: run_dir train resampled out_csv chunk epochs patience n_hvj min_cells sets nmf_K
{PYTHON} -u {SCRIPT} \\
    "$RUN_DIR" "{train_h5ad}" "{resampled_dir}" "$OUT_CSV" \\
    4096 300 10 {n_hvj} 20 "{sets}" 0
""")
    os.chmod(array_script, 0o755)
    print(f"Array script → {array_script}")

    r = subprocess.run(["sbatch", array_script], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"ERROR: {r.stderr.strip()}"); sys.exit(1)
    jid = r.stdout.strip().split()[-1]
    print(f"Submitted array {jid} ({len(tasks)} tasks, <={max_concurrent} at once)")

    agg = os.path.join(eval_out_dir, "withinCT_collect_job.sh")
    with open(agg, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name=wct_collect
#SBATCH --mem=16G
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --dependency=afterany:{jid}
#SBATCH --output={eval_out_dir}/wct_collect_%j.out
#SBATCH --error={eval_out_dir}/wct_collect_%j.err
export PYTHONUNBUFFERED=1
{PYTHON} -u {os.path.abspath(__file__)} --collect "{eval_out_dir}"
""")
    os.chmod(agg, 0o755)
    r2 = subprocess.run(["sbatch", agg], capture_output=True, text=True)
    print(f"Collect job: {r2.stdout.strip() or r2.stderr.strip()}")
    print(f"\nResults → {eval_out_dir}/")


if __name__ == "__main__":
    main()
