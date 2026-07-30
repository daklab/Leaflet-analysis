#!/usr/bin/env python
"""
Step 10c: Submit LeafletFA training as a SLURM job array.

Reads parameter_combinations.json and submits a single SLURM array job with at
most --max-concurrent tasks running simultaneously (default 5).  A single array
job ID makes monitoring and cancellation easy and naturally throttles GPU use so
other lab members get fair access.

Usage:
    python pipelines/05_leafletfa/02_submit_jobs.py config.yaml \\
        <model_dir> <anndata.h5ad> [--max-concurrent N] [--dry-run]

    <model_dir>          directory containing parameter_combinations.json (Step 10b)
    <anndata.h5ad>       model-ready splice AnnData from Step 10a
    --max-concurrent N   cap on simultaneous GPU jobs (default: 5)
    --dry-run            write the array script and print the sbatch command;
                         do not actually submit
"""

import argparse
import sys
import os
import json
import datetime
import subprocess
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# Single bash script submitted as one SLURM array.
# $SLURM_ARRAY_TASK_ID maps directly to param_id in parameter_combinations.json.
ARRAY_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=leaflet
#SBATCH --output={log_dir}/leaflet_%A_%a.out
#SBATCH --error={log_dir}/leaflet_%A_%a.err
#SBATCH --mem=100G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --array=0-{last_idx}%{max_concurrent}
#SBATCH --exclude=ne1dg7-[001-011]

export PYTHONUNBUFFERED=1
{python_bin} -u {run_script} $SLURM_ARRAY_TASK_ID {model_dir} {anndata_file} 2>&1
"""

# Default training script: this repo's run_training.py (uses installed leafletfa)
DEFAULT_RUN_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "run_training.py"
)

# Full Python path avoids conda activation issues in SLURM (sh vs bash).
# Override via leafletfa.python_bin in config.
DEFAULT_PYTHON = "/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("model_dir")
    parser.add_argument("anndata_file")
    parser.add_argument(
        "--max-concurrent", type=int, default=5,
        help="Max simultaneous GPU jobs (default: 5, be fair to lab mates)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Write script and print sbatch command without submitting"
    )
    parser.add_argument(
        "--dependency", type=str, default=None,
        help="SLURM dependency string passed to sbatch, e.g. afterok:17266822"
    )
    args = parser.parse_args()

    cfg     = load_config(args.config)
    lfa_cfg = cfg.get("leafletfa", {})

    run_script = os.path.abspath(lfa_cfg.get("run_script", DEFAULT_RUN_SCRIPT))
    python_bin = lfa_cfg.get("python_bin", DEFAULT_PYTHON)

    for path in [run_script, args.anndata_file]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    param_file = os.path.join(args.model_dir, "parameter_combinations.json")
    with open(param_file) as f:
        param_list = json.load(f)
    n_jobs = len(param_list)
    print(f"Loaded {n_jobs} parameter sets from {param_file}")

    log_dir = os.path.join(args.model_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    script_content = ARRAY_TEMPLATE.format(
        log_dir=log_dir,
        last_idx=n_jobs - 1,
        max_concurrent=args.max_concurrent,
        python_bin=python_bin,
        run_script=run_script,
        model_dir=args.model_dir,
        anndata_file=args.anndata_file,
    )

    script_path = os.path.join(args.model_dir, "leaflet_array_job.sh")
    with open(script_path, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)

    print(f"Array script     → {script_path}")
    print(f"  Tasks          : {n_jobs}  (array indices 0–{n_jobs - 1})")
    print(f"  Max concurrent : {args.max_concurrent} GPU jobs at once")
    print(f"  Python         : {python_bin}")
    print(f"  Training script: {run_script}")
    print(f"  Logs           : {log_dir}/leaflet_<JOBID>_<TASKID>.out")

    sbatch_cmd = ["sbatch"]
    if args.dependency:
        sbatch_cmd += [f"--dependency={args.dependency}"]
    sbatch_cmd.append(script_path)

    if args.dry_run:
        print(f"\n[dry-run] To submit:")
        print(f"  {' '.join(sbatch_cmd)}")
        return

    result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"sbatch FAILED:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    slurm_msg = result.stdout.strip()
    job_id = slurm_msg.split()[-1]
    print(f"\nSubmitted: {slurm_msg}")
    print(f"  Monitor : squeue -u $USER | grep leaflet")
    print(f"  Cancel  : scancel {job_id}")

    log_path = os.path.join(args.model_dir, "job_submission_log.txt")
    with open(log_path, "w") as f:
        f.write(f"Submitted      : {datetime.datetime.now()}\n")
        f.write(f"Config         : {args.config}\n")
        f.write(f"Model dir      : {args.model_dir}\n")
        f.write(f"AnnData        : {args.anndata_file}\n")
        f.write(f"n_jobs         : {n_jobs}\n")
        f.write(f"max_concurrent : {args.max_concurrent}\n")
        f.write(f"dependency     : {args.dependency or 'none'}\n")
        f.write(f"SLURM job      : {slurm_msg}\n")
    print(f"  Log     : {log_path}")


if __name__ == "__main__":
    main()
