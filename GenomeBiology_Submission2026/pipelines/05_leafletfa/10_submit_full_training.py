#!/usr/bin/env python
"""
Step 10: Retrain a FILTERED set of grid configs on the FULL dataset (all cells)
for the final paper model. Selects param_ids from parameter_combinations.json by
hyperparameter criteria (not a manual id list) and submits run_training.py as a
throttled GPU array.

Usage:
    python 10_submit_full_training.py <full_h5ad> <full_model_dir> <source_param_dir>
        [--K 20,50] [--lr 0.001,0.01] [--lik binom,bb] [--jsp True,False]
        [--waypoints True] [--max-concurrent 3] [--dry-run]

    full_h5ad:        all-cells splice model-ready h5ad (must have junc_ratio_centered)
    full_model_dir:   NEW output dir for the final full-data runs
    source_param_dir: dir with parameter_combinations.json/.csv (the trained grid)

Filters (comma lists) default to K{20,50} × lik{binom,bb} × lr{.001,.01} × jsp{T,F},
waypoints=True → 16 configs. GPU dirs mirror 02_submit_jobs (300G, exclude
ne1dg7-[001-011], 2-day). run_training.py reads config from the grid by param_id.
"""
import sys, os, shutil, json, argparse, subprocess

PYTHON = "/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python"
RUN_TRAIN = os.path.join(
    "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026",
    "pipelines/05_leafletfa/run_training.py")
EXCLUDE = "ne1dg7-[001-011]"


def _bools(s):  # "True,False" -> {True, False}
    return {x.strip().lower() in ("true", "1", "yes") for x in s.split(",")}


def likelihood_of(cfg):
    return "binom" if str(cfg.get("input_conc")).lower() in ("inf", "infinity") else "bb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("full_h5ad"); ap.add_argument("full_model_dir"); ap.add_argument("source_param_dir")
    ap.add_argument("--K", default="20,50")
    ap.add_argument("--lr", default="0.001,0.01")
    ap.add_argument("--lik", default="binom,bb")
    ap.add_argument("--jsp", default="True,False")
    ap.add_argument("--waypoints", default="True")
    ap.add_argument("--max-concurrent", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    Kset  = {int(x) for x in a.K.split(",")}
    lrset = {float(x) for x in a.lr.split(",")}
    likset = {x.strip() for x in a.lik.split(",")}
    jspset = _bools(a.jsp); wpset = _bools(a.waypoints)

    grid = json.load(open(os.path.join(a.source_param_dir, "parameter_combinations.json")))
    matched = []
    for pid, c in enumerate(grid):
        if (int(c["K"]) in Kset and float(c["lr"]) in lrset
                and likelihood_of(c) in likset
                and bool(c["junc_specific_prior"]) in jspset
                and bool(c["waypoints_use"]) in wpset):
            matched.append(pid)

    print(f"Filter: K={sorted(Kset)} lr={sorted(lrset)} lik={sorted(likset)} "
          f"jsp={sorted(jspset)} waypoints={sorted(wpset)}")
    print(f"Matched {len(matched)} param_ids: {matched}")
    for pid in matched:
        c = grid[pid]
        print(f"  {pid}: K={c['K']} wp={c['waypoints_use']} jsp={c['junc_specific_prior']} "
              f"lik={likelihood_of(c)} lr={c['lr']}")
    if not matched:
        print("No configs matched — check filters."); sys.exit(1)

    if not os.path.exists(a.full_h5ad):
        print(f"ERROR missing h5ad: {a.full_h5ad}"); sys.exit(1)
    os.makedirs(a.full_model_dir, exist_ok=True)
    for fn in ("parameter_combinations.json", "parameter_combinations.csv"):
        src = os.path.join(a.source_param_dir, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(a.full_model_dir, fn))
    log_dir = os.path.join(a.full_model_dir, "logs"); os.makedirs(log_dir, exist_ok=True)

    array_spec = ",".join(map(str, matched)) + f"%{a.max_concurrent}"
    script = os.path.join(a.full_model_dir, "full_train_array.sh")
    with open(script, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name=lfa_full
#SBATCH --output={log_dir}/full_%A_%a.out
#SBATCH --error={log_dir}/full_%A_%a.err
#SBATCH --mem=300G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --exclude={EXCLUDE}
#SBATCH --array={array_spec}

export PYTHONUNBUFFERED=1
{PYTHON} -u {RUN_TRAIN} $SLURM_ARRAY_TASK_ID {a.full_model_dir} {a.full_h5ad} 2>&1
""")
    os.chmod(script, 0o755)
    print(f"\nFull-data h5ad: {a.full_h5ad}")
    print(f"Array script → {script}   (<={a.max_concurrent} concurrent)")
    if a.dry_run:
        print("[dry-run] not submitting. Run without --dry-run to submit."); return
    r = subprocess.run(["sbatch", script], capture_output=True, text=True)
    print(f"Submitted: {r.stdout.strip()}" if r.returncode == 0 else f"ERROR: {r.stderr.strip()}")
    print(f"Models → {a.full_model_dir}/run_<param_id>/")


if __name__ == "__main__":
    main()
