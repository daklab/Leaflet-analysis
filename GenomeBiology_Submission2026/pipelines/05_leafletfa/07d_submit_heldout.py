#!/usr/bin/env python
"""
Step 07d: Submit 07c (held-out inferred-phi decoding) as a throttled GPU array
+ dependent collect job.

Usage:
    python 07d_submit_heldout.py <model_dir> <train_h5ad> <test_h5mu> <eval_output_dir>
        <runs> [max_concurrent=2]

    runs: comma-separated run names or "all".
    test_h5mu: any RESAMPLED test file (07c reads its *_original full-coverage layers).

GPU array (exclude ne1dg7-[001-011]); low throttle so it shares GPU with the
imputation array. Output: {run}_heldout_decode.csv + all_heldout_decode.csv
"""
import sys, os, glob, subprocess

PYTHON = "/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python"
SCRIPT = os.path.join("/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026",
                      "pipelines/05_leafletfa/07c_classify_heldout_phi.py")
EXCLUDE = "ne1dg7-[001-011]"


def collect(d):
    import pandas as pd
    fs = sorted(glob.glob(os.path.join(d, "*_heldout_decode.csv")))
    if not fs:
        print(f"No *_heldout_decode.csv in {d}"); return
    pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).to_csv(
        os.path.join(d, "all_heldout_decode.csv"), index=False)
    print(f"Aggregated {len(fs)} → {d}/all_heldout_decode.csv")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--collect":
        collect(sys.argv[2]); return
    if len(sys.argv) < 6:
        print(__doc__); sys.exit(1)
    model_dir, train_h5ad, test_h5mu, out_dir, runs_arg = sys.argv[1:6]
    max_conc = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    os.makedirs(out_dir, exist_ok=True)

    if runs_arg == "all":
        run_names = [os.path.basename(os.path.dirname(p)) for p in
                     sorted(glob.glob(os.path.join(model_dir, "run_*", "leafletfa_model.pkl.gz")))]
    else:
        run_names = [r.strip() for r in runs_arg.split(",") if r.strip()]

    tasks, skipped = [], 0
    for rn in run_names:
        rd = os.path.join(model_dir, rn)
        if not os.path.exists(os.path.join(rd, "leafletfa_model.pkl.gz")):
            print(f"  skip {rn} — no pkl"); continue
        oc = os.path.join(out_dir, f"{rn}_heldout_decode.csv")
        if os.path.exists(oc): skipped += 1; continue
        tasks.append((rd, oc))
    if not tasks:
        print(f"Nothing to submit ({skipped} exist). Collecting."); collect(out_dir); return

    tl = os.path.join(out_dir, "heldout_task_list.txt")
    with open(tl, "w") as f:
        for rd, oc in tasks: f.write(f"{rd}\t{oc}\n")
    print(f"Tasks: {len(tasks)}  skipped: {skipped}  max_concurrent: {max_conc}")

    arr = os.path.join(out_dir, "heldout_array_job.sh")
    with open(arr, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name=heldout_dec
#SBATCH --array=0-{len(tasks)-1}%{max_conc}
#SBATCH --mem=200G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude={EXCLUDE}
#SBATCH --time=8:00:00
#SBATCH --output={out_dir}/heldout_%A_%a.out
#SBATCH --error={out_dir}/heldout_%A_%a.err
export PYTHONUNBUFFERED=1
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "{tl}")
RUN_DIR=$(echo "$LINE" | cut -f1); OUT_CSV=$(echo "$LINE" | cut -f2)
echo "Task $SLURM_ARRAY_TASK_ID: $RUN_DIR"
{PYTHON} -u {SCRIPT} "$RUN_DIR" "{train_h5ad}" "{test_h5mu}" "$OUT_CSV" 4096 300 10
""")
    os.chmod(arr, 0o755)
    r = subprocess.run(["sbatch", arr], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"ERROR: {r.stderr.strip()}"); sys.exit(1)
    jid = r.stdout.strip().split()[-1]
    print(f"Submitted array {jid} ({len(tasks)} tasks, <={max_conc})")

    agg = os.path.join(out_dir, "heldout_collect_job.sh")
    with open(agg, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name=heldout_collect
#SBATCH --mem=16G
#SBATCH --partition=cpu
#SBATCH --time=00:20:00
#SBATCH --dependency=afterany:{jid}
#SBATCH --output={out_dir}/heldout_collect_%j.out
#SBATCH --error={out_dir}/heldout_collect_%j.err
export PYTHONUNBUFFERED=1
{PYTHON} -u {os.path.abspath(__file__)} --collect "{out_dir}"
""")
    os.chmod(agg, 0o755)
    r2 = subprocess.run(["sbatch", agg], capture_output=True, text=True)
    print(f"Collect: {r2.stdout.strip() or r2.stderr.strip()}")
    print(f"\nResults → {out_dir}/")


if __name__ == "__main__":
    main()
