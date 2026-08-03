#!/usr/bin/env python
"""
Step 08-05: the LeafCutter half of the donor-split permutation null.

Mirrors pipelines/05_leafletfa/A27 exactly -- same cell types, same donor-split design, same
"both arms are the same cell type so every call is a false positive" logic -- so the two
methods' type-I error can be compared on identical splits.

The cost asymmetry is the point
-------------------------------
LeafletFA needs NO refit: it is unsupervised, so permuting labels afterwards is free, and 100
permutations x 6 cell types x 2 modes ran in ~1 hour. LeafCutter must refit its
Dirichlet-multinomial GLM for every permutation (~16 min each), so the same statement costs a
SLURM array. That difference is itself a reportable result, which is why this script exists
rather than being skipped.

Budget accordingly: ~20 permutations on 2-3 cell types gives a point estimate with a usable
interval; matching LeafletFA's 100 x 6 would be ~200 CPU-hours for no extra insight.

Usage
-----
    python 05_permutation_null_leafcutter.py --lc-dir <LEAFCUTTER_DS_2026-08> \\
        --cell-types Heart_CARDIAC_MUSCLE_CELL Heart_ENDOTHELIAL_CELL \\
        --n-perm 20 --submit
"""

import argparse
import itertools
import os
import subprocess

import numpy as np
import pandas as pd

from lc_utils import safe_name, assert_groups_file_parses, select_covariates

LEAFCUTTER_DS = "/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/leafcutter-ds"
MIN_SAMPLES_PER_GROUP = 4


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lc-dir", required=True)
    ap.add_argument("--out-sub", default="permutation_null")
    ap.add_argument("--cell-types", nargs="+", default=None,
                    help="default: the cell types with the most donors")
    ap.add_argument("--n-cell-types", type=int, default=3)
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--num-threads", type=int, default=4)
    ap.add_argument("--max-concurrent", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    out_root = os.path.join(args.lc_dir, args.out_sub)
    os.makedirs(out_root, exist_ok=True)

    meta = pd.read_csv(os.path.join(args.lc_dir, "sample_metadata.csv"))
    meta = meta[meta.dataset == "tabula_muris_senis"].copy()
    meta["donor"] = meta.donor.fillna("").map(safe_name)
    meta["tct"] = meta.tissue_cell_type.map(safe_name)
    for c in ("sex", "age"):
        meta[c] = meta[c].map(safe_name)

    per = meta.groupby("tct").donor.nunique().sort_values(ascending=False)
    per = per[per >= 2 * MIN_SAMPLES_PER_GROUP]
    chosen = args.cell_types or per.head(args.n_cell_types).index.tolist()
    print(f"[1] {len(per)} cell types with >= {2*MIN_SAMPLES_PER_GROUP} donors; "
          f"using {chosen}", flush=True)

    counts = pd.read_csv(os.path.join(args.lc_dir, "leafcutter_counts.tsv.gz"),
                         sep="\t", index_col=0, compression="gzip")
    print(f"[2] counts {counts.shape[0]:,} x {counts.shape[1]}", flush=True)

    rng = np.random.default_rng(args.seed)
    jobs, manifest = [], []

    for ct in chosen:
        sub = meta[meta.tct == ct]
        donors = sorted(sub.donor.unique())
        print(f"\n[3] {ct}: {len(sub)} samples, {len(donors)} donors", flush=True)

        for p in range(args.n_perm):
            perm = rng.permutation(donors)
            h = len(perm) // 2
            arm = {d: ("armA" if d in set(perm[:h]) else "armB") for d in donors}
            s = sub.copy()
            s["group"] = s.donor.map(arm)
            s = s[s.sample_name.isin(counts.columns)]
            n1 = (s.group == "armA").sum()
            n2 = (s.group == "armB").sum()
            if min(n1, n2) < MIN_SAMPLES_PER_GROUP:
                manifest.append(dict(cell_type=ct, perm=p, status="skipped",
                                     reason=f"arms {n1}/{n2} below {MIN_SAMPLES_PER_GROUP}"))
                continue

            d = os.path.join(out_root, f"{ct}__perm{p:03d}")
            os.makedirs(d, exist_ok=True)
            names = s.sample_name.tolist()
            cs = counts[names]
            cs = cs[cs.sum(axis=1) > 0]
            cs.to_csv(os.path.join(d, "counts_subset.tsv.gz"), sep="\t",
                      compression="gzip", index_label="chrom")

            kept, dropped = select_covariates(s, "group", ["sex", "age"])
            gp = os.path.join(d, "groups.txt")
            with open(gp, "w") as f:
                for _, row in s.iterrows():
                    f.write("\t".join([row.sample_name, row.group]
                                      + [str(row[c]) for c in kept]) + "\n")
            assert_groups_file_parses(gp, cs.columns, {"armA", "armB"},
                                      n_expected_cols=2 + len(kept))

            jobs.append((d, "armA"))
            manifest.append(dict(cell_type=ct, perm=p, status="ready", n_armA=int(n1),
                                 n_armB=int(n2), n_junctions=len(cs),
                                 covariates=",".join(kept)))
        print(f"    {sum(1 for m in manifest if m['cell_type']==ct and m['status']=='ready')}"
              f" permutations prepared", flush=True)

    pd.DataFrame(manifest).to_csv(
        os.path.join(out_root, "permutation_manifest.tsv"), sep="\t", index=False)
    print(f"\n[4] {len(jobs)} jobs ready, manifest written", flush=True)

    if not args.submit:
        print("DRY RUN — re-run with --submit to queue the array.", flush=True)
        return

    lst = os.path.join(out_root, "array_perms.tsv")
    with open(lst, "w") as f:
        for d, base in jobs:
            f.write(f"{d}\t{base}\n")
    logs = os.path.join(out_root, "logs")
    os.makedirs(logs, exist_ok=True)
    wrap = (
        f'export PYTHONUNBUFFERED=1; '
        f'LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" {lst}); '
        f'D=$(echo "$LINE" | cut -f1); B=$(echo "$LINE" | cut -f2); '
        f'{LEAFCUTTER_DS} "$D/counts_subset.tsv.gz" "$D/groups.txt" '
        f'--output_prefix "$D/ds_results" --baseline_group "$B" '
        f'--num_threads {args.num_threads} '
        f'--min_samples_per_intron {MIN_SAMPLES_PER_GROUP} '
        f'--min_samples_per_group {MIN_SAMPLES_PER_GROUP}')
    res = subprocess.run([
        "sbatch", "--mem=16G", "--partition=cpu",
        f"--cpus-per-task={args.num_threads}", "--time=12:00:00",
        f"--array=0-{len(jobs)-1}%{args.max_concurrent}",
        "--job-name=lcnull",
        f"--output={logs}/perm_%A_%a.out", f"--error={logs}/perm_%A_%a.err",
        "--wrap", wrap], capture_output=True, text=True)
    if res.returncode:
        print("ERROR:", res.stderr.strip())
    else:
        print(f"array job {res.stdout.strip().split()[-1]}: {len(jobs)} tasks, "
              f"%{args.max_concurrent}", flush=True)


if __name__ == "__main__":
    main()
