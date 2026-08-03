#!/usr/bin/env python
"""
Step 08-06: run LeafCutter on the same simulations, so both methods are measured against the
same ground truth.

Why this is needed
------------------
LeafletFA's posterior expected FDP came out 3-4x anti-conservative on these simulations: a
nominal 0.05 delivers a realised false discovery proportion of 0.156 at effect 0.25. That number
means nothing on its own. If LeafCutter's BH-adjusted p-values are calibrated on the same data,
we have a real problem to fix. If LeafCutter is also over-calling, the comparison changes from
"our method is broken" to "this is hard, and here is where each method sits".

We already know the two methods differ sharply on a REAL-data donor-split null, where LeafCutter
falsely called 7.7% of tested events against LeafletFA's 0.14%. But that null has donor
structure LeafCutter does not model. The simulations have no donor structure at all, so this
isolates the methods from that confound.

Pseudobulk design, and its one artificial feature
-------------------------------------------------
Simulated cells have no donor, so LeafCutter has no natural replicate. Cells are split at random
within each true program into N synthetic replicates.

That is the only way to run LeafCutter here, but it must be stated plainly: random splits carry
NO between-replicate biological variation, so LeafCutter's overdispersion estimate is as small as
it can be and its p-values are as liberal as they will ever be. This is therefore a best case for
LeafCutter on the null side, and a fair test of whether its calibration holds when its model
assumptions are exactly satisfied.

Truth
-----
`uns["true_psi_skip"]` is [n_programs, n_ATSE], so the true effect for programs p and q is
exactly true_psi_skip[p] - true_psi_skip[q]. An ATSE counts as a true null when that is at or
below --null-tol, matching A28 so the two methods are scored identically.

Usage
-----
    python 06_leafcutter_on_simulations.py --sim <sim.h5ad> --out <dir> --n-reps 6 --submit
"""

import argparse
import os
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lc_utils import safe_name, assert_groups_file_parses  # noqa: E402

LEAFCUTTER_DS = "/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/leafcutter-ds"


def read_sim(path):
    with h5py.File(path, "r") as f:
        g = f["layers"]["cell_by_junction_matrix"]
        shape = tuple(g.attrs["shape"])
        junc = sp.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape)
        node = f["obs"]["true_program"]
        prog = (np.array(node[:]) if not isinstance(node, h5py.Group)
                else np.array(node["codes"][:]))
        ev_n = f["var"]["event_id"]
        if isinstance(ev_n, h5py.Group):
            c = np.array([x.decode() if isinstance(x, bytes) else x
                          for x in ev_n["categories"][:]])
            ev = pd.Categorical.from_codes(ev_n["codes"][:], c).astype(str)
        else:
            ev = np.array([x.decode() if isinstance(x, bytes) else x for x in ev_n[:]])
        jid_n = f["var"]["junction_id"]
        if isinstance(jid_n, h5py.Group):
            c = np.array([x.decode() if isinstance(x, bytes) else x
                          for x in jid_n["categories"][:]])
            jid = pd.Categorical.from_codes(jid_n["codes"][:], c).astype(str)
        else:
            jid = np.array([x.decode() if isinstance(x, bytes) else x for x in jid_n[:]])
        true_psi = f["uns"]["true_psi_skip"][:]
        atse_ids = np.array([x.decode() if isinstance(x, bytes) else x
                             for x in f["uns"]["atse_ids"][:]])
    return junc, prog, ev, jid, true_psi, atse_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-reps", type=int, default=6,
                    help="synthetic replicates per program (>=4 for LeafCutter)")
    ap.add_argument("--min-cells-per-rep", type=int, default=100)
    ap.add_argument("--num-threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    tag = os.path.basename(args.sim).replace(".h5ad", "")
    root = os.path.join(args.out, tag)
    os.makedirs(root, exist_ok=True)

    junc, prog, ev, jid, true_psi, atse_ids = read_sim(args.sim)
    progs = sorted(set(prog.tolist()))
    print(f"[1] {tag}: {junc.shape[0]:,} cells x {junc.shape[1]:,} junctions, "
          f"{len(progs)} programs", flush=True)

    # LeafCutter row names must embed our ATSE id so it tests OUR groupings, matching
    # 00_export_junction_atse_map.py's convention for the real data.
    rown = [f"chr1:{i}:{i+1}:{safe_name(e)}_+" for i, e in enumerate(ev)]

    rng = np.random.default_rng(args.seed)
    cols, meta = {}, []
    for p in progs:
        idx = np.flatnonzero(prog == p)
        if len(idx) < args.n_reps * args.min_cells_per_rep:
            print(f"    program {p}: only {len(idx)} cells, skipping", flush=True)
            continue
        parts = np.array_split(rng.permutation(idx), args.n_reps)
        for r, part in enumerate(parts):
            name = f"prog{p}_rep{r}"
            cols[name] = np.asarray(junc[part].sum(axis=0)).ravel().astype(np.int64)
            meta.append(dict(sample_name=name, program=f"prog{p}", n_cells=len(part)))
    M = pd.DataFrame(meta)
    counts = pd.DataFrame(cols, index=rown)
    print(f"[2] {counts.shape[1]} pseudobulk samples across "
          f"{M.program.nunique()} programs, {M.n_cells.median():.0f} cells each", flush=True)

    jobs = []
    for i in range(len(progs)):
        for j in range(i + 1, len(progs)):
            g1, g2 = f"prog{progs[i]}", f"prog{progs[j]}"
            sub = M[M.program.isin([g1, g2])]
            if sub.program.nunique() < 2 or sub.groupby("program").size().min() < 4:
                continue
            d = os.path.join(root, f"{g1}__vs__{g2}")
            os.makedirs(d, exist_ok=True)
            names = sub.sample_name.tolist()
            cs = counts[names]
            cs = cs[cs.sum(axis=1) > 0]
            cs.to_csv(os.path.join(d, "counts_subset.tsv.gz"), sep="\t",
                      compression="gzip", index_label="chrom")
            gp = os.path.join(d, "groups.txt")
            with open(gp, "w") as f:
                for _, r in sub.iterrows():
                    f.write(f"{r.sample_name}\t{r.program}\n")
            assert_groups_file_parses(gp, cs.columns, {g1, g2}, n_expected_cols=2)
            jobs.append((d, g1))

    # Ground truth for scoring, written alongside so the collector needs nothing else.
    np.savez_compressed(os.path.join(root, "truth.npz"),
                        true_psi=true_psi, atse_ids=atse_ids,
                        programs=np.array([str(p) for p in progs]))
    print(f"[3] {len(jobs)} contrasts prepared", flush=True)

    if not args.submit:
        print("DRY RUN — pass --submit to queue.", flush=True)
        return

    lst = os.path.join(root, "array_pairs.tsv")
    with open(lst, "w") as f:
        for d, b in jobs:
            f.write(f"{d}\t{b}\n")
    logs = os.path.join(root, "logs"); os.makedirs(logs, exist_ok=True)
    wrap = (f'export PYTHONUNBUFFERED=1; '
            f'L=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" {lst}); '
            f'D=$(echo "$L"|cut -f1); B=$(echo "$L"|cut -f2); '
            f'{LEAFCUTTER_DS} "$D/counts_subset.tsv.gz" "$D/groups.txt" '
            f'--output_prefix "$D/ds_results" --baseline_group "$B" '
            f'--num_threads {args.num_threads} '
            f'--min_samples_per_intron 4 --min_samples_per_group 4')
    res = subprocess.run([
        "sbatch", "--mem=16G", "--partition=cpu", f"--cpus-per-task={args.num_threads}",
        "--time=8:00:00", f"--array=0-{len(jobs)-1}%10", "--job-name=lc_sim",
        f"--output={logs}/%A_%a.out", f"--error={logs}/%A_%a.err",
        "--wrap", wrap], capture_output=True, text=True)
    print(res.stdout.strip() if not res.returncode else res.stderr.strip())


if __name__ == "__main__":
    main()
