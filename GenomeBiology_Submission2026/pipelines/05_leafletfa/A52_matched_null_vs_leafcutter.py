#!/usr/bin/env python
"""A52: LeafletFA's LRT on LeafCutter's OWN permutation null, so the calibration comparison is exact.

Why this is needed
------------------
The head-to-head calibration numbers reported so far pair two DIFFERENT null constructions:

    LeafCutter   shuffles DONOR labels within one cell type, across 7 vs 8 pseudobulk samples
    LeafletFA    shuffles phi across single cells, over the whole atlas

Both are permutation nulls on the same atlas and the same ATSEs, but they are not the same
experiment, so the direction of the result is defensible while the exact ratios are not. Worse, the
LeafletFA side is contaminated: permuting phi on a genuinely differential event turns real
between-program variation into an overdispersed mixture and inflates the null. That makes its ratio
an upper bound rather than a measurement.

This removes both problems by running our test on LeafCutter's construction.

The construction, matched
-------------------------
Within a single cell type there is no real between-group splicing difference, so any split of it is
a genuine null -- no contamination, unlike the phi permutation.

DONORS are shuffled, not cells. LeafCutter permutes donor labels, and donor is where batch structure
lives; splitting cells at random would ignore it and produce an artificially clean null that would
flatter LeafletFA. Each donor's cells move to an arm wholesale, with arm sizes matched to
LeafCutter's own manifest (7/8 for Heart cardiac muscle, 6/7 for Heart endothelial, 7/7 for Skin
keratinocyte -- 14-15 donors available in the atlas for each).

What remains different is the UNIT of observation: LeafCutter aggregates each donor to a pseudobulk
sample and tests across 15 of them; LeafletFA tests across the 864-2,299 single cells. That
difference is not a confound to be removed -- it IS the thing being measured.
"""

import argparse, os, re, sys
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import chi2

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from A45_dm_lrt import fit_event   # noqa: E402
from A49_ds_vs_leafcutter_lrt import norm, obs_cat, read_rows, H5, LC   # noqa: E402
from _phi_rows import phi_row_of                                        # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--perms", type=int, default=20, help="permutations per cell type")
    ap.add_argument("--max-atse", type=int, default=4000)
    ap.add_argument("--min-cells", type=int, default=30)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))

    z = np.load(args.fit, allow_pickle=True)
    conc_vec = np.asarray(z["bb_conc_vec"], np.float32) if "bb_conc_vec" in z else None
    with h5py.File(H5, "r") as f:
        n = f["var"]["event_id"]
        c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
        ev = c[n["codes"][:]].astype(str)
        J = len(ev)
        ct = obs_cat(f, "tissue_cell_type")
        donor = obs_cat(f, "mouse.id")
    uq, inv = np.unique(ev, return_inverse=True)
    n_atse = len(uq)
    amap = {}
    for a in np.unique(ct):
        amap.setdefault(norm(a), a)

    man = pd.read_csv(f"{LC}/permutation_null/permutation_manifest.tsv", sep="\t")
    man = man[man.status == "ready"]
    spec = man.groupby("cell_type").first().reset_index()[["cell_type", "n_armA", "n_armB"]]
    print(f"[1] {len(spec)} cell types with a LeafCutter permutation null", flush=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    for _, s in spec.iterrows():
        a = amap.get(norm(s.cell_type))
        if a is None:
            print(f"    {s.cell_type}: not in atlas, skipped"); continue
        cells = np.flatnonzero(ct == a)
        _pr = phi_row_of(z, len(ct)); cells = cells[_pr[cells] >= 0]
        dons = np.unique(donor[cells])
        nA = int(round(len(dons) * s.n_armA / (s.n_armA + s.n_armB)))
        nA = max(1, min(len(dons) - 1, nA))
        print(f"[2] {s.cell_type}: {len(cells):,} cells, {len(dons)} donors, "
              f"splitting {nA}/{len(dons)-nA} (LeafCutter used {int(s.n_armA)}/{int(s.n_armB)})",
              flush=True)
        y_all = read_rows("cell_by_junction_matrix", cells, J)
        cl_all = read_rows("cell_by_cluster_matrix", cells, J)
        dcell = donor[cells]
        idx = np.arange(n_atse)
        if args.max_atse and args.max_atse < n_atse:
            idx = np.sort(rng.choice(n_atse, args.max_atse, replace=False))
        for b in range(args.perms):
            sh = rng.permutation(dons)
            armB = np.isin(dcell, sh[nA:]).astype(np.float32)
            X_all = torch.from_numpy((armB - armB.mean()).reshape(-1, 1))
            n_ok = 0
            for e in idx:
                cols = np.flatnonzero(inv == e)
                m = len(cols)
                cov = cl_all[:, cols[0]] > 0
                if m < 2 or cov.sum() < args.min_cells:
                    continue
                yy = torch.from_numpy(y_all[cov][:, cols].astype(np.float32))
                Xc = X_all[torch.from_numpy(np.flatnonzero(cov))]
                c0 = float(conc_vec[e]) if conc_vec is not None else 50.0
                l0 = fit_event(yy, Xc, c0, args.steps, 1.0, alt=False, free_conc=True)
                l1 = fit_event(yy, Xc, c0, args.steps, 1.0, alt=True, free_conc=True)
                stat = max(2.0 * (l1 - l0), 0.0)
                df = m - 1
                rows.append(dict(cell_type=s.cell_type, perm=b, cluster=str(uq[e]), m=m,
                                 n_cells=int(cov.sum()), loglr=stat / 2.0, df=df,
                                 p=float(chi2.sf(stat, df))))
                n_ok += 1
            print(f"    perm {b}: {n_ok:,} ATSEs", flush=True)

    R = pd.DataFrame(rows)
    fp = os.path.join(args.out, "a52_matched_null.tsv")
    R.to_csv(fp, sep="\t", index=False)
    print(f"\n[3] {len(R):,} null p-values -> {fp}")
    print("\n=== LeafletFA on LeafCutter's OWN null construction ===")
    print(f"  {'nominal':>8} {'observed':>10} {'ratio':>7}")
    for al in (0.01, 0.05, 0.10, 0.20, 0.50):
        o = float((R['p'] <= al).mean())
        print(f"  {al:>8.2f} {o:>10.4f} {o/al:>7.2f}")
    print(f"\n  by event size:\n  {'m':>4} {'n':>9} {'ratio@0.05':>11}")
    for mm in sorted(R.m.unique())[:8]:
        k = R.m == mm
        if k.sum() < 200:
            continue
        print(f"  {int(mm):>4} {int(k.sum()):>9,} {float((R.p[k]<=0.05).mean())/0.05:>11.2f}")
    print("\n  Compare to LeafCutter on the same construction: 11.25 / 4.48 / 3.08 / 2.15 / 1.36")
    print("  and by m: 3.23 (m=2) rising to 11.35 (m=9).")


if __name__ == "__main__":
    main()
