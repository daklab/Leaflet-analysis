#!/usr/bin/env python
"""A61: which programs replicate -- across SEEDS at fixed K, and across K.

Two questions, deliberately kept apart
--------------------------------------
cross-seed   fit the same K three times from different inits, Hungarian-match programs on |corr(psi)|
             and report the matched correlation per program. This MEASURES effective K; it does not
             select K. On the earlier production fit, 8 of 20 programs came back at r 0.978-0.998
             while six sat below 0.6, with nothing at the boundary.

cross-K      match K=20's programs into K=10 and K=30. If the same programs reappear at a different
             rank, the structure survives changing the rank -- a stronger claim than "stable given
             K=20". If each K finds its own set, the honest statement is the weaker one.

Matching is on PSI only. psi is the program's identity (its junction composition); phi is the
per-cell encoding and is the degenerate direction, so matching on phi would manufacture agreement.
For cross-K the assignment is one-sided: every program of the smaller K is matched into the larger,
and the surplus programs of the larger K are simply unmatched -- that asymmetry is the point.
"""

import argparse, glob, itertools, json, os, re, sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"


def corr_matrix(a, b):
    """|corr| between every program of a [Ka, J] and every program of b [Kb, J]."""
    A = (a - a.mean(1, keepdims=True))
    B = (b - b.mean(1, keepdims=True))
    A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return np.abs(np.nan_to_num(A @ B.T))


def match(a, b):
    """Hungarian match a's programs into b's. Returns (cols, r) aligned to a's rows."""
    C = corr_matrix(a, b)
    r_, c_ = linear_sum_assignment(-C)
    cols = np.full(a.shape[0], -1, int)
    rs = np.zeros(a.shape[0])
    cols[r_] = c_
    rs[r_] = C[r_, c_]
    return cols, rs


def load(path):
    z = np.load(path, allow_pickle=True)
    return np.asarray(z["psi_mean"], float)


def arm_fits(root):
    out = {}
    for p in sorted(glob.glob(f"{root}/*/fit.npz")):
        n = os.path.basename(os.path.dirname(p))
        m = re.search(r"K(\d+)_seed(\d+)", n)
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="label=dir_of_run_dirs")
    ap.add_argument("--thresh", type=float, default=0.9,
                    help="matched |r| above which a program counts as replicating")
    ap.add_argument("--out", default=f"{WD}/ATLAS_DM/a61_replication.tsv")
    args = ap.parse_args()

    rows = []
    for spec in args.arms:
        lab, root = spec.split("=", 1)
        fits = arm_fits(root)
        Ks = sorted({k for k, _ in fits})
        print(f"\n{'='*70}\n{lab}: {len(fits)} fits, K in {Ks}\n{'='*70}", flush=True)

        # ---- cross-seed, per K -----------------------------------------------------------
        psi_by_K = {}
        for K in Ks:
            seeds = sorted(s for k, s in fits if k == K)
            P = {s: load(fits[(K, s)]) for s in seeds}
            psi_by_K[K] = P
            if len(seeds) < 2:
                continue
            # every program of the reference seed, best matched r against each other seed,
            # then the MINIMUM over the other seeds: a program replicates only if it comes
            # back in ALL of them, not in a lucky pair.
            ref = seeds[0]
            mins = np.full(K, np.inf)
            for s in seeds[1:]:
                _, r = match(P[ref], P[s])
                mins = np.minimum(mins, r)
            mins = np.sort(mins)[::-1]
            n_rep = int((mins >= args.thresh).sum())
            print(f"  K={K:<3} seeds {seeds}  replicating (min r >= {args.thresh}): "
                  f"{n_rep}/{K}")
            print("        matched r, sorted: " +
                  " ".join(f"{v:.3f}" for v in mins))
            for i, v in enumerate(mins):
                rows.append(dict(arm=lab, comparison="cross_seed", K=K, rank=i + 1,
                                 r=float(v), replicates=bool(v >= args.thresh)))

            # pairwise spread, so a single bad seed is visible
            pw = []
            for a, b in itertools.combinations(seeds, 2):
                _, r = match(P[a], P[b])
                pw.append((a, b, int((r >= args.thresh).sum())))
            print("        pairwise: " + "  ".join(f"s{a}-s{b}:{n}" for a, b, n in pw))

        # ---- cross-K: match each K's programs into every other K (seed 0) -----------------
        if len(Ks) > 1:
            print(f"\n  --- cross-K (seed 0 of each), one-sided into the larger rank ---")
            for Ka, Kb in itertools.combinations(Ks, 2):
                sa = sorted(psi_by_K[Ka]); sb = sorted(psi_by_K[Kb])
                if not sa or not sb:
                    continue
                A, B = psi_by_K[Ka][sa[0]], psi_by_K[Kb][sb[0]]
                _, r = match(A, B)                     # A is the smaller K, matched into B
                n = int((r >= args.thresh).sum())
                print(f"  K={Ka} -> K={Kb}: {n}/{Ka} programs recovered at r >= {args.thresh}"
                      f"   (median r {np.median(r):.3f})")
                print("        " + " ".join(f"{v:.3f}" for v in np.sort(r)[::-1]))
                for i, v in enumerate(np.sort(r)[::-1]):
                    rows.append(dict(arm=lab, comparison=f"K{Ka}_to_K{Kb}", K=Ka, rank=i + 1,
                                     r=float(v), replicates=bool(v >= args.thresh)))

    R = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    R.to_csv(args.out, sep="\t", index=False)
    print(f"\nsaved -> {args.out}")

    print("\n=== the decision numbers ===")
    for (arm, comp, K), d in R.groupby(["arm", "comparison", "K"]):
        print(f"  {arm:<12} {comp:<12} K={K:<3} {int(d.replicates.sum())}/{len(d)} replicate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
