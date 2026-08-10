#!/usr/bin/env python
"""A60: how many cells must be averaged before phi is reproducible?

The claim this exists to scope
------------------------------
phi is estimated per cell from ~1,177 covered events at ~2-3 effective molecules each, so an
individual cell's loading is noisy. Averaging n cells reduces that noise, so "microglia use program
3 at 0.4" can be solid while "cell #48,201 uses program 3 at 0.4" is not.

That is a reasonable position but it is a HEDGE until it carries a number. This measures it: take
two independent seeds, match their programs, and compute the across-seed agreement of the GROUP-MEAN
phi as a function of group size. The output is a single sentence a reader can check every figure
against -- "phi reproduces at r >= 0.9 once averaged over N cells."

Two regimes, because they scope different claims
------------------------------------------------
  mixed        groups drawn from anywhere. Agreement here is dominated by cell-type differences, so
               this scopes CELL-TYPE-level claims (Figure 3's contrasts).
  within-type  groups drawn from ONE cell type. This asks whether the fits agree on variation
               among cells of the same type, i.e. whether sub-structure is resolvable at all. It is
               the harder and more honest number, and the one that scopes any claim about states
               within a cell type.

Programs are Hungarian-matched on psi and the SAME permutation is applied to phi -- matching phi
separately would let the two disagree and manufacture agreement that is not there.
"""

import argparse, glob, itertools, os, re, sys

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _phi_rows import phi_row_of                                     # noqa: E402

WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
H5 = f"{WD}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad"


def match(psi_a, psi_b):
    Ka, Kb = psi_a.shape[0], psi_b.shape[0]
    C = np.zeros((Ka, Kb))
    for i in range(Ka):
        for j in range(Kb):
            C[i, j] = abs(np.corrcoef(psi_a[i], psi_b[j])[0, 1])
    _, c = linear_sum_assignment(-np.nan_to_num(C))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", nargs="+", required=True, help="fit.npz of the SAME K, diff seeds")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[1, 2, 5, 10, 25, 50, 100, 250, 500, 1000])
    ap.add_argument("--groups", type=int, default=300, help="random groups per size")
    ap.add_argument("--min-celltype", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{WD}/ATLAS_DM/a60_phi_aggregation.tsv")
    args = ap.parse_args()

    with h5py.File(H5, "r") as f:
        x = f["obs"]["tissue_cell_type"]
        c = np.array([y.decode() if isinstance(y, bytes) else y for y in x["categories"][:]])
        lab = c[x["codes"][:]]
    N = len(lab)

    Z = [np.load(p, allow_pickle=True) for p in args.fits]
    psis = [np.asarray(z["psi_mean"], float) for z in Z]
    phis = [np.asarray(z["phi"], float)[phi_row_of(z, N)] for z in Z]
    K = phis[0].shape[1]
    print(f"{len(Z)} fits, K={K}, {N:,} cells", flush=True)

    # align every fit onto fit 0's program order
    aligned = [phis[0]]
    for i in range(1, len(Z)):
        aligned.append(phis[i][:, match(psis[0], psis[i])])

    rng = np.random.default_rng(args.seed)
    big = [ct for ct, n in zip(*np.unique(lab, return_counts=True)) if n >= args.min_celltype]
    print(f"cell types with >= {args.min_celltype:,} cells: {len(big)}", flush=True)

    rows = []
    for regime in ("mixed", "within-type"):
        for n in args.sizes:
            groups = []
            for _ in range(args.groups):
                if regime == "mixed":
                    pool = np.arange(N)
                else:
                    ct = big[rng.integers(len(big))]
                    pool = np.flatnonzero(lab == ct)
                if len(pool) < n:
                    continue
                groups.append(rng.choice(pool, n, replace=False))
            if len(groups) < 30:
                continue
            # [n_groups, K] mean phi per fit
            M = [np.stack([P[g].mean(0) for g in groups]) for P in aligned]
            rs = []
            for a, b in itertools.combinations(range(len(M)), 2):
                # correlate across groups, per program, then average over programs: a single
                # flattened correlation would be dominated by differences in program SIZE rather
                # than by whether the fits agree on how groups differ.
                per_k = []
                for k in range(K):
                    x_, y_ = M[a][:, k], M[b][:, k]
                    if x_.std() > 1e-12 and y_.std() > 1e-12:
                        per_k.append(np.corrcoef(x_, y_)[0, 1])
                if per_k:
                    rs.append(float(np.mean(per_k)))
            if rs:
                rows.append(dict(regime=regime, n_cells=n, r=float(np.mean(rs)),
                                 r_min=float(np.min(rs)), n_pairs=len(rs)))
                print(f"  {regime:<12} n={n:<5} r={np.mean(rs):.3f}", flush=True)

    R = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    R.to_csv(args.out, sep="\t", index=False)

    print("\n=== the number that scopes every phi claim ===")
    for regime, d in R.groupby("regime"):
        ok = d[d.r >= 0.9].n_cells
        thr = int(ok.min()) if len(ok) else None
        if thr is None:
            print(f"  {regime:<12} NEVER reaches r >= 0.9 up to n={int(d.n_cells.max())} "
                  f"(best r = {d.r.max():.3f})")
        else:
            print(f"  {regime:<12} phi reproduces at r >= 0.9 once averaged over {thr} cells")
    print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
