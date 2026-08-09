#!/usr/bin/env python
"""A57: are we indexing the right cells and the right junctions, on the REAL atlas and a REAL fit?

Every check here recomputes the answer INDEPENDENTLY -- pandas groupby, boolean masks on the
original h5ad -- rather than calling the function under test and comparing it to itself. Six
indexing bugs have already shipped silently in this project, and four of them produced plausible
numbers rather than an exception:

  * phi rows vs atlas cell indices under --holdout-frac: A54 reported 0.045 where the truth was
    0.31, because the heart cells happened to sit below 119,435 so no bounds error was raised
  * an empirical p-value computed with searchsorted on a DESCENDING array
  * held-out null fitted on the held-out data, making every gain negative
  * array tasks that each ran all contrasts, so the concatenation stitched duplicates

So the standard here is: if a check can only fail loudly, it is not worth much. Each one below is
chosen because it could fail QUIETLY.

Run:
    python A57_index_sanity.py --fit <fit.npz>
"""

import argparse, os, sys

import h5py
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _phi_rows import phi_row_of, train_index          # noqa: E402

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from leafletfa.dirichlet_multinomial import build_atse_index, dm_log_prob   # noqa: E402

WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
H5 = f"{WD}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad"

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""), flush=True)


def cat(f, grp, name):
    n = f[grp][name]
    if isinstance(n, h5py.Group):
        c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
        return c[n["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in n[:]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--label-field", default="tissue_cell_type")
    args = ap.parse_args()

    z = np.load(args.fit, allow_pickle=True)
    with h5py.File(H5, "r") as f:
        ev = cat(f, "var", "event_id").astype(str)
        jid = cat(f, "var", "junction_id").astype(str)
        lab = cat(f, "obs", args.label_field).astype(str)
        N = len(lab)
        J = len(ev)
        gi = f["layers/cell_by_cluster_matrix"]
        indptr = gi["indptr"][:]
        indices = gi["indices"][:]
        data = gi["data"][:]
        gj = f["layers/cell_by_junction_matrix"]
        jptr, jind, jdat = gj["indptr"][:], gj["indices"][:], gj["data"][:]

    print(f"atlas: {N:,} cells x {J:,} junctions; fit phi {np.asarray(z['phi']).shape}\n")

    # ---------------------------------------------------------------- 1. ATSE grouping
    print("1. does atse_idx group the junctions the h5ad says belong together?")
    atse_idx, rep_idx, sizes = build_atse_index(ev)
    ai = atse_idx.numpy()
    # independent: pandas factorize, then compare the PARTITIONS (not the labels, which may differ)
    df = pd.DataFrame({"ev": ev, "ai": ai})
    per_ev_nunique = df.groupby("ev")["ai"].nunique()
    check("every ATSE maps to exactly one index",
          bool((per_ev_nunique == 1).all()),
          f"max distinct indices per ATSE = {int(per_ev_nunique.max())}")
    per_idx_nunique = df.groupby("ai")["ev"].nunique()
    check("no index collides two different ATSEs",
          bool((per_idx_nunique == 1).all()),
          f"max distinct ATSEs per index = {int(per_idx_nunique.max())}")
    check("sizes match a groupby count",
          bool(np.array_equal(np.sort(sizes.numpy()),
                              np.sort(df.groupby("ev").size().values))))
    # rep_idx must point at a junction that really is in that event
    rep = rep_idx.numpy()
    ok_rep = all(ai[rep[e]] == e for e in range(len(rep)) if rep[e] < J)
    check("rep_idx points into its own ATSE", ok_rep)

    # ---------------------------------------------------------------- 2. the denominator
    print("\n2. is N_e really the sum of the event's junctions, in the DATA?")
    # sample cells, rebuild both matrices densely for those rows only
    rng = np.random.default_rng(0)
    rows = rng.choice(N, 200, replace=False)
    bad_tot = 0
    worst = 0
    for c in rows:
        cl = np.zeros(J); cl[indices[indptr[c]:indptr[c+1]]] = data[indptr[c]:indptr[c+1]]
        jc = np.zeros(J); jc[jind[jptr[c]:jptr[c+1]]] = jdat[jptr[c]:jptr[c+1]]
        seg = np.bincount(ai, weights=jc, minlength=len(sizes))
        # cluster matrix holds the ATSE total broadcast to each junction of the event
        stored = np.bincount(ai, weights=cl, minlength=len(sizes)) / np.maximum(sizes.numpy(), 1)
        d = np.abs(seg - stored)
        worst = max(worst, d.max())
        bad_tot += int((d > 1e-6).sum())
    check("sum_j y_j == the stored ATSE total, exactly",
          bad_tot == 0, f"{bad_tot} mismatches over 200 cells, worst |diff| = {worst:.3g}")

    # ---------------------------------------------------------------- 3. psi compositions
    print("\n3. does psi sum to 1 within each ATSE, per program?")
    psi = np.asarray(z["psi_mean"], float)
    seg = np.zeros((psi.shape[0], len(sizes)))
    for k in range(psi.shape[0]):
        seg[k] = np.bincount(ai, weights=psi[k], minlength=len(sizes))
    check("all program-ATSE pairs sum to 1",
          bool(np.abs(seg - 1).max() < 1e-3), f"max deviation {np.abs(seg-1).max():.3e}")

    # ---------------------------------------------------------------- 4. phi rows vs cells
    print("\n4. do phi's rows correspond to the atlas cells we think they do?")
    phi = np.asarray(z["phi"])
    PROW = phi_row_of(z, N)
    ti = train_index(z, N)
    check("phi row count matches the training index", phi.shape[0] == len(ti),
          f"phi {phi.shape[0]:,} vs train_idx {len(ti):,}")
    check("the lookup is a bijection onto phi's rows",
          bool(np.array_equal(np.sort(PROW[PROW >= 0]), np.arange(phi.shape[0]))))
    check("held-out cells map to -1, never to a row",
          int((PROW < 0).sum()) == N - phi.shape[0],
          f"{int((PROW<0).sum()):,} mapped to -1")
    # the check that would have caught the silent 0.045: a cell's row must round-trip
    samp = rng.choice(np.flatnonzero(PROW >= 0), 1000, replace=False)
    check("cell -> row -> cell round-trips", bool(np.array_equal(ti[PROW[samp]], samp)))

    # ---------------------------------------------------------------- 5. group selection
    print("\n5. when we select a cell group by label, do we get those cells?")
    uq = pd.Series(lab).value_counts()
    g = uq.index[0]
    sel = np.flatnonzero(lab == g)
    check(f"label mask selects only '{g}' cells",
          bool((lab[sel] == g).all()), f"{len(sel):,} cells")
    selp = sel[PROW[sel] >= 0]
    check("group cells survive the phi-row filter with the labels intact",
          bool((lab[ti[PROW[selp]]] == g).all()),
          f"{len(selp):,} of {len(sel):,} are training cells")

    # ---------------------------------------------------------------- 6. one event end to end
    print("\n6. for one ATSE, are the junctions used the ones in that ATSE?")
    big = pd.Series(ev).value_counts()
    e_name = big.index[0]
    cols = np.flatnonzero(ev == e_name)
    check("column selection matches the event id",
          bool((ev[cols] == e_name).all()) and len(cols) == big.iloc[0],
          f"{e_name}: {len(cols)} junctions")
    check("those columns share one atse_idx", len(set(ai[cols].tolist())) == 1)
    check("and no other column carries that atse_idx",
          int((ai == ai[cols[0]]).sum()) == len(cols))
    # the model's composition for this event must sum to 1 over exactly these columns
    mu = phi[:200] @ psi
    s = mu[:, cols].sum(1)
    check("phi @ psi sums to 1 over the event's columns",
          bool(np.abs(s - 1).max() < 1e-3), f"max deviation {np.abs(s-1).max():.3e}")

    # ---------------------------------------------------------------- 7. likelihood invariance
    print("\n7. is the likelihood invariant to junction ORDER, as a composition must be?")
    perm = rng.permutation(J)
    y = torch.zeros(4, J)
    for i, c in enumerate(rows[:4]):
        v = np.zeros(J); v[jind[jptr[c]:jptr[c+1]]] = jdat[jptr[c]:jptr[c+1]]
        y[i] = torch.from_numpy(v.astype(np.float32))
    mu_t = torch.from_numpy(np.ascontiguousarray(mu[:4].astype(np.float32)))
    conc = torch.tensor(2.0)
    a1 = build_atse_index(ev)[0]
    a2 = build_atse_index(ev[perm])[0]
    l1 = float(dm_log_prob(y, mu_t, conc, a1, len(sizes)).sum())
    l2 = float(dm_log_prob(y[:, perm], mu_t[:, perm], conc, a2, len(sizes)).sum())
    check("permuting the junction axis leaves the log-likelihood unchanged",
          abs(l1 - l2) < max(1e-3, 1e-6 * abs(l1)), f"{l1:.6f} vs {l2:.6f}")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
