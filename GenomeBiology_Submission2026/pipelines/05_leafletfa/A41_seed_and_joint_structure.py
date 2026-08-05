#!/usr/bin/env python
"""A41: cross-seed reproducibility, and whether the model captures the JOINT structure.

Run: python A41_seed_and_joint_structure.py --dir <DM_TEST> --sim-dir <SIMULATION>

Two questions, both answerable from the fits A36 now saves.

1. CROSS-SEED (goal c)
   The 9x seed-spread reduction claimed for the simplex came from a toy with 60 ATSEs and K=4.
   These sims have 4,268 ATSEs, 12,804 junctions and 6,986 cells, so the claim is testable at
   real scale without touching the atlas. Per (sim, config), across seed pairs:
     - correlation of the ATSE-level TV statistic
     - overlap of the top-500 and top-2000 ATSEs by TV
     - psi split into its two parts: the within-ATSE SUM (the unidentified direction, r 0.51-0.66
       on released fits) and the COMPOSITION (the biology, r 0.95-0.96)
   Programs are matched between seeds by the Hungarian algorithm first -- program k in one fit is
   not program k in another, and comparing them unmatched would manufacture disagreement.

2. JOINT STRUCTURE
   The BetaBinomial gets the per-junction MARGINALS right and the JOINT wrong: it treats the
   junctions of an ATSE as independent, when in truth they compete for a fixed total and are
   therefore negatively correlated. Differential splicing is a statement about how usage is
   redistributed WITHIN an event, so the joint is exactly what it depends on.

   Each model implies a within-event correlation between two junctions of the same ATSE:
     Dirichlet-multinomial :  corr(y_i, y_j) = -sqrt( p_i p_j / ((1-p_i)(1-p_j)) )   < 0
     independent BetaBinomial : 0
   The observed correlation is measured directly from the counts across cells. Whichever model's
   implied value is closer to observed has the joint structure right.

   This needs only psi and phi, both saved, so no re-simulation and no stored concentration.
"""

import argparse
import glob
import itertools
import os
import re

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment


def load_fit(path):
    z = np.load(path, allow_pickle=True)
    return dict(psi=z["psi_mean"], phi=z["phi"], inv=z["inv_ev"].astype(np.int64),
                lik=str(z["likelihood"]), par=str(z["psi_parameterization"]),
                seed=int(z["seed"]), scale=z["psi_scale"])


def _safe_r(a, b, tol=1e-8):
    """Correlation, but NaN when either side is constant.

    Under the simplex every within-ATSE sum is exactly 1, so the 'sum correlation' is a
    correlation between two constant vectors -- undefined, and numpy returns whatever floating
    point noise happens to be there (measured 0.59-0.86, which reads as a real, mediocre
    agreement when in fact the quantity is perfectly pinned). Report NaN and use `sum_max_dev`
    instead for that case.
    """
    if np.std(a) < tol or np.std(b) < tol:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def split_psi(psi, inv, n_atse):
    """psi -> (within-ATSE sum, composition). The sum is the unidentified direction."""
    s = np.zeros((psi.shape[0], n_atse), dtype=np.float64)
    np.add.at(s.T, inv, psi.T.astype(np.float64))
    comp = psi / np.clip(s[:, inv], 1e-12, None)
    return s, comp


def match_programs(a, b):
    """Hungarian match of b's programs onto a's, by psi correlation."""
    C = np.corrcoef(a, b)[:a.shape[0], a.shape[0]:]
    r, c = linear_sum_assignment(-np.nan_to_num(C))
    return c, float(np.mean([C[i, j] for i, j in zip(r, c)]))


def tv_stat(psi, phi, c1, c2, inv, n_atse):
    d = (phi[c1].mean(0) - phi[c2].mean(0)) @ psi
    tv = np.zeros(n_atse)
    np.add.at(tv, inv, np.abs(d))
    return 0.5 * tv


def observed_within_event_corr(junc, clus, inv, n_atse, min_cells=200, max_events=400, seed=0):
    """Correlation between two junctions of the same ATSE, on RATIOS y_j / N_e.

    Must be ratios, not raw counts. Junctions of one event compete for a fixed total, so they
    are negatively correlated -- but only CONDITIONAL on coverage. On raw counts a deep cell has
    high counts on every junction and a shallow one low on every junction, and that shared
    sequencing depth swamps the competition entirely: measured +0.84 on these sims, which made
    the comparison against the model's compositional prediction meaningless. On ratios the same
    data gives -0.81 against the DM's implied -0.80.
    """
    rng = np.random.default_rng(seed)
    out = []
    events = rng.permutation(n_atse)[:max_events]
    for e in events:
        cols = np.flatnonzero(inv == e)
        if len(cols) < 2:
            continue
        cov = clus[:, cols[0]] > 0
        if cov.sum() < min_cells:
            continue
        Ne = clus[cov, cols[0]]
        for i, j in itertools.combinations(range(len(cols)), 2):
            a = junc[cov, cols[i]] / Ne
            b = junc[cov, cols[j]] / Ne
            if a.std() < 1e-8 or b.std() < 1e-8:
                continue
            out.append((e, cols[i], cols[j], float(np.corrcoef(a, b)[0, 1])))
    return pd.DataFrame(out, columns=["event", "j1", "j2", "obs_corr"])


def implied_corr_dm(p_i, p_j):
    """Within-event correlation a Dirichlet-multinomial implies for two junctions."""
    d = np.sqrt(np.clip(p_i * p_j / np.clip((1 - p_i) * (1 - p_j), 1e-12, None), 0, None))
    return -d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--sim-dir", default="/gpfs/commons/groups/knowles_lab/Karin/"
                                         "Leaflet-analysis-WD/SIMULATION")
    ap.add_argument("--top", type=int, nargs="+", default=[500, 2000])
    args = ap.parse_args()

    fits = sorted(glob.glob(os.path.join(args.dir, "fit_*.npz")))
    if not fits:
        print(f"no fit_*.npz under {args.dir} -- A36 must run with the saving change")
        return
    meta = []
    for f in fits:
        m = re.match(r"fit_(.+?)_(betabinomial|dirichletmultinomial)_(beta|simplex)_seed(\d)\.npz",
                     os.path.basename(f))
        if m:
            meta.append(dict(path=f, sim=m.group(1), cell=f"{m.group(2)}_{m.group(3)}",
                             seed=int(m.group(4))))
    M = pd.DataFrame(meta)
    print(f"loaded {len(M)} fits across {M.sim.nunique()} sims and {M.cell.nunique()} configs\n")

    # ── 1. cross-seed ───────────────────────────────────────────────────────────────────
    rows = []
    for (sim, cell), g in M.groupby(["sim", "cell"]):
        if len(g) < 2:
            print(f"  skip {sim} {cell}: only {len(g)} seed(s)")
            continue
        fs = {int(r.seed): load_fit(r.path) for r in g.itertuples()}
        n_atse = int(max(v["inv"].max() for v in fs.values())) + 1
        seeds = sorted(fs)
        for s1, s2 in itertools.combinations(seeds, 2):
            a, b = fs[s1], fs[s2]
            perm, psi_r = match_programs(a["psi"], b["psi"])
            bp = b["psi"][perm]
            sa, ca = split_psi(a["psi"], a["inv"], n_atse)
            sb, cb = split_psi(bp, b["inv"], n_atse)
            mult = np.bincount(a["inv"], minlength=n_atse)[a["inv"]] > 1
            # TV needs a contrast; use the two largest programs of fit a as a stand-in that is
            # defined identically for both fits after matching
            n1, n2 = np.argsort(-a["phi"].sum(0))[:2]
            c1 = np.flatnonzero(a["phi"].argmax(1) == n1)
            c2 = np.flatnonzero(a["phi"].argmax(1) == n2)
            if min(len(c1), len(c2)) < 50:
                tv_r = np.nan
                ov = {t: np.nan for t in args.top}
            else:
                # phi MUST be reordered by the same permutation as psi. tv_stat computes
                # (phi[c1].mean - phi[c2].mean) @ psi, so mixing b's program order in phi with
                # a's order in psi pairs every element with the wrong program and the resulting
                # correlation is meaningless. This was measuring nothing, not measuring poor
                # agreement.
                bphi = b["phi"][:, perm]
                t1 = tv_stat(a["psi"], a["phi"], c1, c2, a["inv"], n_atse)
                t2 = tv_stat(bp, bphi, c1, c2, b["inv"], n_atse)
                tv_r = float(np.corrcoef(t1, t2)[0, 1])
                ov = {t: len(np.intersect1d(np.argsort(-t1)[:t], np.argsort(-t2)[:t])) / t
                      for t in args.top}
            rows.append(dict(sim=sim, cell=cell, pair=f"{s1}-{s2}", psi_r=psi_r,
                             sum_r=_safe_r(sa.ravel(), sb.ravel()),
                             sum_max_dev=float(max(np.abs(sa - 1).max(),
                                                   np.abs(sb - 1).max())),
                             comp_r=float(np.corrcoef(ca[:, mult].ravel(),
                                                      cb[:, mult].ravel())[0, 1]),
                             tv_r=tv_r, **{f"top{t}": ov[t] for t in args.top}))
    S = pd.DataFrame(rows)
    if len(S):
        print("1. CROSS-SEED AGREEMENT (higher is better; mean over seed pairs)")
        cols = (["psi_r", "sum_r", "sum_max_dev", "comp_r", "tv_r"]
                + [f"top{t}" for t in args.top])
        print(S.groupby(["sim", "cell"])[cols].mean().round(4).to_string())
        print("\n   sum_r  NaN under the simplex BY DESIGN: every sum is exactly 1, so the")
        print("          correlation is between two constants and is undefined. Read sum_max_dev")
        print("          instead -- 0 means perfectly pinned.")
        print("   comp_r the biology; tv_r the DS statistic; topN the overlap of the calls\n")
        S.to_csv(os.path.join(args.dir, "a41_cross_seed.tsv"), sep="\t", index=False)

    # ── 2. joint structure ──────────────────────────────────────────────────────────────
    print("2. JOINT STRUCTURE: within-event correlation between junctions of the same ATSE")
    jrows = []
    for sim, g in M.groupby("sim"):
        h5 = os.path.join(args.sim_dir, f"{sim}.h5ad")
        if not os.path.exists(h5):
            print(f"   skip {sim}: {h5} not found")
            continue
        with h5py.File(h5, "r") as f:
            def dense(layer):
                gg = f["layers"][layer]
                shp = tuple(gg.attrs["shape"])
                return np.asarray(sp.csr_matrix((gg["data"][:], gg["indices"][:],
                                                 gg["indptr"][:]), shape=shp).todense(),
                                  dtype=np.float32)
            junc, clus = dense("cell_by_junction_matrix"), dense("cell_by_cluster_matrix")
        any_fit = load_fit(g.path.iloc[0])
        n_atse = int(any_fit["inv"].max()) + 1
        O = observed_within_event_corr(junc, clus, any_fit["inv"], n_atse)
        for cell, gg in g.groupby("cell"):
            fit = load_fit(gg.path.iloc[0])
            mu = fit["phi"] @ fit["psi"]
            pm = mu.mean(0)
            # normalise to the composition within each event before using the DM formula --
            # implied_corr_dm expects probabilities that sum to 1 over the event
            tot = np.zeros(n_atse)
            np.add.at(tot, fit["inv"], pm)
            pc = pm / np.clip(tot[fit["inv"]], 1e-12, None)
            imp = (implied_corr_dm(pc[O.j1.values], pc[O.j2.values])
                   if fit["lik"] == "dirichletmultinomial" else np.zeros(len(O)))
            jrows.append(dict(sim=sim, cell=cell, n_pairs=len(O),
                              observed_mean_corr=float(O.obs_corr.mean()),
                              implied_mean_corr=float(np.mean(imp)),
                              mean_abs_error=float(np.mean(np.abs(imp - O.obs_corr.values)))))
    if jrows:
        J = pd.DataFrame(jrows)
        print(J.round(4).to_string(index=False))
        print("\n   Observed is negative because junctions of one ATSE compete for a fixed total.")
        print("   The BetaBinomial implies exactly 0 -- it cannot represent that at all.")
        print("   Lower mean_abs_error = the joint structure is captured better.")
        J.to_csv(os.path.join(args.dir, "a41_joint_structure.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
