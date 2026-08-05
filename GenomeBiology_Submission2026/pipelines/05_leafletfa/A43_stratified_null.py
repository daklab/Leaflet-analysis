#!/usr/bin/env python
"""A43: is the null's heavy tail a COVERAGE MIXTURE, and does stratifying by coverage fix the FDP?

Why this exists
---------------
Three attempts to turn the TV ranking into an honest error rate have failed, all in the tail:
A39's posterior-derived null was 1.9x too tight, A40's permutation null 15% too tight at p90,
A42's half-normal bulk fit 2.1x too tight at p99. Reading them as three ideas was wrong -- they
are three instances of one problem, and the diagnostic that motivated this script says what it is:

    the pooled null statistic is a MIXTURE over ATSE coverage.

Null median falls 0.099 -> 0.018 across depth deciles (5.6x), while p99/median stays ~5 within
every decile. Same SHAPE, different SCALE. A mixture of same-shaped distributions at different
scales is heavier-tailed than any of its components, so no single-family fit can match its tail --
which is exactly and only where BH makes its decisions.

The test
--------
Stratify by ATSE coverage, compute an empirical null within each stratum, convert to p-values,
BH, and score realised FDP against the known truth.

The null here is an ORACLE: it uses the truly-null ATSEs, which are unavailable on real data. That
is deliberate. It measures the CEILING of what stratification can buy. If the oracle-stratified
null does not calibrate, no estimated version of it will, and coverage is not the missing
ingredient -- which saves building the permutation machinery a fourth time.

Three rules scored side by side, so the comparison is like for like:
    fixed-delta     the current rule, P(TV > 0.10) with Bayesian FDR   -- the incumbent
    oracle-pooled   empirical p against ALL true nulls, one distribution
    oracle-strat    empirical p against true nulls WITHIN the event's coverage stratum

pooled vs stratified is the whole question: same oracle information, differing only in whether
coverage is conditioned on.
"""

import argparse, os, sys
import h5py, numpy as np, pandas as pd, scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

W = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/"


def load_sim(path):
    f = h5py.File(path, "r")
    def dense(l):
        g = f["layers"][l]; s = tuple(g.attrs["shape"])
        return np.asarray(sp.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                                        shape=s).todense(), np.float32)
    def cat(n):
        if isinstance(n, h5py.Group):
            c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
            return c[n["codes"][:]]
        return np.asarray(n[:])
    ev = cat(f["var"]["event_id"]).astype(str); role = cat(f["var"]["junction_role"]).astype(str)
    uq, inv = np.unique(ev, return_inverse=True)
    ids = np.array([x.decode() if isinstance(x, bytes) else x
                    for x in f["uns"]["atse_ids"][:]]).astype(str)
    order = {a: i for i, a in enumerate(ids)}; tidx = np.array([order[a] for a in uq])
    ps = np.asarray(f["uns"]["true_psi_skip"])[:, tidx]
    K = ps.shape[0]; tpsi = np.zeros((K, len(ev))); sk = role == "J3"
    tpsi[:, sk] = ps[:, inv[sk]]; tpsi[:, ~sk] = (1 - ps[:, inv[~sk]]) / 2
    d = dict(clus=dense("cell_by_cluster_matrix"), uq=uq, inv=inv, role=role, tpsi=tpsi,
             delta=np.asarray(f["uns"]["atse_delta_psi"])[tidx],
             prog=cat(f["obs"]["true_program"]).astype(str))
    f.close(); return d


def bh(p, alpha):
    """Benjamini-Hochberg. Returns the boolean reject vector."""
    n = len(p); o = np.argsort(p); thr = alpha * np.arange(1, n + 1) / n
    ok = p[o] <= thr
    sig = np.zeros(n, bool)
    if ok.any():
        sig[o[:np.flatnonzero(ok)[-1] + 1]] = True
    return sig


def emp_p(score, null):
    """Empirical right-tail p. +1 in numerator and denominator: with B null draws no p can be
    below 1/(B+1), which is the honest floor -- claiming p=0 from a finite null is how a tail
    gets over-sold."""
    ns = np.sort(null)                      # ASCENDING -- searchsorted requires it
    n_ge = len(ns) - np.searchsorted(ns, score, side="left")   # how many null draws >= score
    return (1.0 + n_ge) / (len(ns) + 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fits", nargs="+", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-strata", type=int, default=10)
    ap.add_argument("--null-tol", type=float, default=0.05)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.20])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    s = load_sim(args.sim)
    n_atse = len(s["uq"])
    rep = np.array([np.flatnonzero(s["inv"] == e)[0] for e in range(n_atse)])
    skip = np.array([np.flatnonzero(s["inv"] == e)[
        np.flatnonzero(s["role"][s["inv"] == e] == "J3")[0]] for e in range(n_atse)])
    depth = s["clus"][:, rep].mean(0)
    truth = s["delta"] > args.null_tol
    K = s["tpsi"].shape[0]
    print(f"[1] {n_atse:,} ATSEs, {truth.sum():,} truly differential ({100*truth.mean():.1f}%)")
    print(f"    coverage {depth.min():.2f} - {depth.max():.1f}, median {np.median(depth):.2f}")

    # quantile strata on coverage; equal counts so every stratum has enough nulls for a tail
    edges = np.quantile(depth, np.linspace(0, 1, args.n_strata + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    stratum = np.clip(np.searchsorted(edges, depth, side="right") - 1, 0, args.n_strata - 1)

    rows = []
    for fp in args.fits:
        z = np.load(fp, allow_pickle=True)
        seed = int(z["seed"])
        C = np.corrcoef(z["psi_mean"], s["tpsi"])[:K, K:]
        r_, c_ = linear_sum_assignment(-np.nan_to_num(C))
        pm = np.empty(K, int); pm[c_] = r_
        psi = z["psi_mean"][pm]
        score = psi[:, skip].max(0) - psi[:, skip].min(0)

        null = score[~truth]
        p_pool = emp_p(score, null)
        p_str = np.empty(n_atse)
        for k in range(args.n_strata):
            m = stratum == k
            nk = score[m & ~truth]
            if len(nk) < 20:                       # too few nulls to read a tail from
                p_str[m] = p_pool[m]; continue
            p_str[m] = emp_p(score[m], nk)

        for name, p in (("oracle_pooled", p_pool), ("oracle_strat", p_str)):
            for a in args.alphas:
                sig = bh(p, a)
                ns = int(sig.sum()); nf = int((sig & ~truth).sum())
                rows.append(dict(seed=seed, rule=name, alpha=a, n_called=ns, n_false=nf,
                                 realised_fdp=(nf / ns) if ns else np.nan,
                                 power=float((sig & truth).sum() / truth.sum())))
        print(f"[2] seed {seed}: matched r {np.mean([C[i,j] for i,j in zip(r_,c_)]):.4f}, "
              f"null median {np.median(null):.4f} p99 {np.quantile(null,0.99):.4f}")

    R = pd.DataFrame(rows)
    tag = os.path.basename(args.sim).replace(".h5ad", "").replace("sim_heart_K5_", "")
    out = os.path.join(args.out, f"a43_stratified_null_{tag}.tsv")
    R.to_csv(out, sep="\t", index=False)

    print("\n=== realised FDP vs nominal (mean over seeds) ===")
    g = R.groupby(["rule", "alpha"])[["realised_fdp", "power", "n_called"]].mean()
    g["ratio"] = g["realised_fdp"] / g.index.get_level_values("alpha")
    print(g.round(4).to_string())
    print(f"\n-> {out}")
    print("\nratio ~1 = calibrated; >1 = anti-conservative (more false calls than promised).")


if __name__ == "__main__":
    main()
