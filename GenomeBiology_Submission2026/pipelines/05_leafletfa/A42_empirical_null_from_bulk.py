#!/usr/bin/env python
"""A42: turn a good ranking into an honest error rate, using the data's own null bulk.

Run: python A42_empirical_null_from_bulk.py --dir <DM_FRESH> --sim-dir <SIMULATION_DM_FRESH>

Where this sits
---------------
Ranking is not the problem. On fresh compositional simulations the DS statistic reaches
AUPRC 0.977 and AUROC 0.991 at effect 0.5, against a chance AUPRC of 0.279 -- yet the realised
false discovery proportion at nominal 0.05 is 0.147. A method that ranks that well should reach
5% FDR easily. The gap is entirely in the RULE that converts rank into an error rate.

Why the previous attempts failed
--------------------------------
  A39  drew the null from q(psi). The variational posterior is over-confident, so the null came
       out 1.9x too tight and the test was anti-conservative (FDP 0.579 at nominal 0.05).
  A40  built the null by permuting data. Better, but still 15% tight at the 90th percentile, and
       anti-conservative at low effect for the same reason.
  Jensen  moving |.| outside the posterior average reduces the point estimate's null bias
       (median 0.0282 -> 0.0183) but CANNOT fix prob_de, because that is a tail probability over
       individual draws and each draw carries the bias. Worth keeping; not a calibration fix.

The common failure is trying to MEASURE a null while leaving a biased statistic in place.

What this does instead
----------------------
Most ATSEs are not differential -- 72% here, and far more in real data. So the lower bulk of the
observed statistic IS the null distribution, already at the right scale, already including
whatever estimation noise and posterior over-confidence the model has. No permutation, no refit,
and no assumption that the posterior is calibrated.

  1. take the per-ATSE statistic (default: the Jensen point estimate)
  2. estimate the null from its lower quantiles -- everything below `--null-quantile` is treated
     as null-dominated, and a one-sided null is fitted to it
  3. p = P(null >= observed), then Benjamini-Hochberg
  4. score realised FDP against the KNOWN truth, which is the only test that matters

Truth is used ONLY to score, never to build the null, so the procedure is shippable on real data.

The obvious objection, stated up front: if the differential fraction were large enough that it
contaminated the lower bulk, the null would be too wide and the test over-conservative. That is
the safe direction, and `--null-quantile` controls the trade. It is reported alongside the result
rather than tuned to it.
"""

import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")


def bh(p):
    n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def load_sim(p):
    f = h5py.File(p, "r")
    def cat(n):
        if isinstance(n, h5py.Group):
            c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
            return c[n["codes"][:]]
        return np.asarray(n[:])
    ev = cat(f["var"]["event_id"]).astype(str); role = cat(f["var"]["junction_role"]).astype(str)
    uq, inv = np.unique(ev, return_inverse=True)
    ids = np.array([x.decode() if isinstance(x, bytes) else x for x in f["uns"]["atse_ids"][:]]).astype(str)
    order = {x: i for i, x in enumerate(ids)}; tidx = np.array([order[x] for x in uq])
    ps = np.asarray(f["uns"]["true_psi_skip"])[:, tidx]; K = ps.shape[0]
    tpsi = np.zeros((K, len(ev))); sk = role == "J3"
    tpsi[:, sk] = ps[:, inv[sk]]; tpsi[:, ~sk] = (1 - ps[:, inv[~sk]]) / 2
    d = dict(inv=inv, n_atse=len(uq), tpsi=tpsi, role=role,
             delta=np.asarray(f["uns"]["atse_delta_psi"])[tidx],
             prog=cat(f["obs"]["true_program"]).astype(str))
    f.close(); return d


def null_pvalues(score, q_null=0.60):
    """One-sided p from the lower bulk, which is null-dominated.

    A half-normal fitted to the lower `q_null` fraction: with the statistic non-negative and
    concentrated near zero under the null, the lower bulk pins its scale. Using a fitted form
    rather than the empirical quantiles matters because BH needs p-values far below 1/n_atse for
    the top-ranked events, and an empirical null of n draws floors at 1/(n+1).
    """
    from scipy.stats import halfnorm
    cut = np.quantile(score, q_null)
    bulk = score[score <= cut]
    # scale of a half-normal from a truncated sample: match the median, which is robust to the
    # differential events that leak into the upper part of the bulk
    sigma = np.median(bulk) / 0.6745 if len(bulk) else 1.0
    return halfnorm.sf(score, scale=max(sigma, 1e-12)), sigma, cut


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--sim-dir", required=True)
    ap.add_argument("--null-quantile", type=float, default=0.60)
    ap.add_argument("--null-tol", type=float, default=0.05)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.10, 0.20])
    args = ap.parse_args()

    rows = []
    for sp_ in sorted(glob.glob(os.path.join(args.sim_dir, "sim_heart_K5_dm50_eff*_seed0.h5ad"))):
        base = os.path.basename(sp_).replace(".h5ad", "")
        eff = float(re.search(r"eff([\d.]+)_", base).group(1))
        s = load_sim(sp_)
        truth = s["delta"] > args.null_tol
        skip = np.array([np.flatnonzero(s["inv"] == e)[
            np.flatnonzero(s["role"][s["inv"] == e] == "J3")[0]] for e in range(s["n_atse"])])
        for fp in sorted(glob.glob(os.path.join(
                args.dir, f"fit_{base}_dirichletmultinomial_simplex_seed*.npz"))):
            ts = int(re.search(r"seed(\d)\.npz", fp).group(1))
            z = np.load(fp, allow_pickle=True); K = s["tpsi"].shape[0]
            C = np.corrcoef(z["psi_mean"], s["tpsi"])[:K, K:]
            r_, c_ = linear_sum_assignment(-np.nan_to_num(C))
            pm = np.empty(K, int); pm[c_] = r_
            psi = z["psi_mean"][pm]
            score = psi[:, skip].max(0) - psi[:, skip].min(0)
            pv, sigma, cut = null_pvalues(score, args.null_quantile)
            q = bh(pv)
            for al in args.alphas:
                sel = q <= al
                n = int(sel.sum())
                rows.append(dict(eff=eff, seed=ts, alpha=al, n_called=n,
                                 realised_fdp=(int((sel & ~truth).sum()) / n) if n else np.nan,
                                 power=float((sel & truth).sum() / max(truth.sum(), 1)),
                                 null_sigma=sigma, null_cut=cut,
                                 true_null_frac=float((~truth).mean())))
    R = pd.DataFrame(rows)
    if R.empty:
        print("no fits found"); return
    out = os.path.join(args.dir, "A42_bulk_null.tsv")
    R.to_csv(out, sep="\t", index=False)
    print(f"empirical null from the lower {100*args.null_quantile:.0f}% of the statistic")
    print(f"true null fraction in these sims: {R.true_null_frac.iloc[0]:.3f}\n")
    for eff, g in R.groupby("eff"):
        print(f"=== effect {eff} ===")
        print(f"{'alpha':>6} {'realised FDP':>13} {'ratio':>7} {'power':>7} {'n called':>9}")
        for al, gg in g.groupby("alpha"):
            f_ = gg.realised_fdp.mean()
            print(f"{al:>6.2f} {f_:>13.4f} {f_/al:>7.2f} {gg.power.mean():>7.4f} "
                  f"{gg.n_called.mean():>9.0f}")
        print()
    print(f"-> {out}")


if __name__ == "__main__":
    main()
