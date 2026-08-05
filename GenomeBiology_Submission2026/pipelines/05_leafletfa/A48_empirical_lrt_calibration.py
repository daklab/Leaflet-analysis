#!/usr/bin/env python
"""A48: calibrate the DM likelihood-ratio test against its own permuted null, per event size.

Why this is needed
------------------
A45 reads p from the chi-square implied by Wilks' theorem. On the real atlas that reference is
demonstrably wrong, and wrong in the direction that matters:

    nominal 0.50 -> observed 0.543  (ratio 1.09)
    nominal 0.05 -> observed 0.164  (ratio 3.27)
    nominal 0.01 -> observed 0.097  (ratio 9.72)

Fine at the median, failing progressively into the tail -- i.e. the statistic is heavier-tailed
than chi-square. Every FDR decision is made in that tail. Split by event size the ratio is ~3 at
EVERY m (2.95 at m=2, 3.93 at m=3, ~3 for m=4-8), so this is not the small-m problem it was
expected to be; it is uniform.

Two contributions to that, which this does NOT need to separate:
  - real splicing is overdispersed beyond even a per-event DM, and H1's K*(m-1) free parameters
    soak up the excess, inflating the statistic;
  - on real data the permuted null is contaminated, because permuting phi on a genuinely
    differential ATSE does not null it -- it converts real between-program variation into an
    overdispersed mixture. (On simulation, all-events gave ratio 0.36-3.82 while true-nulls-only
    gave 0.28-0.45.)

Either way the remedy is the same: stop asserting a reference distribution and use the one the
permutation actually produces.

What it does
------------
Within each event-size stratum, convert the observed statistic to an empirical right-tail p against
the PERMUTED statistics from the same stratum, then BH. Stratifying by m is not optional: df =
(m-1)*K, so the statistic's scale differs by construction between strata and a pooled null would be
a mixture -- the same mistake that made three earlier null estimates fail in the tail.

Unlike the count-permutation machinery that was abandoned earlier, this needs NO refit: phi is
already fitted and only its row order changes, so null draws are cheap and can be multiplied.

Validation
----------
`--validate` runs the same procedure on simulations, where truth is known, and reports realised FDP
against nominal for chi-square and for the empirical null side by side. If the empirical version is
not better calibrated there, it should not be trusted on the atlas either.
"""

import argparse, glob, os, re
import numpy as np
import pandas as pd
from scipy.stats import chi2


def bh(p, alpha):
    n = len(p)
    if n == 0:
        return np.zeros(0, bool)
    o = np.argsort(p)
    ok = p[o] <= alpha * np.arange(1, n + 1) / n
    sig = np.zeros(n, bool)
    if ok.any():
        sig[o[:np.flatnonzero(ok)[-1] + 1]] = True
    return sig


def empirical_p(stat, null):
    """Right-tail p with the +1 correction: with B null draws no p can be below 1/(B+1), which is
    the honest floor. Claiming p=0 from a finite null is precisely how a tail gets over-sold."""
    ns = np.sort(null)
    return (1.0 + len(ns) - np.searchsorted(ns, stat, side="left")) / (len(ns) + 1.0)


def calibrate(df, min_stratum=200):
    """Empirical p per event-size stratum. Strata too small to support a tail fall back to pooled,
    which is reported rather than silent."""
    p = np.empty(len(df))
    used_pooled = 0
    allnull = df["perm_stat"].values
    for m, idx in df.groupby("m").groups.items():
        i = np.asarray(idx)
        null = df.loc[i, "perm_stat"].values
        if len(null) < min_stratum:
            p[df.index.get_indexer(i)] = empirical_p(df.loc[i, "stat"].values, allnull)
            used_pooled += len(i)
        else:
            p[df.index.get_indexer(i)] = empirical_p(df.loc[i, "stat"].values, null)
    return p, used_pooled


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    W = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
    ap.add_argument("--tsv", nargs="+", help="a45_lrt_*.tsv files")
    ap.add_argument("--validate", action="store_true",
                    help="simulation mode: score realised FDP against known truth")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.20])
    ap.add_argument("--out", default=f"{W}/ATLAS_LRT")
    args = ap.parse_args()

    for fp in args.tsv:
        t = pd.read_csv(fp, sep="\t").reset_index(drop=True)
        if "perm_stat" not in t:
            print(f"{os.path.basename(fp)}: no perm_stat -- rerun A45 with --permute-check")
            continue
        tag = re.sub(r"a45_lrt_|\.tsv", "", os.path.basename(fp))
        p_emp, pooled = calibrate(t)
        p_chi = t["p"].values
        print(f"\n=== {tag} ({len(t):,} ATSEs) ===")
        if pooled:
            print(f"  {pooled:,} events in strata below the size floor fell back to the pooled null")

        # how honest is each reference, measured on the permuted statistics themselves
        pp_chi = chi2.sf(t["perm_stat"].values, t["df"].values)
        pp_emp, _ = calibrate(t.assign(stat=t["perm_stat"]))
        print(f"  {'nominal':>8} {'chi2 ratio':>11} {'empirical ratio':>16}")
        for a in args.alphas:
            print(f"  {a:>8.2f} {float((pp_chi<=a).mean())/a:>11.2f} {float((pp_emp<=a).mean())/a:>16.2f}")
        print("  (ratio 1.0 = honest. The empirical column is near-1 BY CONSTRUCTION on its own")
        print("   draws, so it is a sanity check, not evidence -- the evidence is --validate.)")

        if args.validate and "is_de" in t:
            tr = t["is_de"].values.astype(bool)
            print(f"\n  realised FDP vs nominal, against known truth ({tr.sum():,} truly DS):")
            print(f"  {'alpha':>6} {'chi2 FDP':>10} {'ratio':>7} {'power':>7} | "
                  f"{'emp FDP':>9} {'ratio':>7} {'power':>7}")
            for a in args.alphas:
                out = []
                for p in (p_chi, p_emp):
                    s = bh(p, a); ns = int(s.sum()); nf = int((s & ~tr).sum())
                    out.append(((nf/ns) if ns else np.nan, (nf/ns/a) if ns else np.nan,
                                (s & tr).sum()/max(tr.sum(), 1)))
                print(f"  {a:>6.2f} {out[0][0]:>10.4f} {out[0][1]:>7.2f} {out[0][2]:>7.3f} | "
                      f"{out[1][0]:>9.4f} {out[1][1]:>7.2f} {out[1][2]:>7.3f}")
        else:
            for a in args.alphas:
                print(f"  alpha {a:.2f}: chi2 calls {int(bh(p_chi,a).sum()):,}, "
                      f"empirical calls {int(bh(p_emp,a).sum()):,}")

        t["p_empirical"] = p_emp
        o = os.path.join(args.out, f"a48_calibrated_{tag}.tsv")
        t.to_csv(o, sep="\t", index=False)
        print(f"  -> {o}")


if __name__ == "__main__":
    main()
