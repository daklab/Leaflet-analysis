#!/usr/bin/env python
"""A63: compare the posterior effect-size test against LeafCutter on the SAME question.

The problem this exists to fix
------------------------------
The posterior test asks `P(TV > delta)` with delta = 0.15-0.20, i.e. "does the composition differ by
at least 15-20 points of splicing mass". LeafCutter asks "is the difference non-zero". Those are
different questions, and comparing raw call counts scores the posterior test down for asking the
harder one. Restricting the comparison to events where LEAFCUTTER'S OWN measured effect exceeds
delta puts both methods on the same question.

LeafCutter's effect sizes are per intron (`deltapsi_<B>` in ds_results_effect_sizes.txt), so its
per-cluster total variation is `0.5 * sum_j |deltapsi_j|` -- the same functional the posterior test
uses, computed from LeafCutter's own pseudobulk estimates. That makes it an INDEPENDENT measurement
of the effect size rather than a restatement of our own.

Two directions, and the second is the more useful one
-----------------------------------------------------
  recall     of the events LeafCutter says are large (TV_lc > delta), how many does the posterior
             test call? Comparable to LeafCutter's own call rate on that subset.
  precision  of the events the posterior test calls, how many does LeafCutter independently measure
             as large? This is the direction that actually validates the test, because it is scored
             against a measurement the model did not produce.

Both are reported against LeafCutter as the reference, which means neither is evidence LeafCutter is
right -- it is a concordance analysis. Where they disagree, the honest statement is that they
disagree, not that we found something LeafCutter missed.
"""

import argparse, glob, os, re, sys

import numpy as np
import pandas as pd

WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
LC = f"{WD}/LEAFCUTTER_DS_2026-08"


def norm(s):
    return re.sub(r"[^A-Z0-9]+", "_", str(s).upper()).strip("_")


def bare(c):
    return (pd.Series(c).astype(str)
            .str.replace(r"^chr[^:]*:", "", regex=True)
            .str.replace(r"_[+-]$", "", regex=True).values)


def leafcutter_tv(lc_dir):
    """Per-contrast, per-cluster TV from LeafCutter's own pseudobulk PSI estimates."""
    out = {}
    for d in sorted(glob.glob(f"{lc_dir}/by_pair/*/")):
        fe = os.path.join(d, "ds_results_effect_sizes.txt")
        fs = os.path.join(d, "ds_results_cluster_significance.txt")
        if not (os.path.exists(fe) and os.path.exists(fs)):
            continue
        e = pd.read_csv(fe, sep="\t")
        dcol = [c for c in e.columns if c.startswith("deltapsi")]
        if not dcol:
            continue
        # the intron id carries the cluster id; strip coordinates to get the ATSE
        e["cluster"] = (e["intron"].astype(str)
                        .str.replace(r"^chr[^:]*:\d+:\d+:", "", regex=True)
                        .str.replace(r"_[+-]$", "", regex=True))
        tv = (e.groupby("cluster")[dcol[0]]
                .apply(lambda v: 0.5 * np.abs(v.values).sum()).rename("tv_lc").reset_index())
        s = pd.read_csv(fs, sep="\t")
        s = s[s["status"] == "Success"] if "status" in s else s
        s = s.dropna(subset=["p"]).copy()
        s["cluster"] = bare(s["cluster"])
        out[norm(os.path.basename(d.rstrip("/")))] = tv.merge(
            s[["cluster", "p", "p.adjust"]], on="cluster", how="inner")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--posterior", required=True, help="a51_group_ds_posterior.tsv at the delta used")
    ap.add_argument("--delta", type=float, required=True, help="must match the delta the file used")
    ap.add_argument("--lc-dir", default=LC)
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--lc-q", type=float, default=1.39e-3,
                    help="LeafCutter's threshold at MATCHED realised FDP (A50 measures 1.39e-3 for "
                         "a realised 0.05). Comparing at a common nominal q would score its "
                         "4.48x inflation as sensitivity.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    P = pd.read_csv(args.posterior, sep="\t")
    ccol = "contrast" if "contrast" in P else P.columns[0]
    acol = "atse" if "atse" in P else ("cluster" if "cluster" in P else P.columns[1])
    qcol = "q" if "q" in P else [c for c in P.columns if c.startswith("q")][0]
    P["ckey"] = P[ccol].map(norm)
    lc = leafcutter_tv(args.lc_dir)
    print(f"[1] posterior: {len(P):,} rows over {P.ckey.nunique()} contrasts; "
          f"LeafCutter effect sizes for {len(lc)} contrasts", flush=True)

    rows = []
    for ck, sub in P.groupby("ckey"):
        if ck not in lc:
            continue
        m = sub.rename(columns={acol: "cluster"}).merge(lc[ck], on="cluster", how="inner")
        if len(m) < 100:
            continue
        big = m["tv_lc"] > args.delta                       # LeafCutter says the effect is large
        called = m[qcol] <= args.q                          # the posterior test calls it
        lc_called = m["p.adjust"] <= args.lc_q              # LeafCutter at MATCHED realised FDP
        rows.append(dict(
            contrast=ck, n_shared=len(m), n_big=int(big.sum()),
            # recall on LeafCutter's large-effect set, both methods
            recall_post=float(called[big].mean()) if big.any() else np.nan,
            recall_lc=float(lc_called[big].mean()) if big.any() else np.nan,
            # precision of our calls against LeafCutter's independent effect measurement
            prec_post=float(big[called].mean()) if called.any() else np.nan,
            prec_lc=float(big[lc_called].mean()) if lc_called.any() else np.nan,
            n_called_post=int(called.sum()), n_called_lc=int(lc_called.sum()),
            median_tv_lc=float(m["tv_lc"].median())))
    if not rows:
        print("no comparable contrasts"); return 1
    R = pd.DataFrame(rows)
    o = os.path.join(args.out, f"a63_matched_delta{args.delta:g}.tsv")
    R.to_csv(o, sep="\t", index=False)

    print(f"\n=== on the SAME question: LeafCutter's own TV > {args.delta:g} ===")
    print(f"  contrasts compared: {len(R)}   median shared ATSEs {int(R.n_shared.median()):,}")
    print(f"  median large-effect ATSEs per contrast: {int(R.n_big.median()):,} "
          f"({100*R.n_big.median()/R.n_shared.median():.1f}% of tested)")
    print(f"\n  RECALL on LeafCutter's large-effect set (higher = finds the big effects)")
    print(f"    posterior test  {R.recall_post.median():.3f}")
    print(f"    LeafCutter      {R.recall_lc.median():.3f}   (at matched realised FDP)")
    print(f"\n  PRECISION against LeafCutter's independent effect measurement")
    print(f"    posterior test  {R.prec_post.median():.3f}")
    print(f"    LeafCutter      {R.prec_lc.median():.3f}")
    print(f"\n  median TV that LeafCutter measures across ALL tested ATSEs: "
          f"{R.median_tv_lc.median():.4f}")
    print(f"  -- if that sits well below delta, most of the atlas is outside the regime this test")
    print(f"     can speak to, and that has to be stated rather than discovered by a reader")
    print(f"\n-> {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
