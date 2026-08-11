#!/usr/bin/env python
"""A50: merge LeafletFA's LRT calls with LeafCutter's on the same contrasts and compare.

Both sides now report the same object -- loglr, df, p from a Dirichlet-multinomial likelihood-ratio
test -- so the tables merge on `cluster` and the columns mean the same thing. Under the retired
fixed-delta rule this was not true: a posterior tail probability was being set against a p-value.

What it reports, and why each is needed
---------------------------------------
1. CALIBRATION on the matched null. Each method's own permutation null, ratio of observed to
   nominal. LeafCutter's null shuffles donor labels within a cell type (clean); LeafletFA's shuffles
   phi across cells (contaminated by real signal, so its ratio is an UPPER bound). Reported side by
   side with that asymmetry stated, because it is what makes the comparison conservative in our
   favour rather than an artefact.

2. AGREEMENT on the real contrasts: overlap of called sets, rank correlation of the statistic, and
   the confusion table at matched q. Overlap alone is misleading when the two call very different
   numbers of events, so the count is always shown next to it.

3. POWER AT MATCHED FDR. The honest comparison is not "who calls more" -- a miscalibrated test calls
   more by construction. Each method is thresholded at the q where its OWN permutation null says the
   realised false discovery proportion is alpha, then the call counts are compared. Without this
   step, LeafCutter's larger call count at nominal q=0.05 would look like power when much of it is
   its 4.48x inflation.

4. EVENT SIZE. LeafCutter's null degrades 3.2x -> 11.4x from m=2 to m=9 while LeafletFA's stays near
   3, so any aggregate comparison is confounded by the m distribution of what each calls. Broken out.
"""

import argparse, glob, os, re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
LC = f"{WD}/LEAFCUTTER_DS_2026-08"


def norm(s):
    return re.sub(r"[^A-Z0-9]+", "_", str(s).upper()).strip("_")


def load_leafcutter(lc_dir):
    """LeafCutter's per-contrast cluster significance, keyed by normalised contrast name."""
    out = {}
    for d in sorted(glob.glob(f"{lc_dir}/by_pair/*/")):
        fp = os.path.join(d, "ds_results_cluster_significance.txt")
        if not os.path.exists(fp):
            continue
        t = pd.read_csv(fp, sep="\t")
        t = t[t["status"] == "Success"].dropna(subset=["p"]) if "status" in t else t.dropna(subset=["p"])
        if not len(t):
            continue
        # LeafCutter names a cluster "chr3:<atse_id>_-"; LeafletFA stores the bare "<atse_id>".
        # Merging the two raw silently produced ZERO shared rows for every contrast, which the
        # caller then reported as "no contrasts matched" -- the contrasts matched fine, the JOIN KEY
        # did not. Strip the chromosome prefix and strand suffix so the two agree.
        t = t.copy()
        t["cluster"] = (t["cluster"].astype(str)
                        .str.replace(r"^chr[^:]*:", "", regex=True)
                        .str.replace(r"_[+-]$", "", regex=True))
        out[norm(os.path.basename(d.rstrip("/")))] = t[["cluster", "loglr", "df", "p", "p.adjust"]]
    return out


def null_ratio(p, alpha):
    return float((p <= alpha).mean()) / alpha


def q_at_realised_fdp(null_p, alpha):
    """The nominal q at which a method's OWN null says the realised FDP is alpha.

    A test whose null p-values are inflated k-fold must be thresholded k times tighter to deliver
    the FDR it advertises. Comparing call counts at a common NOMINAL q would otherwise reward
    miscalibration."""
    grid = np.geomspace(1e-6, 0.5, 400)
    for q in grid:
        if null_ratio(null_p, q) * q >= alpha:
            return q
    return grid[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leafletfa", required=True, help="a49_leafletfa_lrt_*.tsv")
    ap.add_argument("--lc-dir", default=LC)
    ap.add_argument("--lfa-null", help="a45_lrt_atlas*.tsv with perm_p, for LeafletFA's null")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lfa = pd.read_csv(args.leafletfa, sep="\t")
    lfa["ckey"] = lfa["contrast"].map(norm)
    lc = load_leafcutter(args.lc_dir)
    print(f"[1] LeafletFA: {len(lfa):,} rows over {lfa.ckey.nunique()} contrasts")
    print(f"    LeafCutter: {len(lc)} contrasts on disk")

    # ---- 1. calibration on each method's own permutation null -------------------------------
    lc_null = []
    for f in sorted(glob.glob(f"{args.lc_dir}/permutation_null/*/ds_results_cluster_significance.txt")):
        t = pd.read_csv(f, sep="\t")
        t = t[t["status"] == "Success"] if "status" in t else t
        t = t.dropna(subset=["p"])
        if len(t):
            lc_null.append(t[["p", "df"]])
    lc_null = pd.concat(lc_null, ignore_index=True) if lc_null else None
    lfa_null = None
    if args.lfa_null and os.path.exists(args.lfa_null):
        t = pd.read_csv(args.lfa_null, sep="\t")
        # Two files can serve as LeafletFA's null and they name the column differently:
        # A45 writes `perm_p` alongside the observed `p`, while A52 -- which runs LeafCutter's OWN
        # null construction and is the one that makes the comparison like-for-like -- writes the
        # null p directly as `p`. Accepting only `perm_p` silently left this None, and every
        # LeafletFA calibration number printed as nan.
        if "perm_p" in t:
            lfa_null = t.rename(columns={"perm_p": "p"})[["p", "m"]]
        elif {"p", "m"} <= set(t.columns):
            lfa_null = t[["p", "m"]]
        else:
            print(f"  WARNING: {args.lfa_null} has neither perm_p nor p+m -- null unusable")

    print("\n=== 1. calibration, each method on its OWN permutation null ===")
    print(f"  {'nominal':>8} {'LeafCutter':>12} {'LeafletFA':>12}")
    for a in (0.01, 0.05, 0.10, 0.20, 0.50):
        r1 = null_ratio(lc_null["p"].values, a) if lc_null is not None else np.nan
        r2 = null_ratio(lfa_null["p"].values, a) if lfa_null is not None else np.nan
        print(f"  {a:>8.2f} {r1:>12.2f} {r2:>12.2f}")
    print("  ratio 1.0 = honest. LeafCutter's null is CLEAN (labels shuffled within one cell type).")
    print("  LeafletFA's is CONTAMINATED by real signal, so its ratio is an UPPER BOUND -- the")
    print("  comparison is conservative in LeafletFA's favour, not flattering to it.")

    q_lc = q_at_realised_fdp(lc_null["p"].values, args.alpha) if lc_null is not None else args.alpha
    q_lfa = q_at_realised_fdp(lfa_null["p"].values, args.alpha) if lfa_null is not None else args.alpha
    print(f"\n  to deliver a realised FDP of {args.alpha}: LeafCutter needs q<={q_lc:.2e}, "
          f"LeafletFA q<={q_lfa:.2e}")

    # ---- 2/3. agreement and power at matched FDR --------------------------------------------
    rows = []
    n_key_hit = n_thin = 0
    for ck, sub in lfa.groupby("ckey"):
        if ck not in lc:
            continue
        n_key_hit += 1
        m = sub.merge(lc[ck], on="cluster", suffixes=("_lfa", "_lc"))
        if len(m) < 100:
            n_thin += 1
            continue
        a_nom = (m["p.adjust_lfa"] <= args.alpha)
        b_nom = (m["p.adjust_lc"] <= args.alpha)
        a_cal = (m["p_lfa"] <= q_lfa)
        b_cal = (m["p_lc"] <= q_lc)
        rho = spearmanr(m["loglr_lfa"], m["loglr_lc"]).statistic if len(m) > 10 else np.nan
        rows.append(dict(contrast=ck, n_shared=len(m),
                         lfa_nom=int(a_nom.sum()), lc_nom=int(b_nom.sum()),
                         both_nom=int((a_nom & b_nom).sum()),
                         lfa_cal=int(a_cal.sum()), lc_cal=int(b_cal.sum()),
                         both_cal=int((a_cal & b_cal).sum()),
                         jaccard_cal=float((a_cal & b_cal).sum() / max((a_cal | b_cal).sum(), 1)),
                         spearman_loglr=rho))
    if not rows:
        # Separate the two failure modes, because they have completely different fixes and the old
        # single message sent the reader after the wrong one.
        if n_key_hit == 0:
            print(f"\nno CONTRAST NAMES in common: LeafletFA has {lfa.ckey.nunique()} keys, "
                  f"LeafCutter {len(lc)} -- e.g. {sorted(lfa.ckey.unique())[:2]} vs "
                  f"{sorted(lc)[:2]}")
        else:
            print(f"\n{n_key_hit} contrast names matched, but {n_thin} of them shared fewer than "
                  f"100 ATSEs -- the join key `cluster` is the suspect, not the contrast name")
        return
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(args.out, "a50_contrast_comparison.tsv"), sep="\t", index=False)
    print(f"\n=== 2/3. agreement over {len(R)} shared contrasts ===")
    print(f"  shared ATSEs per contrast: median {int(R.n_shared.median()):,}")
    print(f"  at NOMINAL q<={args.alpha}: LeafletFA calls {int(R.lfa_nom.median()):,}, "
          f"LeafCutter {int(R.lc_nom.median()):,} (medians)")
    print(f"  at MATCHED realised FDP:    LeafletFA calls {int(R.lfa_cal.median()):,}, "
          f"LeafCutter {int(R.lc_cal.median()):,}")
    print(f"  Jaccard of the calibrated call sets: median {R.jaccard_cal.median():.3f}")
    print(f"  Spearman of loglr: median {R.spearman_loglr.median():.3f}")
    print(f"\n  -> {args.out}/a50_contrast_comparison.tsv")

    # ---- 4. by event size --------------------------------------------------------------------
    if lc_null is not None and lfa_null is not None:
        print("\n=== 4. null calibration by event size (the confound in any aggregate number) ===")
        print(f"  {'m':>4} {'LeafCutter':>12} {'LeafletFA':>12}")
        for mm in range(2, 10):
            k1 = lc_null["df"].values == mm - 1
            k2 = lfa_null["m"].values == mm
            if k1.sum() < 200 or k2.sum() < 200:
                continue
            print(f"  {mm:>4} {null_ratio(lc_null['p'].values[k1], 0.05):>12.2f} "
                  f"{null_ratio(lfa_null['p'].values[k2], 0.05):>12.2f}")


if __name__ == "__main__":
    main()
