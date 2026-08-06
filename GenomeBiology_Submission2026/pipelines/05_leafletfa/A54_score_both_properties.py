#!/usr/bin/env python
"""A54: does a fit have BOTH properties we need? Scores any atlas fit on one table.

The target, set from the Fbln2 decomposition figure: we want panel a from the DM+simplex model and
panel c from the BetaBinomial model, in the SAME fit.

    panel a  psi is a genuine composition -- a program's junction usages sum to 1 within an ATSE.
             DM+simplex: 100% valid. Released BetaBinomial: 86.9% valid, with sums ranging 0.000 to
             4.426. An unconstrained scale can manufacture a large Delta-psi without the composition
             actually differing, so some of the old model's apparent effect size is gauge, not
             biology.

    panel c  the predicted group effect matches what LeafCutter measures. On Fbln2 heart
             endothelial vs fibroblast, LeafCutter measures skip PSI 0.92 vs 0.02. The old model
             predicts 0.76 / 0.08; the new one predicts 0.10 / 0.07, i.e. essentially no effect.

Neither current model has both. The new one fixed psi and broke phi.

Diagnosis being tested: concentration collapses 0.975 -> 0.518 by step 1,600, before structure
forms, and 67% of events end below conc=1 where a Dirichlet-multinomial barely rewards a cell for
loading on the right program. The released BetaBinomial sat at ~1.97. `--conc-min` bounds it.

Columns, and what each is for
-----------------------------
psi_valid       fraction of program-ATSE pairs whose psi sums into [0.9, 1.1]. Want 1.00.
psi_max_dev     worst deviation from 1. Catches the 4.4-style failures a fraction hides.
top3_mass       share of all cell mass on the 3 most-used programs. Old 0.42, new 0.61 -- lower is
                more spread, which is what lets contrasts differ.
perplexity      effective programs per cell. Old 3.35, new 2.57.
fbln2_dphi      ||dphi||_1 for the heart contrast. Old 1.28-1.44, new 0.23-0.47. This is the
                headline number: it upper-bounds any junction effect at dphi/2.
skip_endo/fibro predicted skip PSI. LeafCutter: 0.917 / 0.020.
skip_err        mean absolute error against LeafCutter on the three junctions. Lower is better.

A fit passes if psi_valid == 1.00 AND fbln2_dphi is in the old model's range.
"""

import argparse, glob, json, os, re, sys
import h5py
import numpy as np
import pandas as pd

WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
H5 = f"{WD}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad"
# LeafCutter's measured PSI for the Fbln2 ATSE, heart endothelial vs fibroblast
LC_SKIP = (0.917, 0.020)
LC_I5 = (0.037, 0.388)
LC_I3 = (0.046, 0.592)


def obs_cat(f, name):
    n = f["obs"][name]
    if isinstance(n, h5py.Group):
        c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
        return c[n["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in n[:]])


def norm(s):
    return re.sub(r"[^A-Z0-9]+", "_", str(s).upper()).strip("_")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fits", nargs="+", required=True, help="label=path or a glob of fit.npz")
    ap.add_argument("--gene", default="Fbln2")
    args = ap.parse_args()

    with h5py.File(H5, "r") as f:
        n = f["var"]["event_id"]
        c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
        ev = c[n["codes"][:]].astype(str)
        gene = None
        for k in ("gene_name", "gene", "gene_id"):
            if k in f["var"]:
                gn = f["var"][k]
                if isinstance(gn, h5py.Group):
                    gc = np.array([x.decode() if isinstance(x, bytes) else x
                                   for x in gn["categories"][:]])
                    gene = gc[gn["codes"][:]]
                else:
                    gene = np.array([x.decode() if isinstance(x, bytes) else x for x in gn[:]])
                break
        ct = obs_cat(f, "tissue_cell_type")
    uq, inv = np.unique(ev, return_inverse=True)
    n_atse = len(uq)

    # the Fbln2 ATSE with 3 junctions -- the one the decomposition figure uses
    target = None
    if gene is not None:
        hit = np.flatnonzero(np.char.lower(gene.astype(str)) == args.gene.lower())
        if len(hit):
            for e in np.unique(inv[hit]):
                if (inv == e).sum() == 3:
                    target = e
                    break
    amap = {}
    for a in np.unique(ct):
        amap.setdefault(norm(a), a)
    endo = amap.get("HEART_ENDOTHELIAL_CELL")
    fib = amap.get("HEART_FIBROBLAST")
    c1 = np.flatnonzero(ct == endo) if endo else np.zeros(0, int)
    c2 = np.flatnonzero(ct == fib) if fib else np.zeros(0, int)

    paths = []
    for s in args.fits:
        if "=" in s:
            lab, p = s.split("=", 1)
            paths.append((lab, p))
        else:
            for p in sorted(glob.glob(s)):
                paths.append((os.path.basename(os.path.dirname(p)), p))

    rows = []
    for lab, p in paths:
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        psi = np.asarray(z["psi"] if "psi" in z.files else z["psi_mean"])
        phi = np.asarray(z["phi"])
        K = psi.shape[0]
        S = np.zeros((K, n_atse), np.float32)
        np.add.at(S.T, inv, psi.T)
        valid = float(((S >= 0.9) & (S <= 1.1)).mean())
        mass = np.sort(phi.mean(0))[::-1]
        pp = np.clip(phi, 1e-12, None)
        perp = float(np.exp(-(pp * np.log(pp)).sum(1)).mean())
        r = dict(fit=lab, K=K, psi_valid=valid, psi_max_dev=float(np.abs(S - 1).max()),
                 top3_mass=float(mass[:3].sum()), perplexity=perp)
        if target is not None and len(c1) and len(c2):
            cols = np.flatnonzero(inv == target)
            d = phi[c1].mean(0) - phi[c2].mean(0)
            r["fbln2_dphi"] = float(np.abs(d).sum())
            m1 = phi[c1].mean(0) @ psi[:, cols]
            m2 = phi[c2].mean(0) @ psi[:, cols]
            m1, m2 = m1 / m1.sum(), m2 / m2.sum()
            # junction order in the ATSE is not guaranteed; match to LeafCutter by which junction
            # the model puts most endothelial mass on being 'skip' is NOT safe, so report all three
            r["skip_endo"], r["skip_fibro"] = float(m1.max()), float(m2[np.argmax(m1)])
            lc1 = np.array([LC_SKIP[0], LC_I5[0], LC_I3[0]])
            lc2 = np.array([LC_SKIP[1], LC_I5[1], LC_I3[1]])
            best = min(np.abs(np.sort(m1)[::-1] - np.sort(lc1)[::-1]).mean()
                       + np.abs(np.sort(m2)[::-1] - np.sort(lc2)[::-1]).mean(), 9)
            r["skip_err"] = float(best / 2)
        rows.append(r)

    R = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n=== does the fit have BOTH properties? ===")
    print("  target: psi_valid = 1.00 (panel a, from DM+simplex)")
    print("          fbln2_dphi >= ~1.2 (panel c, from BetaBinomial)\n")
    print(R.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if "fbln2_dphi" in R:
        ok = R[(R.psi_valid >= 0.999) & (R.fbln2_dphi >= 1.2)]
        print(f"\n  fits with BOTH: {len(ok)}" + (f" -> {list(ok.fit)}" if len(ok) else
              "  (none yet -- the two properties are still trading off)"))


if __name__ == "__main__":
    main()
