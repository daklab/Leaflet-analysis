#!/usr/bin/env python
"""
A29: which effect-size estimator generalises to held-out donors?

The question this answers
-------------------------
LeafletFA's effect sizes are ~10x attenuated relative to LeafCutter (median OLS slope 0.094),
because |beta_j| <= |dphi|_1 is a hard low-rank ceiling. Read as "agreement", that is a poor
result. But attenuation is not the same as inaccuracy: a shrunk estimator trades bias for
variance, and LeafCutter's own pseudobulk dPSI is NOISY -- split-half of donors puts its
reliability at only 0.61-0.68 at n=13.

So the fair question is not "do the numbers match" but "which estimate better predicts the
truth on donors neither method saw". That is what this measures.

Design
------
Per contrast, split DONORS (never cells) into train and test halves:

  train estimate, LeafCutter-style : observed pseudobulk dPSI from the TRAIN donors
  train estimate, LeafletFA        : dphi computed from TRAIN cells only, times psi
  target                           : observed pseudobulk dPSI from the TEST donors

Scored by R^2 of the target regressed on each estimate. Regression absorbs a scale factor, so
the 10x attenuation is deliberately NOT penalised -- this isolates whether the RANKING and
relative magnitudes carry information, which is what a marker catalogue needs.

Honest caveat, to be stated in the Methods
------------------------------------------
psi (the program basis) was learned on the full atlas, which includes the test donors' cells.
The model never saw cell-type or contrast labels -- it is unsupervised -- so this is not label
leakage, but it IS an information advantage of the same kind as running PCA before
cross-validation. The comparison is therefore "LeafCutter using train counts" vs "LeafletFA
using train counts plus an unsupervised atlas-wide basis". That is a real deployment
advantage, but it is not a like-for-like estimator comparison and must not be described as one.

A per-junction-mean baseline is included as the floor: any method that cannot beat "predict
the global mean effect" is not using the data.

Usage
-----
    python A29_heldout_effect_prediction.py --fit <fit.npz> --h5ad <atlas> \\
        --lc-dir <LEAFCUTTER_DS_2026-08> --out <dir> --n-splits 20
"""

import argparse
import json
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
import torch

safe = lambda s: re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))


def cat(h5, name, which):
    with h5py.File(h5, "r") as f:
        node = f[which][name]
        if isinstance(node, h5py.Group):
            c = np.array([x.decode() if isinstance(x, bytes) else x
                          for x in node["categories"][:]])
            return pd.Categorical.from_codes(node["codes"][:], c).astype(str)
        return np.array([x.decode() if isinstance(x, bytes) else x for x in node[:]])


def load_fit_strict(path):
    z = np.load(path)
    bad = [k for k in z.files if z[k].dtype.kind == "f" and not np.isfinite(z[k]).all()]
    if bad:
        raise ValueError(f"{path}: non-finite arrays {bad}")
    return z


def pseudobulk_dpsi(counts, inv_ev, n_atse, cols_1, cols_2):
    """Observed dPSI per junction from raw pseudobulk counts. Model-free."""
    def psi(cols):
        if not cols:
            return None
        sub = counts[cols].to_numpy(dtype=np.float64)          # [J, n_samples]
        tot = np.zeros((n_atse, sub.shape[1]))
        np.add.at(tot, inv_ev, sub)
        den = tot[inv_ev]
        with np.errstate(invalid="ignore", divide="ignore"):
            p = np.where(den > 0, sub / den, np.nan)
        return np.nanmean(p, axis=1)
    a, b = psi(cols_1), psi(cols_2)
    if a is None or b is None:
        return None
    return a - b


def r2(target, pred):
    """R^2 of target regressed on pred (scale-free: absorbs any attenuation factor)."""
    ok = np.isfinite(target) & np.isfinite(pred)
    if ok.sum() < 100 or np.std(pred[ok]) == 0:
        return np.nan
    t, p = target[ok], pred[ok]
    b = np.polyfit(p, t, 1)
    resid = t - np.polyval(b, p)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return float(1 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--lc-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--min-donors-per-arm", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    z = load_fit_strict(args.fit)
    phi, psi = z["phi"], z["psi"]
    meta = json.load(open(os.path.join(os.path.dirname(args.fit), "result.json")))
    print(f"[1] fit K={meta['K']} seed={meta['seed']}", flush=True)

    tct = np.array([safe(x) for x in cat(args.h5ad, "tissue_cell_type", "obs")])
    donor = np.array([safe(x) for x in cat(args.h5ad, "mouse.id", "obs")])
    ds = cat(args.h5ad, "dataset", "obs")
    ev = cat(args.h5ad, "event_id", "var")
    jid = cat(args.h5ad, "junction_id", "var")
    uq_ev, inv_ev = np.unique(ev, return_inverse=True)
    n_atse = len(uq_ev)
    tms = ds == "tabula_muris_senis"

    jm = pd.read_csv(f"{os.path.dirname(args.lc_dir.rstrip('/'))}/"
                     "LEAFCUTTER_DS/junction_atse_map.tsv", sep="\t")
    lc2j = dict(zip(jm.leafcutter_junc_id, jm.junction_id))
    jpos = {k: i for i, k in enumerate(jid)}

    smeta = pd.read_csv(f"{args.lc_dir}/sample_metadata.csv")
    smeta["tct_safe"] = smeta.tissue_cell_type.map(safe)
    smeta["donor_safe"] = smeta.donor.fillna("").map(safe)

    counts = pd.read_csv(f"{args.lc_dir}/leafcutter_counts.tsv.gz", sep="\t",
                         index_col=0, compression="gzip")
    counts.index = pd.Series(counts.index).map(lc2j).values
    counts = counts[~pd.isna(counts.index)]
    counts = counts.reindex([j for j in jid if j in set(counts.index)])
    keep_j = np.array([jpos[j] for j in counts.index])
    inv_sub = inv_ev[keep_j]
    print(f"[2] counts aligned: {len(counts):,} junctions, {counts.shape[1]} samples",
          flush=True)

    man = pd.read_csv(f"{args.lc_dir}/pairwise_manifest_tabula_muris_senis_"
                      "tissue_cell_type.tsv", sep="\t")
    man = man[man.status.isin(["ready", "already_done"])]

    rng = np.random.default_rng(args.seed)
    rows = []
    for _, r in man.iterrows():
        g1, g2 = r.group_1, r.group_2
        s1 = smeta[smeta.tct_safe == g1]
        s2 = smeta[smeta.tct_safe == g2]
        shared = sorted(set(s1.donor_safe) & set(s2.donor_safe))
        if len(shared) < 2 * args.min_donors_per_arm:
            continue

        for sp in range(args.n_splits):
            perm = rng.permutation(shared)
            h = len(perm) // 2
            tr_d, te_d = set(perm[:h]), set(perm[h:])

            def cols(sm, dset):
                return [x for x in sm[sm.donor_safe.isin(dset)].sample_name
                        if x in counts.columns]
            tr1, tr2 = cols(s1, tr_d), cols(s2, tr_d)
            te1, te2 = cols(s1, te_d), cols(s2, te_d)
            if min(len(tr1), len(tr2), len(te1), len(te2)) < args.min_donors_per_arm:
                continue

            target = pseudobulk_dpsi(counts, inv_sub, n_atse, te1, te2)
            lc_tr = pseudobulk_dpsi(counts, inv_sub, n_atse, tr1, tr2)
            if target is None or lc_tr is None:
                continue

            # LeafletFA trained-on-train: dphi from TRAIN donors' cells only
            m1 = tms & (tct == g1) & np.isin(donor, list(tr_d))
            m2 = tms & (tct == g2) & np.isin(donor, list(tr_d))
            if m1.sum() < 50 or m2.sum() < 50:
                continue
            dphi = phi[m1].mean(0) - phi[m2].mean(0)
            lfa_tr = (dphi @ psi)[keep_j]

            rows.append(dict(
                contrast=f"{g1}__vs__{g2}", split=sp, n_donors=len(shared),
                r2_leafcutter=r2(target, lc_tr),
                r2_leafletfa=r2(target, lfa_tr),
                r2_baseline=r2(target, np.full_like(target, np.nanmean(lc_tr))),
                spearman_leafcutter=float(pd.Series(target).corr(pd.Series(lc_tr),
                                                                method="spearman")),
                spearman_leafletfa=float(pd.Series(target).corr(pd.Series(lfa_tr),
                                                               method="spearman")),
                dphi_l1=float(np.abs(dphi).sum())))
        if rows and len(rows) % 40 == 0:
            print(f"    {len(rows)} splits scored", flush=True)

    R = pd.DataFrame(rows)
    tag = f"K{meta['K']}_seed{meta['seed']}"
    out = os.path.join(args.out, f"heldout_effect_prediction_{tag}.tsv")
    R.to_csv(out, sep="\t", index=False)

    print(f"\n=== held-out donor prediction, {tag} "
          f"({R.contrast.nunique()} contrasts x {args.n_splits} splits) ===")
    print("R^2 predicting held-out pseudobulk dPSI (scale-free; attenuation not penalised)")
    print(R[["r2_leafcutter", "r2_leafletfa", "r2_baseline"]].describe()
          .loc[["mean", "50%"]].round(4).to_string())
    win = (R.r2_leafletfa > R.r2_leafcutter).mean()
    print(f"\nLeafletFA beats LeafCutter on {100*win:.1f}% of splits")
    print("\nby program-space separation:")
    R["bin"] = pd.qcut(R.dphi_l1, 3, labels=["low", "mid", "high"], duplicates="drop")
    print(R.groupby("bin")[["r2_leafcutter", "r2_leafletfa"]].mean().round(4).to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
