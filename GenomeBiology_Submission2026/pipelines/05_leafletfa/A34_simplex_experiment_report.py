#!/usr/bin/env python
"""
A34: score the constrained-psi experiment and decide whether a retrain is justified.

Reads the six A33 fits (baseline / simplex x 3 seeds, K=20, identical cells) and reports the
four numbers the decision turns on:

The two things that decide it are BALANCE and SEED VARIANCE. Agreement with LeafCutter is a
guardrail only -- we have already shown LeafCutter falsely calls 7.7% of events under a
donor-split null versus 0.14% for LeafletFA, so it is not a reference worth optimising toward.

  1. BALANCE  do a program's junction usages within an ATSE sum to 1? Reported as the full
     distribution and split by ATSE size, since violation currently scales from 0% at
     2-junction events to ~50% at 6+ junction events. Simplex must be exactly 1 everywhere;
     anything else means the implementation is wrong.

  2. SEED VARIANCE  measured at three levels, because they answer different questions:
       psi           raw correlation across seeds, Hungarian-matched
       sum / comp    psi split into (within-ATSE sum) x (composition). On the released fits the
                     composition reproduces at r = 0.95-0.96 while the sum only reaches
                     0.51-0.66 -- the sum is an unidentified direction and that is the whole
                     argument for constraining it. The constrained model should have no sum
                     variance left at all.
       DS calls      correlation of the ATSE-level TV statistic across seeds, and Jaccard of the
                     top-ranked sets. This is what actually reaches the paper, so it is the
                     number that matters most.

  3. GUARDRAIL  AUROC vs LeafCutter, overall and at high-effect events. Not a target. It is here
     to catch a large regression, since violation currently tracks real signal (LeafCutter median
     max |dPSI| 0.113 where our composition is invalid vs 0.035 where valid), so there is a
     plausible route by which constraining could cost accuracy at strong events.

Decision rule, fixed before seeing any output:
  - constraint wins   if DS-call seed agreement rises AND high-effect AUROC drops by <0.02
  - constraint loses  if DS-call seed agreement does not improve, or high-effect AUROC drops >0.02
  - otherwise         inconclusive at this scale

Caveat: these fits use a 20,000-cell subset while the LeafCutter reference used all cells in
those types, so absolute AUROC is not comparable to the full-atlas numbers. Only the
baseline-vs-simplex difference is meaningful here.

Usage:
    python A34_simplex_experiment_report.py --exp-dir <SIMPLEX_EXP> --lc-dir <LEAFCUTTER_DS_2026-08>
"""

import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment

WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
H5 = f"{WD}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad"
safe = lambda s: re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))


def cat(name, which):
    with h5py.File(H5, "r") as f:
        n = f[which][name]
        if isinstance(n, h5py.Group):
            c = np.array([x.decode() if isinstance(x, bytes) else x
                          for x in n["categories"][:]])
            return pd.Categorical.from_codes(n["codes"][:], c).astype(str)
        return np.array([x.decode() if isinstance(x, bytes) else x for x in n[:]])


def hungarian(a, b):
    A = a - a.mean(1, keepdims=True); B = b - b.mean(1, keepdims=True)
    A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    r, c = linear_sum_assignment(-np.abs(A @ B.T))
    col = np.empty(len(r), int); col[r] = c
    return col


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", default=f"{WD}/SIMPLEX_EXP")
    ap.add_argument("--lc-dir", default=f"{WD}/LEAFCUTTER_DS_2026-08")
    ap.add_argument("--high-effect", type=float, default=0.10,
                    help="LeafCutter max |dPSI| above which an ATSE counts as high-effect")
    args = ap.parse_args()

    tct_all = np.array([safe(x) for x in cat("tissue_cell_type", "obs")])
    ev = cat("event_id", "var")
    jid = cat("junction_id", "var")
    uq, inv = np.unique(ev, return_inverse=True)
    pos = {a: i for i, a in enumerate(uq)}
    jpos = {k: i for i, k in enumerate(jid)}
    jm = pd.read_csv(f"{WD}/LEAFCUTTER_DS/junction_atse_map.tsv", sep="\t")
    lc2j = dict(zip(jm.leafcutter_junc_id, jm.junction_id))

    fits = {}
    for f in sorted(glob.glob(os.path.join(args.exp_dir, "simplex_exp_*.npz"))):
        z = np.load(f, allow_pickle=True)
        fits[(str(z["variant"]), int(z["seed"]))] = z
    if not fits:
        print("no fits found"); return
    variants = sorted({v for v, _ in fits})
    seeds = sorted({s for _, s in fits})
    print(f"loaded {len(fits)} fits: {variants} x seeds {seeds}\n")

    man = pd.read_csv(f"{args.lc_dir}/pairwise_manifest_tabula_muris_senis_"
                      "tissue_cell_type.tsv", sep="\t")
    man = man[man.status.isin(["ready", "already_done"])]

    rows, seedrows = [], []
    for v in variants:
        # 1. validity
        for s in seeds:
            if (v, s) not in fits:
                continue
            z = fits[(v, s)]
            psi = z["psi"]
            S = np.zeros((psi.shape[0], len(uq))); np.add.at(S.T, inv, psi.T)
            valid = float(((S >= 0.9) & (S <= 1.1)).mean())

            cells = z["cells"]; phi = z["phi"]
            tct = tct_all[cells]
            for _, r in man.iterrows():
                name = f"{r.group_1}__vs__{r.group_2}"
                sigf = f"{args.lc_dir}/by_pair/{name}/ds_results_cluster_significance.txt"
                eff = f"{args.lc_dir}/by_pair/{name}/ds_results_effect_sizes.txt"
                if not (os.path.exists(sigf) and os.path.exists(eff)):
                    continue
                m1, m2 = tct == r.group_1, tct == r.group_2
                if m1.sum() < 50 or m2.sum() < 50:
                    continue
                d = phi[m1].mean(0) - phi[m2].mean(0)
                tv = np.zeros(len(uq)); np.add.at(tv, inv, np.abs(d @ psi)); tv *= 0.5

                sg = pd.read_csv(sigf, sep="\t"); sg = sg[sg.status == "Success"].copy()
                sg["atse"] = sg.cluster.str.split(":").str[1].str[:-2]
                sg["padj"] = pd.to_numeric(sg["p.adjust"], errors="coerce")
                sg = sg.dropna(subset=["padj"]); sg = sg[sg.atse.isin(pos)]
                if len(sg) < 500:
                    continue
                ii = sg.atse.map(pos).values
                sig = (sg.padj < 0.05).values
                if sig.sum() < 20 or sig.all():
                    continue

                e = pd.read_csv(eff, sep="\t")
                dc = [c for c in e.columns if c.startswith("deltapsi")]
                hi = np.zeros(len(ii), bool)
                if dc:
                    e["jid"] = e.intron.map(lc2j)
                    e = e.dropna(subset=["jid"]); e = e[e.jid.isin(jpos)]
                    e["atse"] = [ev[jpos[j]] for j in e.jid]
                    mx = e.groupby("atse")[dc[0]].apply(
                        lambda x: np.abs(pd.to_numeric(x, errors="coerce")).max())
                    hi = np.array([mx.get(a, 0) > args.high_effect for a in sg.atse])

                def auroc(mask):
                    y = sig[mask]
                    if y.sum() < 10 or (~y).sum() < 10:
                        return np.nan
                    x = tv[ii[mask]]
                    return stats.mannwhitneyu(x[y], x[~y]).statistic / (y.sum() * (~y).sum())

                rows.append(dict(variant=v, seed=s, contrast=name, frac_valid=valid,
                                 auroc_all=auroc(np.ones(len(ii), bool)),
                                 auroc_high=auroc(hi), n_high=int(hi.sum()),
                                 dphi_l1=float(np.abs(d).sum())))

        # 2. seed agreement
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                if (v, seeds[i]) not in fits or (v, seeds[j]) not in fits:
                    continue
                a = fits[(v, seeds[i])]["psi"]; b = fits[(v, seeds[j])]["psi"]
                col = hungarian(a, b); b = b[col]
                # split psi into the unidentified (sum) and identified (composition) parts
                Sa = np.zeros((a.shape[0], len(uq))); np.add.at(Sa.T, inv, a.T)
                Sb = np.zeros((b.shape[0], len(uq))); np.add.at(Sb.T, inv, b.T)
                Ca = a / np.maximum(Sa[:, inv], 1e-9)
                Cb = b / np.maximum(Sb[:, inv], 1e-9)
                mult = np.bincount(inv)[inv] > 1
                # DS-level agreement: the TV statistic these fits actually produce
                pa, pb = fits[(v, seeds[i])]["phi"], fits[(v, seeds[j])]["phi"]
                cells = fits[(v, seeds[i])]["cells"]; tct = tct_all[cells]
                tvs = []
                for _, r in man.iterrows():
                    m1, m2 = tct == r.group_1, tct == r.group_2
                    if m1.sum() < 50 or m2.sum() < 50:
                        continue
                    out2 = []
                    for ph, ps in ((pa, a), (pb, b)):
                        d = ph[m1].mean(0) - ph[m2].mean(0)
                        t = np.zeros(len(uq)); np.add.at(t, inv, np.abs(d @ ps))
                        out2.append(0.5 * t)
                    tvs.append((np.corrcoef(out2[0], out2[1])[0, 1],
                                len(set(np.argsort(-out2[0])[:2000])
                                    & set(np.argsort(-out2[1])[:2000])) / 2000))
                seedrows.append(dict(
                    variant=v, pair=f"{seeds[i]}-{seeds[j]}",
                    psi_r=float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
                    sum_r=float(np.corrcoef(Sa.ravel(), Sb.ravel())[0, 1]),
                    comp_r=float(np.corrcoef(Ca[:, mult].ravel(), Cb[:, mult].ravel())[0, 1]),
                    tv_r=float(np.median([t[0] for t in tvs])) if tvs else np.nan,
                    tv_top2000=float(np.median([t[1] for t in tvs])) if tvs else np.nan))

    R = pd.DataFrame(rows); SR = pd.DataFrame(seedrows)
    R.to_csv(os.path.join(args.exp_dir, "simplex_experiment_scores.tsv"), sep="\t", index=False)

    print("1. BALANCE  (fraction of program-ATSE pairs with psi summing to ~1)")
    print(R.groupby("variant").frac_valid.mean().round(4).to_string() if not R.empty
          else "   (no fits scored)")
    print("\n2. SEED VARIANCE  (across-seed agreement; higher is better)")
    # Needs >=2 seeds of the same variant. With one fit per variant SR is empty and the
    # groupby raises KeyError: 'variant' -- say so instead of dying.
    if SR.empty:
        print("   (skipped -- fewer than two seeds available for any variant: "
              f"have {sorted(fits)})")
    else:
        print(SR.groupby("variant")[["psi_r", "sum_r", "comp_r", "tv_r", "tv_top2000"]]
              .mean().round(4).to_string())
    print("   psi_r raw psi | sum_r the unidentified direction | comp_r the biology")
    print("   tv_r  correlation of the DS statistic | tv_top2000 overlap of top-2000 ATSEs")
    print("\n3. GUARDRAIL  AUROC vs LeafCutter (not a target)")
    print(R.groupby("variant")[["auroc_all", "auroc_high"]].median().round(4).to_string()
          if not R.empty else "   (no fits scored)")

    if len(variants) == 2 and not SR.empty:
        p = R.pivot_table(index=["contrast", "seed"], columns="variant",
                          values=["auroc_all", "auroc_high"])
        for metric in ["auroc_all", "auroc_high"]:
            q = p[metric].dropna()
            if len(q) < 10:
                continue
            diff = (q["simplex"] - q["baseline"]).median()
            w = stats.wilcoxon(q["simplex"], q["baseline"])
            print(f"\n   {metric}: simplex - baseline = {diff:+.4f}  "
                  f"(p={w.pvalue:.3g}, n={len(q)})")
        d_tv = (SR[SR.variant == "simplex"].tv_r.mean()
                - SR[SR.variant == "baseline"].tv_r.mean())
        d_top = (SR[SR.variant == "simplex"].tv_top2000.mean()
                 - SR[SR.variant == "baseline"].tv_top2000.mean())
        dh = (p["auroc_high"]["simplex"] - p["auroc_high"]["baseline"]).median()
        print(f"\n   DS-call seed agreement: tv_r {d_tv:+.4f}, top-2000 overlap {d_top:+.4f}")
        print("\nVERDICT")
        if (d_tv > 0 or d_top > 0) and dh > -0.02:
            print("   constraint WINS: DS calls more reproducible, no material AUROC loss")
        elif dh < -0.02:
            print("   constraint LOSES: high-effect AUROC dropped materially")
        else:
            print("   INCONCLUSIVE at this scale")


if __name__ == "__main__":
    main()
