#!/usr/bin/env python
"""
A31: which ATSE-level statistic best matches LeafCutter, and how reproducible is it across seeds?

Candidates
----------
  raw TV            0.5 * sum_j |rho_A,j - rho_B,j|         (what differential_splicing_atses does)
  compositional TV  same, after renormalising rho within the ATSE so it sums to 1
  logit TV          0.5 * sum_j |logit(rho_A,j) - logit(rho_B,j)|

Why the compositional variant is in the running: LeafCutter models the junction vector jointly
with a Dirichlet-multinomial, so its PSI lies on the simplex BY CONSTRUCTION and its dPSI sums
to exactly 0 within a cluster. LeafletFA scores each junction with an independent Beta-Binomial
-- which is exactly the DM's marginal, but drops the joint dependence -- so the composition is
learned rather than enforced. It mostly holds (median |sum rho - 1| = 0.004) but degrades with
ATSE size: 0.3% of 2-junction ATSEs violate it by >0.05 versus 82% of 11+ junction ATSEs.

Why the logit variant: LeafCutter's own GLM is logit-link (its native effect size is the
`logef_` column; dPSI is derived), and differences of proportions are compressed near 0 and 1
while logit differences are not.

Scored per contrast against LeafCutter by AUROC for ranking its significant ATSEs, plus the
junction-level slope, across all three K=30 seeds.

Usage
-----
    python A31_ds_statistic_comparison.py --model-dir <atlas_FULL> --tasks 6 7 8 \\
        --h5ad <atlas> --lc-dir <LEAFCUTTER_DS_2026-08> --out <dir>
"""

import argparse
import json
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
from scipy import stats

safe = lambda s: re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))


def cat(h5, name, which):
    with h5py.File(h5, "r") as f:
        node = f[which][name]
        if isinstance(node, h5py.Group):
            c = np.array([x.decode() if isinstance(x, bytes) else x
                          for x in node["categories"][:]])
            return pd.Categorical.from_codes(node["codes"][:], c).astype(str)
        return np.array([x.decode() if isinstance(x, bytes) else x for x in node[:]])


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tasks", type=int, nargs="+", default=[6, 7, 8])
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--lc-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tct = np.array([safe(x) for x in cat(args.h5ad, "tissue_cell_type", "obs")])
    ev = cat(args.h5ad, "event_id", "var")
    jid = cat(args.h5ad, "junction_id", "var")
    uq, inv = np.unique(ev, return_inverse=True)
    jpos = {k: i for i, k in enumerate(jid)}

    jm = pd.read_csv(f"{os.path.dirname(args.lc_dir.rstrip('/'))}/"
                     "LEAFCUTTER_DS/junction_atse_map.tsv", sep="\t")
    lc2j = dict(zip(jm.leafcutter_junc_id, jm.junction_id))

    man = pd.read_csv(f"{args.lc_dir}/pairwise_manifest_tabula_muris_senis_"
                      "tissue_cell_type.tsv", sep="\t")
    man = man[man.status.isin(["ready", "already_done"])]
    pos = {a: i for i, a in enumerate(uq)}

    rows = []
    for task in args.tasks:
        z = np.load(f"{args.model_dir}/mb_run_{task}/fit.npz")
        bad = [k for k in z.files if z[k].dtype.kind == "f" and not np.isfinite(z[k]).all()]
        assert not bad, f"mb_run_{task}: non-finite {bad}"
        meta = json.load(open(f"{args.model_dir}/mb_run_{task}/result.json"))
        phi, psi = z["phi"], z["psi"]
        print(f"\n=== mb_run_{task}  K={meta['K']} seed={meta['seed']} ===", flush=True)

        for _, r in man.iterrows():
            name = f"{r.group_1}__vs__{r.group_2}"
            sigf = f"{args.lc_dir}/by_pair/{name}/ds_results_cluster_significance.txt"
            eff = f"{args.lc_dir}/by_pair/{name}/ds_results_effect_sizes.txt"
            if not (os.path.exists(sigf) and os.path.exists(eff)):
                continue
            A = phi[tct == r.group_1].mean(0)
            B = phi[tct == r.group_2].mean(0)
            rA, rB = A @ psi, B @ psi
            sA = np.zeros(len(uq)); np.add.at(sA, inv, rA)
            sB = np.zeros(len(uq)); np.add.at(sB, inv, rB)

            variants = {
                "raw":   rA - rB,
                "comp":  rA / np.maximum(sA[inv], 1e-9) - rB / np.maximum(sB[inv], 1e-9),
                "logit": logit(rA) - logit(rB),
            }
            tvs = {}
            for k, d in variants.items():
                t = np.zeros(len(uq)); np.add.at(t, inv, np.abs(d)); tvs[k] = 0.5 * t

            s = pd.read_csv(sigf, sep="\t")
            s = s[s.status == "Success"].copy()
            s["atse"] = s.cluster.str.split(":").str[1].str[:-2]
            s["padj"] = pd.to_numeric(s["p.adjust"], errors="coerce")
            s = s.dropna(subset=["padj"])
            s = s[s.atse.isin(pos)]
            if len(s) < 500:
                continue
            ii = s.atse.map(pos).values
            lcsig = (s.padj < 0.05).values
            if lcsig.sum() < 20 or lcsig.all():
                continue

            out = dict(task=task, K=meta["K"], seed=meta["seed"], contrast=name,
                       dphi_l1=float(np.abs(A - B).sum()),
                       n_tested=len(s), n_lc_sig=int(lcsig.sum()))
            for k in variants:
                v = tvs[k][ii]
                out[f"auroc_{k}"] = (stats.mannwhitneyu(v[lcsig], v[~lcsig]).statistic
                                     / (lcsig.sum() * (~lcsig).sum()))

            e = pd.read_csv(eff, sep="\t")
            dc = [c for c in e.columns if c.startswith("deltapsi")]
            if dc:
                e["jid"] = e.intron.map(lc2j)
                e = e.dropna(subset=["jid"])
                e = e[e.jid.isin(jpos)]
                k_ = e.jid.map(jpos).values
                # LeafCutter reports deltapsi for group_2 vs the baseline (group_1); negate
                # so both sides are group_1 - group_2.
                lcd = -pd.to_numeric(e[dc[0]], errors="coerce").values
                ok = np.isfinite(lcd)
                if ok.sum() > 200:
                    out["slope_raw"] = stats.linregress(lcd[ok], variants["raw"][k_][ok]).slope
                    out["slope_comp"] = stats.linregress(lcd[ok], variants["comp"][k_][ok]).slope
                    out["spearman_raw"] = stats.spearmanr(lcd[ok], variants["raw"][k_][ok]).statistic
            rows.append(out)
        print(f"    {sum(1 for x in rows if x['task']==task)} contrasts", flush=True)

    R = pd.DataFrame(rows)
    out = os.path.join(args.out, f"ds_statistic_comparison_{len(args.tasks)}models.tsv")
    R.to_csv(out, sep="\t", index=False)

    print("\n=== median AUROC vs LeafCutter-significant ATSEs, per seed ===")
    print(R.groupby("seed")[["auroc_raw", "auroc_comp", "auroc_logit"]].median()
          .round(4).to_string())
    print("\n=== median junction-level slope ===")
    print(R.groupby("seed")[["slope_raw", "slope_comp", "spearman_raw"]].median()
          .round(4).to_string())
    print("\n=== cross-seed agreement (same contrast) ===")
    for c in ["auroc_raw", "auroc_comp", "auroc_logit"]:
        p = R.pivot_table(index="contrast", columns="seed", values=c)
        if p.shape[1] >= 2:
            cc = p.corr().values
            print(f"   {c:12s} pairwise r = {np.round(cc[np.triu_indices_from(cc,1)],3)}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
