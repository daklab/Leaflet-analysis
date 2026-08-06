#!/usr/bin/env python
"""A49: LeafletFA differential splicing on LeafCutter's own contrasts, using the LRT.

Replaces A26, which is built on machinery that has since been retired: prob_de = P(|delta| > 0.10)
against a fixed threshold, scored with bayesian_fdr, on a BetaBinomial fit. That rule was measured
at 2.9x over nominal FDR at alpha=0.05 on simulations with known truth. The current DS test is a
per-ATSE Dirichlet-multinomial likelihood-ratio test (A45).

Why the comparison is now like-for-like
---------------------------------------
LeafCutter's `ds_results_cluster_significance.txt` reports `loglr`, `df`, `p` -- it IS a
likelihood-ratio test on a Dirichlet-multinomial. Our test is the same form on the same likelihood
(verified term-for-term against `dm_glm.stan`: -114.362812 vs -114.362793). Under the old fixed-delta
rule the two were not comparable at all; a posterior tail probability and a p-value are different
objects. Now the same columns mean the same thing and can be merged on `cluster`.

What differs, and is the actual subject of the comparison:
  - LeafCutter aggregates cells into pseudobulk samples per donor, then tests across samples.
    LeafletFA tests across single cells directly, so n is ~1000x larger but each observation is
    far sparser.
  - `--design group` uses a binary group indicator, i.e. exactly LeafCutter's design. This isolates
    single-cell-vs-pseudobulk with the test held fixed.
  - `--design phi` uses the learned program loadings instead, which is LeafletFA's actual claim:
    differential splicing across structure discovered WITHOUT labels. Calls from the two designs
    can then be compared to ask how much of the labelled signal the unsupervised model recovers.

Label matching
--------------
Manifest groups are written `BAT_B_CELL` while the atlas writes `BAT_B CELL`, so a naive join
matched only 8 of 70 contrasts. Normalising both to uppercase-underscore matches all 73 (70 tabula
muris senis + 3 allen brain), every one with at least 218 cells in the smaller arm.

Output columns mirror LeafCutter's so the two tables merge directly on `cluster`.
"""

import argparse, os, re, sys
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import chi2

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from A45_dm_lrt import fit_event, bh   # noqa: E402

WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
H5 = f"{WD}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad"
LC = f"{WD}/LEAFCUTTER_DS_2026-08"


def norm(s):
    return re.sub(r"[^A-Z0-9]+", "_", str(s).upper()).strip("_")


def obs_cat(f, name):
    n = f["obs"][name]
    if isinstance(n, h5py.Group):
        c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
        return c[n["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in n[:]])


def read_rows(layer, rows, n_cols, h5=H5, chunk=2000):
    order = np.argsort(rows); srt = np.asarray(rows)[order]
    out = np.zeros((len(rows), n_cols), np.float32)
    with h5py.File(h5, "r") as f:
        g = f["layers"][layer]; ip_all = g["indptr"]
        for a in range(0, len(srt), chunk):
            blk = srt[a:a + chunk]
            lo, hi = int(blk[0]), int(blk[-1]) + 1
            ip = ip_all[lo:hi + 1]; s, e = int(ip[0]), int(ip[-1])
            m = sp.csr_matrix((g["data"][s:e], g["indices"][s:e], ip - ip[0]), shape=(hi - lo, n_cols))
            out[order[a:a + len(blk)]] = m[blk - lo].toarray()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", required=True, help="DM+simplex atlas fit.npz")
    ap.add_argument("--manifest", default=f"{LC}/pairwise_manifest_tabula_muris_senis_tissue_cell_type.tsv")
    ap.add_argument("--label-field", default="tissue_cell_type")
    ap.add_argument("--design", default="group", choices=["group", "phi"],
                    help="'group' = binary indicator, LeafCutter's design, isolates single-cell vs "
                         "pseudobulk. 'phi' = learned program loadings, LeafletFA's label-free claim.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--contrasts", type=int, default=0, help="0 = all")
    ap.add_argument("--min-cells", type=int, default=50)
    ap.add_argument("--max-atse", type=int, default=0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))

    z = np.load(args.fit, allow_pickle=True)
    psi, phi = np.asarray(z["psi_mean"]), np.asarray(z["phi"])
    conc_vec = np.asarray(z["bb_conc_vec"], np.float32) if "bb_conc_vec" in z else None
    K = psi.shape[0]

    with h5py.File(H5, "r") as f:
        n = f["var"]["event_id"]
        c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
        ev = c[n["codes"][:]].astype(str)
        J = len(ev)
        lab = obs_cat(f, args.label_field)
    uq, inv = np.unique(ev, return_inverse=True)
    n_atse = len(uq)
    amap = {}
    for a in np.unique(lab):
        amap.setdefault(norm(a), a)

    man = pd.read_csv(args.manifest, sep="\t")
    man = man[man.status == "ready"].copy()
    man["a1"] = man.group_1.map(lambda g: amap.get(norm(g)))
    man["a2"] = man.group_2.map(lambda g: amap.get(norm(g)))
    man = man[man.a1.notna() & man.a2.notna()]
    if args.contrasts:
        man = man.head(args.contrasts)
    print(f"[1] {len(man)} contrasts mappable; K={K}, {n_atse:,} ATSEs, design={args.design}",
          flush=True)

    rows_all = []
    for ci, (_, row) in enumerate(man.iterrows()):
        c1 = np.flatnonzero(lab == row.a1)
        c2 = np.flatnonzero(lab == row.a2)
        if min(len(c1), len(c2)) < args.min_cells:
            continue
        cells = np.sort(np.concatenate([c1, c2]))
        y_all = read_rows("cell_by_junction_matrix", cells, J)
        cl_all = read_rows("cell_by_cluster_matrix", cells, J)
        is2 = np.isin(cells, c2).astype(np.float32)
        if args.design == "group":
            X_all = torch.from_numpy((is2 - is2.mean()).reshape(-1, 1))
        else:
            P = phi[cells]
            X_all = torch.from_numpy((P - P.mean(0, keepdims=True)).astype(np.float32))
        idx = np.arange(n_atse)
        if args.max_atse:
            idx = np.sort(np.random.default_rng(0).choice(n_atse, min(args.max_atse, n_atse), False))
        recs = []
        for e in idx:
            cols = np.flatnonzero(inv == e)
            m = len(cols)
            cov = cl_all[:, cols[0]] > 0
            if m < 2 or cov.sum() < args.min_cells:
                continue
            y = torch.from_numpy(y_all[cov][:, cols].astype(np.float32))
            Xc = X_all[torch.from_numpy(np.flatnonzero(cov))]
            conc0 = float(conc_vec[e]) if conc_vec is not None else 50.0
            l0 = fit_event(y, Xc, conc0, args.steps, 1.0, alt=False, free_conc=True)
            l1 = fit_event(y, Xc, conc0, args.steps, 1.0, alt=True, free_conc=True)
            stat = max(2.0 * (l1 - l0), 0.0)
            df = (m - 1) * X_all.shape[1]
            # deltapsi from the model, so the effect size is comparable to LeafCutter's
            mu1 = phi[c1].mean(0) @ psi[:, cols]
            mu2 = phi[c2].mean(0) @ psi[:, cols]
            recs.append(dict(cluster=str(uq[e]), m=m, n_cells=int(cov.sum()), loglr=stat / 2.0,
                             df=df, p=float(chi2.sf(stat, df)),
                             deltapsi_max=float(np.abs(mu2 / mu2.sum() - mu1 / mu1.sum()).max())))
        if not recs:
            continue
        R = pd.DataFrame(recs)
        R["p.adjust"] = np.where(bh(R["p"].values, args.alpha), 1.0, 1.0)  # placeholder, replaced below
        # BH q-values, so the column means what LeafCutter's does
        o = np.argsort(R["p"].values); q = np.empty(len(R))
        pv = R["p"].values[o]; n_ = len(R)
        q_o = np.minimum.accumulate((pv * n_ / np.arange(1, n_ + 1))[::-1])[::-1]
        q[o] = np.clip(q_o, 0, 1)
        R["p.adjust"] = q
        R.insert(0, "contrast", f"{row.group_1}__vs__{row.group_2}")
        R.insert(1, "design", args.design)
        rows_all.append(R)
        print(f"  [{ci+1}/{len(man)}] {row.group_1} vs {row.group_2}: {len(R):,} ATSEs, "
              f"{int((R['p.adjust']<=args.alpha).sum()):,} called at q<={args.alpha}", flush=True)

    if not rows_all:
        print("no contrasts scored"); return
    out = pd.concat(rows_all, ignore_index=True)
    fp = os.path.join(args.out, f"a49_leafletfa_lrt_{args.design}.tsv")
    out.to_csv(fp, sep="\t", index=False)
    print(f"\n[2] {len(out):,} rows over {out.contrast.nunique()} contrasts -> {fp}")
    s = out.groupby("contrast").apply(lambda d: (d["p.adjust"] <= args.alpha).sum())
    print(f"    calls per contrast: median {int(s.median()):,}, range {int(s.min()):,}-{int(s.max()):,}")


if __name__ == "__main__":
    main()
