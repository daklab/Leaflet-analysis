#!/usr/bin/env python
"""
A27: empirical false-positive rate for LeafletFA group DS, by permutation null.

This is Reviewer #1's explicit ask:

    "Although a Bayesian FDR framework is described for differential splicing, the manuscript
     does not provide a clear empirical estimate of false positives ... more explicit reporting
     of type I error control and robustness (e.g., permutation-based null models or age-label
     shuffling) would strengthen confidence."

Design: donor-split within a single cell type
---------------------------------------------
Take one tissue_cell_type, split its DONORS into two arms, and test arm A vs arm B. Both arms
are the same cell type, so the true effect is exactly zero and EVERY call is a false positive.
Any signal is donor-to-donor variation being misread as a cell-type difference -- which is
precisely the failure mode the reviewer is probing.

Two nulls are run so the design choice is evidenced rather than asserted:

  donor  -- permute whole donors  (correct; preserves within-donor correlation)
  cell   -- permute individual cells (WRONG; breaks within-donor correlation)

The meeting notes predict the cell-level null is falsely clean. This measures by how much. If
the two agree, the donor-level machinery is unnecessary; if they diverge, cell-level shuffling
would have understated the false-positive rate and the difference is the size of that error.

Why this is cheap for LeafletFA
-------------------------------
The model is unsupervised -- it never saw cell-type labels -- so permuting labels afterwards
needs NO refit and leaks nothing. PSI is sampled once and reused across every permutation, so
each permutation costs one group-mean phi plus one einsum. LeafCutter, by contrast, must refit
its Dirichlet-multinomial GLM per permutation (~16 min each).

Usage
-----
    python A27_ds_permutation_null.py --fit <fit.npz> --h5ad <atlas> --out <dir> \\
        --n-perm 100 --num-samples 500
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

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from leafletfa.differential_splicing_groups import bayesian_fdr  # noqa: E402

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


def score_batch(dphi, loc, scale, inv_ev, n_atse, S, delta, rng, chunk=20000):
    """TV per (permutation, posterior sample, ATSE), streamed over junctions.

    dphi : [P, K] one row per permutation. PSI is drawn once per junction chunk and SHARED
           across permutations, so they sit on a common posterior draw and are comparable.
    """
    P, K = dphi.shape
    J = loc.shape[1]
    tv = np.zeros((P, S, n_atse), dtype=np.float32)
    dphi_t = torch.from_numpy(dphi)

    for a in range(0, J, chunk):
        b = min(a + chunk, J)
        eps = rng.standard_normal((S, K, b - a)).astype(np.float32)
        psi = torch.from_numpy(
            1.0 / (1.0 + np.exp(-(loc[:, a:b][None] + scale[:, a:b][None] * eps))))
        d = torch.einsum("pk,skj->psj", dphi_t, psi)
        idx = torch.from_numpy(inv_ev[a:b].astype(np.int64))
        blk = torch.zeros((P, S, n_atse), dtype=torch.float32)
        blk.index_add_(2, idx, d.abs())
        tv += blk.numpy()
        del psi, d, blk
    return 0.5 * tv


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--num-samples", type=int, default=500)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.01, 0.02, 0.05, 0.10, 0.20])
    ap.add_argument("--min-donors", type=int, default=8)
    ap.add_argument("--min-cells-per-arm", type=int, default=100)
    ap.add_argument("--batch", type=int, default=20, help="permutations per batch")
    ap.add_argument("--celltypes", type=int, default=6, help="how many cell types to use")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    z = load_fit_strict(args.fit)
    loc, scale, phi = z["psis_loc"], z["psis_scale"], z["phi"]
    meta = json.load(open(os.path.join(os.path.dirname(args.fit), "result.json")))
    K, J = loc.shape
    print(f"[1] fit K={K} seed={meta['seed']}  J={J:,}", flush=True)

    tct = np.array([safe(x) for x in cat(args.h5ad, "tissue_cell_type", "obs")])
    donor = np.array([safe(x) for x in cat(args.h5ad, "mouse.id", "obs")])
    dataset = cat(args.h5ad, "dataset", "obs")
    ev = cat(args.h5ad, "event_id", "var")
    uq, inv = np.unique(ev, return_inverse=True)
    n_atse = len(uq)

    tms = dataset == "tabula_muris_senis"          # donors only meaningful on TMS
    df = pd.DataFrame({"tct": tct, "donor": donor, "tms": tms})
    cand = (df[df.tms].groupby("tct").donor.nunique()
            .sort_values(ascending=False))
    cand = cand[cand >= args.min_donors]
    chosen = cand.head(args.celltypes).index.tolist()
    print(f"[2] {len(cand)} cell types with >= {args.min_donors} donors; using "
          f"{len(chosen)}:", flush=True)
    for c in chosen:
        print(f"      {c}  ({cand[c]} donors)", flush=True)

    rng = np.random.default_rng(args.seed)
    rows, per_perm = [], []

    for ct in chosen:
        cells = np.flatnonzero((tct == ct) & tms)
        dl = donor[cells]
        donors = np.unique(dl)
        print(f"\n[3] {ct}: {len(cells):,} cells, {len(donors)} donors", flush=True)

        for mode in ("donor", "cell"):
            dphis, kept = [], 0
            tries = 0
            while kept < args.n_perm and tries < args.n_perm * 20:
                tries += 1
                if mode == "donor":
                    perm = rng.permutation(donors)
                    h = len(donors) // 2
                    m1 = np.isin(dl, perm[:h])
                else:
                    m1 = rng.random(len(cells)) < 0.5
                a1, a2 = cells[m1], cells[~m1]
                if min(len(a1), len(a2)) < args.min_cells_per_arm:
                    continue
                dphis.append(phi[a1].mean(0) - phi[a2].mean(0))
                kept += 1
            if kept == 0:
                print(f"    {mode}: no valid splits, skipping", flush=True)
                continue
            dphi = np.stack(dphis).astype(np.float32)

            n_sig = {a: [] for a in args.alphas}
            null_tv = []          # [n_perm, n_atse] posterior-mean TV under the null
            for s in range(0, kept, args.batch):
                e = min(s + args.batch, kept)
                tv = score_batch(dphi[s:e], loc, scale, inv, n_atse,
                                 args.num_samples, args.delta,
                                 np.random.default_rng(args.seed + 1000 + s))
                prob = (tv > args.delta).mean(axis=1)          # [batch, n_atse]
                null_tv.append(tv.mean(axis=1).astype(np.float32))   # [batch, n_atse]
                for p in range(prob.shape[0]):
                    for a in args.alphas:
                        _, _, ns = bayesian_fdr(prob[p], alpha=a)
                        n_sig[a].append(int(ns))
                del tv, prob
                print(f"    {mode}: {e}/{kept} permutations", flush=True)

            # Per-permutation counts, not just summaries. The distribution is the result:
            # for Heart cardiac muscle the mean is 185 and the median 5.5, so a summary
            # row cannot show that a handful of donor splits drive everything.
            dl1 = np.abs(dphi).sum(1)
            for a in args.alphas:
                v = np.array(n_sig[a])
                for p_i, val in enumerate(v):
                    per_perm.append(dict(cell_type=ct, mode=mode, alpha=a,
                                         perm=p_i, n_calls=int(val),
                                         dphi_l1=float(dl1[p_i])))
                rows.append(dict(cell_type=ct, mode=mode, alpha=a, n_perm=len(v),
                                 mean_calls=float(v.mean()), median_calls=float(np.median(v)),
                                 q75_calls=float(np.percentile(v, 75)),
                                 q95_calls=float(np.percentile(v, 95)),
                                 max_calls=int(v.max()),
                                 frac_perm_with_any=float((v > 0).mean()),
                                 fpr=float(v.mean() / n_atse),
                                 n_atse=n_atse,
                                 mean_dphi_l1=float(dl1.mean())))
            if mode == "donor" and null_tv:
                nt = np.concatenate(null_tv, axis=0)
                np.savez_compressed(
                    os.path.join(args.out, f"null_tv_{safe(ct)}_K{meta['K']}_seed{meta['seed']}.npz"),
                    null_tv=nt, atse=uq, cell_type=ct, delta=args.delta)
                print(f"    saved null TV {nt.shape} for empirical calibration", flush=True)
            print(f"    {mode}: |dphi|_1 mean {np.abs(dphi).sum(1).mean():.4f}; "
                  f"calls at alpha=0.05 mean {np.mean(n_sig[0.05]):.1f}", flush=True)

    R = pd.DataFrame(rows)
    tag = f"K{meta['K']}_seed{meta['seed']}"
    out = os.path.join(args.out, f"permutation_null_{tag}.tsv")
    R.to_csv(out, sep="\t", index=False)
    pp = pd.DataFrame(per_perm)
    pp["K"] = meta["K"]; pp["model_seed"] = meta["seed"]
    pp_out = os.path.join(args.out, f"permutation_null_perperm_{tag}.tsv.gz")
    pp.to_csv(pp_out, sep="\t", index=False, compression="gzip")
    print(f"    per-permutation counts ({len(pp):,} rows) -> {pp_out}", flush=True)

    print(f"\n=== Empirical false positives, {tag} ===")
    print("(every call is a false positive by construction)\n")
    piv = R.pivot_table(index="alpha", columns="mode", values="mean_calls", aggfunc="mean")
    print("mean ATSEs called per permutation, of "
          f"{n_atse:,} tested:")
    print(piv.round(2).to_string())
    print("\nrealised FPR (called / total ATSEs):")
    print(R.pivot_table(index="alpha", columns="mode", values="fpr",
                        aggfunc="mean").round(5).to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
