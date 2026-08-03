#!/usr/bin/env python
"""
A30: how does DS detection degrade as cells per cell type fall? (Figure 3, rare-cell-type panel)

The claim being tested is the Notion outline's panel D: "markers recovered by LeafletFA in rare
cell types where pseudobulk testing is underpowered, shown by downsampling."

Two distinct advantages get measured, and they should not be conflated:

  1. GRACEFUL DEGRADATION -- how much of the full-data marker set survives as cells are
     removed. Measured here for LeafletFA. The LeafCutter arm of the same ladder is prepared
     by pipelines/08_leafcutter_ds (it needs a pseudobulk rebuild and a GLM refit per rung).

  2. A HARD FLOOR that only LeafCutter has. It needs >= 4 pseudobulk samples per group, and a
     pseudobulk sample needs >= 25 cells, so a cell type thins out of testability entirely.
     Already visible in the real data: 34 of 96 TMS tissue_cell_types clear the pseudobulk bar
     while 107 clear it at cell level -- and Lung_MONOCYTE has 14 donors' worth of cells but
     only 2 pseudobulk samples, so LeafCutter cannot test it at any threshold.

Reference set
-------------
"Recovery" is measured against the FULL-DATA LeafletFA calls for the same contrast, not
against LeafCutter. Using LeafCutter as truth would confound degradation-with-depth against
the between-method disagreement already quantified in A26, and that disagreement is large
(AUROC 0.62-0.66) -- it would swamp the signal this panel is after.

Subsampling is done WITHIN DONOR so the donor composition is held fixed down the ladder;
otherwise thinning would silently drop whole donors and confound depth with donor identity.

Usage
-----
    python A30_downsampling_power.py --fit <fit.npz> --h5ad <atlas> \\
        --manifest <pairwise_manifest.tsv> --out <dir> \\
        --ladder 1.0 0.5 0.25 0.1 0.05 0.02 --n-reps 5
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


def subsample_within_donor(cells, donors, frac, rng):
    """Keep `frac` of each donor's cells, so donor composition is preserved down the ladder."""
    keep = []
    for d in np.unique(donors):
        idx = cells[donors == d]
        n = max(1, int(round(len(idx) * frac)))
        keep.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
    return np.concatenate(keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ladder", type=float, nargs="+",
                    default=[1.0, 0.5, 0.25, 0.1, 0.05, 0.02])
    ap.add_argument("--n-reps", type=int, default=5)
    ap.add_argument("--n-contrasts", type=int, default=12)
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    z = load_fit_strict(args.fit)
    loc, scale, phi = z["psis_loc"], z["psis_scale"], z["phi"]
    meta = json.load(open(os.path.join(os.path.dirname(args.fit), "result.json")))
    K, J = loc.shape
    print(f"[1] fit K={meta['K']} seed={meta['seed']}", flush=True)

    tct = np.array([safe(x) for x in cat(args.h5ad, "tissue_cell_type", "obs")])
    donor = np.array([safe(x) for x in cat(args.h5ad, "mouse.id", "obs")])
    ev = cat(args.h5ad, "event_id", "var")
    uq_ev, inv_ev = np.unique(ev, return_inverse=True)
    n_atse = len(uq_ev)
    inv_t = torch.as_tensor(inv_ev.astype(np.int64))

    man = pd.read_csv(args.manifest, sep="\t")
    man = man[man.status.isin(["ready", "already_done"])]
    man = man.sort_values("n_junctions", ascending=False).head(args.n_contrasts)
    print(f"[2] {len(man)} contrasts", flush=True)

    # PSI drawn once and reused at every rung: the ladder must vary ONLY in cell count,
    # not in which posterior draw each rung happened to get.
    rng0 = np.random.default_rng(args.seed)
    eps = rng0.standard_normal((args.num_samples, K, J)).astype(np.float32)
    psi_s = torch.from_numpy(
        (1.0 / (1.0 + np.exp(-(loc[None] + scale[None] * eps)))).astype(np.float32))
    del eps
    print(f"[3] PSI sampled: S={args.num_samples}", flush=True)

    def score(c1, c2):
        dphi = torch.as_tensor((phi[c1].mean(0) - phi[c2].mean(0)).astype(np.float32))
        d = torch.einsum("k,skj->sj", dphi, psi_s)
        tv = torch.zeros((args.num_samples, n_atse), dtype=torch.float32)
        tv.index_add_(1, inv_t, d.abs())
        tv = 0.5 * tv
        prob = (tv > args.delta).float().mean(0).numpy()
        sig, _, n_sig = bayesian_fdr(prob, alpha=args.alpha)
        return sig, int(n_sig), float(np.abs(dphi.numpy()).sum())

    rows = []
    for _, r in man.iterrows():
        g1, g2 = r.group_1, r.group_2
        c1 = np.flatnonzero(tct == g1)
        c2 = np.flatnonzero(tct == g2)
        if min(len(c1), len(c2)) < 100:
            continue
        ref_sig, ref_n, ref_dphi = score(c1, c2)
        if ref_n < 20:
            continue
        print(f"\n  {g1} vs {g2}: {len(c1)}/{len(c2)} cells, full-data calls {ref_n}",
              flush=True)

        for frac in args.ladder:
            for rep in range(1 if frac == 1.0 else args.n_reps):
                rng = np.random.default_rng(args.seed + rep * 977)
                s1 = subsample_within_donor(c1, donor[c1], frac, rng)
                s2 = subsample_within_donor(c2, donor[c2], frac, rng)
                sig, n_sig, dl1 = score(s1, s2)
                inter = int((sig & ref_sig).sum())
                rows.append(dict(
                    contrast=f"{g1}__vs__{g2}", frac=frac, rep=rep,
                    n_cells_1=len(s1), n_cells_2=len(s2),
                    cells_per_type=int((len(s1) + len(s2)) / 2),
                    n_called=n_sig, n_called_full=ref_n,
                    recall_of_full=inter / ref_n if ref_n else np.nan,
                    precision_vs_full=inter / n_sig if n_sig else np.nan,
                    dphi_l1=dl1, dphi_l1_full=ref_dphi))
            done = [x for x in rows if x["contrast"].startswith(g1) and x["frac"] == frac]
            print(f"    frac={frac:<5} cells/type~{done[-1]['cells_per_type']:>5}  "
                  f"calls {np.mean([d['n_called'] for d in done]):>7.0f}  "
                  f"recall {np.mean([d['recall_of_full'] for d in done]):.3f}", flush=True)

    R = pd.DataFrame(rows)
    tag = f"K{meta['K']}_seed{meta['seed']}"
    out = os.path.join(args.out, f"downsampling_power_{tag}.tsv")
    R.to_csv(out, sep="\t", index=False)

    print(f"\n=== recovery of the full-data marker set, {tag} ===")
    print(R.groupby("frac")[["cells_per_type", "n_called", "recall_of_full",
                             "precision_vs_full", "dphi_l1"]]
          .mean().round(3).to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
