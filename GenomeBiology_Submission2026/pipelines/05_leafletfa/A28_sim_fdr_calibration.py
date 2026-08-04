#!/usr/bin/env python
"""
A28: nominal vs REALISED FDR for LeafletFA group DS, on simulations with ground truth.

Why this exists, and how it differs from A27
--------------------------------------------
A27 runs a permutation null on real data. Under that GLOBAL null every discovery is false,
so the realised FDR is 1.0 whenever anything is called -- A27 measures type-I error VOLUME
(how many false calls, and whether donor structure inflates it). It is not, and cannot be, a
calibration curve.

A calibration curve needs a mixture of true positives and true nulls, so that the realised
false discovery PROPORTION among the called set is a meaningful fraction between 0 and 1.
Only the simulations have that: `uns["atse_delta_psi"]` gives the true per-ATSE effect and
~31% of the 4,268 ATSEs are truly differential, leaving ~69% true nulls.

    realised FDP(alpha) = #{called ATSEs with |true dPSI| <= null_tol} / #{called ATSEs}

and a well-calibrated procedure has realised FDP <= nominal alpha.

Contrast definition
-------------------
The simulator assigns each cell a `true_program` and stores `uns["true_psi_skip"]` as
[n_programs, n_ATSE]. So for programs p and q the TRUE effect per ATSE is exactly
true_psi_skip[p] - true_psi_skip[q], and the cells are grouped by true_program. No label
inference is involved -- the ground truth is the generative parameter.

The effect ladder (eff0.1/0.25/0.5/1.0, median |dPSI| 0.09/0.21/0.42/0.84) lets the curve be
reported against effect size, which matters because the atlas contrasts sit at the low end.

Usage
-----
    python A28_sim_fdr_calibration.py --sims <a.h5ad> <b.h5ad> --out <dir> --K 5
"""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from leafletfa.minibatch import LeafletFAMinibatch          # noqa: E402
from leafletfa.differential_splicing_groups import bayesian_fdr  # noqa: E402


def read_h5ad_bits(path):
    """Layers, the true program per cell, and the generative truth. No anndata needed."""
    import scipy.sparse as sp
    with h5py.File(path, "r") as f:
        def dense(layer):
            g = f["layers"][layer]
            if isinstance(g, h5py.Group):                     # sparse
                shape = tuple(g.attrs["shape"])
                m = sp.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape)
                return np.asarray(m.todense(), dtype=np.float32)
            return np.asarray(g[:], dtype=np.float32)

        junc = dense("cell_by_junction_matrix")
        clus = dense("cell_by_cluster_matrix")

        node = f["obs"]["true_program"]
        if isinstance(node, h5py.Group):
            cats = np.array([x.decode() if isinstance(x, bytes) else x
                             for x in node["categories"][:]])
            prog = pd.Categorical.from_codes(node["codes"][:], cats).astype(str)
        else:
            prog = np.array([x.decode() if isinstance(x, bytes) else x for x in node[:]])

        ev_node = f["var"]["event_id"]
        if isinstance(ev_node, h5py.Group):
            cats = np.array([x.decode() if isinstance(x, bytes) else x
                             for x in ev_node["categories"][:]])
            ev = pd.Categorical.from_codes(ev_node["codes"][:], cats).astype(str)
        else:
            ev = np.array([x.decode() if isinstance(x, bytes) else x for x in ev_node[:]])

        true_psi = f["uns"]["true_psi_skip"][:]               # [n_prog, n_atse]
        atse_ids = np.array([x.decode() if isinstance(x, bytes) else x
                             for x in f["uns"]["atse_ids"][:]])
    return junc, clus, prog, ev, true_psi, atse_ids


def check_tv_wellformed(d, tv, inv_ev, n_atse, label="", tol_sum=1e-4, verbose=True):
    """Assert TV is actually a total variation. Cheap, and would have caught three real bugs.

    TV = 0.5 * sum_j |Delta_j| is the total variation distance between the mean junction usage of
    two groups. That interpretation requires psi to be a composition within each ATSE. Two ways it
    silently stops being one:

      1. sum_j Delta_j must be EXACTLY 0 per ATSE, because Delta is the difference of two
         compositions and dphi itself sums to zero. It is non-zero when psi's within-event scale
         differs ACROSS PROGRAMS -- the gauge direction where seeds disagree (sum r 0.51-0.66 vs
         composition r 0.95-0.96), and the direction the simplex removes.
      2. TV must lie in [0, 1]. Nothing in the code enforced this, so an inflated psi could push
         TV past 1 and be compared against delta=0.10 as if nothing were wrong.

    Reports rather than raises: the released Beta-psi fits legitimately violate (1), and the point
    is that it becomes visible rather than that the run dies. Returns a dict of the diagnostics.
    """
    idx = torch.as_tensor(inv_ev.astype(np.int64))
    dsum = torch.zeros((d.shape[0], n_atse), dtype=torch.float32).index_add_(1, idx, d)
    max_abs_sum = float(dsum.abs().max())
    tv_min, tv_max = float(tv.min()), float(tv.max())
    bad_sum = int((dsum.abs() > tol_sum).float().mean(0).gt(0.5).sum())
    out = dict(max_abs_delta_sum=max_abs_sum, frac_atse_sum_violating=bad_sum / max(n_atse, 1),
               tv_min=tv_min, tv_max=tv_max)
    if verbose and (max_abs_sum > tol_sum or tv_max > 1.0 or tv_min < 0.0):
        print(f"  [TV CHECK{(' ' + label) if label else ''}] "
              f"max|sum_j Delta_j| = {max_abs_sum:.3e} (should be ~0; non-zero means per-program "
              f"psi scale mismatch, so TV is not a total variation); "
              f"{100*bad_sum/max(n_atse,1):.1f}% of ATSEs violating; "
              f"TV range [{tv_min:.4f}, {tv_max:.4f}] (should be within [0, 1])", flush=True)
    return out


def score_contrast(psi_s, phi, cells_1, cells_2, inv_ev, n_atse, delta, check=True, label=""):
    """Per-sample ATSE total variation for one contrast. psi_s is [S, K, J]."""
    dphi = torch.as_tensor((phi[cells_1].mean(0) - phi[cells_2].mean(0)).astype(np.float32))
    d = torch.einsum("k,skj->sj", dphi, psi_s)                # [S, J]
    tv = torch.zeros((psi_s.shape[0], n_atse), dtype=torch.float32)
    tv.index_add_(1, torch.as_tensor(inv_ev.astype(np.int64)), d.abs())
    tv = 0.5 * tv
    if check:
        check_tv_wellformed(d, tv, inv_ev, n_atse, label=label)
    # Jeffreys count, not the raw proportion. S/S records prob_de = 1 exactly, so 1 - prob_de
    # is 0 and the Bayesian-FDP rule reports zero expected false discoveries over a top set that
    # actually contains ~2% false ones. (n_exceed + 0.5)/(S + 1) caps prob_de below 1 and floors
    # the FDP at a value the draws can actually support.
    S = tv.shape[0]
    prob = ((tv > delta).float().sum(0) + 0.5) / (S + 1.0)
    return tv.mean(0).numpy(), prob.numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sims", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--num-samples", type=int, default=500)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--null-tol", type=float, default=0.05,
                    help="|true dPSI| at or below this counts as a TRUE NULL")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.01, 0.02, 0.05, 0.10, 0.20, 0.30])
    ap.add_argument("--model-seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for path in args.sims:
        tag = os.path.basename(path).replace(".h5ad", "")
        print(f"\n=== {tag} ===", flush=True)
        junc, clus, prog, ev, true_psi, atse_ids = read_h5ad_bits(path)
        uq_ev, inv_ev = np.unique(ev, return_inverse=True)
        n_atse = len(uq_ev)
        # uns['atse_ids'] and the var event order must agree, or truth is joined to the
        # wrong ATSE. Compare by identity, never by length.
        assert n_atse == len(atse_ids), f"{n_atse} ATSEs in var vs {len(atse_ids)} in uns"
        assert (uq_ev == np.sort(atse_ids)).all() or (uq_ev == atse_ids).all(), \
            "ATSE id order differs between var/event_id and uns/atse_ids"
        order = {a: i for i, a in enumerate(atse_ids)}
        truth_idx = np.array([order[a] for a in uq_ev])       # map var-order -> uns-order

        print(f"  {junc.shape[0]:,} cells x {junc.shape[1]:,} junctions, {n_atse:,} ATSEs",
              flush=True)

        t0 = time.time()
        m = LeafletFAMinibatch(junc, clus, K=args.K, device="cpu")
        m.fit(num_steps=args.steps, subsample_size=min(args.batch, junc.shape[0]),
              lr=args.lr, gamma=args.gamma, seed=args.model_seed, verbose=False)
        _pm = m.phi_mean()
        phi = np.asarray(_pm.cpu().numpy() if hasattr(_pm, 'cpu') else _pm)
        # psi_loc_scale returns numpy on the minibatch trainer; tolerate either.
        loc, scale = [np.asarray(x.cpu().numpy() if hasattr(x, 'cpu') else x)
                      for x in m.psi_loc_scale()]
        print(f"  fit in {time.time()-t0:.0f}s", flush=True)
        np.savez_compressed(os.path.join(args.out, f"simfit_{tag}.npz"),
                            phi=phi, psi_loc=loc, psi_scale=scale)
        assert np.isfinite(loc).all() and np.isfinite(scale).all(), "non-finite PSI posterior"

        rng = np.random.default_rng(0)
        eps = rng.standard_normal((args.num_samples,) + loc.shape).astype(np.float32)
        psi_s = torch.from_numpy(
            (1.0 / (1.0 + np.exp(-(loc[None] + scale[None] * eps)))).astype(np.float32))

        progs = sorted(set(prog))
        for i in range(len(progs)):
            for j in range(i + 1, len(progs)):
                p, q = progs[i], progs[j]
                c1 = np.flatnonzero(prog == p)
                c2 = np.flatnonzero(prog == q)
                if min(len(c1), len(c2)) < 50:
                    continue
                # TRUE effect from the generative parameters, in var-ATSE order
                tp = int(p.split("_")[-1]) if "_" in str(p) else int(p)
                tq = int(q.split("_")[-1]) if "_" in str(q) else int(q)
                true_d = np.abs(true_psi[tp] - true_psi[tq])[truth_idx]
                is_null = true_d <= args.null_tol
                if is_null.sum() < 50 or (~is_null).sum() < 50:
                    continue

                tv, prob = score_contrast(psi_s, phi, c1, c2, inv_ev, n_atse, args.delta)
                np.savez_compressed(
                    os.path.join(args.out, f"simscores_{tag}_{p}_vs_{q}.npz"),
                    prob_de=prob.astype(np.float32), tv=tv.astype(np.float32),
                    true_delta=true_d.astype(np.float32), is_null=is_null,
                    atse=uq_ev)
                for a in args.alphas:
                    sig, _, n_sig = bayesian_fdr(prob, alpha=a)
                    n_false = int((sig & is_null).sum())
                    rows.append(dict(
                        sim=tag, contrast=f"{p}_vs_{q}", alpha=a,
                        n_called=int(n_sig), n_false=n_false,
                        realised_fdp=(n_false / n_sig) if n_sig else np.nan,
                        power=float((sig & ~is_null).sum() / max((~is_null).sum(), 1)),
                        n_true_alt=int((~is_null).sum()), n_true_null=int(is_null.sum()),
                        median_true_effect=float(np.median(true_d[~is_null]))))
                print(f"    {p} vs {q}: alt={int((~is_null).sum())} null={int(is_null.sum())}"
                      f"  called@0.05={rows[-len(args.alphas)+2]['n_called']}", flush=True)

    R = pd.DataFrame(rows)
    out = os.path.join(args.out, "sim_fdr_calibration.tsv")
    R.to_csv(out, sep="\t", index=False)
    print("\n=== nominal vs realised FDP (mean over contrasts) ===")
    print(R.groupby(["sim", "alpha"])[["realised_fdp", "power", "n_called"]]
          .mean().round(3).to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
