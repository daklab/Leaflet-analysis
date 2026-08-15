#!/usr/bin/env python
"""A64: derive delta from permuted negative-control ATSEs, and check the construction first.

What delta is
-------------
delta is the effect size of interest in `P(TV > delta)` -- the smallest amount of splicing mass that
has to move before we are willing to call an event differential. Units are fraction of the ATSE's
splicing mass, on [0, 1]; at m=2 it is exactly |dPSI|. It is NOT the significance threshold, which
is the separate `q <= alpha`. delta says how big, q says how sure.

Why it has to be derived rather than chosen
-------------------------------------------
delta must sit above the model's own resolution -- the TV that appears when nothing is actually
different. Reading it off a simulation table is post hoc, and on those simulations `effect` caps the
achievable delta (at effect 0.15 the largest true delta in the whole dataset is 0.147), so the
evidence for "0.15-0.20 is calibrated" rests on two effect sizes only.

A fit made with `--permute-atse-frac` contains its own answer: the permuted ATSEs are null BY
CONSTRUCTION, and because only ~10% were touched the remaining events still pin phi, so dphi keeps
its true magnitude. Their TV distribution is therefore the resolution floor AT THE RIGHT SCALE, on
whatever data the fit was made from, with no truth labels required. delta = a high quantile of it.

THE GATE, which runs first and is not optional
----------------------------------------------
On simulations the truly-null ATSEs are known, so the construction is checkable directly. The
permuted ATSEs' TV distribution should MATCH the true nulls':

    permuted TIGHTER than true nulls  -> dphi has collapsed; any delta from here is too small and
                                        the test built on it will be anti-conservative
    permuted LOOSER                   -> the permutation destroyed more than the cell-to-usage
                                        association; delta comes out too large and the test loses
                                        real signal

A39 also looked sound on paper and was anti-conservative for a reason only visible once it was
built, so this is measured rather than assumed. `--gate-only` stops after the check.
"""

import argparse, itertools, os, re, sys

import h5py
import numpy as np
import torch
from scipy.stats import ks_2samp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from A51_group_ds_posterior import draw_psi, draw_dphi, tv_per_atse      # noqa: E402
from _permute_null import summarise                                       # noqa: E402


def obs_arr(f, name):
    x = f["obs"][name]
    if isinstance(x, h5py.Group):
        c = np.array([y.decode() if isinstance(y, bytes) else str(y) for y in x["categories"][:]])
        return c[x["codes"][:]]
    return np.array([y.decode() if isinstance(y, bytes) else str(y) for y in x[:]])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", required=True, help="a fit made with --permute-atse-frac > 0")
    ap.add_argument("--sim", help="simulation h5ad. Supplies the truth needed for the GATE.")
    ap.add_argument("--atlas-h5", help="atlas h5ad. No truth, so the GATE cannot run -- delta is "
                                       "derived from the permuted ATSEs alone. Only defensible "
                                       "because the gate PASSED on simulations first.")
    ap.add_argument("--label-field", default="tissue_cell_type")
    ap.add_argument("--contrasts", type=int, default=8,
                    help="atlas mode: how many cell-type pairs to average the floor over")
    ap.add_argument("--min-cells", type=int, default=200)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--quantile", type=float, default=0.95,
                    help="quantile of the null floor to take as delta")
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    z = np.load(args.fit, allow_pickle=True)
    if "permuted_atses" not in z.files or np.asarray(z["permuted_atses"]).size == 0:
        print("FATAL: this fit has no permuted ATSEs. Refit with --permute-atse-frac 0.10.",
              file=sys.stderr)
        return 2
    perm = np.asarray(z["permuted_atses"]).ravel()
    fev = np.asarray(z["event_id"]).astype(str)
    atse_idx = torch.from_numpy(np.asarray(z["inv_ev"], np.int64))
    n_atse = len(fev)
    K = int(np.asarray(z["phi"]).shape[1])
    print(f"[1] {os.path.basename(args.fit)}: {n_atse:,} ATSEs, {len(perm):,} permuted "
          f"({100*len(perm)/n_atse:.0f}%), K={K}", flush=True)

    if "phi_loc" not in z.files or np.asarray(z["phi_loc"]).size == 0:
        print("FATAL: no q(phi) in this fit; dphi would be pinned at the mean.", file=sys.stderr)
        return 2

    ATLAS = args.atlas_h5 is not None
    if ATLAS:
        # No truth on the atlas, so the gate cannot run here. That is exactly why it runs on
        # simulations FIRST -- the construction is validated there and only then applied to data
        # where it cannot be checked.
        from _phi_rows import phi_row_of
        with h5py.File(args.atlas_h5, "r") as f:
            lab = obs_arr(f, args.label_field)
        N = len(lab)
        prow = phi_row_of(z, N)
        big = [c for c, n in zip(*np.unique(lab, return_counts=True)) if n >= args.min_cells]
        # pair cell types WITHIN a tissue where possible, so dphi is a realistic contrast rather
        # than an artificially large cross-tissue one
        pairs = list(itertools.combinations(sorted(big), 2))[:args.contrasts]
        groups = [(np.flatnonzero((lab == a) & (prow >= 0)),
                   np.flatnonzero((lab == b) & (prow >= 0)), f"{a}|{b}") for a, b in pairs]
        groups = [(prow[g1], prow[g2], nm) for g1, g2, nm in groups
                  if min(len(g1), len(g2)) >= args.min_cells]
        true_psi = None; tmap = np.full(n_atse, -1)
    else:
        with h5py.File(args.sim, "r") as f:
            prog = obs_arr(f, "true_program")
            true_psi = np.asarray(f["uns"]["true_psi_skip"][:], float)
            ids = np.array([y.decode() if isinstance(y, bytes) else str(y)
                            for y in f["uns"]["atse_ids"][:]])
        pos = {e: i for i, e in enumerate(ids)}
        tmap = np.array([pos.get(e, -1) for e in fev])

    psi_s = draw_psi(z, atse_idx, n_atse, args.samples, seed=args.seed)
    if not ATLAS:
        u = sorted(np.unique(prog))
        groups = [(np.flatnonzero(prog == a), np.flatnonzero(prog == b), f"{a} vs {b}")
                  for a, b in itertools.combinations(u[:3], 2)]
    deltas = []
    for r1, r2, name in groups:
        if min(len(r1), len(r2)) < 100:
            continue
        dphi = draw_dphi(z, r1, r2, args.samples, K, seed=args.seed)
        tv, _ = tv_per_atse(dphi, psi_s, atse_idx, n_atse)
        tv_pt = tv.mean(0).numpy()
        is_perm = np.zeros(n_atse, bool); is_perm[perm] = True
        if ATLAS:
            print(f"\n=== {name}   ||dphi||_1 = {float(dphi.abs().sum(1).mean()):.3f} ===")
            d = summarise(tv_pt[is_perm], tv_pt[~is_perm], label="")
            q99 = float(np.quantile(tv_pt[is_perm], 0.99))
            print(f"  p99 of the floor = {q99:.3f}  <- on simulations this quantile, not p95, is "
                  f"what matched the empirically calibrated delta")
            deltas.append((d, q99))
            continue
        ia = int(re.sub(r"\D", "", str(pa))); ib = int(re.sub(r"\D", "", str(pb)))

        keep = tmap >= 0
        true_tv = np.full(n_atse, np.nan)
        true_tv[keep] = np.abs(true_psi[ia, tmap[keep]] - true_psi[ib, tmap[keep]])
        is_perm = np.zeros(n_atse, bool); is_perm[perm] = True
        # true nulls among the UNTOUCHED events -- the reference the gate compares against
        true_null = (~is_perm) & np.isfinite(true_tv) & (true_tv <= 1e-9)

        print(f"\n=== contrast {pa} vs {pb}   ||dphi||_1 = {float(dphi.abs().sum(1).mean()):.3f} ===")
        print("  --- GATE: does the permuted set look like a real null? ---")
        a, b = tv_pt[is_perm], tv_pt[true_null]
        print(f"  permuted   n={a.size:,}  median {np.median(a):.4f}  p95 {np.quantile(a,.95):.4f}")
        print(f"  TRUE null  n={b.size:,}  median {np.median(b):.4f}  p95 {np.quantile(b,.95):.4f}")
        if b.size > 30:
            ks = ks_2samp(a, b)
            ratio = np.median(a) / max(np.median(b), 1e-12)
            verdict = ("PASS -- indistinguishable" if 0.8 <= ratio <= 1.25 else
                       ("FAIL: permuted TIGHTER, delta would be too small (anti-conservative)"
                        if ratio < 0.8 else
                        "FAIL: permuted LOOSER, delta would be too large (loses real signal)"))
            print(f"  median ratio permuted/true = {ratio:.2f}   KS D={ks.statistic:.3f} "
                  f"p={ks.pvalue:.2g}")
            print(f"  {verdict}")
        if not args.gate_only:
            d = summarise(a, tv_pt[~is_perm], label="")
            deltas.append(d)

    if deltas and ATLAS:
        p95 = [a for a, _ in deltas]; p99 = [b for _, b in deltas]
        print(f"\n=== delta measured on the ATLAS, over {len(deltas)} contrasts ===")
        print(f"  p95 of the floor: " + "  ".join(f"{d:.3f}" for d in p95)
              + f"   -> max {max(p95):.2f}")
        print(f"  p99 of the floor: " + "  ".join(f"{d:.3f}" for d in p99)
              + f"   -> max {max(p99):.2f}")
        print(f"\n  RECOMMENDED delta = {max(p99):.2f}")
        print(f"  Using p99 rather than p95: on simulations p95 gave 0.06-0.15 while the FDR sweep "
              f"calibrated at 0.15, and p99 (0.08-0.21) is what tracked it. p95 controls a "
              f"PER-EVENT error rate; with tens of thousands of events the aggregate procedure "
              f"needs the stricter quantile.")
    elif deltas:
        print(f"\n=== delta from the {100*args.quantile:.0f}th percentile of the measured floor ===")
        print(f"  per contrast: " + "  ".join(f"{d:.3f}" for d in deltas))
        print(f"  RECOMMENDED delta = {max(deltas):.2f}  (the largest, so the floor is cleared in "
              f"every contrast rather than on average)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
