#!/usr/bin/env python
"""A62: the SHIPPED posterior DS test, scored on simulations against ground truth.

Why a new script rather than A28
--------------------------------
A28 is the only existing sim scorer for the posterior test, and it runs the RETIRED rule: a fixed
`delta = 0.10` compared against a TV built from a POINT estimate of dphi, with `1 - prob` pushed
through Benjamini-Hochberg. Every one of those three pieces has since been replaced --
    * dphi is now DRAWN from q(phi) rather than pinned at the posterior mean, because dphi
      multiplies psi in every term of Delta and it is the half of the statistic that carries the
      biology; pinning it understated the spread of TV;
    * `1 - prob` is a local false discovery rate, not a p-value, so BH's guarantee never applied to
      it -- `bayesian_fdr` takes the running mean of the lfdr instead, which IS the expected fraction
      of false calls in the called set;
    * delta is stated as the effect size of interest rather than tuned.
A28's numbers therefore cannot be quoted for the shipped test, and re-running A28 would only
reproduce the old rule.

Why not just point A51 at a simulation
--------------------------------------
A51 hardcodes the atlas h5ad and takes its groups from a LeafCutter cell-type manifest. Rather than
fork the statistic -- which is exactly how two implementations drift apart -- this imports A51's
`draw_psi`, `draw_dphi`, `bayesian_fdr` and `tv_per_atse` UNCHANGED and supplies simulation groups.
The test being validated here is literally the code that runs on the atlas.

Truth, and the scoring trap it avoids
-------------------------------------
The simulator stores `uns/true_psi_skip` as [n_programs, n_ATSE], so for programs p and q the true
per-ATSE effect is exactly the TV between those two rows -- a generative parameter, not an estimate.
Cells are grouped by `obs/true_program`.

Power MUST be scored against the truth for THIS contrast, not against "differential across any
program pair". An event that differs between programs 2 and 3 is not something a 0-vs-1 test was
ever asked to find, and counting it as a miss understated power by ~1.5x in an earlier evaluation.

    realised FDP(alpha) = #{called with true TV <= null_tol} / #{called}

so a calibrated procedure has realised FDP <= alpha. Reported per contrast AND pooled, because a
single contrast is one draw and the spread across the six pairs is the honest error bar.
"""

import argparse, glob, itertools, os, re, sys

import h5py
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from leafletfa.dirichlet_multinomial import build_atse_index                     # noqa: E402
# the shipped statistic, imported rather than reimplemented
from A51_group_ds_posterior import (draw_psi, draw_dphi, bayesian_fdr,           # noqa: E402
                                    tv_per_atse)


def obs_arr(f, name):
    n = f["obs"][name]
    if isinstance(n, h5py.Group):
        c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
        return c[n["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in n[:]])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--fit", required=True, help="fit npz from the SHIPPED recipe (needs phi_loc)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--delta", type=float, default=0.05,
                    help="effect size of interest. 0.05 matches how the simulator defines a true "
                         "positive, so it is a statement about the truth, not a tuning knob.")
    ap.add_argument("--null-tol", type=float, default=0.05,
                    help="an ATSE is a true null for a contrast when its true TV is <= this")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.20])
    ap.add_argument("--min-cells", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tag = os.path.basename(args.sim).replace(".h5ad", "").replace("sim_", "")
    fit_seed = re.search(r"seed(\d+)\.npz$", args.fit)
    fit_seed = int(fit_seed.group(1)) if fit_seed else -1

    with h5py.File(args.sim, "r") as f:
        prog = obs_arr(f, "true_program").astype(str)
        n_ev = f["var"]["event_id"]
        c = np.array([x.decode() if isinstance(x, bytes) else x for x in n_ev["categories"][:]])
        ev = c[n_ev["codes"][:]].astype(str)
        true_psi = np.asarray(f["uns"]["true_psi_skip"][:], float)      # [n_prog, n_atse]
        atse_ids = np.array([x.decode() if isinstance(x, bytes) else x
                             for x in f["uns"]["atse_ids"][:]]).astype(str)

    z = np.load(args.fit, allow_pickle=True)
    K = int(np.asarray(z["phi"]).shape[1])
    if "phi_loc" not in z.files or np.asarray(z["phi_loc"]).size == 0:
        print("FATAL: this fit has no q(phi). dphi would be pinned at the posterior mean, which is\n"
              "       the exact understatement of TV's spread this test was changed to remove.\n"
              "       Refit with a build that saves phi_loc/phi_scale.", file=sys.stderr)
        return 2

    # the fit's own event order is authoritative -- align truth onto it rather than assuming
    fit_ev = np.asarray(z["event_id"]).astype(str)
    atse_idx = torch.from_numpy(np.asarray(z["inv_ev"], np.int64))
    n_atse = len(fit_ev)
    pos = {e: i for i, e in enumerate(atse_ids)}
    tmap = np.array([pos.get(e, -1) for e in fit_ev])
    if (tmap < 0).any():
        print(f"  WARNING: {(tmap<0).sum()} of {n_atse} fit ATSEs absent from the truth table; "
              f"they are dropped from scoring", flush=True)

    print(f"[1] {tag} fit-seed {fit_seed}: {len(prog):,} cells, {n_atse:,} ATSEs, K={K}", flush=True)

    psi_s = draw_psi(z, atse_idx, n_atse, args.samples, seed=args.seed)          # [S, K, J]

    progs = sorted(np.unique(prog), key=lambda s: str(s))
    usable = [p for p in progs if (prog == p).sum() >= args.min_cells]
    print(f"[2] programs with >= {args.min_cells} cells: {usable}", flush=True)

    rows, per_atse = [], []
    for pa, pb in itertools.combinations(usable, 2):
        r1 = np.flatnonzero(prog == pa)
        r2 = np.flatnonzero(prog == pb)
        dphi = draw_dphi(z, r1, r2, args.samples, K, seed=args.seed)
        if dphi is None:
            print("  q(phi) unusable -- stopping rather than silently pinning dphi", file=sys.stderr)
            return 2
        tv, _ = tv_per_atse(dphi, psi_s, atse_idx, n_atse)                       # [S, n_atse]
        prob = (tv > args.delta).float().mean(0).numpy()
        q = bayesian_fdr(prob)

        # contrast-specific truth: TV between the two generating compositions
        ia = int(re.sub(r"\D", "", str(pa)) or -1)
        ib = int(re.sub(r"\D", "", str(pb)) or -1)
        if not (0 <= ia < true_psi.shape[0] and 0 <= ib < true_psi.shape[0]):
            print(f"  cannot map {pa}/{pb} onto true_psi_skip rows -- skipped", flush=True)
            continue
        keep = tmap >= 0
        true_tv = np.abs(true_psi[ia, tmap[keep]] - true_psi[ib, tmap[keep]])
        # true_psi_skip is one junction's usage; at m=2 |dPSI| IS the TV, and for m>2 it is a lower
        # bound. Scoring on it therefore cannot inflate power -- it can only understate it.
        is_de = true_tv > args.null_tol
        qq, pp = q[keep], prob[keep]

        for a in args.alphas:
            called = qq <= a
            n_c = int(called.sum())
            fdp = float((~is_de[called]).mean()) if n_c else 0.0
            power = float(called[is_de].mean()) if is_de.any() else float("nan")
            rows.append(dict(sim=tag, fit_seed=fit_seed, contrast=f"{pa}_vs_{pb}", alpha=a,
                             n_called=n_c, n_true_de=int(is_de.sum()), n_tested=int(keep.sum()),
                             realised_fdp=fdp, fdp_ratio=fdp / a, power=power))
        per_atse.append(pd.DataFrame(dict(sim=tag, fit_seed=fit_seed,
                                          contrast=f"{pa}_vs_{pb}", atse=fit_ev[keep],
                                          prob=pp, q=qq, true_tv=true_tv, is_de=is_de)))
        m = [r for r in rows if r["contrast"] == f"{pa}_vs_{pb}" and r["alpha"] == 0.05]
        if m:
            print(f"  {pa} vs {pb}: alpha 0.05 -> called {m[0]['n_called']:,}  "
                  f"FDP {m[0]['realised_fdp']:.3f} (ratio {m[0]['fdp_ratio']:.2f})  "
                  f"power {m[0]['power']:.3f}", flush=True)

    if not rows:
        print("no scorable contrasts"); return 1
    R = pd.DataFrame(rows)
    o1 = os.path.join(args.out, f"a62_{tag}_fitseed{fit_seed}.tsv")
    R.to_csv(o1, sep="\t", index=False)
    pd.concat(per_atse).to_csv(o1.replace(".tsv", "_peratse.tsv.gz"), sep="\t", index=False)

    print("\n=== pooled over contrasts: nominal alpha vs realised FDP ===")
    g = R.groupby("alpha").agg(realised_fdp=("realised_fdp", "mean"),
                               fdp_sd=("realised_fdp", "std"),
                               ratio=("fdp_ratio", "mean"),
                               power=("power", "mean"),
                               n_called=("n_called", "mean")).round(4)
    print(g.to_string())
    print(f"\n-> {o1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
