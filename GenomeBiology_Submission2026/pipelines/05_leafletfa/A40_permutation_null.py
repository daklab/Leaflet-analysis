#!/usr/bin/env python
"""A40: calibrate TV against a null built by resampling DATA, not by drawing from q(psi).

Run (gate only, one refit):
    python A40_permutation_null.py --sim <sim.h5ad> --out <dir> --rounds 1 --gate-only
Run (full):
    python A40_permutation_null.py --sim <sim.h5ad> --out <dir> --rounds 5

Why
---
A38 located the miscalibration: TV = 0.5 * sum_j |Delta_j| is a sum of ABSOLUTE values, so psi
estimation noise cannot cancel and a truly null ATSE scores TV 0.017-0.025 instead of 0. That is
compared against a FIXED delta = 0.10, while at effect 0.25 the median TRUE positive scores only
0.096 -- below the threshold a tail of nulls clears. Low power and high FDP are the same fact,
and no likelihood change reaches it.

A39 tried the obvious fix -- a null drawn from q(psi) -- and it was anti-conservative (FDP 0.579
against nominal 0.05), because the null inherited the posterior's own over-confidence and came
out 1.9x too tight. So the null must come from the data.

The design
----------
Permuting EVERY ATSE destroys all signal and collapses dphi = phibar_A - phibar_B, which is the
trap that made A39 fail: null TV scales with |dphi|, so a null computed at dphi ~ 0 is far too
small and the test is anti-conservative for exactly the same reason.

Instead permute a random FRACTION and leave the rest intact:

  * pick `--frac-permuted` of ATSEs at random (default 10%, ~427 of 4,268)
  * permute their junction count VECTORS across cells. The vector and its total move TOGETHER --
    both `junc[:, cols]` and `clus[:, cols]` get the same row permutation -- so
    `sum_j y_j == N_e` stays exact and the DM likelihood stays defined.
  * by default permute WITHIN coverage strata (deciles of N_e). Free permutation would also
    shuffle sequencing depth across cells, and if depth correlates with program the null would
    sit at a different coverage than the observed data. `--no-stratify` disables this.
  * the other 90% still pin phi, so dphi survives at the correct magnitude
  * refit, and score every contrast. The permuted ATSEs are nulls BY CONSTRUCTION -- no truth
    labels are used to build them, so this is a procedure that could actually ship.

THE GATE
--------
This design is untested and A39 looked equally sound. On simulations the true nulls are known, so
it can be checked after ONE refit before spending the rest:

  1. |dphi|_1 on the permuted refit within `--dphi-tol` of the unpermuted fit.
     This is the specific property A39's null failed.
  2. the permuted-ATSE TV distribution should MATCH the true-null TV distribution
     (median ratio within `--tv-tol`, plus a KS statistic for the record).

If the permuted nulls are systematically TIGHTER -> dphi has collapsed -> the null will be
anti-conservative like A39. Adjustment: LOWER `--frac-permuted` so more ATSEs pin phi.
If they are LOOSER -> the permutation is destroying more than the psi-program association
(most likely coverage) -> adjustment: keep `--stratify` on, or raise `--n-strata`.

Both adjustments are flags, not a rewrite. The gate prints which way it failed and which flag to
move, so a failed run tells you what to do next.
"""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import pyro

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leafletfa.minibatch import LeafletFAMinibatch                      # noqa: E402
from A28_sim_fdr_calibration import read_h5ad_bits, score_contrast, bayesian_fdr  # noqa: E402


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def permute_atses(junc, clus, inv_ev, which, rng, stratify=True, n_strata=10):
    """Permute the count VECTORS of the chosen ATSEs across cells.

    The vector and its ATSE total move together, so `sum_j y_j == N_e` remains exact. With
    `stratify`, cells are permuted only within deciles of that ATSE's coverage, so sequencing
    depth is not shuffled across cells -- only WHICH composition a cell received.
    """
    junc, clus = junc.copy(), clus.copy()
    N = junc.shape[0]
    for e in which:
        cols = np.flatnonzero(inv_ev == e)
        N_e = clus[:, cols[0]]
        if stratify:
            # rank cells by coverage, cut into strata, permute inside each
            order = np.argsort(N_e, kind="stable")
            perm = np.arange(N)
            for chunk in np.array_split(order, n_strata):
                perm[chunk] = rng.permutation(chunk)
        else:
            perm = rng.permutation(N)
        junc[:, cols] = junc[perm][:, cols]
        clus[:, cols] = clus[perm][:, cols]
    return junc, clus


def fit_and_score(junc, clus, ev, K, args, seed, label):
    """One fit, then TV + prob_de for every eligible contrast. Returns (fit, per-contrast dict)."""
    uq_ev, inv_ev = np.unique(ev, return_inverse=True)
    n_atse = len(uq_ev)
    pyro.clear_param_store()
    t0 = time.time()
    mod = LeafletFAMinibatch(junc, clus, K=K, device=getattr(args, "device", "cpu"),
                             likelihood=args.likelihood,
                             psi_parameterization=args.psi_parameterization,
                             event_id=ev)
    mod.fit(num_steps=args.steps, subsample_size=min(args.batch, junc.shape[0]),
            lr=args.lr, gamma=args.gamma, seed=seed, verbose=False)
    _pm = mod.phi_mean()
    phi = np.asarray(_pm.cpu().numpy() if hasattr(_pm, "cpu") else _pm)
    psi_s = mod.sample_psi(args.num_samples, seed=0).cpu()
    print(f"    [{label}] fit in {time.time()-t0:.0f}s", flush=True)
    return phi, psi_s, inv_ev, n_atse


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--likelihood", default="dirichletmultinomial")
    ap.add_argument("--psi-parameterization", default="simplex")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--num-samples", type=int, default=500)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--null-tol", type=float, default=0.05)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu",
                    help="cuda makes each round minutes rather than ~23 min, which is what makes "
                         "5 permuted rounds plus a reference fit affordable at all.")
    # --- the knobs the gate tells you to move ---
    ap.add_argument("--frac-permuted", type=float, default=0.10,
                    help="fraction of ATSEs permuted per round. LOWER this if the gate says the "
                         "permuted nulls are too tight (dphi collapsing).")
    ap.add_argument("--stratify", dest="stratify", action="store_true", default=True,
                    help="permute within coverage strata (default on)")
    ap.add_argument("--no-stratify", dest="stratify", action="store_false")
    ap.add_argument("--n-strata", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--pool-contrasts", dest="pool_contrasts", action="store_true", default=True,
                    help="pool the null across contrasts after normalising by |dphi|_1. Needed "
                         "for p-value resolution: per-contrast draws floor the top-ranked events "
                         "at q=1.0 under BH.")
    ap.add_argument("--no-pool-contrasts", dest="pool_contrasts", action="store_false")
    ap.add_argument("--gate-only", action="store_true",
                    help="run one permuted refit, report the gate, and stop")
    ap.add_argument("--dphi-tol", type=float, default=0.35,
                    help="max relative change in |dphi|_1 allowed by the gate. Calibrated against "
                         "the model's OWN seed-to-seed variability, not picked: three unpermuted "
                         "fits of the same data differing only by seed give |dphi|_1 relative "
                         "changes of median 0.128 and max 0.346. A tighter tolerance measures "
                         "optimiser noise rather than permutation damage -- the first run used "
                         "0.15 and failed a permutation that had in fact moved dphi LESS "
                         "(median 0.076, max 0.197) than two unpermuted refits differ from "
                         "each other.")
    ap.add_argument("--tv-tol", type=float, default=0.35,
                    help="max relative gap between permuted-null and true-null median TV")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tag = os.path.basename(args.sim).replace(".h5ad", "")
    junc, clus, prog, ev, true_psi, atse_ids = read_h5ad_bits(args.sim)
    uq_ev, inv_ev = np.unique(ev, return_inverse=True)
    n_atse = len(uq_ev)
    order = {a: i for i, a in enumerate(atse_ids)}
    truth_idx = np.array([order[a] for a in uq_ev])
    progs = sorted(set(prog))
    print(f"[1] {tag} | {args.likelihood} + {args.psi_parameterization}")
    print(f"    {junc.shape[0]:,} cells x {junc.shape[1]:,} junctions, {n_atse:,} ATSEs",
          flush=True)

    # ── reference fit on the untouched data ────────────────────────────────────────────
    print(f"[2] reference fit (no permutation)", flush=True)
    phi0, psi0_s, _, _ = fit_and_score(junc, clus, ev, args.K, args, args.seed, "reference")

    pairs = []
    for i in range(len(progs)):
        for j in range(i + 1, len(progs)):
            p, q = progs[i], progs[j]
            c1 = np.flatnonzero(prog == p)
            c2 = np.flatnonzero(prog == q)
            if min(len(c1), len(c2)) < 50:
                continue
            td = np.abs(true_psi[int(p)] - true_psi[int(q)])[truth_idx]
            is_null = td <= args.null_tol
            if is_null.sum() < 50 or (~is_null).sum() < 50:
                continue
            pairs.append((f"{p}_vs_{q}", c1, c2, is_null))
    print(f"    {len(pairs)} eligible contrasts", flush=True)

    obs_tv, obs_prob, dphi_ref = {}, {}, {}
    for name, c1, c2, _ in pairs:
        tv, prob = score_contrast(psi0_s, phi0, c1, c2, inv_ev, n_atse, args.delta,
                                  label=f"ref {name}")
        obs_tv[name], obs_prob[name] = tv, prob
        dphi_ref[name] = float(np.abs(phi0[c1].mean(0) - phi0[c2].mean(0)).sum())

    # ── permuted rounds ────────────────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    n_perm = max(1, int(round(args.frac_permuted * n_atse)))
    null_tv = {name: [] for name, _, _, _ in pairs}
    gate = {}
    rounds = 1 if args.gate_only else args.rounds
    for r in range(rounds):
        which = rng.choice(n_atse, size=n_perm, replace=False)
        pj, pc = permute_atses(junc, clus, inv_ev, which, rng,
                               stratify=args.stratify, n_strata=args.n_strata)
        print(f"[3.{r}] round {r}: permuted {n_perm:,} ATSEs "
              f"({100*args.frac_permuted:.0f}%), stratify={args.stratify}", flush=True)
        # the identity must survive the permutation, or the DM is undefined
        seg = np.zeros((pj.shape[0], n_atse), np.float32)
        np.add.at(seg.T, inv_ev, pj.T)
        rep = np.array([np.flatnonzero(inv_ev == e)[0] for e in range(n_atse)])
        gap = float(np.abs(seg - pc[:, rep]).max())
        if gap != 0:
            raise AssertionError(f"permutation broke sum_j y_j == N_e (max gap {gap})")
        phi_p, psi_p, _, _ = fit_and_score(pj, pc, ev, args.K, args, args.seed, f"perm {r}")
        for name, c1, c2, _ in pairs:
            tv, _ = score_contrast(psi_p, phi_p, c1, c2, inv_ev, n_atse, args.delta,
                                   check=False)
            null_tv[name].append(tv[which])
        if r == 0:
            dphi_p = {n: float(np.abs(phi_p[c1].mean(0) - phi_p[c2].mean(0)).sum())
                      for n, c1, c2, _ in pairs}
            rel = np.array([abs(dphi_p[n] - dphi_ref[n]) / max(dphi_ref[n], 1e-9)
                            for n, _, _, _ in pairs])
            tn = np.concatenate([obs_tv[n][isn] for n, _, _, isn in pairs])
            pn = np.concatenate([null_tv[n][0] for n, _, _, _ in pairs])
            med_ratio = float(np.median(pn) / max(np.median(tn), 1e-12))
            try:
                from scipy.stats import ks_2samp
                ks = float(ks_2samp(pn, tn).statistic)
            except Exception:
                ks = float("nan")
            gate = dict(dphi_rel_max=float(rel.max()), dphi_rel_median=float(np.median(rel)),
                        true_null_median=float(np.median(tn)),
                        perm_null_median=float(np.median(pn)),
                        median_ratio=med_ratio, ks=ks,
                        true_null_p90=float(np.percentile(tn, 90)),
                        perm_null_p90=float(np.percentile(pn, 90)))
            print("\n" + "=" * 78)
            print("GATE  (design is untested -- this decides whether to spend the rest)")
            print("=" * 78)
            print(f"  1. |dphi|_1 preserved?   max rel change {rel.max():.4f}, "
                  f"median {np.median(rel):.4f}   (tol {args.dphi_tol})")
            print(f"  2. null TV matches?      true-null median {np.median(tn):.5f}, "
                  f"permuted median {np.median(pn):.5f}")
            print(f"                           ratio {med_ratio:.3f} (want ~1.0, "
                  f"tol {1-args.tv_tol:.2f}-{1+args.tv_tol:.2f}), KS {ks:.4f}")
            print(f"                           p90 {np.percentile(tn,90):.5f} vs "
                  f"{np.percentile(pn,90):.5f}")
            ok_dphi = rel.max() <= args.dphi_tol
            ok_tv = abs(med_ratio - 1.0) <= args.tv_tol
            gate["pass"] = bool(ok_dphi and ok_tv)
            if not ok_dphi:
                print(f"\n  FAIL on |dphi|: the permutation moved phi too much. "
                      f"LOWER --frac-permuted (now {args.frac_permuted}).")
            if not ok_tv and med_ratio < 1:
                print(f"\n  FAIL: permuted nulls TOO TIGHT (ratio {med_ratio:.3f}). dphi is "
                      f"collapsing -> anti-conservative, the A39 failure. LOWER --frac-permuted.")
            if not ok_tv and med_ratio > 1:
                print(f"\n  FAIL: permuted nulls TOO LOOSE (ratio {med_ratio:.3f}). The "
                      f"permutation is destroying more than the psi-program association -> "
                      f"over-conservative. Keep --stratify on and/or raise --n-strata "
                      f"(now {args.n_strata}).")
            if gate["pass"]:
                print("\n  PASS -- the permuted ATSEs behave like real nulls at the right dphi.")
            print("=" * 78 + "\n", flush=True)
            with open(os.path.join(args.out, f"gate_{tag}.json"), "w") as fh:
                json.dump(gate, fh, indent=2)
            if args.gate_only or not gate["pass"]:
                print("stopping after the gate." if args.gate_only else
                      "GATE FAILED -- stopping rather than spending the remaining rounds.")
                return

    # ── empirical p -> BH -> realised FDP ──────────────────────────────────────────────
    # RESOLUTION. The smallest empirical p a null of n draws can produce is 1/(n+1), and BH needs
    # p <= alpha/n_atse for the TOP-ranked event. With 5 rounds x 20% that is 4,265 draws per
    # contrast, min p 2.3e-4, against a required 1.2e-5 -- so the highest-ranked ATSEs would floor
    # at q = 1.0 and the run would return zeros that look like a bug rather than a result.
    #
    # Pooling the null across contrasts multiplies the draws by the number of contrasts. But null
    # TV scales with |dphi|_1, which differs per contrast, so raw pooling mixes scales. Normalise
    # each contrast's null by its own |dphi|_1, pool, then rescale to the target contrast.
    if args.pool_contrasts and len(pairs) > 1:
        unit = np.concatenate([np.concatenate(null_tv[n]) / max(dphi_ref[n], 1e-12)
                               for n, _, _, _ in pairs])
        unit.sort()
        print(f"[4] pooled null: {len(unit):,} draws across {len(pairs)} contrasts "
              f"(min achievable p = {1/(len(unit)+1):.2e}); "
              f"per-contrast would have been {len(np.concatenate(null_tv[pairs[0][0]])):,}",
              flush=True)
    else:
        unit = None

    rows = []
    for name, c1, c2, is_null in pairs:
        if unit is not None:
            pool = unit * dphi_ref[name]          # back to this contrast's scale
        else:
            pool = np.sort(np.concatenate(null_tv[name]))
        # searchsorted rather than a [n_atse x n_draws] broadcast, which would be 182M booleans
        idx = np.searchsorted(pool, obs_tv[name], side="left")
        pv = (1.0 + (len(pool) - idx)) / (len(pool) + 1.0)
        q = bh(pv)
        for a in args.alphas:
            sel_emp = q <= a
            sig_fix, _, _ = bayesian_fdr(obs_prob[name], alpha=a)
            for meth, sel in (("empirical_null", sel_emp), ("fixed_delta", sig_fix)):
                n = int(sel.sum())
                rows.append(dict(sim=tag, method=meth, contrast=name, alpha=a, n_called=n,
                                 n_false=int((sel & is_null).sum()),
                                 realised_fdp=(int((sel & is_null).sum()) / n) if n else np.nan,
                                 power=float((sel & ~is_null).sum() / max((~is_null).sum(), 1)),
                                 n_null_draws=len(pool), pooled=bool(unit is not None)))
    R = pd.DataFrame(rows)
    out = os.path.join(args.out, f"a40_{tag}.tsv")
    R.to_csv(out, sep="\t", index=False)
    print("\n=== realised FDP: fixed delta vs empirical null ===")
    print(R.pivot_table(index="alpha", columns="method",
                        values=["realised_fdp", "power"]).round(4).to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
