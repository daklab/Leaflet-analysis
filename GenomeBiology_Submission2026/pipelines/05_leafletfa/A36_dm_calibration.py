#!/usr/bin/env python
"""A36: does the Dirichlet-multinomial fix the FDR miscalibration, and at what cost in power?

The question
------------
A35 established the cause. Down-weighting each junction's log-likelihood by (m-1)/m -- a scalar
stand-in for the effective-sample-size correction -- brought realised FDP at effect 0.25 from
0.110 to 0.0525 against a nominal 0.05, but left effect 0.50 at 0.144. The weight is crude: it
applies one number where the correction is per-event, and `02_posterior_width_vs_event_size.py`
shows it barely moves the posterior for large events (width 0.050 -> 0.053 at m=8 where the DM
says 0.154).

The Dirichlet-multinomial applies the same correction exactly, per event. This measures whether
that closes the remaining gap.

The design
----------
Four cells, because the likelihood and the psi parameterisation are separable and we need to
know which one is doing the work:

    betabinomial        + beta      the current released model -- the baseline to beat
    betabinomial        + simplex   isolates the constraint on its own
    dirichletmultinomial + beta     isolates the likelihood on its own. Expected to leave the
                                    within-event scale of psi unidentified (the DM sees only
                                    conc * psi), so it is included to demonstrate the coupling
                                    rather than because it is a candidate.
    dirichletmultinomial + simplex  the intended model

Reported per cell, all three properties from the tracker, so a change that fixes one and breaks
another is visible:
  calibration  realised FDP vs nominal alpha, on simulations with known truth
  power        fraction of truly-differential ATSEs called
  sensible     fraction of program-ATSE pairs whose psi sums into [0.9, 1.1]

A note on the simulation
------------------------
`SIMULATION/sim_heart_K5_*.h5ad` draws each junction as an INDEPENDENT Binomial against the
shared ATSE total, so sum_j y_j equals the stored total only in expectation
(`pipelines/09_simulation/00_simulate_splicing.py:366`). The true psi IS a proper composition
(asserted at line 233 of that file), so recovery targets are well defined, but the emission is
the one the current model assumes rather than the one the DM assumes. `dm_log_prob` derives
N_e = sum_j y_j rather than reading the stored total, so this is handled and not silently wrong
-- but it does mean the DM is being tested slightly out of its own generative family, which is
the conservative direction: a win here is a win despite the sim favouring the incumbent.
`--report-denominator-gap` prints the size of that mismatch.

Usage
-----
    python A36_dm_calibration.py --sim <sim.h5ad> --likelihood dirichletmultinomial \\
        --psi-parameterization simplex --out <dir>
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import pyro

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leafletfa.minibatch import LeafletFAMinibatch                      # noqa: E402
from leafletfa.dirichlet_multinomial import (                           # noqa: E402
    build_atse_index, denominator_consistency, check_A_e_is_conc)
from A28_sim_fdr_calibration import read_h5ad_bits, score_contrast, bayesian_fdr  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--likelihood", default="dirichletmultinomial",
                    choices=["betabinomial", "dirichletmultinomial"])
    ap.add_argument("--psi-parameterization", default="simplex", choices=["beta", "simplex"])
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
    ap.add_argument("--report-denominator-gap", action="store_true")
    ap.add_argument("--device", default="cpu",
                    help="cpu or cuda. Measured on an L40S: 74 ms/step against ~1,240 ms/step on "
                         "4 CPU threads, i.e. a 3000-step fit is 3.7 min instead of 62-118 min. "
                         "The bottleneck is lgamma over [B, J], which is exactly what a GPU is for.")
    ap.add_argument("--psi-logit-prior-sd", type=float, default=1.0,
                    help="prior sd on the softmax logits. LeafCutter's beta has a free scale, so "
                         "its effective prior is weak; Normal(0,1) is not flat.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tag = os.path.basename(args.sim).replace(".h5ad", "")
    cell = f"{args.likelihood}_{args.psi_parameterization}"
    if args.psi_logit_prior_sd != 1.0:
        cell += f"_sd{args.psi_logit_prior_sd:g}"
    junc, clus, prog, ev, true_psi, atse_ids = read_h5ad_bits(args.sim)
    uq_ev, inv_ev = np.unique(ev, return_inverse=True)
    n_atse = len(uq_ev)
    print(f"[1] {tag} | {cell} | seed {args.seed}", flush=True)
    print(f"    {junc.shape[0]:,} cells x {junc.shape[1]:,} junctions, {n_atse:,} ATSEs",
          flush=True)

    atse_idx, rep_idx, sizes = build_atse_index(ev)
    if args.report_denominator_gap:
        mx, frac = denominator_consistency(torch.from_numpy(junc.astype(np.float32)),
                                           torch.from_numpy(clus.astype(np.float32)),
                                           atse_idx, n_atse, rep_idx)
        print(f"[1b] sum_j y_j vs stored ATSE total: max gap {mx:.0f}, "
              f"{100*frac:.1f}% of (cell, ATSE) entries disagree. dm_log_prob derives N_e from "
              f"the counts, so this does not affect the fit.", flush=True)

    pyro.clear_param_store()
    t0 = time.time()
    mod = LeafletFAMinibatch(junc, clus, K=args.K, device=args.device,
                             likelihood=args.likelihood,
                             psi_parameterization=args.psi_parameterization,
                             psi_logit_prior_sd=args.psi_logit_prior_sd,
                             event_id=ev)
    mod.fit(num_steps=args.steps, subsample_size=min(args.batch, junc.shape[0]),
            lr=args.lr, gamma=args.gamma, seed=args.seed, verbose=False)
    _pm = mod.phi_mean()
    phi = np.asarray(_pm.cpu().numpy() if hasattr(_pm, "cpu") else _pm)
    psi_point = mod.psi_mean()
    psi_point = np.asarray(psi_point.cpu().numpy() if hasattr(psi_point, "cpu") else psi_point)
    print(f"[2] fit in {time.time()-t0:.0f}s", flush=True)

    # PROPERTY 3 -- sensible: does a program's psi sum to 1 within an ATSE?
    S_sum = np.zeros((args.K, n_atse))
    np.add.at(S_sum.T, inv_ev, psi_point.T)
    frac_valid = float(((S_sum >= 0.9) & (S_sum <= 1.1)).mean())
    print(f"[3] within-ATSE psi sums: median {np.median(S_sum):.4f}, "
          f"valid[0.9,1.1] {100*frac_valid:.1f}%", flush=True)

    # A_e diagnostic: under the simplex this should be ~conc for every event
    mu_pt = torch.from_numpy((phi @ psi_point).astype(np.float32))
    a_max, a_med = check_A_e_is_conc(mu_pt, torch.tensor(1.0), atse_idx, n_atse)
    print(f"[3b] A_e/conc deviation: max {a_max:.4f}, median {a_med:.4f} "
          f"(~0 means the simplex is active)", flush=True)

    # psi posterior draws, on the constrained scale for whichever parameterisation is in use
    psi_s = mod.sample_psi(args.num_samples, seed=0).cpu()
    print(f"[4] drew {args.num_samples} psi samples {tuple(psi_s.shape)}", flush=True)

    order = {a: i for i, a in enumerate(atse_ids)}
    truth_idx = np.array([order[a] for a in uq_ev])
    rows = []
    progs = sorted(set(prog))
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
            tv, prob = score_contrast(psi_s, phi, c1, c2, inv_ev, n_atse, args.delta)
            for a in args.alphas:
                sig, _, ns = bayesian_fdr(prob, alpha=a)
                nf = int((sig & is_null).sum())
                rows.append(dict(sim=tag, likelihood=args.likelihood,
                                 psi_parameterization=args.psi_parameterization,
                                 cell=cell, seed=args.seed,
                                 steps=args.steps, num_samples=args.num_samples,
                                 batch=args.batch,
                                 contrast=f"{p}_vs_{q}", alpha=a, n_called=int(ns), n_false=nf,
                                 realised_fdp=(nf / ns) if ns else np.nan,
                                 power=float((sig & ~is_null).sum() / max((~is_null).sum(), 1)),
                                 frac_valid=frac_valid,
                                 n_saturated=int((prob >= 0.999).sum())))

    # SAVE THE FIT. Without this, cross-seed agreement cannot be computed after the fact --
    # the first 24 runs discarded psi/phi and are unusable for the seed question, which is the
    # property the simplex is meant to fix. loc/scale rather than point estimates so the posterior
    # width is recoverable too.
    # bb_conc is the overdispersion the model learned. On DM-generated sims we know the truth
    # (--dm-conc), so recovering it is a direct check the likelihood is doing what it claims.
    # It is a guide site, not returned by any accessor, so read it out of the param store.
    try:
        _med = mod._median()
        conc_hat = float(_med["bb_conc"]) if "bb_conc" in _med else float("nan")
    except Exception:
        conc_hat = float("nan")
    print(f"[4b] learned bb_conc = {conc_hat:.3f}", flush=True)

    loc, scale = mod.psi_loc_scale()
    fit_out = os.path.join(args.out, f"fit_{tag}_{cell}_seed{args.seed}.npz")
    np.savez_compressed(
        fit_out,
        psi_mean=psi_point.astype(np.float32),
        psi_loc=(np.asarray(loc.cpu().numpy() if hasattr(loc, "cpu") else loc, dtype=np.float32)
                 if loc is not None else np.zeros(0, dtype=np.float32)),
        psi_scale=(np.asarray(scale.cpu().numpy() if hasattr(scale, "cpu") else scale,
                              dtype=np.float32) if scale is not None else np.zeros(0, np.float32)),
        phi=phi.astype(np.float32),
        event_id=np.asarray(uq_ev, dtype=object), inv_ev=inv_ev.astype(np.int32),
        likelihood=args.likelihood, psi_parameterization=args.psi_parameterization,
        seed=args.seed, steps=args.steps, frac_valid=frac_valid, bb_conc=conc_hat,
        a_e_dev_max=a_max, a_e_dev_median=a_med)
    print(f"[5] fit saved -> {os.path.basename(fit_out)}", flush=True)

    R = pd.DataFrame(rows)
    out = os.path.join(args.out, f"a36_{tag}_{cell}_seed{args.seed}.tsv")
    R.to_csv(out, sep="\t", index=False)
    print("\n=== nominal alpha vs realised FDP ===")
    print(R.groupby("alpha")[["realised_fdp", "power", "n_called", "n_saturated"]]
          .mean().round(4).to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
