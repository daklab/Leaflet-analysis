#!/usr/bin/env python
"""
A35: fix the FDR miscalibration by correcting the effective sample size.

The problem
-----------
Measured on simulations with known truth, the posterior expected FDP is 3-4x anti-conservative:
a nominal 0.05 delivers a realised false discovery proportion of 0.156 at effect 0.25 and 0.199
at effect 0.50. Every combination of effect size and threshold tested came out over-called.

The cause
---------
An ATSE's junction counts are ONE observation split m ways -- they share a denominator and sum
to it exactly. The likelihood scores them as m independent Beta-Binomials, each with the full
ATSE total as its trial count. That counts the same reads m times, inflates the information the
event carries, and narrows the psi posterior. Since prob_de = P(TV > delta) is read off that
posterior, an over-narrow posterior pushes prob_de toward 0 and 1, which makes 1 - prob_de an
underestimate of the false-discovery probability and the Bayesian FDP too small.

The clearest case is m=2: junction 2's count is completely determined by junction 1's, so
scoring both doubles the evidence for what is really a single binomial observation.

The fix tested here
-------------------
Down-weight each junction's likelihood contribution so the total information per ATSE matches
its free dimensions. A multinomial over m categories has m-1 free parameters, so weight each of
the m junctions by (m-1)/m. That gives exactly 1/2 per junction at m=2 (total 1, one
observation) and approaches 1 for large m, where the redundancy is proportionally smaller.

This is an approximation, not a derivation -- the exact correction is the Dirichlet-multinomial
likelihood, which models the joint directly. But it is three lines, needs no data reshaping, and
if it substantially closes the calibration gap it tells us the effective-n diagnosis is right
before anyone commits to the larger change.

Run with --scale-mode none to reproduce the current behaviour on identical data.

Usage
-----
    python A35_effective_n_calibration.py --sim <sim.h5ad> --scale-mode mminus1 --out <dir>
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leafletfa.minibatch import LeafletFAMinibatch          # noqa: E402
from leafletfa.differential_splicing_groups import bayesian_fdr  # noqa: E402
from A28_sim_fdr_calibration import read_h5ad_bits, score_contrast  # noqa: E402


class ScaledLikelihood(LeafletFAMinibatch):
    """Identical to the parent except each junction's likelihood is down-weighted."""

    def set_weights(self, w):
        self.jw = torch.as_tensor(np.asarray(w, dtype=np.float32), device=self.device)
        return self

    def model(self, subsample_size=None):
        subsample_size = self._B if subsample_size is None else subsample_size
        K, J, N = self.K, self.J, self.N
        one = torch.tensor(1.0, device=self.device)

        if self.fixed_psi is None:
            a = pyro.sample("a", dist.Gamma(one, one).expand([J]).to_event(1))
            b = pyro.sample("b", dist.Gamma(one, one).expand([J]).to_event(1))
            psi = pyro.sample("psi", dist.Beta(a + self.eps, b + self.eps)
                              .expand([K, J]).to_event(2))
        else:
            psi = self.fixed_psi

        alpha_pi = pyro.sample("alpha_pi", dist.Gamma(one, one))
        pi = pyro.sample("pi", dist.Dirichlet((alpha_pi / K).expand([K])))
        dir_conc = torch.clamp(pyro.sample("dir_conc", dist.Gamma(one, one)), 1e-6, 1e6)
        bb_conc = torch.clamp(
            pyro.sample("bb_conc", dist.Gamma(torch.tensor(2.0, device=self.device),
                                              torch.tensor(0.1, device=self.device))),
            1e-3, 1e6)

        with pyro.plate("cells", N, subsample_size=subsample_size) as idx:
            phi = pyro.sample("phi", dist.Dirichlet(pi * dir_conc))
            mu = torch.clamp(phi @ psi, self.clip, 1 - self.clip)
            i = idx.to(self.junc.device)
            b_junc = self.junc[i].to(self.device, non_blocking=True)
            b_clus = self.clus[i].to(self.device, non_blocking=True)
            b_mask = self.mask[i].to(self.device, non_blocking=True)
            obs_dist = dist.BetaBinomial(concentration1=mu * bb_conc,
                                         concentration0=(1.0 - mu) * bb_conc,
                                         total_count=b_clus)
            # THE ONLY CHANGE: a per-junction weight on the observation term, so an ATSE
            # contributes information proportional to its free dimensions rather than to its
            # junction count.
            #
            # This cannot be done with poutine.scale on the sample site: .to_event(1) sums the
            # log-prob over junctions FIRST, leaving a [B] tensor, so a [J] weight has nothing
            # to multiply. Compute the per-junction log-prob, weight it, mask it, and sum --
            # then add it as a factor. Identical to the sample statement when jw is all ones.
            lp = obs_dist.log_prob(b_junc)                       # [B, J]
            lp = torch.where(b_mask.bool(), lp, torch.zeros_like(lp))
            pyro.factor("obs", (lp * self.jw).sum(-1))           # [B], plate scales it


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--scale-mode", default="mminus1", choices=["none", "mminus1", "inv_m"])
    ap.add_argument("--out", required=True)
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
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tag = os.path.basename(args.sim).replace(".h5ad", "")
    junc, clus, prog, ev, true_psi, atse_ids = read_h5ad_bits(args.sim)
    uq_ev, inv_ev = np.unique(ev, return_inverse=True)
    n_atse = len(uq_ev)
    m = np.bincount(inv_ev)[inv_ev]                       # junctions in each junction's ATSE
    if args.scale_mode == "none":
        w = np.ones_like(m, dtype=np.float32)
    elif args.scale_mode == "mminus1":
        w = ((m - 1) / m).astype(np.float32)
    else:
        w = (1.0 / m).astype(np.float32)
    print(f"[1] {tag}: {junc.shape[0]:,} cells x {junc.shape[1]:,} junctions, "
          f"{n_atse:,} ATSEs", flush=True)
    print(f"[2] scale-mode={args.scale_mode}: weight median {np.median(w):.3f}, "
          f"range {w.min():.3f}-{w.max():.3f}", flush=True)

    pyro.clear_param_store()
    t0 = time.time()
    mod = ScaledLikelihood(junc, clus, K=args.K, device="cpu").set_weights(w)
    mod.fit(num_steps=args.steps, subsample_size=min(args.batch, junc.shape[0]),
            lr=args.lr, gamma=args.gamma, seed=args.seed, verbose=False)
    _pm = mod.phi_mean()
    phi = np.asarray(_pm.cpu().numpy() if hasattr(_pm, "cpu") else _pm)
    loc, scale = [np.asarray(x.cpu().numpy() if hasattr(x, "cpu") else x)
                  for x in mod.psi_loc_scale()]
    print(f"[3] fit in {time.time()-t0:.0f}s; median psi posterior sd (logit) "
          f"{np.median(scale):.4f}", flush=True)

    rng = np.random.default_rng(0)
    eps = rng.standard_normal((args.num_samples,) + loc.shape).astype(np.float32)
    psi_s = torch.from_numpy((1.0 / (1.0 + np.exp(-(loc[None] + scale[None] * eps))))
                             .astype(np.float32))

    order = {a: i for i, a in enumerate(atse_ids)}
    truth_idx = np.array([order[a] for a in uq_ev])
    rows = []
    progs = sorted(set(prog))
    for i in range(len(progs)):
        for j in range(i + 1, len(progs)):
            p, q = progs[i], progs[j]
            c1 = np.flatnonzero(prog == p); c2 = np.flatnonzero(prog == q)
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
                rows.append(dict(sim=tag, scale_mode=args.scale_mode,
                                 contrast=f"{p}_vs_{q}", alpha=a, n_called=int(ns),
                                 n_false=nf,
                                 realised_fdp=(nf / ns) if ns else np.nan,
                                 power=float((sig & ~is_null).sum() / max((~is_null).sum(), 1)),
                                 median_psi_sd=float(np.median(scale))))

    R = pd.DataFrame(rows)
    out = os.path.join(args.out, f"effn_{tag}_{args.scale_mode}.tsv")
    R.to_csv(out, sep="\t", index=False)
    print("\n=== nominal alpha vs realised FDP ===")
    print(R.groupby("alpha")[["realised_fdp", "power", "n_called"]].mean().round(4).to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
