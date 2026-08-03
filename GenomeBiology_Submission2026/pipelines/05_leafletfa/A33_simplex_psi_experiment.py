#!/usr/bin/env python
"""
A33: does constraining psi to a within-ATSE simplex help, before committing to a retrain?

The question
------------
psi is currently drawn as independent Betas per (program, junction), so nothing forces a
program's junction usages within one ATSE to sum to 1 -- even though the data always does,
since the ATSE total IS the sum of its junction counts. Measured on the released fits, only
~86% of program-ATSE pairs land in [0.9, 1.1], uniformly across K=10/20/30 and all seeds.

Two competing expectations:

  FOR the constraint -- the sum is a gauge freedom the likelihood barely penalises, and it is
  where the seeds disagree. Splitting psi into (sum x composition) and matching programs across
  seeds gives composition r = 0.95-0.96 but sum r = 0.51-0.66. Removing that direction should
  make fits more reproducible. It also makes TV mean what the Methods says it means.

  AGAINST -- agreement with LeafCutter is WORSE on ATSEs whose composition is valid (AUROC
  0.587 vs 0.610 at K=30, and the same at K=10 and K=20), and violation tracks real signal:
  LeafCutter's max |dPSI| is a median 0.035 where our composition is valid and 0.113 where it
  is not. So the slack may be what lets the low-rank model represent strong effects at all, and
  removing it could hurt exactly the high-effect events.

This trains both variants on the same cells with the same seeds and scores them together, so
the trade is measured rather than argued.

Implementation
--------------
`SimplexPsi` overrides only the psi block: instead of Beta per junction, an unconstrained
Normal logit per (program, junction) is exponentiated and normalised within its ATSE via a
segment sum. Everything else -- phi, the Dirichlet prior, the Beta-Binomial likelihood, the
guide -- is unchanged, so any difference is attributable to the constraint.

Usage
-----
    python A33_simplex_psi_experiment.py --variant simplex --seed 0 --out <dir>
    python A33_simplex_psi_experiment.py --variant baseline --seed 0 --out <dir>
"""

import argparse
import json
import os
import re
import sys
import time

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import pyro
import pyro.distributions as dist

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from leafletfa.minibatch import LeafletFAMinibatch  # noqa: E402

WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
H5 = f"{WD}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad"
LC = f"{WD}/LEAFCUTTER_DS_2026-08"
safe = lambda s: re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))


class SimplexPsi(LeafletFAMinibatch):
    """psi constrained to a composition within each ATSE, by construction."""

    def set_atse(self, atse_index):
        self.atse_idx = torch.as_tensor(np.asarray(atse_index), dtype=torch.long,
                                        device=self.device)
        self.n_atse = int(self.atse_idx.max().item()) + 1
        return self

    def _segment_softmax(self, logit):
        """Softmax within each ATSE, numerically stable.

        The max must be subtracted PER ATSE, not globally. A global max makes every entry of
        every other ATSE underflow to zero, the segment denominator becomes zero, and the fit
        diverges -- which is what produced the NaN at step 1143 in the first run.
        """
        idx = self.atse_idx.expand(logit.shape[0], -1)
        seg_max = torch.full((logit.shape[0], self.n_atse), -1e30, device=logit.device)
        seg_max = seg_max.scatter_reduce(1, idx, logit, reduce="amax", include_self=True)
        e = torch.exp(logit - seg_max.gather(1, idx))
        denom = torch.zeros(logit.shape[0], self.n_atse, device=logit.device)
        denom = denom.index_add(1, self.atse_idx, e)
        return (e / denom.gather(1, idx).clamp_min(1e-30)).clamp(self.clip, 1 - self.clip)

    def model(self, subsample_size=None):
        subsample_size = self._B if subsample_size is None else subsample_size
        K, J, N = self.K, self.J, self.N
        one = torch.tensor(1.0, device=self.device)

        if self.fixed_psi is None:
            # THE ONLY CHANGE. Unconstrained logits, then a within-ATSE softmax so that
            # sum_{j in e} psi_kj == 1 exactly for every program k and every event e.
            psi_logit = pyro.sample(
                "psi_logit", dist.Normal(torch.zeros(K, J, device=self.device),
                                         torch.ones(K, J, device=self.device)).to_event(2))
            psi = self._segment_softmax(psi_logit)
        else:
            psi = self.fixed_psi

        alpha_pi = self.fixed_alpha_pi if self.fixed_alpha_pi is not None else \
            pyro.sample("alpha_pi", dist.Gamma(one, one))
        pi = self.fixed_pi if self.fixed_pi is not None else \
            pyro.sample("pi", dist.Dirichlet((alpha_pi / K).expand([K])))
        dir_conc = self.fixed_dir_conc if self.fixed_dir_conc is not None else \
            pyro.sample("dir_conc", dist.Gamma(one, one))
        dir_conc = torch.clamp(dir_conc, min=1e-6, max=1e6)

        if self.likelihood == "betabinomial":
            if self.fix_bb_conc is not None:
                bb_conc = torch.tensor(self.fix_bb_conc, device=self.device)
            else:
                bb_conc = pyro.sample("bb_conc", dist.Gamma(
                    torch.tensor(2.0, device=self.device),
                    torch.tensor(0.1, device=self.device)))
                bb_conc = torch.clamp(bb_conc, min=1e-3, max=1e6)

        with pyro.plate("cells", N, subsample_size=subsample_size) as idx:
            phi = pyro.sample("phi", dist.Dirichlet(pi * dir_conc))
            mu = torch.clamp(phi @ psi, self.clip, 1 - self.clip)
            i = idx.to(self.junc.device)
            b_junc = self.junc[i].to(self.device, non_blocking=True)
            b_clus = self.clus[i].to(self.device, non_blocking=True)
            b_mask = self.mask[i].to(self.device, non_blocking=True)
            if self.likelihood == "binomial":
                obs_dist = dist.Binomial(total_count=b_clus, probs=mu)
            else:
                obs_dist = dist.BetaBinomial(concentration1=mu * bb_conc,
                                             concentration0=(1.0 - mu) * bb_conc,
                                             total_count=b_clus)
            pyro.sample("obs", obs_dist.mask(b_mask).to_event(1), obs=b_junc)

    def psi_mean(self):
        """Reconstruct psi on the constrained scale from the guide's logit median."""
        if self.fixed_psi is not None:
            return self.fixed_psi.detach()
        with torch.no_grad():
            return self._segment_softmax(self._median()["psi_logit"].detach())


def cat(name, which):
    with h5py.File(H5, "r") as f:
        n = f[which][name]
        if isinstance(n, h5py.Group):
            c = np.array([x.decode() if isinstance(x, bytes) else x
                          for x in n["categories"][:]])
            return pd.Categorical.from_codes(n["codes"][:], c).astype(str)
        return np.array([x.decode() if isinstance(x, bytes) else x for x in n[:]])


def read_rows(layer, rows, n_cols, chunk=4000):
    order = np.argsort(rows); srt = np.asarray(rows)[order]
    out = np.zeros((len(rows), n_cols), dtype=np.float32)
    with h5py.File(H5, "r") as f:
        g = f["layers"][layer]; ip_all = g["indptr"]
        for a in range(0, len(srt), chunk):
            b = min(a + chunk, len(srt)); blk = srt[a:b]
            lo, hi = int(blk[0]), int(blk[-1]) + 1
            ip = ip_all[lo:hi + 1]; s, e = int(ip[0]), int(ip[-1])
            m = sp.csr_matrix((g["data"][s:e], g["indices"][s:e], ip - ip[0]),
                              shape=(hi - lo, n_cols))
            out[order[a:b]] = m[blk - lo].toarray()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=["baseline", "simplex"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-contrasts", type=int, default=12)
    ap.add_argument("--max-cells", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tct = np.array([safe(x) for x in cat("tissue_cell_type", "obs")])
    ev = cat("event_id", "var")
    uq, inv = np.unique(ev, return_inverse=True)
    J = len(ev)

    man = pd.read_csv(f"{LC}/pairwise_manifest_tabula_muris_senis_tissue_cell_type.tsv", sep="\t")
    man = man[man.status.isin(["ready", "already_done"])]
    man = man.sort_values("n_junctions", ascending=False).head(args.n_contrasts)
    cts = sorted(set(man.group_1) | set(man.group_2))

    rng = np.random.default_rng(0)          # cell selection fixed across variants and seeds
    per = args.max_cells // len(cts)
    cells = np.sort(np.concatenate(
        [rng.choice(np.flatnonzero(tct == c), size=min(per, (tct == c).sum()), replace=False)
         for c in cts]))
    print(f"[1] {len(cts)} cell types, {len(cells):,} cells, {J:,} junctions, "
          f"{len(uq):,} ATSEs", flush=True)

    junc = read_rows("cell_by_junction_matrix", cells, J)
    clus = read_rows("cell_by_cluster_matrix", cells, J)
    print(f"[2] layers loaded ({junc.nbytes/1e9:.1f} GB each)", flush=True)

    pyro.clear_param_store()
    t0 = time.time()
    if args.variant == "simplex":
        m = SimplexPsi(junc, clus, K=args.K, device="cpu").set_atse(inv)
    else:
        m = LeafletFAMinibatch(junc, clus, K=args.K, device="cpu")
    m.fit(num_steps=args.steps, subsample_size=min(args.batch, len(cells)),
          lr=args.lr, gamma=args.gamma, seed=args.seed, verbose=False)
    psi = np.asarray(m.psi_mean().cpu().numpy())
    phi = np.asarray(m.phi_mean().cpu().numpy())
    print(f"[3] fit in {time.time()-t0:.0f}s", flush=True)

    S = np.zeros((args.K, len(uq))); np.add.at(S.T, inv, psi.T)
    frac_valid = float(((S >= 0.9) & (S <= 1.1)).mean())
    print(f"[4] per-program within-ATSE psi sums: median {np.median(S):.4f}, "
          f"valid {100*frac_valid:.1f}%", flush=True)

    out = os.path.join(args.out, f"simplex_exp_{args.variant}_K{args.K}_seed{args.seed}.npz")
    np.savez_compressed(out, psi=psi, phi=phi, cells=cells,
                        cell_types=np.array(cts), frac_valid=frac_valid,
                        variant=args.variant, K=args.K, seed=args.seed)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
