#!/usr/bin/env python
"""Is `spearman_delta` penalising NMF for a gauge choice rather than for accuracy?

NMF is scale-indeterminate: W H = (W S)(S^-1 H) for any positive diagonal S, so the split of
magnitude between loadings and components is arbitrary. The two metrics react differently:

  psi_r           per-program CORRELATION            scale-invariant
  spearman_delta  ranks d = max_k psi - min_k psi    scale-DEPENDENT

`fit_nmf` returns `mm.components_` untouched, so if sklearn happens to put unequal scale on the
components, d is dominated by that scale rather than by how usage differs between programs. On the
DM sims NMF scored psi_r 0.913 and spearman_delta -0.036 at effect 0.1 -- recovering the programs
well while apparently ranking the effect at chance, which is not a coherent pair of numbers.

This refits NMF once and recomputes the metric under three gauges. If NMF's score moves materially,
the published comparison is unfair and has to be fixed; if it does not, the near-zero is real and
NMF genuinely cannot rank per-ATSE effects.
"""

import os, sys

import anndata as ad
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim_benchmark as SB          # noqa: E402

SIM = ("/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION/"
       "sim_heart_K5_dm50_eff0.5_frac0.3_seed0.h5ad")


def delta_spearman(T, psi_hat):
    """Exactly the metric sim_benchmark.metrics() uses, isolated."""
    ph = psi_hat[:, T["skip_idx"]][:, np.argsort(T["order"])]
    d = ph.max(0) - ph.min(0)
    ok = np.isfinite(d) & np.isfinite(T["true_delta"]) & T["is_diff"]
    return float(spearmanr(T["true_delta"][ok], d[ok]).correlation), ph


def main():
    sim = ad.read_h5ad(SIM)
    T = SB.truth_of(sim)
    fits = SB.fit_nmf(sim, T["K_true"])

    print(f"{'gauge':<34}{'0-impute':>12}{'mean-impute':>14}")
    res = {}
    for tag, (H, W, *_rest) in fits.items():
        col_scale = W.mean(0)                      # per-component magnitude carried by loadings
        gauges = {
            "raw components_ (as published)": H,
            "scale absorbed from W (W mean=1)": H * col_scale[:, None],
            "rows L1-normalised": H / np.maximum(H.sum(1, keepdims=True), 1e-12),
        }
        for g, M in gauges.items():
            r, _ = delta_spearman(T, np.asarray(M, float))
            res.setdefault(g, {})[tag] = r
    for g, v in res.items():
        print(f"{g:<34}{v.get('0-impute', float('nan')):>12.3f}"
              f"{v.get('mean-impute', float('nan')):>14.3f}")

    # For reference: the model on the same sim and the same metric.
    f = ("/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/"
         "DM_FRESH/fit_sim_heart_K5_dm50_eff0.5_frac0.3_seed0_"
         "dirichletmultinomial_simplex_seed0.npz")
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True)
        r, _ = delta_spearman(T, np.asarray(z["psi_mean"], float))
        print(f"\n{'LeafletFA (DM+simplex), same metric':<34}{r:>12.3f}")
        print("LeafletFA needs no gauge choice: psi sums to 1 within each ATSE by construction,")
        print("so max-min across programs is already on the usage scale.")


if __name__ == "__main__":
    main()
