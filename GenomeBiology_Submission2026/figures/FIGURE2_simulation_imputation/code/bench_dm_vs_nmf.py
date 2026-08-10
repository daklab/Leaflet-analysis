#!/usr/bin/env python
"""LeafletFA (DM + simplex) against NMF, on the DIRICHLET-MULTINOMIAL simulations.

Why this exists
---------------
The published Figure 2 benchmark ran on Binomial-generated simulations and compared the
BetaBinomial and Binomial models -- neither of which is the shipped model any more. Every LeafletFA
row in it validates something the paper no longer proposes, and the NMF baseline was fitted on data
generated the competitor's way (junctions drawn independently against a shared ATSE total, so the
compositional identity holds only in expectation).

This puts both methods on the DM-generated simulations, where sum_j y_j == N_e exactly, as it does
on the real atlas.

NMF is the right baseline at matched rank K. `mb_ct` is not: it uses the cell-type labels, so it is
an unfair bar in LeafletFA's favour.

Metrics come from sim_benchmark.metrics() unchanged, so numbers stay comparable to the previous
benchmark. The headline is `spearman_delta` -- how well the recovered per-ATSE effect size ranks
against the truth.
"""

import argparse, glob, os, re, sys, time

import anndata as ad
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim_benchmark as SB          # noqa: E402  -- reuse metrics() and fit_nmf() unchanged

SIM_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION"
DM_FRESH = ("/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/"
            "MOUSE_SPLICING_FOUNDATION/DM_FRESH")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--effects", nargs="+", type=float, default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--conc", default="50")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "dm_vs_nmf.tsv"))
    args = ap.parse_args()

    rows = []
    for eff in args.effects:
        h5 = f"{SIM_DIR}/sim_heart_K5_dm{args.conc}_eff{eff}_frac0.3_seed0.h5ad"
        if not os.path.exists(h5):
            print(f"  MISSING {os.path.basename(h5)}", flush=True)
            continue
        sim = ad.read_h5ad(h5)
        T = SB.truth_of(sim)

        # --- NMF, once per sim: it has no model seed, so seeds would be identical work ---------
        #
        # NMF is scale-indeterminate (W H = (W S)(S^-1 H)), and `spearman_delta` ranks
        # max_k psi - min_k psi, which is NOT scale-invariant -- so the gauge sklearn happens to
        # return can change the score. Measured on eff 0.5 (check_nmf_scale_fairness.py):
        #
        #   gauge                       0-impute   mean-impute
        #   raw components_ (published)    0.023         0.026
        #   scale absorbed from W          0.023        -0.038
        #   rows L1-normalised             0.041         0.103
        #
        # The near-zero is therefore REAL and not an artefact -- but the published benchmark used
        # the worst gauge for NMF. Scoring at its BEST gauge removes the objection entirely and
        # still leaves LeafletFA ~7x ahead, so the baseline is given every advantage.
        t0 = time.time()
        for tag, (H, W, secs, n_it, conv) in SB.fit_nmf(sim, args.K).items():
            cand = {"raw": np.asarray(H, float),
                    "l1_rows": np.asarray(H, float) / np.maximum(
                        np.asarray(H, float).sum(1, keepdims=True), 1e-12),
                    "w_absorbed": np.asarray(H, float) * W.mean(0)[:, None]}
            scored = {g: SB.metrics(T, M, W) for g, M in cand.items()}
            best_g = max(scored, key=lambda g: (scored[g]["spearman_delta"]
                                                if np.isfinite(scored[g]["spearman_delta"])
                                                else -np.inf))
            m = scored[best_g]
            m["nmf_gauge"] = best_g
            m.update(model=f"NMF ({tag})", effect=eff, seed=0, seconds=secs,
                     steps_used=n_it, converged=bool(conv))
            rows.append(m)
            print(f"  eff {eff:<5g} NMF {tag:<12} spearman_delta {m['spearman_delta']:.3f} "
                  f"psi_r {m['psi_r']:.3f} (n_iter {n_it}, conv={conv}, "
                  f"best gauge: {best_g})", flush=True)

        # --- LeafletFA DM + simplex: reuse the fits already on disk ---------------------------
        for s in args.seeds:
            f = (f"{DM_FRESH}/fit_sim_heart_K5_dm{args.conc}_eff{eff}_frac0.3_seed0_"
                 f"dirichletmultinomial_simplex_seed{s}.npz")
            if not os.path.exists(f):
                print(f"  MISSING {os.path.basename(f)}", flush=True)
                continue
            z = np.load(f, allow_pickle=True)
            m = SB.metrics(T, np.asarray(z["psi_mean"], float), np.asarray(z["phi"], float))
            m.update(model="LeafletFA (DM+simplex)", effect=eff, seed=s,
                     seconds=np.nan, steps_used=np.nan, converged=True)
            rows.append(m)
            print(f"  eff {eff:<5g} LeafletFA s{s}   spearman_delta {m['spearman_delta']:.3f} "
                  f"psi_r {m['psi_r']:.3f} ari {m['ari']:.3f}", flush=True)

    if not rows:
        print("nothing benchmarked"); return 1
    R = pd.DataFrame(rows)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    R.to_csv(out, sep="\t", index=False)
    print("\n=== effect-size recovery (spearman_delta), mean over seeds ===")
    print(R.groupby(["model", "effect"])["spearman_delta"].mean().unstack().round(3).to_string())
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
