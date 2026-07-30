#!/usr/bin/env python
"""Select ONE model per (K, lr, init arm) from its seed restarts, and report initialization consistency.

WHY THIS EXISTS
---------------
The Methods says the random-init model "is fit three times, and the one with the best ELBO was
selected". That was NEVER done -- run_training.py called train(num_initializations=1) with a
hardcoded seed, and no step ever grouped the restarts. This is that step.

It does TWO things:

  1. SELECTION. Pick one run per (K, lr, arm). Prefers a HELD-OUT metric if one is available
     (08_collect_evaluation.py / 08b_collect_classification.py output), and falls back to the ELBO.
     Selecting on the ELBO is only valid if the ELBO tracks recovery -- see
     LeafletFA/experiments/seed_variance_and_elbo_selection.py, which measures exactly that on the
     simulation. READ THAT RESULT BEFORE TRUSTING ELBO SELECTION.

  2. INITIALIZATION CONSISTENCY -- the robustness result the reviewers asked for. Across the seeds of
     a given config, report:
        * psi agreement  : Hungarian-matched program correlation between every pair of seeds
        * phi agreement  : per-cell correlation of the imputed program activities
        * psi_contrast   : does every seed even escape the symmetric collapse?
     A config whose seeds disagree is not reproducible, whatever its best ELBO says.

Usage:
    python 03b_select_across_seeds.py <model_dir> [--metric-csv <held_out_metrics.csv>]

<model_dir> is the directory containing parameter_combinations.json and run_*/ .
"""
import argparse
import gzip
import json
import os
import pickle
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def load_model(run_dir):
    d = {}
    p = os.path.join(run_dir, "leafletfa_model.pkl.gz")
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rb") as f:
        while True:
            try:
                d.update(pickle.load(f))
            except EOFError:
                break
    return d


def matched_psi_r(A, B):
    """Median Hungarian-matched correlation. Programs are identifiable only up to permutation."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    ka, kb = A.shape[0], B.shape[0]
    C = np.zeros((ka, kb))
    for i in range(ka):
        for j in range(kb):
            C[i, j] = np.corrcoef(A[i], B[j])[0, 1]
    C = np.nan_to_num(C)
    r, c = linear_sum_assignment(-np.abs(C))
    return float(np.median([C[i, j] for i, j in zip(r, c)])), r, c


def phi_agreement(phi_a, phi_b, row, col):
    """Per-cell correlation of program activities, after aligning the programs."""
    a = np.asarray(phi_a, float)[:, row]
    b = np.asarray(phi_b, float)[:, col]
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--metric-csv", default=None,
                    help="CSV of HELD-OUT metrics with a `run_dir` (or param_id) column. If given, "
                         "selection uses it instead of the ELBO.")
    ap.add_argument("--metric-col", default=None,
                    help="Column in --metric-csv to maximise (e.g. heldout_celltype_balacc, spearman)")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    params = json.load(open(os.path.join(args.model_dir, "parameter_combinations.json")))

    held = None
    if args.metric_csv:
        held = pd.read_csv(args.metric_csv)
        if args.metric_col is None:
            sys.exit("--metric-col is required with --metric-csv")

    rows = []
    for pid, p in enumerate(params):
        rd = os.path.join(args.model_dir, f"run_{pid}")
        md = load_model(rd)
        if md is None:
            continue
        psi = md.get("psi")
        phi = md.get("assign_post")
        if psi is None:
            continue
        psi = np.asarray(psi, float)
        rows.append(dict(
            param_id=pid, run_dir=rd,
            K=p.get("K"), lr=p.get("lr"), arm=p.get("arm", "?"), seed=p.get("seed", 0),
            waypoints=p.get("waypoints_use"),
            elbo=float(md.get("best_elbo", np.nan)),
            psi_contrast=float(psi.std(axis=0).mean()),
            _psi=psi, _phi=None if phi is None else np.asarray(phi, float),
        ))

    if not rows:
        sys.exit(f"no models found under {args.model_dir}")
    df = pd.DataFrame(rows)

    print(f"{len(df)} models loaded\n")
    print("=" * 92)
    print("INITIALIZATION CONSISTENCY — do the seeds of a config agree with each other?")
    print("=" * 92)
    print(f"{'K':>4} {'lr':>6} {'arm':>16} {'n':>3} {'psi_contrast':>20} "
          f"{'psi agree':>11} {'phi agree':>11}")

    summary = []
    for (K, lr, arm), g in df.groupby(["K", "lr", "arm"]):
        g = g.sort_values("seed")
        contrasts = g.psi_contrast.values
        pairs_psi, pairs_phi = [], []
        for a, b in combinations(range(len(g)), 2):
            r, row, col = matched_psi_r(g._psi.iloc[a], g._psi.iloc[b])
            pairs_psi.append(r)
            if g._phi.iloc[a] is not None and g._phi.iloc[b] is not None \
               and g._phi.iloc[a].shape == g._phi.iloc[b].shape:
                pairs_phi.append(phi_agreement(g._phi.iloc[a], g._phi.iloc[b], row, col))
        mpsi = float(np.median(pairs_psi)) if pairs_psi else np.nan
        mphi = float(np.median(pairs_phi)) if pairs_phi else np.nan

        # SELECT
        if held is not None:
            m = held[held.run_dir.isin(g.run_dir)] if "run_dir" in held.columns else \
                held[held.param_id.isin(g.param_id)]
            best = g.loc[g.param_id == int(m.loc[m[args.metric_col].idxmax()].param_id)].iloc[0] \
                if len(m) else g.loc[g.elbo.idxmin()]
            basis = f"held-out {args.metric_col}"
        else:
            best = g.loc[g.elbo.idxmin()]     # LeafletFA stores the LOSS: lower is better
            basis = "ELBO"

        flag = ""
        if np.nanmin(contrasts) < 0.04:
            flag += "  !! a seed COLLAPSED (psi_contrast < 0.04)"
        if not np.isnan(mpsi) and mpsi < 0.7:
            flag += "  !! seeds DISAGREE (psi r < 0.7) — not reproducible"

        print(f"{K:>4} {lr:>6} {arm:>16} {len(g):>3} "
              f"{contrasts.min():>7.4f}-{contrasts.max():<7.4f} "
              f"{mpsi:>11.3f} {mphi:>11.3f}{flag}")

        summary.append(dict(K=K, lr=lr, arm=arm, n_seeds=len(g),
                            psi_contrast_min=contrasts.min(), psi_contrast_max=contrasts.max(),
                            seed_psi_agreement=mpsi, seed_phi_agreement=mphi,
                            selected_param_id=int(best.param_id), selected_seed=int(best.seed),
                            selected_on=basis, selected_elbo=float(best.elbo)))

    out = pd.DataFrame(summary)
    print()
    print("=" * 92)
    print(f"SELECTED (one per config, on {out.selected_on.iloc[0]})")
    print("=" * 92)
    print(out[["K", "lr", "arm", "selected_seed", "selected_param_id",
               "seed_psi_agreement", "psi_contrast_min"]].to_string(index=False))

    if held is None:
        print("\n*** SELECTED ON THE ELBO. That is only valid if the ELBO tracks recovery.")
        print("*** Check LeafletFA/experiments/seed_variance_and_elbo_selection.py FIRST.")
        print("*** Prefer --metric-csv with a HELD-OUT metric.")

    core = out[out.arm.str.startswith("core_")].copy()
    if len(core):
        core["init"]  = np.where(core.arm.str.contains("waypoints"), "waypoints", "random")
        core["prior"] = np.where(core.arm.str.contains("globalprior"), "global", "junction-specific")

        # The two comparisons this grid exists for. Both are read off SEED AGREEMENT: a config whose
        # seeds land in different basins is not reproducible, whatever its best ELBO says.
        for label, col in (("WAYPOINTS vs RANDOM init", "init"),
                           ("PSI PRIOR: junction-specific vs global", "prior")):
            print(f"\n{label} — median seed agreement (psi) and worst psi_contrast:")
            for lvl, g in core.groupby(col):
                print(f"  {lvl:>18}  psi_agreement={g.seed_psi_agreement.median():.3f}   "
                      f"worst psi_contrast={g.psi_contrast_min.min():.4f}   "
                      f"collapsed seeds={int((g.psi_contrast_min < 0.04).sum())}/{len(g)}")

        print("\n  The psi prior is the one to watch. On simulated data the junction-specific prior "
              "(the paper's\n  default, never tested) scored psi_r 0.31 with huge seed variance; the "
              "global prior scored 0.99\n  with none. a_j/b_j are shared across the K factors, so the "
              "junction-specific prior SHRINKS THE\n  PROGRAMS TOWARD EACH OTHER. If that reproduces "
              "here, the atlas models need the global prior.")
    print("\n  (decide on HELD-OUT metrics, not these; this only shows reproducibility across seeds)")

    p = args.out_csv or os.path.join(args.model_dir, "seed_selection.csv")
    out.to_csv(p, index=False)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
