#!/usr/bin/env python
"""
08_consistency_check.py — Check consistency of LeafletFA factors across random initializations.

Two approaches per same-K run group:

A. Factor-level: cosine similarity between psi rows across run pairs,
   solved with Hungarian assignment. High mean matched similarity → reproducible factors.

B. Cell-level: for a random subset of cells, compare imputed PSI (assign_post @ psi)
   across run pairs using per-cell Pearson r. Permutation-invariant — directly
   measures whether the model produces the same cell representation regardless of init.
   Uses assign_post already stored in the model pkl (no need to reload h5ad).

Usage:
    python 08_consistency_check.py <model_dir> <output_dir> [--n-cells N] [--seed S]

    model_dir   LEAFLETFA_MODELS/YYYY-MM-DD/
    output_dir  where to write CSVs and plots
"""

import sys
import os
import argparse
import glob
import gzip
import pickle
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(run_dir):
    pkl = os.path.join(run_dir, "leafletfa_model.pkl.gz")
    d = {}
    with gzip.open(pkl, "rb") as f:
        while True:
            try:
                d.update(pickle.load(f))
            except EOFError:
                break
    return d


def cosine_sim_matrix(psi_a, psi_b):
    """(K x K) cosine similarity matrix between two psi arrays."""
    na = np.linalg.norm(psi_a, axis=1, keepdims=True)
    nb = np.linalg.norm(psi_b, axis=1, keepdims=True)
    na[na == 0] = 1.0
    nb[nb == 0] = 1.0
    return (psi_a / na) @ (psi_b / nb).T


def pearson_rows(A, B):
    """Per-row Pearson r between two (N x J) float32 arrays. Returns (N,) array."""
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    num = (A * B).sum(axis=1)
    den = np.sqrt((A ** 2).sum(axis=1) * (B ** 2).sum(axis=1))
    den[den < 1e-12] = 1.0
    return (num / den).astype(np.float32)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--n-cells", type=int, default=500,
                    help="Cells for cell-level comparison (default 500)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ── Load all run summaries ────────────────────────────────────────────────
    summaries = []
    for run_dir in sorted(glob.glob(os.path.join(args.model_dir, "run_*"))):
        csv = os.path.join(run_dir, "run_summary.csv")
        if not os.path.exists(csv):
            continue
        row = pd.read_csv(csv).iloc[0].to_dict()
        row["run_dir"] = run_dir
        row["run_name"] = os.path.basename(run_dir)
        summaries.append(row)
    df_meta = pd.DataFrame(summaries)
    print(f"Loaded {len(df_meta)} runs from {args.model_dir}")

    # Determine n_total from first model
    first_m = load_model(df_meta.iloc[0]["run_dir"])
    n_total = first_m["assign_post"].shape[0]
    print(f"Training cells in model: {n_total}")
    del first_m

    # Fixed cell subset (same indices across all models)
    subset_idx = np.sort(rng.choice(n_total, size=min(args.n_cells, n_total), replace=False))
    print(f"Cell subset for approach B: {len(subset_idx)} cells (seed={args.seed})")

    results_A = []
    results_B = []

    for K, grp in df_meta.groupby("K"):
        run_dirs  = grp["run_dir"].tolist()
        run_names = grp["run_name"].tolist()
        print(f"\n{'='*60}")
        print(f"K={K}: {len(run_dirs)} runs")

        # Load psi and subset of assign_post for each run
        models = {}
        for rd in run_dirs:
            name = os.path.basename(rd)
            print(f"  Loading {name}...", flush=True)
            m = load_model(rd)
            models[name] = {
                "psi":     m["psi"].astype(np.float32),                        # (K, J)
                "phi_sub": m["assign_post"][subset_idx].astype(np.float32),    # (n_sub, K)
            }

        pairs = list(itertools.combinations(run_names, 2))
        print(f"  {len(pairs)} pairwise comparisons...")

        for name_a, name_b in pairs:
            psi_a   = models[name_a]["psi"]
            psi_b   = models[name_b]["psi"]
            phi_a   = models[name_a]["phi_sub"]
            phi_b   = models[name_b]["phi_sub"]

            # ── Approach A ────────────────────────────────────────────────────
            cos = cosine_sim_matrix(psi_a, psi_b)           # (K x K)
            row_ind, col_ind = linear_sum_assignment(-cos)   # maximise
            matched_cos = cos[row_ind, col_ind]

            results_A.append({
                "K":                      K,
                "run_a":                  name_a,
                "run_b":                  name_b,
                "mean_matched_cosine":    float(matched_cos.mean()),
                "min_matched_cosine":     float(matched_cos.min()),
                "max_matched_cosine":     float(matched_cos.max()),
            })

            # ── Approach B ────────────────────────────────────────────────────
            imp_a = (phi_a @ psi_a).astype(np.float32)   # (n_sub, J)
            imp_b = (phi_b @ psi_b).astype(np.float32)
            cell_r = pearson_rows(imp_a, imp_b)           # (n_sub,)
            del imp_a, imp_b

            results_B.append({
                "K":                      K,
                "run_a":                  name_a,
                "run_b":                  name_b,
                "n_cells":                len(subset_idx),
                "mean_cell_pearson_r":    float(cell_r.mean()),
                "median_cell_pearson_r":  float(np.median(cell_r)),
                "pct_r_gt_0.9":           float(100 * (cell_r > 0.9).mean()),
            })

            print(f"    {name_a} vs {name_b}: "
                  f"cosine={matched_cos.mean():.3f}  "
                  f"cell_r={cell_r.mean():.3f}")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    df_A = pd.DataFrame(results_A)
    df_B = pd.DataFrame(results_B)
    df_A.to_csv(os.path.join(args.output_dir, "consistency_factor_level.csv"), index=False)
    df_B.to_csv(os.path.join(args.output_dir, "consistency_cell_level.csv"),   index=False)

    # ── Plots ─────────────────────────────────────────────────────────────────
    K_vals = sorted(df_A["K"].unique())
    ncols  = len(K_vals)

    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]
    for ax, K in zip(axes, K_vals):
        vals = df_A[df_A["K"] == K]["mean_matched_cosine"].values
        ax.boxplot(vals, widths=0.5)
        ax.scatter([1] * len(vals), vals, color="steelblue", zorder=3, s=40)
        ax.set_title(f"K={K}", fontsize=11)
        ax.set_xticks([1]); ax.set_xticklabels(["run pairs"])
        ax.set_ylim(0, 1.05)
        ax.axhline(0.9, ls="--", c="gray", alpha=0.5, label="0.9")
    axes[0].set_ylabel("mean matched cosine similarity")
    fig.suptitle("Factor reproducibility — approach A (cosine similarity + Hungarian)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "consistency_factor_level.png"), dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]
    for ax, K in zip(axes, K_vals):
        vals = df_B[df_B["K"] == K]["mean_cell_pearson_r"].values
        ax.boxplot(vals, widths=0.5)
        ax.scatter([1] * len(vals), vals, color="darkorange", zorder=3, s=40)
        ax.set_title(f"K={K}", fontsize=11)
        ax.set_xticks([1]); ax.set_xticklabels(["run pairs"])
        ax.set_ylim(0, 1.05)
        ax.axhline(0.9, ls="--", c="gray", alpha=0.5)
    axes[0].set_ylabel("mean per-cell Pearson r (imputed PSI)")
    fig.suptitle("Cell representation consistency — approach B (imputed PSI)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "consistency_cell_level.png"), dpi=150)
    plt.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print("\nApproach A — mean matched cosine similarity per K:")
    print(df_A.groupby("K")["mean_matched_cosine"].describe()[["mean", "min", "max"]].to_string())
    print("\nApproach B — mean per-cell Pearson r per K:")
    print(df_B.groupby("K")["mean_cell_pearson_r"].describe()[["mean", "min", "max"]].to_string())
    print(f"\nOutputs in {args.output_dir}")


if __name__ == "__main__":
    main()
