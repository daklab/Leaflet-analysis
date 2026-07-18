#!/usr/bin/env python
"""
Collect imputation evaluation results across all runs and make figures.

Usage:
    python 08_collect_evaluation.py <eval_output_dir>

Reads all run_*_imputation_results.csv files, merges them, picks best model
per K (by Pearson r at highest downsampling level), and plots all imputation
metrics vs. downsampling percentage — matching SpliceVI metric naming.
"""

import sys
import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS_TO_PLOT = [
    ("pearson_r",    "Global Pearson r (predicted vs original PSI)",         (0.0, 1.0)),
    ("spearman_r",   "Global Spearman r (predicted vs original PSI)",        (0.0, 1.0)),
    ("l1_mean",      "L1 mean (|predicted − original PSI|)",                 (0.0, None)),
    ("l1_median",    "L1 median (|predicted − original PSI|)",               (0.0, None)),
    ("l1_p90",       "L1 P90 (90th pct |predicted − original PSI|)",        (0.0, None)),
    ("rmse",         "RMSE",                                                  (0.0, None)),
    ("smape",        "SMAPE",                                                 (0.0, None)),
    ("cosine_sim",   "Cosine similarity",                                     (0.0, 1.0)),
    ("minmax_ratio", "Min-max ratio",                                         (0.0, 1.0)),
]

colors_K = {10: "#1f77b4", 20: "#ff7f0e", 50: "#2ca02c"}


def main():
    if len(sys.argv) < 2:
        print("Usage: python 08_collect_evaluation.py <eval_output_dir>")
        sys.exit(1)

    eval_out_dir = sys.argv[1]

    # ── Collect all per-run CSVs ──────────────────────────────────────────────
    csvs = sorted(glob.glob(os.path.join(eval_out_dir, "run_*_imputation_results.csv")))
    print(f"Found {len(csvs)} result files")

    if not csvs:
        print("No results yet.")
        sys.exit(0)

    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)

    # Keep best row per (run, downsampling_pct) in case of duplicate runs
    before = len(df)
    df = df.loc[df.groupby(["run_dir", "downsampling_pct"])["pearson_r"].idxmax()].copy()
    if len(df) < before:
        print(f"  Deduplicated {before} → {len(df)} rows "
              f"(kept best row per run × downsampling level)")

    df = df.sort_values(["K", "downsampling_pct"]).reset_index(drop=True)

    # Save merged table
    merged_path = os.path.join(eval_out_dir, "all_runs_imputation_results.csv")
    df.to_csv(merged_path, index=False)
    print(f"Merged results → {merged_path}")
    print(f"  {len(df)} rows, {df['run_dir'].nunique()} runs, "
          f"{df['downsampling_pct'].nunique()} downsampling levels")

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY: Pearson r and L1 mean by K and downsampling level")
    print(f"{'='*70}")
    for metric, label, _ in METRICS_TO_PLOT[:2]:
        pivot = df.groupby(["K", "downsampling_pct"])[metric].agg(["mean", "max"]).reset_index()
        pivot.columns = ["K", "downsampling_pct", "avg_across_runs", "best_run"]
        print(f"\n{label}:")
        with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 100):
            print(pivot.to_string(index=False))

    # ── Select best model per K (by Pearson r at highest downsampling level) ─
    ref_pct = df["downsampling_pct"].max()
    best_per_K = (
        df[df["downsampling_pct"] == ref_pct]
        .groupby("K")
        .apply(lambda g: g.loc[g["pearson_r"].idxmax()])
        .reset_index(drop=True)
    )
    print(f"\n--- Best run per K (by Pearson r at {ref_pct}% reads removed) ---")
    for _, row in best_per_K.iterrows():
        print(f"  K={int(row['K'])}: {row['run_dir']}  "
              f"waypoints={row['waypoints_use']}  prior={row['junc_specific_prior']}  "
              f"batch={int(row['batch_size'])}  r={row['pearson_r']:.4f}")

    best_run_dirs = best_per_K["run_dir"].tolist()
    best_df = df[df["run_dir"].isin(best_run_dirs)].copy()

    pcts    = sorted(df["downsampling_pct"].unique())
    xlabels = [f"{p}%" for p in pcts]

    def _setup_ax(ax, ylabel, ylim, title):
        ax.set_xlabel("% reads removed (10%=mild, 99%=very sparse)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(pcts)
        ax.set_xticklabels(xlabels, fontsize=8)
        if ylim[0] is not None or ylim[1] is not None:
            lo = ylim[0] if ylim[0] is not None else ax.get_ylim()[0]
            hi = ylim[1] if ylim[1] is not None else ax.get_ylim()[1]
            ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.3)

    # ── Figure 1: best model per K, Pearson r ────────────────────────────────
    fig1, ax = plt.subplots(figsize=(7, 5))
    for _, row_best in best_per_K.iterrows():
        K   = int(row_best["K"])
        sub = best_df[best_df["run_dir"] == row_best["run_dir"]].sort_values("downsampling_pct")
        ax.plot(sub["downsampling_pct"], sub["pearson_r"],
                marker="o", label=f"K={K}", color=colors_K.get(K, "gray"), linewidth=2)
    _setup_ax(ax, "Global Pearson r", (0.0, 1.0),
              "LeafletFA imputation accuracy (best model per K)")
    ax.legend(title="Factors (K)")
    plt.tight_layout()
    fig1.savefig(os.path.join(eval_out_dir, "imputation_best_per_K.png"), dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # ── Figure 2: mean ± SD by (K, waypoints_use) ────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    ymin = max(0, df["pearson_r"].min() - 0.05)
    for ax, wp in zip(axes2, [True, False]):
        sub = df[df["waypoints_use"] == wp]
        for K in sorted(df["K"].unique()):
            ksub = sub[sub["K"] == K]
            if ksub.empty:
                continue
            stats = ksub.groupby("downsampling_pct")["pearson_r"].agg(["mean", "std"]).reindex(pcts)
            ax.plot(pcts, stats["mean"], marker="o", label=f"K={K}",
                    color=colors_K.get(K, "gray"), linewidth=2)
            ax.fill_between(pcts,
                            stats["mean"] - stats["std"].fillna(0),
                            stats["mean"] + stats["std"].fillna(0),
                            color=colors_K.get(K, "gray"), alpha=0.2)
        label = "waypoints init" if wp else "random init"
        _setup_ax(ax, "Global Pearson r", (ymin, 1.0),
                  f"LeafletFA imputation — {label}\n(mean ± SD across runs same K)")
        ax.legend(title="Factors (K)")
    plt.tight_layout()
    fig2.savefig(os.path.join(eval_out_dir, "imputation_by_waypoints.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ── Figure 3: mean ± SD by (K, junc_specific_prior) ──────────────────────
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, pr in zip(axes3, [True, False]):
        sub = df[df["junc_specific_prior"] == pr]
        for K in sorted(df["K"].unique()):
            ksub = sub[sub["K"] == K]
            if ksub.empty:
                continue
            stats = ksub.groupby("downsampling_pct")["pearson_r"].agg(["mean", "std"]).reindex(pcts)
            ax.plot(pcts, stats["mean"], marker="o", label=f"K={K}",
                    color=colors_K.get(K, "gray"), linewidth=2)
            ax.fill_between(pcts,
                            stats["mean"] - stats["std"].fillna(0),
                            stats["mean"] + stats["std"].fillna(0),
                            color=colors_K.get(K, "gray"), alpha=0.2)
        label = "junction-specific prior" if pr else "no prior"
        _setup_ax(ax, "Global Pearson r", (ymin, 1.0),
                  f"LeafletFA imputation — {label}\n(mean ± SD across runs same K)")
        ax.legend(title="Factors (K)")
    plt.tight_layout()
    fig3.savefig(os.path.join(eval_out_dir, "imputation_by_prior.png"), dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # ── Figure 4: all runs (thin lines) grouped by K ─────────────────────────
    fig4, ax = plt.subplots(figsize=(8, 5))
    for K, grp in df.groupby("K"):
        for _, rgrp in grp.groupby("run_dir"):
            rgrp = rgrp.sort_values("downsampling_pct")
            ax.plot(rgrp["downsampling_pct"], rgrp["pearson_r"],
                    color=colors_K.get(K, "gray"), alpha=0.35, linewidth=1)
    for K, c in colors_K.items():
        if K in df["K"].values:
            ax.plot([], [], color=c, linewidth=2, label=f"K={K}")
    _setup_ax(ax, "Global Pearson r", (ymin, 1.0),
              "All runs — variation across hyperparameters")
    ax.legend(title="Factors (K)")
    plt.tight_layout()
    fig4.savefig(os.path.join(eval_out_dir, "imputation_all_runs.png"), dpi=150, bbox_inches="tight")
    plt.close(fig4)

    # ── Figures 5+: additional metrics (best model per K) ────────────────────
    for metric, ylabel, ylim in METRICS_TO_PLOT[1:]:
        if metric not in df.columns:
            print(f"  Skipping {metric} (column not found in results)")
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        for _, row_best in best_per_K.iterrows():
            K   = int(row_best["K"])
            sub = best_df[best_df["run_dir"] == row_best["run_dir"]].sort_values("downsampling_pct")
            if metric not in sub.columns:
                continue
            ax.plot(sub["downsampling_pct"], sub[metric],
                    marker="o", label=f"K={K}", color=colors_K.get(K, "gray"), linewidth=2)
        _setup_ax(ax, ylabel, ylim, f"LeafletFA imputation — {ylabel} (best model per K)")
        ax.legend(title="Factors (K)")
        plt.tight_layout()
        fig.savefig(os.path.join(eval_out_dir, f"imputation_{metric}_best_per_K.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  imputation_{metric}_best_per_K.png")

    # Save best-model subset
    best_path = os.path.join(eval_out_dir, "best_models_imputation_results.csv")
    best_df.to_csv(best_path, index=False)
    print(f"\nFigures saved to {eval_out_dir}")
    print(f"Best models table → {best_path}")


if __name__ == "__main__":
    main()
