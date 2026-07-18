#!/usr/bin/env python
"""
Step 10d: Collect and compare results across all LeafletFA runs.

Scans all run_*/run_summary.csv files in a model directory, merges them with
the parameter grid, and saves a ranked comparison table.

Usage:
    python pipelines/05_leafletfa/03_collect_results.py <model_dir>

    <model_dir>   directory containing parameter_combinations.json and run_*/
"""

import sys
import os
import json
import glob
import pandas as pd
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: python 03_collect_results.py <model_dir>")
        sys.exit(1)

    model_dir = sys.argv[1]

    param_file = os.path.join(model_dir, "parameter_combinations.json")
    with open(param_file) as f:
        param_list = json.load(f)
    params_df = pd.DataFrame(param_list)
    params_df.index.name = "param_id"
    params_df = params_df.reset_index()

    # Collect per-run summaries
    run_dirs = sorted(glob.glob(os.path.join(model_dir, "run_*")))
    print(f"Found {len(run_dirs)} run directories (expecting {len(param_list)})")

    summaries = []
    missing = []
    for run_dir in run_dirs:
        summary_file = os.path.join(run_dir, "run_summary.csv")
        if not os.path.exists(summary_file):
            missing.append(run_dir)
            continue
        df = pd.read_csv(summary_file)
        summaries.append(df)

    if missing:
        print(f"  ⚠️  {len(missing)} runs have no run_summary.csv yet (still running or failed):")
        for d in missing[:10]:
            print(f"    {os.path.basename(d)}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")

    if not summaries:
        print("No completed runs found.")
        sys.exit(0)

    results_df = pd.concat(summaries, ignore_index=True)

    # Merge with full parameter grid so all params are visible
    keep_params = ["param_id", "K", "batch_size", "num_passes", "junc_specific_prior",
                   "waypoints_use", "lr", "gamma", "max_junctions", "num_inits"]
    keep_params = [c for c in keep_params if c in params_df.columns]
    merged = results_df.merge(params_df[keep_params], on="param_id", how="left",
                              suffixes=("", "_cfg"))

    # Sort by best_elbo ascending — this is Pyro's SVI loss (negative ELBO), lower = better
    merged = merged.sort_values("best_elbo", ascending=True).reset_index(drop=True)
    merged.insert(0, "rank", merged.index + 1)

    # Summary table — key columns only
    display_cols = [
        "rank", "param_id", "best_elbo", "pruned_K", "waypoints_use",
        "junc_specific_prior", "batch_size", "num_passes", "cells_coverage",
        "avg_updates_per_cell",
    ]
    display_cols = [c for c in display_cols if c in merged.columns]

    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY  ({len(results_df)} completed / {len(param_list)} total)")
    print(f"{'='*70}")
    with pd.option_context("display.max_columns", None, "display.width", 120,
                           "display.float_format", "{:.4g}".format):
        print(merged[display_cols].to_string(index=False))

    # Best run per waypoints_use setting
    print(f"\n--- Best run by initialization type ---")
    for wp in [True, False]:
        sub = merged[merged["waypoints_use"] == wp]
        if sub.empty:
            continue
        best = sub.iloc[0]
        label = "waypoints" if wp else "random init"
        print(f"  {label:12s}: param_id={int(best['param_id'])}, "
              f"ELBO={best['best_elbo']:.4g}, K={int(best['pruned_K'])}, "
              f"batch={int(best['batch_size'])}, passes={int(best['num_passes'])}")

    # Check pass-level convergence for top 5
    print(f"\n--- Convergence check (top 5 runs) ---")
    for _, row in merged.head(5).iterrows():
        pass_file = os.path.join(model_dir, f"run_{int(row['param_id'])}", "pass_summary_results.csv")
        if not os.path.exists(pass_file):
            continue
        pass_df = pd.read_csv(pass_file)
        elbos = pass_df["avg_elbo"].values
        delta = abs(elbos[-1] - elbos[-2]) if len(elbos) >= 2 else float("nan")
        trend = "↓ converging" if len(elbos) >= 2 and elbos[-1] < elbos[-2] else "→ flat/diverging"
        print(f"  param_id={int(row['param_id'])}: {len(elbos)} passes, "
              f"final ELBO={elbos[-1]:.4g}, last delta={delta:.4g}  {trend}")

    # Save full merged table
    out_file = os.path.join(model_dir, "all_runs_summary.csv")
    merged.to_csv(out_file, index=False)
    print(f"\nFull results → {out_file}")

    # Save display table
    display_file = os.path.join(model_dir, "all_runs_display.csv")
    merged[display_cols].to_csv(display_file, index=False)
    print(f"Display table → {display_file}")


if __name__ == "__main__":
    main()
