#!/usr/bin/env python
"""
Collect per-run classifier CSVs written by 07_classify_splicing_programs.py --run-dir,
merge them, deduplicate baselines, generate plots and summary tables.

Usage:
    python 08b_collect_classification.py <classifiers_dir>

    classifiers_dir   directory containing run_*_classifier_results.csv files
                      (same dir passed as output_dir to 07b_submit_classification.py)
"""

import sys
import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REG_TARGETS = {"age"}


def main():
    if len(sys.argv) < 2:
        print("Usage: python 08b_collect_classification.py <classifiers_dir>")
        sys.exit(1)

    clf_dir = sys.argv[1]

    # ── Load and merge ────────────────────────────────────────────────────────
    csvs = sorted(glob.glob(os.path.join(clf_dir, "run_*_classifier_results.csv")))
    if not csvs:
        print(f"ERROR: no run_*_classifier_results.csv found in {clf_dir}")
        sys.exit(1)
    print(f"Found {len(csvs)} per-run CSVs")

    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    print(f"  Total rows before dedup: {len(df)}")

    # Baselines (scVI, label_permuted) are repeated once per run job — keep one copy each
    baselines = df[df["model_type"].isin(["scVI_linear", "label_permuted"])].drop_duplicates(
        subset=["model_type", "target"]
    )
    lfa = df[df["model_type"].isin(["LeafletFA", "shuffled"])]
    df  = pd.concat([baselines, lfa], ignore_index=True)
    print(f"  Total rows after dedup:  {len(df)}")

    # ── Save merged ───────────────────────────────────────────────────────────
    out_csv = os.path.join(clf_dir, "classifier_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nMerged results → {out_csv}")

    # ── Summary table ─────────────────────────────────────────────────────────
    available_targets = df["target"].unique().tolist()

    lfa_df    = df[df["model_type"] == "LeafletFA"]
    scvi_df   = df[df["model_type"] == "scVI_linear"]
    null_df   = df[df["model_type"] == "shuffled"].groupby("target")["score"].mean()
    perm_df   = df[df["model_type"] == "label_permuted"].groupby("target")["score"].mean()

    best_per_K = (lfa_df.sort_values("best_elbo", ascending=True)
                        .groupby(["K", "target"])
                        .first()
                        .reset_index())

    print("\n=== Best-per-K score (balanced_accuracy or R²) ===")
    pivot = best_per_K.pivot(index="K", columns="target", values="score")
    print(pivot.round(3).to_string())

    print("\n=== scVI baseline ===")
    print(scvi_df[["target", "metric_name", "score", "macro_f1", "n_classes", "pearson_r"]]
          .to_string(index=False))

    print("\n=== Shuffled-phi null (mean across runs) ===")
    print(null_df.round(3).to_string())

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(available_targets),
                             figsize=(4 * len(available_targets), 5), sharey=False)
    if len(available_targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, available_targets):
        sub = best_per_K[best_per_K["target"] == target].sort_values("K")
        if sub.empty:
            ax.set_visible(False)
            continue

        metric_name = sub["metric_name"].iloc[0]
        K_labels    = sub["K"].astype(str).values
        ax.bar(K_labels, sub["score"], color="steelblue", alpha=0.8, label="LeafletFA (best/K)")

        scvi_val = scvi_df[scvi_df["target"] == target]["score"]
        if not scvi_val.empty:
            ax.axhline(scvi_val.values[0], ls="--", c="orange", lw=1.5, label="scVI")

        if target in null_df.index:
            ax.axhline(null_df[target], ls=":", c="gray", lw=1.5, label="null (shuffled φ)")

        if target in perm_df.index:
            ax.axhline(perm_df[target], ls="-", c="darkred", lw=1, alpha=0.5,
                       label="null (label permuted)")

        if target not in REG_TARGETS and not sub.empty:
            ax.axhline(sub["chance_level"].iloc[0], ls="-.", c="red", lw=1,
                       alpha=0.6, label="chance")

        ax.set_title(target, fontsize=10)
        ax.set_xlabel("K")
        ax.set_ylabel(metric_name)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)

    fig.suptitle("LeafletFA vs scVI — classifier/regression evaluation (5-fold CV)", fontsize=11)
    plt.tight_layout()
    out_png = os.path.join(clf_dir, "classifier_balanced_accuracy.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"\nPlot → {out_png}")


if __name__ == "__main__":
    main()
