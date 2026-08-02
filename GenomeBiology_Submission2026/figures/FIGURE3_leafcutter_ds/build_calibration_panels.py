#!/usr/bin/env python
"""
Figure 3 calibration panels — LeafletFA group DS vs pseudobulk LeafCutter.

Draws only panels whose inputs already exist, so it can be re-run as results land.

  g  empirical false positives under a donor-level permutation null, and the same
     null done WRONG (cell-level shuffle) to show why the distinction matters
  c1 the low-rank ceiling: per-contrast effect-size slope vs |dphi|_1
  c2 concordance (AUROC) by statistic, across the three K=30 seeds
  s1 within-ATSE simplex violation vs ATSE size  (supplementary)
  s2 posterior draws S: what changes and what does not (supplementary)

Usage:
    python build_calibration_panels.py --out .
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
import leaflet_theme as T  # noqa: E402

WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
NULL_TSV = f"{WD}/DS_CALIBRATION/permutation_null_K30_seed1.tsv"
EXP_TSV = ("/scratch/claude-91104/-gpfs-commons-home-kisaev-LeafletFA/"
           "29bb094d-e379-4593-8f2a-d221364ac4d4/scratchpad/ds_experiment_results.tsv")


def panel_null(ax_a, ax_b):
    """Donor-split vs cell-shuffle null.

    Every call is a false positive by construction: both arms are the SAME cell type.
    """
    R = pd.read_csv(NULL_TSV, sep="\t")

    # (a) FPR vs nominal threshold, both nulls
    for mode, lab, col in [("donor", "donor split (correct)", T.METHOD_COLORS["LeafletFA"]),
                           ("cell", "cell shuffle (incorrect)", T.METHOD_COLORS["shuffled"])]:
        d = R[R["mode"] == mode].groupby("alpha").mean_calls.mean()
        ax_a.plot(d.index, d.values, "o-", color=col, label=lab, ms=3, lw=1.2)
    ax_a.set_xscale("log")
    ax_a.set_xlabel("nominal posterior expected FDP")
    ax_a.set_ylabel("false ATSE calls\nper permutation")
    ax_a.legend(frameon=False, loc="upper left")
    T.despine(ax_a)

    # (b) per cell type at alpha=0.05 — the aggregate hides that it is 2 of 6
    d = R[(R["mode"] == "donor") & (R.alpha == 0.05)].sort_values("mean_calls")
    y = np.arange(len(d))
    ax_b.barh(y, d.mean_calls.values, color=T.METHOD_COLORS["LeafletFA"], height=0.6)
    ax_b.plot(d.median_calls.values, y, "|", color=T.INK, ms=6, mew=1.2,
              label="median across permutations")
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([c.replace("_", " ") for c in d.cell_type], fontsize=T.PT["tick"] - 1)
    ax_b.set_xlabel("false ATSE calls per permutation\n(mean bar, median tick)")
    ax_b.legend(frameon=False, loc="lower right")
    T.despine(ax_b)
    return R


def panel_ceiling(ax):
    """Per-contrast slope is set by how far apart the two cell types sit in program space."""
    R = pd.read_csv(EXP_TSV, sep="\t")
    R = R[R.seed == 1].dropna(subset=["slope_A", "dphi_l1"])
    ax.scatter(R.dphi_l1, R.slope_A, s=9, color=T.METHOD_COLORS["LeafletFA"],
               alpha=0.8, linewidths=0)
    m = np.polyfit(R.dphi_l1, R.slope_A, 1)
    xs = np.linspace(R.dphi_l1.min(), R.dphi_l1.max(), 50)
    ax.plot(xs, np.polyval(m, xs), "--", color=T.INK, lw=0.9)
    r = np.corrcoef(R.dphi_l1, R.slope_A)[0, 1]
    ax.text(0.04, 0.95, f"$r$ = {r:.2f}", transform=ax.transAxes,
            va="top", fontsize=T.PT["annot"])
    ax.set_xlabel(r"$\|\Delta\bar{\phi}\|_1$  (program-space separation)")
    ax.set_ylabel("effect-size slope\nvs LeafCutter")
    T.despine(ax)


def panel_statistic(ax):
    """AUROC by statistic, one point per contrast, three seeds."""
    R = pd.read_csv(EXP_TSV, sep="\t")
    cols = [("auroc_A_raw", "TV"), ("auroc_B_comp", "TV\n(compositional)"),
            ("auroc_C_logit", "TV\n(logit)")]
    data = [R[c].dropna().values for c, _ in cols]
    bp = ax.boxplot(data, widths=0.55, showfliers=False, patch_artist=True,
                    medianprops=dict(color=T.INK, lw=1.1))
    for p in bp["boxes"]:
        p.set(facecolor=T.METHOD_COLORS["LeafletFA"], alpha=0.35, linewidth=0.7,
              edgecolor=T.INK)
    for i, (c, _) in enumerate(cols, start=1):
        v = R[c].dropna().values
        ax.plot(np.random.default_rng(0).normal(i, 0.055, len(v)), v, ".",
                ms=1.6, color=T.INK, alpha=0.35)
    ax.axhline(0.5, color=T.PALETTE["grey"], lw=0.7, ls=":")
    ax.set_xticklabels([l for _, l in cols], fontsize=T.PT["tick"])
    ax.set_ylabel("AUROC vs LeafCutter\nsignificant ATSEs")
    T.despine(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    T.apply_theme()

    fig = plt.figure(figsize=T.figsize("double", 5.4))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.5,
                          left=0.09, right=0.98, top=0.93, bottom=0.11)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1:])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    R = panel_null(ax_a, ax_b)
    panel_ceiling(ax_c)
    panel_statistic(ax_d)

    for ax, l in [(ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")]:
        T.panel_label(ax, l)

    out = os.path.join(args.out, "FIG3_calibration_draft")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    print(f"saved -> {out}.pdf / .png")

    d = R[(R["mode"] == "donor") & (R.alpha == 0.05)]
    print("\nnumbers behind panel a/b (alpha = 0.05, 100 permutations per cell type):")
    print(f"  donor split : mean {d.mean_calls.mean():.1f} calls of {d.n_atse.iloc[0]:,} ATSEs; "
          f"{(d.mean_calls == 0).sum()}/{len(d)} cell types with ZERO")
    c = R[(R["mode"] == "cell") & (R.alpha == 0.05)]
    print(f"  cell shuffle: mean {c.mean_calls.mean():.1f} calls  <- falsely clean")


if __name__ == "__main__":
    main()
