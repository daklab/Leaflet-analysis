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
CAL = f"{WD}/DS_CALIBRATION"
NULL_TSV = f"{CAL}/permutation_null_K30_seed1.tsv"
NULL_PERPERM = f"{CAL}/permutation_null_perperm_K30_seed1.tsv.gz"
SIM_FDR = f"{CAL}/sim_fdr_calibration.tsv"
HELDOUT = f"{CAL}/heldout_effect_prediction_K30_seed1.tsv"
DOWNSAMP = f"{CAL}/downsampling_power_K30_seed1.tsv"
EXP_TSV = f"{CAL}/ds_statistic_comparison.tsv"


def panel_sim_calibration(ax):
    """The REAL calibration curve: nominal alpha vs realised FDP, with true positives.

    Distinct from the permutation null, which has no true positives and therefore measures
    type-I error volume rather than a false discovery PROPORTION.
    """
    R = pd.read_csv(SIM_FDR, sep="\t")
    R["eff"] = R.sim.str.extract(r"eff([0-9.]+)")[0].astype(float)
    cmap = plt.get_cmap(T.CMAP["sequential"])
    effs = sorted(R.eff.dropna().unique())
    for i, e in enumerate(effs):
        d = R[R.eff == e].groupby("alpha").realised_fdp.mean()
        ax.plot(d.index, d.values, "o-", ms=3, lw=1.2,
                color=cmap(0.15 + 0.7 * i / max(len(effs) - 1, 1)),
                label=f"effect {e:g}")
    lim = [0, max(R.alpha.max(), R.realised_fdp.max()) * 1.05]
    ax.plot(lim, lim, ":", color=T.PALETTE["grey"], lw=0.9, zorder=0)
    ax.set_xlabel("nominal posterior expected FDP")
    ax.set_ylabel("realised false\ndiscovery proportion")
    ax.legend(frameon=False, fontsize=T.PT["legend"] - 0.5, loc="upper left")
    T.despine(ax)


def panel_downsampling(ax):
    """Recovery of the full-data marker set as cells are removed."""
    R = pd.read_csv(DOWNSAMP, sep="\t")
    g = R.groupby("frac").agg(cells=("cells_per_type", "mean"),
                              recall=("recall_of_full", "mean"),
                              sd=("recall_of_full", "std")).reset_index()
    g = g.sort_values("cells")
    ax.errorbar(g.cells, g.recall, yerr=g.sd, fmt="o-", ms=3, lw=1.2, capsize=1.5,
                color=T.METHOD_COLORS["LeafletFA"], label="LeafletFA")
    # LeafCutter's hard floor: >=4 pseudobulk samples, each needing >=25 cells
    ax.axvline(100, color=T.PALETTE["grey"], ls="--", lw=0.9)
    ax.text(105, 0.05, "LeafCutter\nfloor", fontsize=T.PT["annot"] - 0.5,
            color=T.PALETTE["grey"], va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("cells per cell type")
    ax.set_ylabel("recall of full-data\nmarker ATSEs")
    ax.set_ylim(0, 1.02)
    T.despine(ax)


def panel_heldout(ax):
    """Which estimator predicts held-out donors better? Scale-free R^2."""
    R = pd.read_csv(HELDOUT, sep="\t")
    lim = [min(R.r2_leafcutter.min(), R.r2_leafletfa.min()),
           max(R.r2_leafcutter.max(), R.r2_leafletfa.max())]
    ax.scatter(R.r2_leafcutter, R.r2_leafletfa, s=7, alpha=0.6, linewidths=0,
               color=T.METHOD_COLORS["LeafletFA"])
    ax.plot(lim, lim, ":", color=T.PALETTE["grey"], lw=0.9)
    win = (R.r2_leafletfa > R.r2_leafcutter).mean()
    ax.text(0.04, 0.95, f"LeafletFA better\non {100*win:.0f}% of splits",
            transform=ax.transAxes, va="top", fontsize=T.PT["annot"])
    ax.set_xlabel(r"$R^2$, LeafCutter-style estimate")
    ax.set_ylabel(r"$R^2$, LeafletFA")
    T.despine(ax)


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

    # (b) per cell type at alpha=0.05 — the aggregate hides that it is 2 of 6, and that
    # within those two it is a handful of extreme donor splits (mean 185 vs median 5.5).
    # symlog: median sits at 0-6 while the worst split reaches ~1200.
    d = R[(R["mode"] == "donor") & (R.alpha == 0.05)].sort_values("mean_calls")
    y = np.arange(len(d))
    ax_b.hlines(y, 0, d.max_calls.values, color=T.PALETTE["grey"], lw=0.8, zorder=1)
    ax_b.plot(d.max_calls.values, y, "o", color=T.PALETTE["grey"], ms=3.2,
              label="worst split", zorder=2)
    ax_b.plot(d.mean_calls.values, y, "D", color=T.METHOD_COLORS["LeafletFA"], ms=4,
              label="mean", zorder=3)
    ax_b.plot(d.median_calls.values, y, "|", color=T.INK, ms=7, mew=1.4,
              label="median", zorder=4)
    ax_b.set_xscale("symlog", linthresh=1)
    ax_b.set_xlim(-0.3, 2500)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([c.replace("_", " ").title() for c in d.cell_type],
                         fontsize=T.PT["tick"] - 1)
    ax_b.set_ylim(-0.6, len(d) - 0.4)
    ax_b.set_xlabel("false ATSE calls per permutation (symlog)")
    ax_b.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3,
                handletextpad=0.4, columnspacing=1.2)
    T.despine(ax_b)
    return R


def panel_ceiling(ax):
    """Per-contrast slope is set by how far apart the two cell types sit in program space."""
    R = pd.read_csv(EXP_TSV, sep="\t")
    R = R[R.seed == 1].dropna(subset=["slope_raw", "dphi_l1"])
    ax.scatter(R.dphi_l1, R.slope_raw, s=9, color=T.METHOD_COLORS["LeafletFA"],
               alpha=0.8, linewidths=0)
    m = np.polyfit(R.dphi_l1, R.slope_raw, 1)
    xs = np.linspace(R.dphi_l1.min(), R.dphi_l1.max(), 50)
    ax.plot(xs, np.polyval(m, xs), "--", color=T.INK, lw=0.9)
    r = np.corrcoef(R.dphi_l1, R.slope_raw)[0, 1]
    ax.text(0.04, 0.95, f"$r$ = {r:.2f}", transform=ax.transAxes,
            va="top", fontsize=T.PT["annot"])
    ax.set_xlabel(r"$\|\Delta\bar{\phi}\|_1$  (program-space separation)")
    ax.set_ylabel("effect-size slope\nvs LeafCutter")
    T.despine(ax)


def panel_statistic(ax):
    """AUROC by statistic, one point per contrast, three seeds."""
    R = pd.read_csv(EXP_TSV, sep="\t")
    cols = [("auroc_raw", "TV"), ("auroc_comp", "TV\n(compositional)"),
            ("auroc_logit", "TV\n(logit)")]
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

    # Optional panels are drawn only when their job has landed, so this stays runnable
    # while the queue drains.
    optional = [(SIM_FDR, "FDR calibration (simulation)", panel_sim_calibration),
                (DOWNSAMP, "downsampling recovery", panel_downsampling),
                (HELDOUT, "held-out prediction", panel_heldout)]
    ready = [(p, n, f) for p, n, f in optional if os.path.exists(p)]
    for p, n, _ in optional:
        if not os.path.exists(p):
            print(f"  (skipping {n} — {os.path.basename(p)} not written yet)")

    n_rows = 2 + int(np.ceil(len(ready) / 2))
    fig = plt.figure(figsize=T.figsize("double", 2.3 * n_rows))
    gs = fig.add_gridspec(n_rows, 4, hspace=0.85, wspace=1.25,
                          left=0.10, right=0.99, top=0.94, bottom=0.09)

    ax_a = fig.add_subplot(gs[0, :2])
    ax_b = fig.add_subplot(gs[0, 2:])
    ax_c = fig.add_subplot(gs[1, :2])
    ax_d = fig.add_subplot(gs[1, 2:])

    R = panel_null(ax_a, ax_b)
    panel_ceiling(ax_c)
    panel_statistic(ax_d)
    axes = [(ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")]

    for i, (_, name, fn) in enumerate(ready):
        r, c = 2 + i // 2, (i % 2) * 2
        ax = fig.add_subplot(gs[r, c:c + 2])
        try:
            fn(ax)
            axes.append((ax, "efghij"[i]))
        except Exception as e:                      # a half-written TSV must not kill the run
            print(f"  WARNING: {name} panel failed: {e}")
            ax.axis("off")

    for ax, l in axes:
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
