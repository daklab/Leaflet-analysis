#!/usr/bin/env python
"""Fig 2 supp: is the per-cell imputation uncertainty calibrated? (A68: 400 ATSEs x 6,000 cells, seed-0
posterior of rho_cj from 200 joint q(phi) q(psi) draws, vs the other two seeds' posterior-mean rho_cj)
a  across-seed sd vs posterior sd (hexbin, binned medians, y = x)
b  fraction of other-seed values inside seed-0's 90% credible interval, per posterior-sd decile (target 0.9)
c  posterior sd vs reads at the (cell, event): uncertainty rises where coverage falls"""
import os, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); FIGROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, FIGROOT); sys.path.insert(0, os.path.join(FIGROOT, "_common"))
from leaflet_theme import apply_theme, PALETTE, METHOD_COLORS, figsize, panel_label, despine, save
M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
LFA, GREY = METHOD_COLORS["LeafletFA"], PALETTE["grey"]

def main():
    apply_theme(); d = pd.read_csv(f"{M}/IMPUTATION_DM/a68_uncertainty.tsv.gz", sep="\t")
    fig, axes = plt.subplots(1, 3, figsize=figsize("double", h=2.4))
    ax = axes[0]; x = d.post_sd.clip(1e-4, 0.5); y = d.cross_sd.clip(1e-4, 0.5)
    ax.hexbin(np.log10(x), np.log10(y), gridsize=45, bins="log", cmap="Greys", mincnt=1, linewidths=0)
    d["bin"] = pd.qcut(d.post_sd, 20, labels=False, duplicates="drop"); g = d.groupby("bin").agg(px=("post_sd", "median"), py=("cross_sd", "median"))
    ax.plot(np.log10(g.px), np.log10(g.py), "o-", color=LFA, ms=3, lw=1.3, label="binned median")
    lim = [-3.5, -0.3]; ax.plot(lim, lim, ls=":", color=PALETTE["rose"], lw=1, label="y = x")
    ax.set_xlabel("posterior sd of imputed usage (seed 0)  [log10]"); ax.set_ylabel("sd of imputed usage across 3 seeds  [log10]")
    ax.text(0.03, 0.97, f"Spearman {spearmanr(d.post_sd, d.cross_sd).correlation:.2f}\n{len(d):,} (cell, junction) pairs\n{d.event.nunique()} ATSEs, {d.cell.nunique():,} cells", transform=ax.transAxes, va="top", fontsize=5.5)
    ax.legend(frameon=False, fontsize=5.5, loc="lower right")
    ax = axes[1]; d["dec"] = pd.qcut(d.post_sd, 10, labels=False, duplicates="drop"); g = d.groupby("dec").agg(px=("post_sd", "median"), covg=("inside90", "mean"))
    ax.plot(g.px, g.covg, "o-", color=LFA, ms=3.5, lw=1.4); ax.axhline(0.9, color=PALETTE["rose"], ls=":", lw=1); ax.set_xscale("log")
    ax.set_ylim(0.5, 1.0); ax.set_xlabel("posterior sd (decile medians)"); ax.set_ylabel("fraction of other-seed values\ninside seed-0's 90% credible interval")
    ax.text(0.03, 0.05, f"overall {d.inside90.mean():.2f} (target 0.90)", transform=ax.transAxes, fontsize=6, color=LFA)
    ax = axes[2]; d["rb"] = pd.cut(d.reads, [-1, 0, 2, 5, 10, 20, 50, 1e9], labels=["0", "1-2", "3-5", "6-10", "11-20", "21-50", ">50"])
    g = d.groupby("rb", observed=True).agg(sd=("post_sd", "median"), lo=("post_sd", lambda v: v.quantile(.25)), hi=("post_sd", lambda v: v.quantile(.75)), n=("post_sd", "size"))
    xx = np.arange(len(g)); ax.errorbar(xx, g.sd, yerr=[g.sd - g.lo, g.hi - g.sd], fmt="o-", color=LFA, ms=3.5, lw=1.3, capsize=2)
    ax.set_xticks(xx); ax.set_xticklabels(g.index, fontsize=6); ax.set_xlabel("reads at the (cell, ATSE)"); ax.set_ylabel("posterior sd of imputed usage\n(median, IQR)")
    for ax, L in zip(axes, "abc"): despine(ax); panel_label(ax, L)
    fig.suptitle("Per-cell imputation uncertainty is calibrated against seed-to-seed variation", fontsize=8, y=1.02)
    fig.tight_layout(); save(fig, os.path.join(FIGROOT, "FIGURE2_simulation_imputation", "plots", "FIG2_supp_imputation_uncertainty"))
    print(g.round(4).to_string())

if __name__ == "__main__":
    main()
