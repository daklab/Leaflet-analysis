#!/usr/bin/env python
"""Plot A66 v2 (all 39,812 held-out cells, HVJ junctions): imputed vs observed at 0/10/50/90% removed.
Rows: LeafletFA DM posterior mean; NMF (when nmf_level_*.npz exists); junction-mean baseline.
Hexbins from the 1% entry subsample; statistics from the full sufficient stats: all-entry r,
median within-junction r, median within-cell r, and the same on entries whose reads were all removed."""
import os, sys, glob
import numpy as np, pandas as pd, matplotlib.pyplot as plt
FIGROOT = "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures"
sys.path.insert(0, FIGROOT); sys.path.insert(0, os.path.join(FIGROOT, "_common"))
sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa")
from leaflet_theme import apply_theme, PALETTE, METHOD_COLORS, figsize, panel_label, despine, save
from A66_imputed_vs_raw_heldout import Acc
M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
D = f"{M}/IMPUTATION_DM/a66_all_cells"
LFA, LC, GREY = METHOD_COLORS["LeafletFA"], METHOD_COLORS["NMF"], PALETTE["grey"]


def st(z, key, lost=False):
    p = "lost" if lost else ""
    J = z[f"J{p}_{key}"]; C = z[f"C{p}_{key}"]; G = z[f"G{p}_{key}"]
    if G[0] < 1000: return None
    return dict(r=Acc.r(G), wj=np.nanmedian(Acc.r(J[J[:, 0] >= 30])), wc=np.nanmedian(Acc.r(C[C[:, 0] >= 30])), n=int(G[0]))


def main():
    apply_theme()
    files = sorted(glob.glob(f"{D}/level_*.npz")); Z = [np.load(f) for f in files]
    NM = {int(np.load(f)["level"]): np.load(f) for f in glob.glob(f"{D}/nmf_level_*.npz")}
    rows = [("post", "LeafletFA\nDM posterior mean", LFA)] + ([("nmf", "NMF (rank 20,\nmean-imputed input)", LC)] if NM else []) + [("jmean", "junction mean\n(baseline)", GREY)]
    fig, axes = plt.subplots(len(rows), len(Z), figsize=figsize("double", h=2.3 * len(rows)), squeeze=False)
    tab = []
    for c, z in enumerate(Z):
        lev = int(z["level"]); obs = z["sub_obs"]
        for r_, (key, lab, col) in enumerate(rows):
            ax = axes[r_, c]
            if key == "nmf":
                y = NM[lev]["sub_nmf"]; S = st(NM[lev], "nmf"); SL = st(NM[lev], "nmf", True)
            else:
                y = z[f"sub_{key}"]; S = st(z, key); SL = st(z, key, True)
            ax.hexbin(obs, y, gridsize=40, bins="log", cmap="Greys", mincnt=1, extent=(0, 1, 0, 1), linewidths=0)
            ax.plot([0, 1], [0, 1], ls=":", color=PALETTE["rose"], lw=0.9)
            txt = f"all entries: r {S['r']:.2f}; within junction {S['wj']:.2f}; within cell {S['wc']:.2f}"
            tab.append(dict(level=lev, method=key, subset="all", **S))
            if SL: txt += f"\nno reads left ({100*SL['n']/S['n']:.0f}%): r {SL['r']:.2f}; within junction {SL['wj']:.2f}; within cell {SL['wc']:.2f}"; tab.append(dict(level=lev, method=key, subset="no_reads_left", **SL))
            ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=4.6, color=col)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
            if r_ == 0: ax.set_title(f"{lev}% of reads removed" if lev else "full coverage", fontsize=7)
            if c == 0: ax.set_ylabel(lab + "\nimputed PSI", fontsize=6.5)
            if r_ == len(rows) - 1: ax.set_xlabel("observed PSI at full coverage\n(held-out cells, $\\geq$5 reads)", fontsize=6.5)
            despine(ax)
    for ax, l in zip(axes[:, 0], "abc"): panel_label(ax, l)
    fig.suptitle("All 39,812 held-out cells (70/30 fit), HVJ junctions: imputed vs observed junction usage", fontsize=8, y=0.995)
    fig.tight_layout()
    pd.DataFrame(tab).to_csv(f"{D}/summary_stats.tsv", sep="\t", index=False); print(pd.DataFrame(tab).round(3).to_string())
    save(fig, os.path.join(FIGROOT, "FIGURE2_simulation_imputation", "plots", "SANITY_imputed_vs_raw_heldout"))


if __name__ == "__main__":
    main()
