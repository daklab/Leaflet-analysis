#!/usr/bin/env python
"""Supplementary: per-ATSE effect sizes, LeafletFA posterior TV against LeafCutter's pseudobulk TV,
for six contrasts, and both against the observed pooled TV of the same cells (A65).

Why three columns: LeafCutter's TV is its fitted deltapsi on 7 v 8 pseudobulks (shrunk toward zero
when the replicates are noisy); the posterior TV is the model's imputed group means (can be far
from the cells when the programs are pinned elsewhere -- Gna13). The observed pooled TV is the
neutral reference both are trying to estimate. Coverage-gated (>= 25 covered cells / arm).
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
FIGROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, FIGROOT); sys.path.insert(0, os.path.join(FIGROOT, "_common"))
from leaflet_theme import apply_theme, PALETTE, METHOD_COLORS, figsize, panel_label, despine, save

M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
O = f"{M}/LEAFCUTTER_CMP_DELTAONLY"
LFA, LC, GREY, OLIVE = METHOD_COLORS["LeafletFA"], METHOD_COLORS["NMF"], PALETTE["grey"], PALETTE["olive"]

CONTRASTS = [
    "Brain_Non-Myeloid_ENDOTHELIAL_CELL__vs__Brain_Non-Myeloid_OLIGODENDROCYTE",
    "Heart_ENDOTHELIAL_CELL__vs__Heart_FIBROBLAST",
    "Lung_FIBROBLAST__vs__Lung_SMOOTH_MUSCLE_CELL",
    "Marrow_B_CELL__vs__Marrow_MONOCYTE",
    "Skin_BASAL_CELL__vs__Skin_KERATINOCYTE",
    "Large_Intestine_EPITHELIAL_CELL__vs__Large_Intestine_INTESTINAL_CELL",
]


def short(c):
    a, b = c.split("__vs__")
    t = a.split("_")[0] if not a.startswith(("Brain", "Large")) else a.split("_")[0] + ("" if a.startswith("Brain") else " intestine")
    f = lambda s: (s.split("_", 1)[-1] if not s.startswith("Large") else s.split("_", 2)[-1]).replace("Non-Myeloid_", "").replace("_", " ").lower()
    return f"{t}: {f(a)} vs {f(b)}"


def main():
    apply_theme()
    g = pd.read_csv(f"{O}/analysis_matched/posterior_consensus_gated.tsv.gz", sep="\t")
    d = pd.read_csv(f"{O}/analysis_consistency/data_consistency.tsv.gz", sep="\t",
                    usecols=["contrast", "cluster", "obs_tv_pooled"])
    g = g.merge(d, on=["contrast", "cluster"], how="left")
    g = g[(g.mincov >= 25) & g.tv_lc.notna()]
    cats = np.where(g.lfa_cons & g.lc_sig, "both", np.where(g.lfa_cons, "LeafletFA only",
                    np.where(g.lc_sig, "LeafCutter only", "neither")))
    g = g.assign(cat=cats)
    col = {"both": OLIVE, "LeafletFA only": LFA, "LeafCutter only": LC, "neither": GREY}
    fig, axes = plt.subplots(len(CONTRASTS), 3, figsize=figsize("double", h=1.4 * len(CONTRASTS)))
    for i, c in enumerate(CONTRASTS):
        s = g[g.contrast == c]
        if len(s) == 0:
            for ax in axes[i]: ax.axis("off")
            continue
        for k, (x, y, xl, yl) in enumerate([
                ("tv_lc", "tv", "LeafCutter TV (pseudobulk)", "LeafletFA posterior TV"),
                ("obs_tv_pooled", "tv_lc", "observed pooled TV", "LeafCutter TV (pseudobulk)"),
                ("obs_tv_pooled", "tv", "observed pooled TV", "LeafletFA posterior TV")]):
            ax = axes[i, k]
            for cat in ["neither", "LeafCutter only", "LeafletFA only", "both"]:
                ss = s[s.cat == cat]
                ax.scatter(ss[x], ss[y], s=3 if cat == "neither" else 6, color=col[cat], alpha=0.35 if cat == "neither" else 0.7,
                           lw=0, label=f"{cat} ({len(ss):,})" if i == 0 and k == 0 else None)
            ax.plot([0, 1], [0, 1], ls=":", color=GREY, lw=0.8)
            ax.axhline(0.15, color=LFA if "posterior" in yl else GREY, lw=0.5, ls="--")
            ok = np.isfinite(s[x]) & np.isfinite(s[y])
            rho = spearmanr(s.loc[ok, x], s.loc[ok, y]).correlation
            ax.text(0.03, 0.95, f"Spearman {rho:.2f}\nn = {ok.sum():,}", transform=ax.transAxes, va="top", fontsize=6)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel(xl, fontsize=6.5); ax.set_ylabel(yl, fontsize=6.5)
            if k == 0: ax.set_title(short(c), fontsize=7, loc="left")
            despine(ax)
    axes[0, 0].legend(frameon=False, fontsize=5.5, loc="lower right", markerscale=2)
    fig.suptitle("Per-ATSE effect sizes by contrast (>= 25 covered cells per arm; dashed = delta 0.15)", fontsize=8, y=0.995)
    fig.tight_layout()
    out = os.path.join(FIGROOT, "FIGURE3_leafcutter_ds", "plots", "FIG3_supp_effect_sizes_by_contrast")
    save(fig, out); print("->", out)


if __name__ == "__main__":
    main()
