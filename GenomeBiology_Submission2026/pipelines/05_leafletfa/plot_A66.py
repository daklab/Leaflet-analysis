#!/usr/bin/env python
"""Plot A66: imputed vs raw PSI on held-out cells, full and reduced coverage.

Rows: pure prior (phi . psi) and DM posterior mean (prior updated by the cell's own reads).
Columns: 0 / 10 / 50 / 90 % of reads removed. x = observed PSI at FULL coverage (entries with
>= 5 original reads so the truth is not a coin flip), y = imputed at that level. Two Pearsons per
panel: over all entries (mostly between-junction level, easy) and the median WITHIN junction across
cells (ranking cells, hard). Bottom row: within-junction Pearson distributions per level.
"""
import os, sys, glob
import numpy as np, matplotlib.pyplot as plt
FIGROOT = "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures"
sys.path.insert(0, FIGROOT); sys.path.insert(0, os.path.join(FIGROOT, "_common"))
from leaflet_theme import apply_theme, PALETTE, METHOD_COLORS, figsize, panel_label, despine, save
M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
D = f"{M}/IMPUTATION_DM/a66_imputed_vs_raw"
LFA, GREY = METHOD_COLORS["LeafletFA"], PALETTE["grey"]


def within_junction_r(junc, obs, imp, min_cells=30):
    o = np.argsort(junc, kind="stable"); j = junc[o]; x = obs[o]; y = imp[o]
    b = np.flatnonzero(np.r_[True, j[1:] != j[:-1], True]); rs = []
    for s, e in zip(b[:-1], b[1:]):
        if e - s < min_cells: continue
        xs, ys = x[s:e], y[s:e]
        if xs.std() < 1e-6 or ys.std() < 1e-6: continue
        rs.append(np.corrcoef(xs, ys)[0, 1])
    return np.array(rs)


def main():
    apply_theme()
    files = sorted(glob.glob(f"{D}/level_*.npz"))
    levels = [np.load(f) for f in files]
    fig, axes = plt.subplots(3, len(levels), figsize=figsize("double", h=6.4))
    for c, z in enumerate(levels):
        lev = int(z["level"]); keep = z["N_orig"] >= 5
        obs = z["obs_full"][keep]; junc = z["junc"][keep]; nl = z["N_level"][keep]
        for r, (key, lab) in enumerate([("pure", "pure prior $\\phi\\cdot\\psi$"), ("post", "DM posterior mean\n(prior + the cell's own reads)")]):
            ax = axes[r, c]; imp = z[key][keep]
            ax.hexbin(obs, imp, gridsize=40, bins="log", cmap="Greys", mincnt=1, extent=(0, 1, 0, 1), linewidths=0)
            ax.plot([0, 1], [0, 1], ls=":", color=PALETTE["rose"], lw=0.9)
            r_all = np.corrcoef(obs, imp)[0, 1]; wr = within_junction_r(junc, obs, imp)
            lost = nl == 0
            txt = f"r all = {r_all:.2f}\nmedian r within junction = {np.median(wr):.2f}"
            if lost.any() and key == "post":
                txt += f"\n{100*lost.mean():.0f}% of entries have no reads left"
            ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=5.5)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
            if r == 0: ax.set_title(f"{lev}% of reads removed" if lev else "full coverage", fontsize=7)
            if c == 0: ax.set_ylabel(lab + "\nimputed PSI", fontsize=6.5)
            if r == 1: ax.set_xlabel("observed PSI at full coverage\n(held-out cells, $\\geq$5 reads)", fontsize=6.5)
            despine(ax)
        ax = axes[2, c]
        for key, col, lab in [("pure", GREY, "pure prior"), ("post", LFA, "posterior mean")]:
            wr = within_junction_r(junc, obs, z[key][keep])
            ax.hist(wr, bins=np.linspace(-1, 1, 41), histtype="step", color=col, lw=1.2, label=f"{lab} (median {np.median(wr):.2f})", density=True)
        # posterior mean on the entries whose reads were REMOVED at this level -- what imputation has to do
        if lev:
            lost = nl == 0
            if lost.sum() > 1000:
                wr = within_junction_r(junc[lost], obs[lost], z["post"][keep][lost])
                ax.hist(wr, bins=np.linspace(-1, 1, 41), histtype="step", color=PALETTE["ochre"], lw=1.2, ls="--",
                        label=f"posterior mean, entries with\nno reads left (median {np.median(wr):.2f})", density=True)
        ax.axvline(0, color=GREY, lw=0.5, ls=":")
        ax.set_xlabel("within-junction Pearson\n(imputed vs observed, across cells)", fontsize=6.5)
        if c == 0: ax.set_ylabel("density of junctions", fontsize=6.5)
        ax.legend(frameon=False, fontsize=5, loc="upper left"); despine(ax)
    for ax, l in zip(axes[:, 0], "abc"): panel_label(ax, l)
    fig.suptitle(f"Held-out cells ({len(np.load(f'{D}/cells.npy')):,}, 70/30 fit): imputed vs observed junction usage", fontsize=8, y=0.995)
    fig.tight_layout()
    out = os.path.join(FIGROOT, "FIGURE2_simulation_imputation", "plots", "SANITY_imputed_vs_raw_heldout")
    save(fig, out); print("->", out)


if __name__ == "__main__":
    main()
