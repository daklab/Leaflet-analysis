#!/usr/bin/env python
"""Plot A66 (v2): imputed vs observed PSI on held-out cells, LeafletFA against NMF, by coverage.

Rows: LeafletFA DM posterior mean (the model's prior phi.psi updated by the cell's own remaining
reads -- what one uses in practice) and NMF at rank 20 on the mean-imputed usage matrix at the same
coverage. Columns: 0 / 10 / 50 / 90% of reads removed. Entries: (cell, junction) pairs that were
COVERED at full coverage with >= 5 original reads (so the truth is not a coin flip); x = observed
PSI at full coverage, y = imputed at that level. Per panel: Pearson and Spearman over all entries
(dominated by between-junction level), and the median within-junction (ranking cells) and
within-cell (ranking junctions) Pearson. At 50/90% a second line gives the same numbers on the
entries whose reads were ALL removed -- the actual imputation task.
"""
import os, sys, glob
import numpy as np, matplotlib.pyplot as plt
from scipy.stats import spearmanr
FIGROOT = "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures"
sys.path.insert(0, FIGROOT); sys.path.insert(0, os.path.join(FIGROOT, "_common"))
from leaflet_theme import apply_theme, PALETTE, METHOD_COLORS, figsize, panel_label, despine, save
M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
D = f"{M}/IMPUTATION_DM/a66_imputed_vs_raw"
LFA, LC, GREY = METHOD_COLORS["LeafletFA"], METHOD_COLORS["NMF"], PALETTE["grey"]


def grouped_r(key, obs, imp, min_n=30):
    o = np.argsort(key, kind="stable"); k = key[o]; x = obs[o]; y = imp[o]
    b = np.flatnonzero(np.r_[True, k[1:] != k[:-1], True]); rs = []
    for s, e in zip(b[:-1], b[1:]):
        if e - s < min_n: continue
        xs, ys = x[s:e], y[s:e]
        if xs.std() < 1e-6 or ys.std() < 1e-6: continue
        rs.append(np.corrcoef(xs, ys)[0, 1])
    return np.median(rs) if rs else np.nan


def stats(obs, imp, junc, cell):
    return dict(r=np.corrcoef(obs, imp)[0, 1], rho=spearmanr(obs, imp).correlation,
                wj=grouped_r(junc, obs, imp), wc=grouped_r(cell, obs, imp))


def main():
    apply_theme()
    files = sorted(glob.glob(f"{D}/level_*.npz"))
    fig, axes = plt.subplots(2, len(files), figsize=figsize("double", h=4.6))
    rows = []
    for c, f in enumerate(files):
        z = np.load(f); lev = int(z["level"]); keep = z["N_orig"] >= 5
        obs, junc, cell, nl = z["obs_full"][keep], z["junc"][keep], z["cell"][keep], z["N_level"][keep]
        nmf_f = f"{D}/nmf_level_{lev:03d}.npz"
        series = [("post", z["post"][keep], "LeafletFA\nDM posterior mean", LFA)]
        if os.path.exists(nmf_f):
            series.append(("nmf", np.load(nmf_f)["nmf"][keep], "NMF (rank 20,\nmean-imputed input)", LC))
        for r_, (key, imp, lab, col) in enumerate(series):
            ax = axes[r_, c]
            ax.hexbin(obs, imp, gridsize=40, bins="log", cmap="Greys", mincnt=1, extent=(0, 1, 0, 1), linewidths=0)
            ax.plot([0, 1], [0, 1], ls=":", color=PALETTE["rose"], lw=0.9)
            st = stats(obs, imp, junc, cell)
            txt = (f"all entries: r {st['r']:.2f}, $\\rho$ {st['rho']:.2f}\n"
                   f"within junction r {st['wj']:.2f}; within cell r {st['wc']:.2f}")
            rows.append(dict(level=lev, method=key, subset="all", n=len(obs), **st))
            lost = nl == 0
            if lost.sum() > 5000:
                sl = stats(obs[lost], imp[lost], junc[lost], cell[lost])
                txt += (f"\nno reads left ({100*lost.mean():.0f}%): r {sl['r']:.2f}, $\\rho$ {sl['rho']:.2f}\n"
                        f"   within junction {sl['wj']:.2f}; within cell {sl['wc']:.2f}")
                rows.append(dict(level=lev, method=key, subset="no_reads_left", n=int(lost.sum()), **sl))
            ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=4.8, color=col)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
            if r_ == 0: ax.set_title(f"{lev}% of reads removed" if lev else "full coverage", fontsize=7)
            if c == 0: ax.set_ylabel(lab + "\nimputed PSI", fontsize=6.5)
            if r_ == len(series) - 1: ax.set_xlabel("observed PSI at full coverage\n(held-out cells, $\\geq$5 reads)", fontsize=6.5)
            despine(ax)
    for ax, l in zip(axes[:, 0], "ab"): panel_label(ax, l)
    fig.suptitle(f"Held-out cells ({len(np.load(f'{D}/cells.npy')):,}, never in training; 70/30 fit): imputed vs observed junction usage", fontsize=8, y=0.995)
    fig.tight_layout()
    import pandas as pd
    pd.DataFrame(rows).to_csv(f"{D}/summary_stats.tsv", sep="\t", index=False)
    print(pd.DataFrame(rows).round(3).to_string())
    out = os.path.join(FIGROOT, "FIGURE2_simulation_imputation", "plots", "SANITY_imputed_vs_raw_heldout")
    save(fig, out); print("->", out)


if __name__ == "__main__":
    main()
