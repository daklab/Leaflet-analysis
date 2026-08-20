#!/usr/bin/env python
"""Fig 4 worked example: Kcnip3 ATSE 2 in marrow hematopoietic progenitors -- junction usage moves
with age while the gene's expression does not. Chosen by A69b (screen of the aging program's top-200
junctions over all TMS tissue-cell-types with >= 25 covered cells at each of 3/18/24m; largest
monotone PSI shift with |rho_GE| < 0.1).
a  gene model with the ATSE's junctions as arcs
b  per-age violins of the changing junction's usage over covered cells (median/IQR), pooled PSI marked
c  the gene's log-normalised expression by age in the same cells -- flat
"""
import os, sys, importlib.util
import numpy as np, pandas as pd, h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
HERE = os.path.dirname(os.path.abspath(__file__)); FIGROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, FIGROOT); sys.path.insert(0, os.path.join(FIGROOT, "_common"))
from leaflet_theme import apply_theme, PALETTE, figsize, panel_label, despine, save
_spec = importlib.util.spec_from_file_location("viz", "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/08_leafcutter_ds/04_visualize_ds_atse.py")
viz = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(viz)
import gffutils
M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
EV = "ENSMUSG00000079056.12_atse_2"; GENE = "Kcnip3"; TCT = "Marrow_HEMATOPOIETIC PROGENITOR"; JFOCUS = "chr2_127465978_127482004_-"
AGES = [3, 18, 24]; ACOL = {3: PALETTE["teal"], 18: PALETTE["ochre"], 24: PALETTE["rose"]}


def cat(g):
    if hasattr(g, "keys") and "categories" in g:
        c = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]); return c[g["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])


def arc(ax, x0, x1, h, color, lw, label=None):
    ax.add_patch(mpatches.PathPatch(Path([(x0, 0), ((x0 + x1) / 2, h), (x1, 0)],
                 [Path.MOVETO, Path.CURVE3, Path.CURVE3]), fc="none", ec=color, lw=lw, zorder=4))
    if label: ax.text((x0 + x1) / 2, h * 0.7, label, ha="center", va="bottom", fontsize=6, color=color)


def main():
    apply_theme()
    with h5py.File(f"{M}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad", "r") as f:
        ev = cat(f["var"]["event_id"]); jid = cat(f["var"]["junction_id"])
        cols = np.flatnonzero(ev == EV); jn = jid[cols]
        obs = {k: cat(f["obs"][k]) for k in ["dataset", "age", "tissue_cell_type"]}
        rows = np.flatnonzero((obs["dataset"] == "tabula_muris_senis") & (obs["tissue_cell_type"] == TCT))
        gl = f["layers"]["cell_by_junction_matrix"]; ip = gl["indptr"][:]
        Y = np.zeros((len(rows), len(cols)))
        for i, r in enumerate(rows):
            s, e = int(ip[r]), int(ip[r + 1]); idx = gl["indices"][s:e]; dat = gl["data"][s:e]
            m = np.isin(idx, cols)
            if m.any(): Y[i, np.searchsorted(cols, idx[m])] = dat[m]
    with h5py.File(f"{M}/MODEL_INPUT/20260526/ge_adata_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5ad", "r") as f:
        gn = cat(f["var"]["gene_name"]); gi = int(np.flatnonzero(gn == GENE)[0])
        X = f["layers"]["log_norm"]
        if hasattr(X, "keys"):
            import scipy.sparse as sp
            ge = sp.csr_matrix((X["data"][:], X["indices"][:], X["indptr"][:]))[rows, gi].toarray().ravel()
        else:
            ge = X[rows, gi]
    age = pd.Series(obs["age"][rows]).str.replace("m", "").astype(int).values
    jF = int(np.flatnonzero(jn == JFOCUS)[0]); Nt = Y.sum(1); cov = Nt > 0
    psi = np.where(cov, Y[:, jF] / np.maximum(Nt, 1), np.nan)
    fig = plt.figure(figsize=figsize("double", h=4.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.4], hspace=0.55, wspace=0.3)
    ax = fig.add_subplot(gs[0, :])
    parts = [j.split("_") for j in jn]; chrom = parts[0][0]; strand = parts[0][3]
    st = np.array([int(p[1]) for p in parts]); en = np.array([int(p[2]) for p in parts])
    lo, hi = st.min() - 2000, en.max() + 2000
    tx = viz.get_transcripts(gffutils.FeatureDB(viz.DEFAULT_DB), chrom, lo, hi, strand, max_tx=4)
    ybot = viz.draw_exons(ax, tx, lo, hi) if tx else -0.7
    for k in range(len(cols)):
        arc(ax, st[k], en[k], 1.5 + 0.5 * (k % 2), PALETTE["indigo"] if k != jF else PALETTE["teal"], 1.6,
            f"J{k+1}" + (" (aging)" if k == jF else ""))
    ax.set_xlim(lo, hi); ax.set_ylim(ybot - 0.4, 2.8); ax.set_yticks([])
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.set_xlabel(f"{chrom} ({strand})", fontsize=6); ax.ticklabel_format(axis="x", style="plain", useOffset=False); ax.tick_params(axis="x", labelsize=5.5)
    ax.set_title(f"$Kcnip3$ ATSE 2 in marrow hematopoietic progenitors", fontsize=8, loc="left"); panel_label(ax, "a", dx=-0.02)
    ax = fig.add_subplot(gs[1, 0])
    for i, a_ in enumerate(AGES):
        v = psi[(age == a_) & cov]; v = v[np.isfinite(v)]
        vp = ax.violinplot([v], positions=[i], widths=0.8, showextrema=False, points=50)
        for b in vp["bodies"]: b.set_facecolor(ACOL[a_]); b.set_alpha(0.55); b.set_edgecolor("none")
        rng = np.random.default_rng(a_); ax.scatter(i + rng.normal(0, 0.06, len(v)), v, s=3, color=ACOL[a_], alpha=0.6, lw=0)
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        ax.hlines(q2, i - 0.3, i + 0.3, color="black", lw=1.3, zorder=5); ax.hlines([q1, q3], i - 0.18, i + 0.18, color="black", lw=0.7, zorder=5)
        pooled = Y[(age == a_) & cov, jF].sum() / Nt[(age == a_) & cov].sum()
        ax.scatter(i + 0.38, pooled, s=26, marker="D", color="black", zorder=6, label="pooled PSI" if i == 0 else None)
        ax.text(i, -0.12, f"n={len(v)}", ha="center", fontsize=5)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{a}m" for a in AGES]); ax.set_ylim(-0.18, 1.05)
    ax.set_ylabel("J1 usage per covered cell"); ax.legend(frameon=False, fontsize=5)
    ax.set_title("junction usage falls with age\n(pooled 0.35 / 0.12 / 0.03)", fontsize=7, loc="left"); despine(ax); panel_label(ax, "b")
    ax = fig.add_subplot(gs[1, 1])
    for i, a_ in enumerate(AGES):
        v = ge[age == a_]
        vp = ax.violinplot([v], positions=[i], widths=0.8, showextrema=False, points=50)
        for b in vp["bodies"]: b.set_facecolor(ACOL[a_]); b.set_alpha(0.55); b.set_edgecolor("none")
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        ax.hlines(q2, i - 0.3, i + 0.3, color="black", lw=1.3, zorder=5); ax.hlines([q1, q3], i - 0.18, i + 0.18, color="black", lw=0.7, zorder=5)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{a}m" for a in AGES])
    ax.set_ylabel(f"$Kcnip3$ expression (log-norm)")
    ax.set_title("gene expression does not change\n(Spearman vs age 0.03)", fontsize=7, loc="left"); despine(ax); panel_label(ax, "c")
    save(fig, os.path.join(HERE, "..", "plots", "FIG4_example_Kcnip3")); print("-> FIG4_example_Kcnip3")


if __name__ == "__main__":
    main()
