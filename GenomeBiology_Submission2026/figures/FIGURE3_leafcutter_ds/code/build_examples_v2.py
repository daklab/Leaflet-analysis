#!/usr/bin/env python
"""Figure 3 worked examples, v2: genomic context + per-junction usage, single cells vs pseudobulk.

Replaces build_examples.py after the Usp54 lesson: a "LeafletFA only" event with 14 covered cells
was the model imputing, not the data speaking. Every example here is gated at >= 50 covered cells in
each arm, so what is shown is what the cells actually do.

Two events, same layout per event:
  row 1  gene model: transcripts as exon boxes, the ATSE's junctions as arcs, one arc per junction,
         labelled J1..Jm. This is the "what is being spliced" panel.
  row 2  per-junction usage: for each junction, the distribution across SINGLE CELLS in each cell type
         (violin over covered cells) with the PSEUDOBULK per-mouse values overlaid as points. One
         panel per junction. This is where the single-cell vs pseudobulk contrast is visible: many
         cells at 0 and 1 with a mean in between (bimodal, cell-level commitment) versus a handful
         of pseudobulks near the mean.
The posterior TV histogram from v1 is dropped: it repeats the P(TV > delta) number and adds little.

  Apbb2  ATSE 3   endothelial vs oligodendrocyte, brain    found by BOTH   (LC q 3e-18)
  Gna13  ATSE 1   endothelial vs fibroblast, heart         LeafletFA ONLY  (LC q 0.63) -- and the
                  same event is LeafletFA-only in four other tissues' mesenchymal-vs-immune contrasts
"""
import os, sys, re
import numpy as np, pandas as pd, h5py, scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import gffutils

HERE = os.path.dirname(os.path.abspath(__file__))
FIGROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, FIGROOT); sys.path.insert(0, os.path.join(FIGROOT, "_common"))
sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/08_leafcutter_ds")
from leaflet_theme import apply_theme, PALETTE, METHOD_COLORS, figsize, panel_label, despine, save
import importlib.util
_spec = importlib.util.spec_from_file_location("viz", "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/08_leafcutter_ds/04_visualize_ds_atse.py")
viz = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(viz)

M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
H5 = f"{M}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad"
LCROOT = f"{M}/LEAFCUTTER_DS_2026-08/by_pair"
DB = gffutils.FeatureDB(viz.DEFAULT_DB)
C1, C2 = PALETTE["teal"], PALETTE["ochre"]

EXAMPLES = [
    dict(gene="Apbb2", event="ENSMUSG00000029207.16_atse_3", tag="found by both",
         g1="Brain_Non-Myeloid_ENDOTHELIAL CELL", g2="Brain_Non-Myeloid_OLIGODENDROCYTE",
         lab1="endothelial", lab2="oligodendrocyte",
         lcdir="Brain_Non-Myeloid_ENDOTHELIAL_CELL__vs__Brain_Non-Myeloid_OLIGODENDROCYTE"),
    dict(gene="Gna13", event="ENSMUSG00000020611.14_atse_1", tag="LeafletFA only",
         g1="Heart_ENDOTHELIAL CELL", g2="Heart_FIBROBLAST",
         lab1="endothelial", lab2="fibroblast",
         lcdir="Heart_ENDOTHELIAL_CELL__vs__Heart_FIBROBLAST"),
]


def obs_cat(f, name):
    n = f["obs"][name]
    c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
    return c[n["codes"][:]]


def rows_csr(f, layer, rows, cols, J):
    g = f["layers"][layer]; ip = g["indptr"][:]
    out = np.zeros((len(rows), len(cols)), np.float32)
    colpos = {c: i for i, c in enumerate(cols)}
    for k, r in enumerate(rows):
        s, e = int(ip[r]), int(ip[r + 1])
        idx = g["indices"][s:e]; dat = g["data"][s:e]
        for i, d in zip(idx, dat):
            if i in colpos: out[k, colpos[i]] = d
    return out


def arc(ax, x0, x1, h, color, lw, label=None):
    verts = [(x0, 0), ((x0 + x1) / 2, h), (x1, 0)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    ax.add_patch(mpatches.PathPatch(Path(verts, codes), fc="none", ec=color, lw=lw, zorder=4))
    if label:
        ax.text((x0 + x1) / 2, h * 0.62, label, ha="center", va="bottom", fontsize=6.5, color=color, zorder=5)


def main():
    apply_theme()
    with h5py.File(H5, "r") as f:
        ct = obs_cat(f, "tissue_cell_type"); mouse = obs_cat(f, "mouse.id")
        n = f["var"]["event_id"]; c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
        ev = c[n["codes"][:]]; J = len(ev)
        vn = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var"]["_index"][:]])
        data = {}
        for ex in EXAMPLES:
            cols = np.flatnonzero(ev == ex["event"])
            r1 = np.flatnonzero(ct == ex["g1"]); r2 = np.flatnonzero(ct == ex["g2"])
            rows = np.concatenate([r1, r2])
            Jm = rows_csr(f, "cell_by_junction_matrix", rows, cols, J)
            Cm = rows_csr(f, "cell_by_cluster_matrix", rows, cols, J)
            data[ex["event"]] = (cols, r1, r2, Jm, Cm, mouse[rows])

    fig = plt.figure(figsize=figsize("double", h=6.4))
    outer = fig.add_gridspec(2, 1, hspace=0.45)
    letters = iter("abcdefgh")
    for row, ex in enumerate(EXAMPLES):
        cols, r1, r2, Jm, Cm, mice = data[ex["event"]]
        m = len(cols)
        # junction coordinates: var names chrN_start_end_strand
        parts = [vn[i].split("_") for i in cols]
        chrom = parts[0][0]; strand = parts[0][3]
        starts = np.array([int(p[1]) for p in parts]); ends = np.array([int(p[2]) for p in parts])
        order = np.argsort(starts); cols = cols[order]; starts = starts[order]; ends = ends[order]
        Jm = Jm[:, order]; Cm = Cm[:, order]
        lo, hi = starts.min(), ends.max(); pad = int(0.08 * (hi - lo)) + 200
        inner = outer[row].subgridspec(2, m, height_ratios=[1.0, 1.6], hspace=0.55, wspace=0.35)

        # ---- gene model + arcs
        axg = fig.add_subplot(inner[0, :])
        tx = viz.get_transcripts(DB, chrom, lo - pad, hi + pad, strand, max_tx=4)
        ybot = viz.draw_exons(axg, tx, lo - pad, hi + pad) if tx else -0.7
        jcols = [PALETTE["indigo"], PALETTE["olive"], PALETTE["rose"], PALETTE["grey"]] * 3
        for j in range(m):
            arc(axg, starts[j], ends[j], 1.4 + 0.5 * (j % 2), jcols[j], 1.6, label=f"J{j+1}")
        axg.set_xlim(lo - pad, hi + pad); axg.set_ylim(ybot - 0.4, 2.6)
        axg.set_yticks([]); axg.spines[["left", "top", "right"]].set_visible(False)
        axg.set_xlabel(f"{chrom} ({strand})", fontsize=6)
        axg.ticklabel_format(axis="x", style="plain", useOffset=False)
        axg.tick_params(axis="x", labelsize=5.5)
        axg.set_title(f"{ex['gene']} — {ex['tag']}    {ex['lab1']} vs {ex['lab2']}", fontsize=8, loc="left")
        panel_label(axg, next(letters), dx=-0.03)

        # ---- LeafCutter's pseudobulk PSI per mouse
        lcd = f"{LCROOT}/{ex['lcdir']}"
        pb = pd.read_csv(f"{lcd}/counts_subset.tsv.gz", sep="\t", index_col=0)
        grp = pd.read_csv(f"{lcd}/groups.txt", sep="\t", header=None, index_col=0)[1]
        pbe = pb.loc[[i for i in pb.index if ex["event"] in i]]
        # map LC intron rows to our junctions by start/end (LC uses start:end in its id)
        def lc_row_for(s, e):
            for i in pbe.index:
                a, b = i.split(":")[1:3]
                if abs(int(a) - s) <= 1 and abs(int(b) - e) <= 1: return i
            return None
        tot = pbe.sum(0).replace(0, np.nan)
        sig = pd.read_csv(f"{lcd}/ds_results_cluster_significance.txt", sep="\t")
        sig["cluster"] = sig.cluster.str.replace(r"^chr[^:]*:", "", regex=True).str.replace(r"_[+-]$", "", regex=True)
        q_lc = float(sig.loc[sig.cluster == ex["event"], "p.adjust"].iloc[0])

        # ---- per-junction usage panels
        n1 = len(r1); is1 = np.r_[np.ones(n1, bool), np.zeros(len(r2), bool)]
        Ne = Cm[:, 0]; covered = Ne > 0
        for j in range(m):
            ax = fig.add_subplot(inner[1, j])
            psi = np.where(covered, Jm[:, j] / np.maximum(Ne, 1), np.nan)
            parts_v = []
            for k, (msk, col, lab) in enumerate([(is1, C1, ex["lab1"]), (~is1, C2, ex["lab2"])]):
                v = psi[msk & covered]; v = v[np.isfinite(v)]
                if len(v) < 5: continue
                vp = ax.violinplot([v], positions=[k], widths=0.75, showextrema=False, showmeans=False, points=60)
                for b in vp["bodies"]:
                    b.set_facecolor(col); b.set_alpha(0.45); b.set_edgecolor("none")
                # jittered cells, thinned if many
                rng = np.random.default_rng(j)
                sel = v if len(v) <= 400 else rng.choice(v, 400, replace=False)
                ax.scatter(k + rng.normal(0, 0.06, len(sel)), sel, s=2, color=col, alpha=0.35, lw=0, zorder=2)
                ax.hlines(v.mean(), k - 0.32, k + 0.32, color=col, lw=1.6, zorder=4)
                ax.text(k, -0.12, f"n={len(v)}", ha="center", fontsize=5, color=col)
                # pseudobulk overlay
                r = lc_row_for(starts[j], ends[j])
                if r is not None:
                    samp = [s_ for s_ in pbe.columns if str(grp.get(s_, "")).replace(" ", "_") == ex[["g1", "g2"][k]].replace(" ", "_")]
                    pv = (pbe.loc[r, samp] / tot[samp]).values
                    ax.scatter(np.full(len(pv), k + 0.42), pv, s=14, color=col, edgecolor="black", lw=0.5, zorder=5,
                               label="pseudobulk (per mouse)" if (j == 0 and k == 0) else None)
            ax.set_xlim(-0.6, 1.8); ax.set_ylim(-0.18, 1.05)
            ax.set_xticks([0, 1]); ax.set_xticklabels([ex["lab1"], ex["lab2"]], fontsize=5.5, rotation=20)
            ax.set_title(f"J{j+1}", fontsize=7, color=jcols[j], loc="left")
            if j == 0:
                ax.set_ylabel("junction usage\n(single cells, violin;\npseudobulk, dots)", fontsize=6)
                ax.legend(frameon=False, fontsize=5, loc="upper right")
            else:
                ax.set_yticklabels([])
            despine(ax)
        fig.text(0.99, outer[row].get_position(fig).y0 - 0.005,
                 f"LeafCutter q = {q_lc:.2g}", ha="right", fontsize=6.5, color=METHOD_COLORS["NMF"])
    out = os.path.join(FIGROOT, "FIGURE3_leafcutter_ds", "plots", "FIG3_examples_v2")
    save(fig, out); print(f"-> {out}.[pdf|png]")


if __name__ == "__main__":
    main()
