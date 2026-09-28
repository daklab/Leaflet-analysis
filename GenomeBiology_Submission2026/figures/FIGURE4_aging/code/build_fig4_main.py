#!/usr/bin/env python
"""Figure 4 main, v2 layout (Karin 2026-08-19): titles stripped to a few words (full text in
CAPTION_FIG4.md); the delta circle heatmap gets the full middle row; the top-junction bar is compact
(old-figure style: direction-coloured, gene + coordinates); the Kcnip3 worked example (genomic
context + usage by age) is folded into the main figure.
Layout: row 1 a ANOVA | b aging violin; row 2 c delta circle heatmap (full width);
row 3 d top junctions | e Kcnip3 gene model | f Kcnip3 usage by age + GE inset."""
import os, sys, importlib.util
import numpy as np, pandas as pd, h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
HERE = os.path.dirname(os.path.abspath(__file__)); FIGROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, FIGROOT); sys.path.insert(0, os.path.join(FIGROOT, "_common"))
sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa")
sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from leaflet_theme import apply_theme, PALETTE, figsize, panel_label, despine, save
from programs import sp_rank
from _phi_rows import phi_row_of
_spec = importlib.util.spec_from_file_location("viz", "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/08_leafcutter_ds/04_visualize_ds_atse.py")
viz = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(viz)
import gffutils
M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
FIT = f"{M}/ATLAS_DM/runs_final_deltaonly/dm_simplex_K20_seed0_b8192/fit.npz"
AGING_FIT_PROG = 6
EV = "ENSMUSG00000079056.12_atse_2"; TCT = "Marrow_HEMATOPOIETIC PROGENITOR"; JFOCUS = "chr2_127465978_127482004_-"
AGES = [3, 18, 24]; ACOL = {3: PALETTE["teal"], 18: PALETTE["ochre"], 24: PALETTE["rose"]}


def cat(g):
    if hasattr(g, "keys") and "categories" in g:
        c = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]); return c[g["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])


def arc(ax, x0, x1, h, color, lw, label=None):
    ax.add_patch(mpatches.PathPatch(Path([(x0, 0), ((x0 + x1) / 2, h), (x1, 0)],
                 [Path.MOVETO, Path.CURVE3, Path.CURVE3]), fc="none", ec=color, lw=lw, zorder=4))
    if label: ax.text((x0 + x1) / 2, h * 0.72, label, ha="center", va="bottom", fontsize=5.5, color=color)


def main():
    apply_theme()
    z = np.load(FIT, allow_pickle=True); phi = np.asarray(z["phi"]); K = phi.shape[1]; spr = sp_rank(z)
    aging_sp = f"SP{spr[AGING_FIT_PROG]}"
    with h5py.File(f"{M}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad", "r") as f:
        obs = {k: cat(f["obs"][k]) for k in ["dataset", "age", "sex", "mouse.id", "tissue_cell_type"]}
    prow = phi_row_of(z, len(obs["age"]))
    tms = (obs["dataset"] == "tabula_muris_senis") & (prow >= 0)
    df = pd.DataFrame({k.replace(".", "_"): v for k, v in obs.items()})[tms].reset_index(drop=True)
    df["age_m"] = df.age.str.replace("m", "").astype(int)
    P = phi[prow[np.flatnonzero(tms)]]

    fig = plt.figure(figsize=figsize("double", h=8.6))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.35, 1.0], hspace=0.42, wspace=0.9)

    # a: variance partition (precomputed table)
    ax = fig.add_subplot(gs[0, :3])
    va = pd.read_csv(os.path.join(HERE, "..", "data", "data_variance_partition.tsv"), sep="\t", index_col=0)
    cols = [c for c in va.columns if c != "R2"]
    V = va[cols].values
    im = ax.imshow(V, aspect="auto", cmap="Greens", vmin=0, vmax=max(0.2, np.nanmax(V)))
    for r_ in range(V.shape[0]):
        for c_ in range(V.shape[1]):
            if V[r_, c_] >= 0.05:
                ax.text(c_, r_, f"{V[r_, c_]:.2f}", ha="center", va="center", fontsize=4,
                        color="white" if V[r_, c_] > 0.5 * np.nanmax(V) else "black")
    ax.set_yticks(range(len(va))); ax.set_yticklabels([f"{i} ({va.loc[i,'R2']:.2f})" for i in va.index], fontsize=4.2)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("C(tissue_cell_type)", "cell type").replace("C(sex)", "sex").replace("age_m", "age") for c in cols], rotation=30, ha="right", fontsize=5.5)
    plt.colorbar(im, ax=ax, fraction=0.035).ax.tick_params(labelsize=5)
    ax.set_title("variance partition", fontsize=7, loc="left"); panel_label(ax, "a")

    # b: aging SP by age
    ax = fig.add_subplot(gs[0, 3:])
    df["p"] = P[:, AGING_FIT_PROG]
    for i, a_ in enumerate(AGES):
        v = df[df.age_m == a_].p.values
        vp = ax.violinplot([v], positions=[i], widths=0.8, showextrema=False, points=60)
        for b in vp["bodies"]: b.set_facecolor(ACOL[a_]); b.set_alpha(0.5); b.set_edgecolor("none")
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        ax.hlines(q2, i - 0.3, i + 0.3, color="black", lw=1.4, zorder=5)
        ax.hlines([q1, q3], i - 0.18, i + 0.18, color="black", lw=0.7, zorder=5)
        for sex, mk in [("male", "o"), ("female", "^")]:
            mice = df[(df.age_m == a_) & (df.sex == sex)].groupby("mouse_id").p.mean()
            if not len(mice): continue
            rng = np.random.default_rng(a_ + (0 if sex == "male" else 7))
            ax.scatter(i + rng.normal(0, 0.06, len(mice)), mice.values, s=13, marker=mk,
                       facecolor="white", edgecolor="black", lw=0.8, zorder=6,
                       label=sex if i == 0 else None)
    ax.set_xticks(range(3)); ax.set_xticklabels(["3m", "18m", "24m"])
    ax.set_ylabel(f"{aging_sp} activity"); ax.set_title(f"the aging program ({aging_sp})", fontsize=7, loc="left")
    ax.legend(frameon=False, fontsize=5, title="per-mouse mean", title_fontsize=5); despine(ax); panel_label(ax, "b")

    # c: delta circle heatmap, full width
    ax = fig.add_subplot(gs[1, :])
    de = pd.read_csv(f"{M}/ATLAS_DM/aging/a69_dm_simplex_K20_seed0_b8192_delta_bytype.tsv", sep="\t")
    med = df.groupby("tissue_cell_type").apply(lambda s_: pd.Series(np.median(P[s_.index], 0))).rename(columns=lambda k: f"SP{spr[k]}")
    piv = de.pivot(index="tct", columns="SP", values="delta")[[f"SP{i+1}" for i in range(K)]]
    piv = piv.loc[piv[aging_sp].sort_values().index]
    piv = piv.T                                              # SPs as rows, cell types as columns -> wide
    v = np.nanmax(np.abs(piv.values))
    im = ax.imshow(piv.values, aspect="auto", cmap="BrBG", vmin=-v, vmax=v)
    for r_ in range(piv.shape[0]):
        for c_, t in enumerate(piv.columns):
            mval = med.loc[t, piv.index[r_]] if t in med.index else 0
            ax.scatter(c_, r_, s=max(1.2, (np.log10(mval + 0.01) + 2) * 11), facecolor="none", edgecolor="black", lw=0.3, zorder=5)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([t.replace("_", " ") for t in piv.columns], rotation=65, ha="right", fontsize=3.8)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=4.5)
    cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.set_label("$\\Delta$ activity (24m$-$3m)", fontsize=5.5); cb.ax.tick_params(labelsize=5)
    ax.set_title("aging shift by tissue-cell-type (circle = median activity)", fontsize=7, loc="left"); panel_label(ax, "c")

    # d: compact top junctions, direction-coloured
    ax = fig.add_subplot(gs[2, :2])
    tj = pd.read_csv(f"{M}/ATLAS_DM/aging/a69_dm_simplex_K20_seed0_b8192_top_junctions.tsv", sep="\t").head(14)[::-1]
    cols_ = np.where(tj.dpsi > 0, PALETTE["rose"], PALETTE["indigo"])
    ax.barh(range(len(tj)), tj.dpsi, color=cols_, height=0.7)
    ax.set_yticks(range(len(tj)))
    ax.set_yticklabels([f"{g}  {j.replace('_',':',1).rsplit('_',1)[0]}" for g, j in zip(tj.gene, tj.junction)], fontsize=4.2)
    ax.axvline(0, color=PALETTE["grey"], lw=0.6)
    ax.set_xlabel("$\\Delta\\psi$ vs other SPs", fontsize=6)
    ax.legend(handles=[mpatches.Patch(color=PALETTE["rose"], label="higher in aging SP"),
                       mpatches.Patch(color=PALETTE["indigo"], label="lower")], frameon=False, fontsize=4.5, loc="lower right")
    despine(ax); panel_label(ax, "d")

    # e/f: Kcnip3 example
    with h5py.File(f"{M}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad", "r") as f:
        ev = cat(f["var"]["event_id"]); jid = cat(f["var"]["junction_id"])
        cols2 = np.flatnonzero(ev == EV); jn = jid[cols2]
        rows = np.flatnonzero((obs["dataset"] == "tabula_muris_senis") & (obs["tissue_cell_type"] == TCT))
        gl = f["layers"]["cell_by_junction_matrix"]; ip = gl["indptr"][:]
        Y = np.zeros((len(rows), len(cols2)))
        for i, r in enumerate(rows):
            s, e = int(ip[r]), int(ip[r + 1]); idx = gl["indices"][s:e]; dat = gl["data"][s:e]
            m = np.isin(idx, cols2)
            if m.any(): Y[i, np.searchsorted(cols2, idx[m])] = dat[m]
    with h5py.File(f"{M}/MODEL_INPUT/20260526/ge_adata_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5ad", "r") as f:
        gn = cat(f["var"]["gene_name"]); gi = int(np.flatnonzero(gn == "Kcnip3")[0])
        X = f["layers"]["log_norm"]
        if hasattr(X, "keys"):
            import scipy.sparse as sp
            ge = sp.csr_matrix((X["data"][:], X["indices"][:], X["indptr"][:]))[rows, gi].toarray().ravel()
        else:
            ge = X[rows, gi]
    age = pd.Series(obs["age"][rows]).str.replace("m", "").astype(int).values
    jF = int(np.flatnonzero(jn == JFOCUS)[0]); Nt = Y.sum(1); covd = Nt > 0
    psi = np.where(covd, Y[:, jF] / np.maximum(Nt, 1), np.nan)
    ax = fig.add_subplot(gs[2, 2:4])
    parts = [j.split("_") for j in jn]; chrom = parts[0][0]; strand = parts[0][3]
    st = np.array([int(p[1]) for p in parts]); en = np.array([int(p[2]) for p in parts])
    lo, hi = st.min() - 2000, en.max() + 2000
    tx = viz.get_transcripts(gffutils.FeatureDB(viz.DEFAULT_DB), chrom, lo, hi, strand, max_tx=3)
    ybot = viz.draw_exons(ax, tx, lo, hi) if tx else -0.7
    for k in range(len(cols2)):
        arc(ax, st[k], en[k], 1.4 + 0.5 * (k % 2), PALETTE["teal"] if k == jF else PALETTE["grey"], 1.5,
            "aging junction" if k == jF else None)
    ax.set_xlim(lo, hi); ax.set_ylim(ybot - 0.4, 2.6); ax.set_yticks([])
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.set_xlabel(f"{chrom} ({strand})", fontsize=5.5); ax.ticklabel_format(axis="x", style="plain", useOffset=False); ax.tick_params(axis="x", labelsize=4.5)
    ax.set_title("$Kcnip3$ (marrow HSC)", fontsize=7, loc="left"); panel_label(ax, "e")
    ax = fig.add_subplot(gs[2, 4:])
    for i, a_ in enumerate(AGES):
        v = psi[(age == a_) & covd]; v = v[np.isfinite(v)]
        vp = ax.violinplot([v], positions=[i], widths=0.8, showextrema=False, points=50)
        for b in vp["bodies"]: b.set_facecolor(ACOL[a_]); b.set_alpha(0.55); b.set_edgecolor("none")
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        ax.hlines(q2, i - 0.3, i + 0.3, color="black", lw=1.2, zorder=5); ax.hlines([q1, q3], i - 0.18, i + 0.18, color="black", lw=0.6, zorder=5)
        pooled_ = Y[(age == a_) & covd, jF].sum() / Nt[(age == a_) & covd].sum()
        ax.scatter(i + 0.36, pooled_, s=22, marker="D", color="black", zorder=6)
        ax.text(i, -0.13, f"{len(v)}", ha="center", fontsize=4.5)
    ins = ax.inset_axes([0.55, 0.55, 0.44, 0.42])
    for i, a_ in enumerate(AGES):
        v = ge[age == a_]
        vp = ins.violinplot([v], positions=[i], widths=0.8, showextrema=False, points=40)
        for b in vp["bodies"]: b.set_facecolor(ACOL[a_]); b.set_alpha(0.55); b.set_edgecolor("none")
        ins.hlines(np.median(v), i - 0.28, i + 0.28, color="black", lw=0.9, zorder=5)
    ins.set_xticks(range(3)); ins.set_xticklabels(["3", "18", "24"], fontsize=4); ins.tick_params(labelsize=4)
    ins.set_title("expression (flat)", fontsize=4.5); despine(ins)
    ax.set_xticks(range(3)); ax.set_xticklabels(["3m", "18m", "24m"]); ax.set_ylim(-0.2, 1.05)
    ax.set_ylabel("aging-junction usage", fontsize=6)
    ax.set_title("usage falls with age; expression flat", fontsize=7, loc="left")
    despine(ax); panel_label(ax, "f")
    save(fig, os.path.join(HERE, "..", "plots", "FIG4_main")); print("-> FIG4_main")


if __name__ == "__main__":
    main()
