#!/usr/bin/env python
"""Fig 4 supplement, NON-aging panels (aging panels paused pending the confound audit).
Shipped atlas fit (K=20 seed 0), SP-numbered by pi.
a  pi per SP (lollipop) coloured by PVE: share of the across-cell variance of the reconstruction
   rho = phi.psi carried by each program, PVE_k = mean_j Var_c(phi_ck) * (psi_kj - psibar_j)^2 / sum_k(.)
b  SP-SP Spearman correlation of phi across cells, hierarchically clustered
c  median SP activity by broad cell type, Allen Brain vs TMS side by side (types with >= 50 cells in both)
d  top 2 junctions per SP by psi_k - pi-weighted mean of the other SPs (gene + coordinates)
e/f Clstn1 ATSE: genomic context + denoised usage (DM posterior mean) of the 2025 example junction
   chr4:149638395-149643194(+) by broad cell type -- no age contrast."""
import os, sys, importlib.util
import numpy as np, pandas as pd, h5py, torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, leaves_list
HERE = os.path.dirname(os.path.abspath(__file__)); FIGROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, FIGROOT); sys.path.insert(0, os.path.join(FIGROOT, "_common"))
sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa")
sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from leaflet_theme import apply_theme, PALETTE, figsize, panel_label, despine, save
from programs import sp_order
from _phi_rows import phi_row_of
from leafletfa.dirichlet_multinomial import segment_softmax
_spec = importlib.util.spec_from_file_location("viz", "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/08_leafcutter_ds/04_visualize_ds_atse.py")
viz = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(viz)
import gffutils
M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
FIT = f"{M}/ATLAS_DM/runs_final_deltaonly/dm_simplex_K20_seed0_b8192/fit.npz"
H5 = f"{M}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad"
CLSTN1_J = "chr4_149638395_149643194_+"


def cat(g):
    if hasattr(g, "keys") and "categories" in g:
        c = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]); return c[g["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])


def arc(ax, x0, x1, h, color, lw):
    ax.add_patch(mpatches.PathPatch(Path([(x0, 0), ((x0 + x1) / 2, h), (x1, 0)], [Path.MOVETO, Path.CURVE3, Path.CURVE3]), fc="none", ec=color, lw=lw, zorder=4))


def main():
    apply_theme()
    z = np.load(FIT, allow_pickle=True); phi = np.asarray(z["phi"]); K = phi.shape[1]
    order = sp_order(z); P = phi[:, order]; names = [f"SP{i+1}" for i in range(K)]
    inv = np.asarray(z["inv_ev"]).astype(np.int64); fev = np.asarray(z["event_id"]).astype(str)
    psi = segment_softmax(torch.from_numpy(np.asarray(z["psi_loc"], np.float32)), torch.from_numpy(inv), len(fev)).numpy()[order]
    pi = np.asarray(z["trace_pi"])[-1][order]
    with h5py.File(H5, "r") as f:
        obs = {k: cat(f["obs"][k]) for k in ["dataset", "broad_cell_type"]}
        jid = cat(f["var"]["junction_id"]); gene = cat(f["var"]["gene_name"]); ev = cat(f["var"]["event_id"])
    prow = phi_row_of(z, len(obs["dataset"])); tr = prow >= 0
    fig = plt.figure(figsize=figsize("double", h=8.6)); gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.2, 1.1], hspace=0.5, wspace=0.35)
    # a
    ax = fig.add_subplot(gs[0, 0])
    psibar = pi @ psi; pve = np.array([np.mean(P[:, k].var() * (psi[k] - psibar) ** 2) for k in range(K)]); pve /= pve.sum()
    sc = ax.scatter(range(K), pi, c=pve, cmap="viridis", s=30, zorder=3, edgecolor="black", lw=0.4)
    ax.vlines(range(K), 0, pi, color=PALETTE["grey"], lw=0.8)
    ax.set_xticks(range(K)); ax.set_xticklabels(names, rotation=90, fontsize=5); ax.set_ylabel("global importance ($\\pi$)")
    plt.colorbar(sc, ax=ax, fraction=0.04).set_label("PVE", fontsize=6); despine(ax); panel_label(ax, "a")
    # b
    ax = fig.add_subplot(gs[0, 1])
    sub = np.random.default_rng(0).choice(len(P), min(20000, len(P)), replace=False)
    R = spearmanr(P[sub]).correlation; o = leaves_list(linkage(R, "average"))
    im = ax.imshow(R[np.ix_(o, o)], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(K)); ax.set_xticklabels([names[i] for i in o], rotation=90, fontsize=4.5)
    ax.set_yticks(range(K)); ax.set_yticklabels([names[i] for i in o], fontsize=4.5)
    plt.colorbar(im, ax=ax, fraction=0.04).set_label("Spearman $\\rho$", fontsize=6); panel_label(ax, "b")
    # c
    ax = fig.add_subplot(gs[1, 0])
    df = pd.DataFrame(dict(ds=obs["dataset"], bct=obs["broad_cell_type"]))[tr]; Pa = phi[prow[tr]][:, order]
    common = [b for b in df.bct.unique() if ((df.bct == b) & (df.ds == "allen_brain")).sum() >= 50 and ((df.bct == b) & (df.ds == "tabula_muris_senis")).sum() >= 50]
    rows, labs = [], []
    for b in sorted(common):
        for d_, tag in [("allen_brain", "AB"), ("tabula_muris_senis", "TMS")]:
            rows.append(np.median(Pa[((df.bct == b) & (df.ds == d_)).values], 0)); labs.append(f"{b.title()} ({tag})")
    Mx = np.vstack(rows)
    im = ax.imshow(Mx, aspect="auto", cmap="Greens", vmin=0, vmax=0.8)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=5); ax.set_xticks(range(K)); ax.set_xticklabels(names, rotation=90, fontsize=4.5)
    plt.colorbar(im, ax=ax, fraction=0.04).set_label("median SP activity", fontsize=6); panel_label(ax, "c")
    # d
    ax = fig.add_subplot(gs[1, 1])
    rec = []
    for k in range(K):
        w = pi.copy(); w[k] = 0; w /= w.sum(); d_ = psi[k] - w @ psi
        for j in np.argsort(-np.abs(d_))[:2]:
            rec.append((names[k], gene[j], jid[j].replace("_", ":", 1).rsplit("_", 1)[0], d_[j]))
    rec = rec[::-1]
    ax.barh(range(len(rec)), [r[3] for r in rec], color=[PALETTE["rose"] if r[3] > 0 else PALETTE["indigo"] for r in rec], height=0.7)
    ax.set_yticks(range(len(rec))); ax.set_yticklabels([f"{r[1]} {r[2]} ({r[0]})" for r in rec], fontsize=3.8)
    ax.axvline(0, color=PALETTE["grey"], lw=0.6); ax.set_xlabel("$\\psi_k - \\pi$-weighted rest", fontsize=6); despine(ax); panel_label(ax, "d")
    # e/f Clstn1
    jcol = int(np.flatnonzero(jid == CLSTN1_J)[0]); e_ = ev[jcol]; cols = np.flatnonzero(ev == e_); jn = jid[cols]
    ax = fig.add_subplot(gs[2, 0])
    parts = [j.split("_") for j in jn]; chrom = parts[0][0]; strand = parts[0][3]
    st = np.array([int(p[1]) for p in parts]); en = np.array([int(p[2]) for p in parts]); lo, hi = st.min() - 1500, en.max() + 1500
    tx = viz.get_transcripts(gffutils.FeatureDB(viz.DEFAULT_DB), chrom, lo, hi, strand, max_tx=4)
    ybot = viz.draw_exons(ax, tx, lo, hi) if tx else -0.7
    for k, j in enumerate(jn): arc(ax, st[k], en[k], 1.4 + 0.5 * (k % 2), PALETTE["teal"] if j == CLSTN1_J else PALETTE["grey"], 1.5)
    ax.set_xlim(lo, hi); ax.set_ylim(ybot - 0.4, 2.6); ax.set_yticks([]); ax.spines[["left", "top", "right"]].set_visible(False)
    ax.set_xlabel(f"{chrom} ({strand})", fontsize=5.5); ax.ticklabel_format(axis="x", style="plain", useOffset=False); ax.tick_params(axis="x", labelsize=4.5)
    ax.set_title("$Clstn1$", fontsize=7, loc="left"); panel_label(ax, "e")
    ax = fig.add_subplot(gs[2, 1])
    k_in = int(np.flatnonzero(cols == jcol)[0]); conc = float(np.asarray(z["bb_conc_vec"])[int(np.flatnonzero(fev == e_)[0])])
    with h5py.File(H5, "r") as f:
        gl = f["layers"]["cell_by_junction_matrix"]; ip = gl["indptr"][:]; N = len(ip) - 1; Y = np.zeros((N, len(cols)), np.float32)
        for s0 in range(0, N, 20000):
            s1 = min(N, s0 + 20000); a, b = int(ip[s0]), int(ip[s1]); idx = gl["indices"][a:b]; dat = gl["data"][a:b]
            rid = np.repeat(np.arange(s0, s1), np.diff(ip[s0:s1 + 1]))
            for kk, cc in enumerate(cols): m = idx == cc; Y[rid[m], kk] += dat[m]
    rho = np.full(N, np.nan); rho[tr] = phi[prow[tr]] @ segment_softmax(torch.from_numpy(np.asarray(z["psi_loc"], np.float32)), torch.from_numpy(inv), len(fev)).numpy()[:, jcol]
    post = (conc * rho + Y[:, k_in]) / (conc + Y.sum(1))
    d2 = pd.DataFrame(dict(b=obs["broad_cell_type"], v=post))[tr]
    top = d2.groupby("b").v.median().sort_values().index[-14:]
    data = [d2[d2.b == b].v.dropna().values for b in top]
    vp = ax.violinplot(data, positions=range(len(top)), widths=0.8, showextrema=False)
    for bd in vp["bodies"]: bd.set_facecolor(PALETTE["teal"]); bd.set_alpha(0.5); bd.set_edgecolor("none")
    for i, v in enumerate(data): ax.hlines(np.median(v), i - 0.3, i + 0.3, color="black", lw=1)
    ax.set_xticks(range(len(top))); ax.set_xticklabels([t.title() for t in top], rotation=60, ha="right", fontsize=4.5)
    ax.set_ylabel("denoised junction usage", fontsize=6); despine(ax); panel_label(ax, "f")
    save(fig, os.path.join(HERE, "..", "plots", "FIG4_supp_nonaging")); print("-> FIG4_supp_nonaging; PVE:", np.round(pve, 3))


if __name__ == "__main__":
    main()
