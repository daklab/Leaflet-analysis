#!/usr/bin/env python
"""Two worked events, single cells vs pseudobulk, for Figure 3.

  Apbb2  ATSE 3   endothelial vs oligodendrocyte, brain    found by BOTH
                  LeafCutter p.adj 3e-18, TV 0.996; LeafletFA consensus 3/3, TV 0.70
  Usp54  ATSE 1   same contrast                             LeafletFA ONLY
                  LeafCutter p.adj 0.60, TV 0.05; LeafletFA consensus 3/3, TV 0.58

For each event, one row of three panels:
  left    per-cell PSI of the reference junction, every cell as a point, coloured by cell type,
          x = the cell's ATSE read depth (log). This is what the model sees.
  middle  the same event pseudobulked: one point per mouse (7 vs 8) -- what LeafCutter sees.
  right   the model's posterior: TV distribution across draws, with delta marked, and the
          LeafCutter effect (its own pseudobulk TV) as a tick.

Nothing is smoothed. If the "LeafletFA only" event looks like noise at the single-cell level, this
panel will say so, and that is the point of drawing it.
"""
import os, sys, re, gzip
import numpy as np, pandas as pd, h5py, scipy.sparse as sp, torch
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, FIGROOT); sys.path.insert(0, os.path.join(FIGROOT, "_common"))
sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa")
sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from leaflet_theme import apply_theme, PALETTE, METHOD_COLORS, figsize, panel_label, despine, save
from A51_group_ds_posterior import draw_psi, draw_dphi, tv_per_atse
from _phi_rows import phi_row_of

M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
H5 = f"{M}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad"
FIT = f"{M}/ATLAS_DM/runs_final_deltaonly/dm_simplex_K20_seed0_b8192/fit.npz"
LCD = f"{M}/LEAFCUTTER_DS_2026-08/by_pair/Brain_Non-Myeloid_ENDOTHELIAL_CELL__vs__Brain_Non-Myeloid_OLIGODENDROCYTE"
G1, G2 = "Brain_Non-Myeloid_ENDOTHELIAL CELL", "Brain_Non-Myeloid_OLIGODENDROCYTE"   # atlas label has a SPACE; LeafCutter dir names normalise it
LAB1, LAB2 = "endothelial", "oligodendrocyte"
C1, C2 = PALETTE["teal"], PALETTE["ochre"]
DELTA = 0.15
EVENTS = [("Apbb2", "ENSMUSG00000029207.16_atse_3", "found by both"),
          ("Usp54", "ENSMUSG00000034235.17_atse_1", "LeafletFA only")]


def obs_cat(f, name):
    n = f["obs"][name]
    c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
    return c[n["codes"][:]]


def rows_csr(f, layer, rows, J):
    g = f["layers"][layer]; ip = g["indptr"][:]
    data, ind, ptr = [], [], [0]
    for r in rows:
        s, e = int(ip[r]), int(ip[r + 1])
        data.append(g["data"][s:e]); ind.append(g["indices"][s:e]); ptr.append(ptr[-1] + e - s)
    return sp.csr_matrix((np.concatenate(data), np.concatenate(ind), np.array(ptr)), shape=(len(rows), J))


def main():
    apply_theme()
    with h5py.File(H5, "r") as f:
        ct = obs_cat(f, "tissue_cell_type"); mouse = obs_cat(f, "mouse.id")
        ev = obs_cat.__wrapped__(f, "event_id") if hasattr(obs_cat, "__wrapped__") else None
        n = f["var"]["event_id"]; c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
        ev = c[n["codes"][:]]; J = len(ev)
        vn = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var"]["_index"][:]])
        r1 = np.flatnonzero(ct == G1); r2 = np.flatnonzero(ct == G2)
        rows = np.concatenate([r1, r2])
        Jm = rows_csr(f, "cell_by_junction_matrix", rows, J)
        Cm = rows_csr(f, "cell_by_cluster_matrix", rows, J)
    z = np.load(FIT, allow_pickle=True)
    N = len(ct); prow = phi_row_of(z, N); K = z["phi"].shape[1]
    fev = np.asarray(z["event_id"]).astype(str); ai = torch.from_numpy(np.asarray(z["inv_ev"], np.int64))
    n_atse = len(fev); epos = {e: i for i, e in enumerate(fev)}
    S = 300
    psi_s = draw_psi(z, ai, n_atse, S, seed=0)
    # only cells that have a phi row (phi_row_of returns -1 for cells absent from the fit); a -1
    # index would silently read the LAST row and poison every draw with a wrong cell
    p1 = prow[r1]; p2 = prow[r2]; p1 = p1[p1 >= 0]; p2 = p2[p2 >= 0]
    dphi = draw_dphi(z, p1, p2, S, K, seed=0)
    assert len(p1) and len(p2), f"empty group: {len(p1)} vs {len(p2)} -- check the label strings"
    assert torch.isfinite(dphi).all(), "non-finite dphi"
    tv, _ = tv_per_atse(dphi, psi_s, ai, n_atse)                       # [S, n_atse]
    lc_eff = pd.read_csv(f"{LCD}/ds_results_effect_sizes.txt", sep="\t")
    lc_eff["cluster"] = lc_eff.intron.str.replace(r"^chr[^:]*:\d+:\d+:", "", regex=True).str.replace(r"_[+-]$", "", regex=True)
    lc_sig = pd.read_csv(f"{LCD}/ds_results_cluster_significance.txt", sep="\t")
    lc_sig["cluster"] = lc_sig.cluster.str.replace(r"^chr[^:]*:", "", regex=True).str.replace(r"_[+-]$", "", regex=True)
    pb = pd.read_csv(f"{LCD}/counts_subset.tsv.gz", sep="\t", index_col=0)
    grp = pd.read_csv(f"{LCD}/groups.txt", sep="\t", header=None, index_col=0)
    grp_ct = grp[1]

    fig, axes = plt.subplots(2, 3, figsize=figsize("double", h=4.6))
    n_cells = np.array([len(r1), len(r2)])
    for row, (gene, event, tag) in enumerate(EVENTS):
        cols = np.flatnonzero(ev == event)
        # reference junction: LeafCutter's most differential intron for this cluster
        e = lc_eff[lc_eff.cluster == event].copy()
        e["absd"] = e.iloc[:, 4].abs()
        ref_intron = e.sort_values("absd", ascending=False).iloc[0]["intron"]
        chrom, s0, e0 = ref_intron.split(":")[:3]
        # match to var name chrN_start_end_strand -- start/end conventions can differ by 1
        cand = [i for i in cols if vn[i].split("_")[1] in (s0, str(int(s0) - 1), str(int(s0) + 1))
                and vn[i].split("_")[2] in (e0, str(int(e0) - 1), str(int(e0) + 1))]
        jref = cand[0] if cand else cols[0]
        y = np.asarray(Jm[:, jref].todense()).ravel(); nn = np.asarray(Cm[:, jref].todense()).ravel()
        keep = nn > 0
        psi_cell = np.where(keep, y / np.maximum(nn, 1), np.nan)
        is1 = np.r_[np.ones(len(r1), bool), np.zeros(len(r2), bool)]

        # ---- left: single cells
        ax = axes[row, 0]
        rng = np.random.default_rng(0)
        for msk, col, lab in [(is1, C1, LAB1), (~is1, C2, LAB2)]:
            k = msk & keep
            ax.scatter(nn[k] * np.exp(rng.normal(0, 0.05, k.sum())), psi_cell[k] + rng.normal(0, 0.012, k.sum()),
                       s=4, alpha=0.35, color=col, lw=0, label=f"{lab} (n={k.sum():,} covered)")
        ax.set_xscale("log"); ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("reads in ATSE (this cell)"); ax.set_ylabel(f"{gene}: junction usage\nper cell")
        ax.set_title(f"{gene} — {tag}\nsingle cells (what LeafletFA models)", fontsize=7, loc="left")
        ax.legend(frameon=False, fontsize=5, loc="center right", markerscale=2)

        # ---- middle: pseudobulk
        ax = axes[row, 1]
        pi_rows = [i for i in pb.index if event in i]
        pbe = pb.loc[pi_rows]
        tot = pbe.sum(0)
        pref = [i for i in pi_rows if i.startswith(ref_intron.rsplit(":", 1)[0])] or [pi_rows[0]]
        psi_pb = pbe.loc[pref[0]] / tot.replace(0, np.nan)
        for g, col, lab in [(G1, C1, LAB1), (G2, C2, LAB2)]:
            gn = g.replace(" ", "_")
            samp = [s for s in psi_pb.index if str(grp_ct.get(s, "")).replace(" ", "_") == gn]
            xs = tot[samp].values; ys = psi_pb[samp].values
            ax.scatter(xs, ys, s=28, color=col, edgecolor="white", lw=0.5, label=f"{lab} ({len(samp)} mice)", zorder=3)
        ax.set_xscale("log"); ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("reads in ATSE (this pseudobulk)"); ax.set_ylabel("junction usage\nper pseudobulk")
        p_adj = float(lc_sig.loc[lc_sig.cluster == event, "p.adjust"].iloc[0])
        tv_lc = 0.5 * e.iloc[:, 4].abs().sum()
        ax.set_title(f"pseudobulk (what LeafCutter tests)\nLeafCutter q = {p_adj:.2g}, TV = {tv_lc:.2f}", fontsize=7, loc="left")
        ax.legend(frameon=False, fontsize=5, loc="center right")

        # ---- right: posterior TV
        ax = axes[row, 2]
        tve = tv[:, epos[event]].numpy()
        ax.hist(tve, bins=30, color=METHOD_COLORS["LeafletFA"], alpha=0.85)
        ax.axvline(DELTA, color=PALETTE["olive"], ls="--", lw=1)
        ax.annotate(f"$\\delta$ = {DELTA}", (DELTA, ax.get_ylim()[1] * 0.92), fontsize=6, color=PALETTE["olive"], ha="right", xytext=(-3, 0), textcoords="offset points")
        ax.axvline(tv_lc, color=METHOD_COLORS["NMF"], ls=":", lw=1.2)
        ax.annotate("LeafCutter\npseudobulk TV", (tv_lc, ax.get_ylim()[1] * 0.55), fontsize=5.5, color=METHOD_COLORS["NMF"], ha="left", xytext=(3, 0), textcoords="offset points")
        P = float((tve > DELTA).mean())
        ax.set_xlim(0, 1); ax.set_xlabel("TV between groups (posterior draws)"); ax.set_ylabel("draws")
        ax.set_title(f"LeafletFA posterior\nP(TV > {DELTA}) = {P:.2f}, mean TV = {tve.mean():.2f}", fontsize=7, loc="left")

    for ax, L in zip(axes.ravel(), "abcdef"):
        despine(ax); panel_label(ax, L)
    fig.tight_layout(w_pad=1.6, h_pad=2.2)
    out = os.path.join(FIGROOT, "FIGURE3_leafcutter_ds", "plots", "FIG3_examples")
    save(fig, out); print(f"-> {out}.[pdf|png]")


if __name__ == "__main__":
    main()
