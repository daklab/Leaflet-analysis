#!/usr/bin/env python
"""Fig 3 supp: the posterior test's reach per contrast is set by how far apart the two cell types'
program loadings are -- and which cell types share programs.

a  per contrast: x = ||phibar_A - phibar_B||_1 (posterior-mean phi, seed 0), y = the contrast's
   posterior-TV ceiling (99th percentile of posterior-mean TV over eligible ATSEs), point size =
   number of gated calls. Contrasts between cell types that share programs sit at the bottom left
   and cannot clear delta at any ATSE (marrow B cell vs monocyte; skin basal vs keratinocyte).
b  mean phi per cell type (57 types in the 70 contrasts), rows clustered: which types share programs.
"""
import os, sys
import numpy as np, pandas as pd, h5py
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list

HERE = os.path.dirname(os.path.abspath(__file__))
FIGROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, FIGROOT); sys.path.insert(0, os.path.join(FIGROOT, "_common"))
sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa")
from leaflet_theme import apply_theme, PALETTE, METHOD_COLORS, figsize, panel_label, despine, save
from _phi_rows import phi_row_of

M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
O = f"{M}/LEAFCUTTER_CMP_DELTAONLY"
H5 = f"{M}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad"
FIT = f"{M}/ATLAS_DM/runs_final_deltaonly/dm_simplex_K20_seed0_b8192/fit.npz"
LFA, LC, GREY = METHOD_COLORS["LeafletFA"], METHOD_COLORS["NMF"], PALETTE["grey"]


def main():
    apply_theme()
    g = pd.read_csv(f"{O}/analysis_matched/posterior_consensus_gated.tsv.gz", sep="\t")
    g = g[g.mincov >= 25]
    z = np.load(FIT, allow_pickle=True); phi = np.asarray(z["phi"]); K = phi.shape[1]
    with h5py.File(H5, "r") as f:
        n = f["obs"]["tissue_cell_type"]; c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
        lab = np.array([l.replace(" ", "_") for l in c[n["codes"][:]]])
    prow = phi_row_of(z, len(lab)); tr = prow >= 0
    types = sorted(set(t for cc in g.contrast.unique() for t in cc.split("__vs__")))
    PH = pd.DataFrame({t: phi[prow[(lab == t) & tr]].mean(0) for t in types}).T
    PH.columns = [f"P{k+1}" for k in range(K)]
    rows = []
    for cc, s in g.groupby("contrast"):
        a, b = cc.split("__vs__")
        rows.append(dict(contrast=cc, tissue=a.split("_")[0], dphi=np.abs(PH.loc[a] - PH.loc[b]).sum(),
                         ceiling=np.nanquantile(s.tv, 0.99), n_calls=int(s.lfa_cons.sum()), n_lc=int(s.lc_sig.sum()), n=len(s)))
    d = pd.DataFrame(rows)
    d.to_csv(f"{O}/analysis_consistency/contrast_ceiling_vs_dphi.tsv", sep="\t", index=False)
    print(d.sort_values("dphi").to_string())

    fig = plt.figure(figsize=figsize("double", h=4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    tis = sorted(d.tissue.unique()); cols = plt.cm.tab20(np.linspace(0, 1, len(tis)))
    for t, col in zip(tis, cols):
        s = d[d.tissue == t]
        ax.scatter(s.dphi, s.ceiling, s=8 + 0.06 * s.n_calls, color=col, alpha=0.8, lw=0.4, edgecolor="black", label=t)
    ax.axhline(0.15, color=LFA, ls="--", lw=0.8); ax.annotate("$\\delta$ = 0.15", (0.02, 0.155), fontsize=6, color=LFA)
    xx = np.linspace(0, d.dphi.max() * 1.05, 50)
    ax.plot(xx, 0.5 * xx, color=GREY, lw=0.8, ls=":"); ax.annotate("hard bound: TV $\\leq \\frac{1}{2}\\|\\Delta\\bar\\phi\\|_1$", (xx[-1], 0.5 * xx[-1]), fontsize=5.5, color=GREY, ha="right", va="bottom")
    a1, a0 = np.polyfit(d.dphi, d.ceiling, 1); r = np.corrcoef(d.dphi, d.ceiling)[0, 1]
    ax.plot(xx, a1 * xx + a0, color=LFA, lw=0.8, ls="-", alpha=0.6); ax.annotate(f"fit: {a1:.2f}x {a0:+.2f}  (r = {r:.2f})", (xx[-1], a1 * xx[-1] + a0), fontsize=5.5, color=LFA, ha="right", va="top")
    n0 = int((d.n_calls == 0).sum())
    ax.annotate(f"{n0} of {len(d)} contrasts: no posterior call at any ATSE\n(LeafCutter calls {int(d[d.n_calls == 0].n_lc.sum()):,} there)", (0.98, 0.02), xycoords="axes fraction", ha="right", fontsize=5.5, color=GREY)
    SHOW = {"Marrow_B_CELL__vs__Marrow_MONOCYTE": "marrow B cell v monocyte", "Skin_BASAL_CELL__vs__Skin_KERATINOCYTE": "skin basal v keratinocyte",
            "Heart_ENDOTHELIAL_CELL__vs__Heart_FIBROBLAST": "heart endothelial v fibroblast", "Brain_Non-Myeloid_ENDOTHELIAL_CELL__vs__Brain_Non-Myeloid_OLIGODENDROCYTE": "brain endothelial v oligodendrocyte",
            "Marrow_B_CELL__vs__Marrow_T_CELL": "marrow B v T cell", "Large_Intestine_EPITHELIAL_CELL__vs__Large_Intestine_INTESTINAL_CELL": "intestine epithelial v intestinal",
            "Lung_FIBROBLAST__vs__Lung_SMOOTH_MUSCLE_CELL": "lung fibroblast v smooth muscle", "MAT_B_CELL__vs__MAT_MESENCHYMAL_STEM_CELL": "MAT B cell v MSC"}
    for _, r_ in d.iterrows():
        if r_.contrast in SHOW:
            ax.annotate(SHOW[r_.contrast], (r_.dphi, r_.ceiling), fontsize=5, color="black", textcoords="offset points", xytext=(4, -2))
    ax.set_xlabel("$\\|\\bar\\phi_A - \\bar\\phi_B\\|_1$  (program-loading distance between the two cell types)")
    ax.set_ylabel("posterior TV ceiling of the contrast\n(99th percentile over eligible ATSEs)")
    ax.legend(frameon=False, fontsize=4.5, ncol=2, loc="upper left", title="tissue (size = # posterior calls)", title_fontsize=5)
    ax.set_xlim(0, None); ax.set_ylim(0, None); despine(ax); panel_label(ax, "a")

    ax2 = fig.add_subplot(gs[1])
    order = leaves_list(linkage(PH.values, "average", metric="cityblock"))
    im = ax2.imshow(PH.values[order], aspect="auto", cmap="Greys", vmin=0, vmax=0.8)
    ax2.set_yticks(range(len(types))); ax2.set_yticklabels([types[i].replace("_", " ").replace("Non-Myeloid ", "") for i in order], fontsize=4)
    ax2.set_xticks(range(K)); ax2.set_xticklabels(PH.columns, fontsize=5, rotation=90)
    ax2.set_xlabel("program"); ax2.set_title("mean $\\phi$ per cell type (clustered)", fontsize=7, loc="left")
    cb = fig.colorbar(im, ax=ax2, fraction=0.03, pad=0.02); cb.set_label("mean loading", fontsize=6); cb.ax.tick_params(labelsize=5)
    panel_label(ax2, "b")
    out = os.path.join(FIGROOT, "FIGURE3_leafcutter_ds", "plots", "FIG3_supp_program_sharing")
    save(fig, out); print("->", out)


if __name__ == "__main__":
    main()
