#!/usr/bin/env python
"""Splicing-program activity by tissue-cell-type: training 70% (fit) vs held-out 30% (E-step only).

Rows: tissue_cell_types with >= 30 cells in BOTH the training and held-out sets, clustered on the
TRAINING mean phi; columns: SP1..SPK in pi order of that fit. Left = training cells' phi from the
70/30 fit; right = held-out cells' phi inferred by the E-step from their own reads (full coverage),
same rows, same columns. Per-type Spearman across SPs annotated. One panel pair per 70/30 seed
(seeds available). Requires a66_all_cells*/phi_level_000.npy from A66 (--levels 0,...)."""
import os, sys, glob
import numpy as np, pandas as pd, h5py
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); FIGROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, FIGROOT); sys.path.insert(0, os.path.join(FIGROOT, "_common"))
sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa")
from leaflet_theme import apply_theme, PALETTE, figsize, panel_label, save
from programs import sp_order
from _phi_rows import phi_row_of
M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
H5 = f"{M}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad"
SEEDS = {0: f"{M}/IMPUTATION_DM/a66_all_cells", 1: f"{M}/IMPUTATION_DM/a66_all_cells_seed1", 2: f"{M}/IMPUTATION_DM/a66_all_cells_seed2"}
MINC = 30


def main():
    apply_theme()
    with h5py.File(H5, "r") as f:
        n = f["obs/tissue_cell_type"]; c = np.array([x.decode() for x in n["categories"][:]]); tct = c[n["codes"][:]]
    import glob as _g
    ref = sorted(_g.glob(f"{M}/IMPUTATION_DM/resampled_combined/RESAMPLED_10_*.h5mu"))[0]
    with h5py.File(ref, "r") as f:
        ho_ids = np.array([x.decode() for x in f["mod/splicing/obs/_index"][:]]).astype(int)   # atlas row indices of held-out cells
    avail = [s for s, d in SEEDS.items() if os.path.exists(f"{d}/phi_level_000.npy")]
    if not avail: raise SystemExit("no phi_level_000.npy yet -- A66 still queued")
    fig, axes = plt.subplots(len(avail), 2, figsize=figsize("double", h=3.2 * len(avail)), squeeze=False)
    for row, s in enumerate(avail):
        z = np.load(f"{M}/ATLAS_DM/runs_heldout70/dm_simplex_K20_seed{s}_b8192/fit.npz", allow_pickle=True)
        phi_tr = np.asarray(z["phi"]); prow = phi_row_of(z, len(tct)); tr = np.flatnonzero(prow >= 0)
        phi_ho = np.load(f"{SEEDS[s]}/phi_level_000.npy"); order = sp_order(z); K = len(order)
        assert (prow[ho_ids] < 0).all(), "held-out ids overlap training rows"
        types = sorted(t for t in np.unique(tct) if (tct[tr] == t).sum() >= MINC and (tct[ho_ids] == t).sum() >= MINC)
        A = np.vstack([phi_tr[prow[tr[tct[tr] == t]]].mean(0)[order] for t in types])
        B = np.vstack([phi_ho[tct[ho_ids] == t].mean(0)[order] for t in types])
        o = leaves_list(linkage(A, "average", metric="cityblock")); A, B = A[o], B[o]; types = [types[i] for i in o]
        rho = [spearmanr(A[i], B[i]).correlation for i in range(len(types))]
        for k, (Mx, title) in enumerate([(A, f"training 70% (fit, seed {s})"), (B, f"held-out 30% (E-step from own reads, seed {s})")]):
            ax = axes[row, k]
            im = ax.imshow(Mx, aspect="auto", cmap="Greens", vmin=0, vmax=0.8)
            ax.set_xticks(range(K)); ax.set_xticklabels([f"SP{i+1}" for i in range(K)], fontsize=4.5, rotation=90)
            ax.set_yticks(range(len(types)))
            ax.set_yticklabels([t.replace("_", " ").replace("Non-Myeloid ", "") for t in types] if k == 0 else [f"$\\rho$ {r:.2f}" for r in rho], fontsize=3.6)
            if k == 1: ax.yaxis.tick_right()
            ax.set_title(title, fontsize=7, loc="left")
            panel_label(ax, "abcdef"[2 * row + k], dx=-0.02 if k == 0 else -0.01)
        axes[row, 1].annotate(f"median per-type Spearman across SPs = {np.median(rho):.2f}; overall Pearson = {np.corrcoef(A.ravel(), B.ravel())[0,1]:.3f}",
                              (0.5, -0.12), xycoords="axes fraction", ha="center", fontsize=6)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.06); cb.set_label("mean SP activity ($\\phi$)", fontsize=6); cb.ax.tick_params(labelsize=5)
    fig.suptitle(f"Splicing programs by tissue-cell-type (>= {MINC} cells in both sets): learned on 70%, recovered on the unseen 30%", fontsize=8, y=0.995)
    out = os.path.join(FIGROOT, "FIGURE2_simulation_imputation", "plots", "FIG2_supp_SP_train_vs_heldout"); save(fig, out); print("->", out)


if __name__ == "__main__":
    main()
