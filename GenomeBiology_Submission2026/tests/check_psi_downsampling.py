#!/usr/bin/env python
"""
check_psi_downsampling.py

Sample a few cells from the original test file and all downsampled versions
to visualize how PSI values change as reads are removed.

Shows:
  1. For sampled cells × junctions: PSI value at each downsampling level
     (heatmap: rows=junctions, cols=levels; 0 means junction dropped out)
  2. Scatter plots of PSI_resampled vs PSI_original at each level
     (colored by retained=blue / dropped=red)

Usage:
    python check_psi_downsampling.py
"""

import os
import numpy as np
import pandas as pd
import h5py
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob

# ── Paths ─────────────────────────────────────────────────────────────────────
WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
ORIGINAL_TEST = os.path.join(WD, "MODEL_INPUT/20260526",
    "test_30_70_model_ready_combined_ge_splicing_aligned_filtered_minCells75_minJuncs500_2026-06-13.h5mu")
RESAMPLED_DIR = os.path.join(WD, "MODEL_INPUT/20260526/resampled")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

N_CELLS_SAMPLE = 5     # how many cells to inspect
N_JUNCS_SHOW   = 30    # how many junctions to show in heatmap
SEED           = 42


def read_sparse_layer(grp):
    data    = grp["data"][:]
    indices = grp["indices"][:]
    indptr  = grp["indptr"][:]
    shape   = tuple(grp.attrs.get("shape", (len(indptr) - 1, indices.max() + 1)))
    return csr_matrix((data, indices, indptr), shape=shape)


def load_psi_for_cells(h5mu_path, cell_indices, junc_indices, layer="junc_ratio"):
    """Load PSI values for specific (cell_indices, junc_indices) from an h5mu file.
    Handles both dense (HDF5 dataset) and sparse (CSR group) storage.
    """
    with h5py.File(h5mu_path, "r") as f:
        splice = f["mod/splicing"]
        obj = splice[f"layers/{layer}"]
        if isinstance(obj, h5py.Dataset):
            # Dense 2D dataset — read selected rows, then column-filter in numpy
            rows = np.array(sorted(cell_indices), dtype=np.intp)
            arr = obj[rows, :]                          # (n_cells × J) via h5py
            return np.array(arr[:, junc_indices], dtype=np.float32)
        else:
            # Sparse group (CSR)
            mat = read_sparse_layer(obj)
            sub = mat[np.ix_(cell_indices, junc_indices)]
            return np.asarray(sub.todense(), dtype=np.float32)


def main():
    rng = np.random.default_rng(SEED)

    # ── Find resampled files ──────────────────────────────────────────────────
    resampled_files = sorted(glob.glob(os.path.join(RESAMPLED_DIR, "RESAMPLED_*_PERCENT_test_*.h5mu")))
    print(f"Found {len(resampled_files)} resampled files")
    for f in resampled_files:
        print(f"  {os.path.basename(f)}")

    # ── Load original PSI to pick interesting junctions ───────────────────────
    print(f"\nLoading original test file (sampling cells only)...")
    with h5py.File(ORIGINAL_TEST, "r") as f:
        splice = f["mod/splicing"]
        N = splice["obs/_index"].shape[0]
        J = splice["layers/junc_ratio"].shape[1]
        # Junction names
        if "junction_id" in splice["var"]:
            junc_names = splice["var"]["junction_id"][:].astype(str)
        elif "ID" in splice["var"]:
            junc_names = splice["var"]["ID"][:].astype(str)
        else:
            junc_names = np.array([str(i) for i in range(J)])

        print(f"  N={N} cells, J={J} junctions")

        # Pick cells uniformly, then load only those rows (not the full N×J matrix)
        cell_indices = sorted(rng.choice(N, size=N_CELLS_SAMPLE, replace=False).tolist())
        print(f"  Sampled cell indices: {cell_indices}")

        # Load only sampled rows from the dense junc_ratio (5 × J)
        rows = np.array(cell_indices, dtype=np.intp)
        psi_cells = splice["layers/junc_ratio"][rows, :].astype(np.float32)
    # Keep junctions covered in at least 3 of the 5 sampled cells with 0 < PSI < 1
    interesting = ((psi_cells > 0) & (psi_cells < 1)).sum(axis=0)  # (J,)
    good_juncs = np.where(interesting >= 3)[0]
    print(f"  Junctions with 0<PSI<1 in ≥3 sampled cells: {len(good_juncs)}")

    if len(good_juncs) < N_JUNCS_SHOW:
        # Fall back to any covered junction
        good_juncs = np.where((psi_cells > 0).sum(axis=0) >= 3)[0]
        print(f"  Fallback — junctions covered in ≥3 sampled cells: {len(good_juncs)}")

    junc_indices = sorted(rng.choice(good_juncs, size=min(N_JUNCS_SHOW, len(good_juncs)), replace=False))
    junc_labels  = junc_names[junc_indices]
    print(f"  Selected {len(junc_indices)} junctions for visualization")

    # ── Collect PSI across all levels ─────────────────────────────────────────
    levels = {"original (0%)": ORIGINAL_TEST}
    for fp in resampled_files:
        pct = fp.split("RESAMPLED_")[1].split("_PERCENT")[0]
        levels[f"{pct}% removed"] = fp

    # Load post-resampling PSI (junc_ratio) from each file
    # For original: junc_ratio = full PSI
    # For resampled: junc_ratio = zeroed for dropped junctions (what the model sees)
    psi_by_level = {}
    for label, path in levels.items():
        print(f"  Loading {label} ...")
        psi_by_level[label] = load_psi_for_cells(path, cell_indices, junc_indices, "junc_ratio")

    # ── Figure 1: Heatmap of PSI across downsampling levels, per cell ─────────
    n_levels = len(psi_by_level)
    level_names = list(psi_by_level.keys())

    fig, axes = plt.subplots(N_CELLS_SAMPLE, n_levels,
                              figsize=(3 * n_levels, 2.5 * N_CELLS_SAMPLE),
                              squeeze=False)
    for ci, cell_idx in enumerate(cell_indices):
        for li, label in enumerate(level_names):
            ax = axes[ci, li]
            psi_row = psi_by_level[label][ci, :]   # (J_show,)
            im = ax.imshow(psi_row[np.newaxis, :], vmin=0, vmax=1,
                           cmap="viridis", aspect="auto")
            ax.set_yticks([])
            ax.set_xticks([])
            if ci == 0:
                ax.set_title(label, fontsize=7, rotation=30, ha="left")
            if li == 0:
                ax.set_ylabel(f"cell {cell_idx}", fontsize=8, rotation=0, labelpad=50, va="center")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    plt.colorbar(im, cax=cbar_ax, label="PSI")
    fig.suptitle(f"PSI for {len(junc_indices)} junctions × {N_CELLS_SAMPLE} cells\n"
                 f"across downsampling levels (0=dropped/NA)", fontsize=11, y=1.01)
    plt.tight_layout(rect=[0, 0, 0.91, 1])
    out1 = os.path.join(OUT_DIR, "psi_downsampling_heatmap.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out1}")

    # ── Figure 2: Scatter PSI_resampled vs PSI_original per level ────────────
    psi_orig = psi_by_level["original (0%)"]
    resampled_levels = [(l, p) for l, p in levels.items() if "RESAMPLED" in p]

    n_r = len(resampled_levels)
    ncols = min(4, n_r)
    nrows = int(np.ceil(n_r / ncols))
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    for idx, (label, path) in enumerate(resampled_levels):
        ax = axes2[idx // ncols, idx % ncols]
        psi_res = psi_by_level[label]   # already loaded

        orig_flat = psi_orig.ravel()
        res_flat  = psi_res.ravel()

        # Color by retained (both > 0) vs dropped (resampled = 0 but original > 0)
        retained = (orig_flat > 0) & (res_flat > 0)
        dropped  = (orig_flat > 0) & (res_flat == 0)
        covered  = orig_flat > 0

        n_tot = covered.sum()
        n_ret = retained.sum()
        n_drp = dropped.sum()

        ax.scatter(orig_flat[retained], res_flat[retained],
                   alpha=0.4, s=8, color="#2196F3", label=f"retained ({n_ret})")
        ax.scatter(orig_flat[dropped], np.zeros(dropped.sum()),
                   alpha=0.4, s=8, color="#F44336", label=f"dropped ({n_drp})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("PSI original", fontsize=9)
        ax.set_ylabel("PSI resampled", fontsize=9)
        ax.set_title(label, fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="upper left")
        pct_dropped = 100 * n_drp / max(n_tot, 1)
        ax.text(0.98, 0.02, f"{pct_dropped:.1f}% dropped", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="#F44336")

    # Hide unused subplots
    for idx in range(n_r, nrows * ncols):
        axes2[idx // ncols, idx % ncols].set_visible(False)

    fig2.suptitle(f"PSI_resampled vs PSI_original\n"
                  f"({N_CELLS_SAMPLE} sampled cells, {len(junc_indices)} junctions)",
                  fontsize=11)
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, "psi_downsampling_scatter.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out2}")

    # ── Figure 3: Line plot — PSI for individual (cell, junction) pairs ───────
    # Pick a few junction-cell pairs that are covered in original
    pair_coverage = (psi_orig > 0)   # (N_CELLS × J_show)
    pair_rows, pair_cols = np.where(pair_coverage)
    if len(pair_rows) > 12:
        chosen = rng.choice(len(pair_rows), size=12, replace=False)
        pair_rows = pair_rows[chosen]
        pair_cols = pair_cols[chosen]

    all_labels  = list(psi_by_level.keys())
    x_positions = np.arange(len(all_labels))

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    cmap3 = plt.cm.tab20
    for k, (pr, pc) in enumerate(zip(pair_rows, pair_cols)):
        vals = [psi_by_level[lab][pr, pc] for lab in all_labels]
        ax3.plot(x_positions, vals, marker="o", linewidth=1.5, markersize=5,
                 color=cmap3(k / max(len(pair_rows), 1)),
                 label=f"cell {cell_indices[pr]}, junc {pc}")

    ax3.set_xticks(x_positions)
    ax3.set_xticklabels(all_labels, rotation=30, ha="right", fontsize=8)
    ax3.set_ylabel("PSI value", fontsize=10)
    ax3.set_ylim(0, 1.05)
    ax3.set_title("PSI for individual (cell, junction) pairs across downsampling levels\n"
                  "(0 = junction dropped out due to read removal)", fontsize=10)
    ax3.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left", ncol=1)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    out3 = os.path.join(OUT_DIR, "psi_downsampling_lines.png")
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {out3}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── PSI dropout summary (sampled cells × junctions) ──────────────────")
    psi_o_flat = psi_orig.ravel()
    n_covered_orig = (psi_o_flat > 0).sum()
    print(f"{'Level':<25} {'n_eval_pairs':>12} {'n_dropped':>10} {'% dropped':>10} {'mean_PSI_retained':>18}")
    for label, _ in resampled_levels:
        pr = psi_by_level[label].ravel()
        covered = psi_o_flat > 0
        n_retained = ((psi_o_flat > 0) & (pr > 0)).sum()
        n_dropped  = ((psi_o_flat > 0) & (pr == 0)).sum()
        mean_retained = pr[(psi_o_flat > 0) & (pr > 0)].mean() if n_retained > 0 else float("nan")
        print(f"{label:<25} {n_covered_orig:>12,} {n_dropped:>10,} {100*n_dropped/max(n_covered_orig,1):>10.1f} {mean_retained:>18.4f}")


if __name__ == "__main__":
    main()
