#!/usr/bin/env python
"""
fgfr2_atse_violin.py

For every ATSE in Fgfr2, plot violin plots of PSI for each junction
across broad cell types — showing only cells where the ATSE is detected.

Uses the full aligned splice adata (132,875 cells × 83,593 junctions).
PSI = cell_by_junction_matrix / cell_by_cluster_matrix (computed on the fly).

Output: tests/fgfr2_atse_<event_id>.png  (one per ATSE)
"""

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = (
    "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/"
    "MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/"
    "splice_adata_aligned_filtered_minCells75_minJuncs500_2026-06-13.h5ad"
)
GENE  = "Fgfr2"
OUT   = os.path.dirname(os.path.abspath(__file__))


def read_cat(grp):
    """Read a categorically-stored h5py group into a numpy str array."""
    if hasattr(grp, "keys"):          # categorical encoding
        cats = grp["categories"][:].astype(str)
        return cats[grp["codes"][:]]
    return grp[:].astype(str)


def main():
    with h5py.File(DATA, "r") as f:
        # ── Load var ─────────────────────────────────────────────────────────
        gene_names  = read_cat(f["var"]["gene_name"])
        event_ids   = read_cat(f["var"]["event_id"])
        junction_ids= read_cat(f["var"]["junction_id"])

        gene_mask   = gene_names == GENE
        col_indices = np.where(gene_mask)[0]   # Fgfr2 column positions
        fgfr2_event_ids  = event_ids[col_indices]
        fgfr2_junc_ids   = junction_ids[col_indices]

        print(f"{GENE}: {len(col_indices)} junctions across "
              f"{len(np.unique(fgfr2_event_ids))} ATSEs")

        # ── Load obs metadata ─────────────────────────────────────────────────
        broad_ct = read_cat(f["obs"]["broad_cell_type"])   # (N_cells,)
        N = len(broad_ct)

        # ── Load count matrices for Fgfr2 columns only ───────────────────────
        # Both layers are sparse CSR stored as groups
        def load_sparse_cols(layer_name):
            grp    = f["layers"][layer_name]
            data   = grp["data"][:]
            idxs   = grp["indices"][:]
            indptr = grp["indptr"][:]
            shape  = tuple(grp.attrs.get("shape", (len(indptr) - 1, idxs.max() + 1)))
            from scipy.sparse import csr_matrix
            mat = csr_matrix((data, idxs, indptr), shape=shape)
            return mat[:, col_indices]   # (N × n_fgfr2_juncs) sparse

        print("Loading cell_by_junction_matrix ...")
        junc_mat    = load_sparse_cols("cell_by_junction_matrix")    # reads per junction
        print("Loading cell_by_cluster_matrix ...")
        cluster_mat = load_sparse_cols("cell_by_cluster_matrix")     # total 5'-end reads

    # Convert to dense (16 junctions × 132K cells — manageable)
    junc_dense    = np.array(junc_mat.todense(),    dtype=np.float32)    # N × 16
    cluster_dense = np.array(cluster_mat.todense(), dtype=np.float32)    # N × 16

    # PSI — only where cluster > 0; else NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = np.where(cluster_dense > 0, junc_dense / cluster_dense, np.nan)

    # ── One figure per ATSE ──────────────────────────────────────────────────
    atses = pd.Series(fgfr2_event_ids).unique()
    # Sort by ATSE number
    atses = sorted(atses, key=lambda x: int(x.split("_atse_")[-1]) if "_atse_" in x else 0)

    for atse in atses:
        atse_local = np.where(fgfr2_event_ids == atse)[0]   # local indices within Fgfr2
        n_juncs    = len(atse_local)
        junc_labels= fgfr2_junc_ids[atse_local]

        # Cells covered in this ATSE (cluster > 0 for any junction in this ATSE)
        atse_covered = np.any(cluster_dense[:, atse_local] > 0, axis=1)
        n_covered    = atse_covered.sum()
        print(f"\n{atse}: {n_juncs} junctions, {n_covered:,} cells with coverage")

        # Per-junction PSI for covered cells
        psi_atse   = psi[atse_covered, :][:, atse_local]   # (n_covered × n_juncs)
        broad_sub  = broad_ct[atse_covered]

        # Order cell types by median PSI of first junction (descending)
        ct_medians = {}
        for ct in np.unique(broad_sub):
            ct_mask = broad_sub == ct
            vals = psi_atse[ct_mask, 0]
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                ct_medians[ct] = (np.median(vals), ct_mask.sum())

        # Only show cell types with ≥10 cells detected in this ATSE
        ct_order = [ct for ct, (_, n) in sorted(ct_medians.items(),
                    key=lambda x: -x[1][0]) if n >= 10]

        if not ct_order:
            print(f"  Skipping {atse} — no cell type has ≥10 covered cells")
            continue

        # ── Plot ─────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(
            n_juncs, 1,
            figsize=(max(10, len(ct_order) * 0.5), 3.5 * n_juncs),
            squeeze=False
        )
        fig.suptitle(f"{GENE}  |  {atse}\n"
                     f"{n_covered:,} cells with ATSE coverage  "
                     f"(showing cell types with ≥10 covered cells)",
                     fontsize=12, y=1.01)

        colors = plt.cm.tab20(np.linspace(0, 1, len(ct_order)))

        for ji, (local_ji, junc_label) in enumerate(zip(atse_local, junc_labels)):
            ax = axes[ji, 0]

            data_per_ct = []
            labels_with_n = []
            for ct in ct_order:
                ct_mask  = broad_sub == ct
                # Only cells where THIS junction is detected (cluster > 0)
                detected = ct_mask & (cluster_dense[atse_covered, local_ji] > 0)
                vals = psi_atse[detected, ji]
                vals = vals[~np.isnan(vals)]
                data_per_ct.append(vals if len(vals) > 0 else np.array([np.nan]))
                labels_with_n.append(f"{ct}\n(n={detected.sum()})")

            # Violin
            parts = ax.violinplot(
                [v[~np.isnan(v)] if len(v) > 0 else [0] for v in data_per_ct],
                positions=range(len(ct_order)),
                showmedians=True, showextrema=False,
            )
            for pc, col in zip(parts["bodies"], colors):
                pc.set_facecolor(col)
                pc.set_alpha(0.7)
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.5)

            # Overlay strip plot (jittered dots)
            rng = np.random.default_rng(42)
            for xi, vals in enumerate(data_per_ct):
                v = vals[~np.isnan(vals)] if len(vals) > 0 else np.array([])
                if len(v) == 0:
                    continue
                # subsample to 300 max
                if len(v) > 300:
                    v = rng.choice(v, 300, replace=False)
                jitter = rng.uniform(-0.15, 0.15, len(v))
                ax.scatter(xi + jitter, v, s=4, alpha=0.3,
                           color=colors[xi], zorder=3)

            ax.set_xticks(range(len(ct_order)))
            ax.set_xticklabels(labels_with_n, fontsize=7, rotation=40, ha="right")
            ax.set_ylabel("PSI", fontsize=9)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(junc_label, fontsize=8, pad=2)
            ax.grid(True, axis="y", alpha=0.3)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.axhline(1, color="gray", linewidth=0.5, linestyle="--")

        plt.tight_layout()
        safe_name = atse.replace("/", "_").replace(":", "_")
        out_path  = os.path.join(OUT, f"fgfr2_{safe_name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
