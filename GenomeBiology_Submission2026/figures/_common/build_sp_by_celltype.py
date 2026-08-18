#!/usr/bin/env python
"""Generic: mean SP activity per cell type for one or more fits (SP order by pi per fit, rows clustered
on the first fit's matrix and kept for the others). Usage:
  python build_sp_by_celltype.py --h5 <h5ad> --fits label=fit.npz [...] --label-field tissue_cell_type --min-cells 30 --out <path>"""
import argparse, os, sys
import numpy as np, pandas as pd, h5py, matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa")
from leaflet_theme import apply_theme, figsize, panel_label, save
from programs import sp_order
from _phi_rows import phi_row_of


def cat(g):
    c = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]); return c[g["codes"][:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True); ap.add_argument("--fits", nargs="+", required=True); ap.add_argument("--label-field", default="tissue_cell_type")
    ap.add_argument("--min-cells", type=int, default=30); ap.add_argument("--out", required=True); ap.add_argument("--title", default="")
    a = ap.parse_args(); apply_theme()
    with h5py.File(a.h5, "r") as f:
        lab = cat(f["obs"][a.label_field])
    fits = [(x.split("=", 1)[0], x.split("=", 1)[1]) for x in a.fits]
    mats = []; types = None; order0 = None
    for name, fp in fits:
        z = np.load(fp, allow_pickle=True); prow = phi_row_of(z, len(lab)); tr = prow >= 0; phi = np.asarray(z["phi"]); order = sp_order(z); K = len(order)
        if types is None: types = sorted(t for t in np.unique(lab[tr]) if (lab[tr] == t).sum() >= a.min_cells)
        Mx = np.vstack([phi[prow[(lab == t) & tr]].mean(0)[order] for t in types]); ncell = [int(((lab == t) & tr).sum()) for t in types]
        mats.append((name, Mx, K, ncell))
    ro = leaves_list(linkage(mats[0][1], "average", metric="cityblock"))
    n = len(mats); fig, axes = plt.subplots(1, n, figsize=figsize("double", h=min(8.6, 0.09 * len(types) + 1.2)), squeeze=False)
    for i, (name, Mx, K, ncell) in enumerate(mats):
        ax = axes[0, i]; im = ax.imshow(Mx[ro], aspect="auto", cmap="Greens", vmin=0, vmax=0.8)
        ax.set_xticks(range(K)); ax.set_xticklabels([f"SP{k+1}" for k in range(K)], fontsize=4.5, rotation=90)
        if i == 0:
            ax.set_yticks(range(len(types))); ax.set_yticklabels([f"{types[j].replace('_', ' ')} ({ncell[j]:,})" for j in ro], fontsize=3.8)
        else: ax.set_yticks([])
        ax.set_title(name, fontsize=7, loc="left"); panel_label(ax, "abcdef"[i])
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02); cb.set_label("mean SP activity ($\\phi$)", fontsize=6); cb.ax.tick_params(labelsize=5)
    if a.title: fig.suptitle(a.title, fontsize=8, y=0.995)
    save(fig, a.out); print("->", a.out)


if __name__ == "__main__":
    main()
