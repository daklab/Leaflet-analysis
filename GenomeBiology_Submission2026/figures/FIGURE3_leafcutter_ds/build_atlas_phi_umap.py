#!/usr/bin/env python
"""phi UMAPs for the atlas fits, old BetaBinomial against new DM+simplex.

The question this answers: the new model puts 61% of all cell mass on its top 3 programs against
the old model's 42%, and its Fbln2 contrast collapses (||dphi||_1 0.23 vs 1.28). Is the new phi
actually failing to separate cell types, or is it separating them along fewer axes while still
resolving them?

A UMAP cannot settle that on its own -- it is a 2-D view of a K-dim simplex and will find structure
if any exists -- so silhouette against held-out `broad_cell_type` labels is printed beside every
panel. The labels are never seen during training; they are only a yardstick.

Reads phi straight from fit.npz. Old fits store psi/phi; new ones psi_mean/phi.
"""

import argparse, os, sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
import leaflet_theme as T  # noqa: E402
sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/"
                   "GenomeBiology_Submission2026/pipelines/05_leafletfa")
from _phi_rows import phi_row_of  # noqa: E402

WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
H5 = f"{WD}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad"


def obs_cat(f, name):
    n = f["obs"][name]
    if isinstance(n, h5py.Group):
        c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
        return c[n["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in n[:]])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fits", nargs="+", required=True, help="label=path/to/fit.npz")
    ap.add_argument("--cells", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    T.apply_theme()

    with h5py.File(H5, "r") as f:
        ct = obs_cat(f, "broad_cell_type")
        N = len(ct)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(N, min(args.cells, N), replace=False))
    lab = ct[idx]
    uq = np.array(sorted(set(lab)))
    code = np.searchsorted(uq, lab)
    print(f"[1] {len(idx):,} cells, {len(uq)} broad cell types", flush=True)

    import umap
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import LabelEncoder

    specs = []
    for s in args.fits:
        name, path = s.split("=", 1)
        z = np.load(path, allow_pickle=True)
        PROW = phi_row_of(z, N)
        keep = idx[PROW[idx] >= 0]          # cells this fit actually has loadings for
        phi = np.asarray(z["phi"], np.float32)[PROW[keep]]
        spec_code = np.searchsorted(uq, ct[keep])
        specs.append((name, phi, spec_code))

    fig, axes = plt.subplots(1, len(specs), figsize=T.figsize("double", h=4.2), squeeze=False)
    cmap = plt.get_cmap("tab20")
    for ax, (name, phi, code) in zip(axes[0], specs):
        # silhouette on the FULL phi, not the 2-D embedding: the embedding is a lossy view and
        # scoring it would measure UMAP as much as the model.
        sil = float(silhouette_score(phi, code, sample_size=5000, random_state=0))
        emb = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(phi)
        sil_e = float(silhouette_score(emb, code, sample_size=5000, random_state=0))
        ax.scatter(emb[:, 0], emb[:, 1], c=[cmap(c % 20) for c in code], s=1.2, linewidths=0,
                   alpha=0.55, rasterized=True)
        top3 = np.sort(phi.mean(0))[::-1][:3].sum()
        p = np.clip(phi, 1e-12, None)
        perp = float(np.exp(-(p * np.log(p)).sum(1)).mean())
        ax.set_title(f"{name}\nsilhouette on $\\phi$ {sil:.3f} (embedding {sil_e:.3f})\n"
                     f"top-3 mass {top3:.2f}   perplexity {perp:.2f}", fontsize=T.PT["annot"])
        ax.set_xticks([]); ax.set_yticks([])
        print(f"[2] {name}: silhouette(phi)={sil:.4f} silhouette(umap)={sil_e:.4f} "
              f"top3={top3:.3f} perp={perp:.2f}", flush=True)
    fig.suptitle("Atlas $\\phi$, coloured by held-out broad cell type (never seen in training)",
                 fontsize=T.PT["panel"], y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight")
    print(f"\nsaved -> {args.out}.png")


if __name__ == "__main__":
    main()
