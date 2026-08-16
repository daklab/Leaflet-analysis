#!/usr/bin/env python
"""UMAP of the LEARNED phi, across seeds and across models.

Run: python panels_phi_umap.py --dir <DM_TEST> --sim-base <tag> --out <panels>

The simulation QC panel showed a UMAP of the TRUE phi -- the programs are separable in principle.
This shows what each fit actually learned, so the question becomes whether the latent space is
reproduced: same structure across model seeds, and same structure across likelihoods.

Layout: rows are models, columns are seeds. Every point is a cell, coloured by its TRUE program,
so a fit that has recovered the structure shows K clean clusters coloured consistently. A fit
that has not shows colours mixed within clusters.

Two things this catches that a correlation cannot:
  - a model that splits one true program across two learned ones, or merges two into one
  - a seed that finds a qualitatively different decomposition rather than a noisier version

All panels are embedded in ONE shared UMAP space, fit on the concatenation of every phi after
program matching. Embedding each panel separately would produce arbitrary rotations and make
identical structure look different.

Labels 1-indexed; colours from the project palette.
"""
import argparse, glob, os, re, sys
import h5py, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/_common")
import leaflet_theme as th  # noqa: E402

LAB = {"betabinomial_beta": "BB + beta (current)", "betabinomial_simplex": "BB + simplex",
       "dirichletmultinomial_beta": "DM + beta", "dirichletmultinomial_simplex": "DM + simplex"}


def true_prog(path):
    with h5py.File(path, "r") as f:
        n = f["obs"]["true_program"]
        if isinstance(n, h5py.Group):
            c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
            return c[n["codes"][:]].astype(str)
        return np.asarray(n[:]).astype(str)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--sim-dir", default="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION")
    ap.add_argument("--sim-base", default="sim_heart_K5_dm50_eff0.5_frac0.3")
    ap.add_argument("--sim-seed", type=int, default=0)
    ap.add_argument("--cells", nargs="+",
                    default=["dirichletmultinomial_simplex", "dirichletmultinomial_beta",
                             "betabinomial_beta"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-cells", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    th.apply_theme()

    simp = os.path.join(args.sim_dir, f"{args.sim_base}_seed{args.sim_seed}.h5ad")
    prog = true_prog(simp)
    got = {}
    for cell in args.cells:
        for s in range(3):
            # ONE simulation, varying MODEL seed -- the cells are then identical across panels,
            # which is what makes a shared embedding meaningful.
            f = os.path.join(args.dir,
                             f"fit_{args.sim_base}_seed{args.sim_seed}_{cell}_seed{s}.npz")
            if os.path.exists(f):
                got[(cell, s)] = np.load(f, allow_pickle=True)
    if not got:
        print("no fits found"); return
    cells_present = [c for c in args.cells if any(k[0] == c for k in got)]
    seeds = sorted({k[1] for k in got})
    print(f"[1] {len(got)} fits: {len(cells_present)} models x {len(seeds)} seeds")

    ref_key = (cells_present[0], seeds[0])
    K = got[ref_key]["psi_mean"].shape[0]
    # match every fit's programs onto the reference by psi, and apply to phi
    phis = {}
    for k, z in got.items():
        C = np.corrcoef(got[ref_key]["psi_mean"], z["psi_mean"])[:K, K:]
        r_, c_ = linear_sum_assignment(-np.nan_to_num(C))
        p = np.empty(K, dtype=int); p[r_] = c_
        phis[k] = z["phi"][:, p]

    rng = np.random.default_rng(0)
    sub = rng.choice(len(prog), size=min(args.max_cells, len(prog)), replace=False)
    stacked = np.concatenate([phis[k][sub] for k in got], axis=0)
    try:
        import umap
        emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(stacked)
        method = "UMAP"
    except Exception:
        x = stacked - stacked.mean(0)
        u, sv, _ = np.linalg.svd(x, full_matrices=False)
        emb, method = u[:, :2] * sv[:2], "PCA"
    n = len(sub)
    cols = [th.PALETTE[c] for c in ("teal", "indigo", "ochre", "rose", "olive")][:K]

    fig, axes = plt.subplots(len(cells_present), len(seeds),
                             figsize=th.figsize("double", h=2.4 * len(cells_present)),
                             squeeze=False)
    for r, cell in enumerate(cells_present):
        for c, s in enumerate(seeds):
            ax = axes[r][c]
            if (cell, s) not in got:
                ax.axis("off"); continue
            i = list(got).index((cell, s))
            e = emb[i * n:(i + 1) * n]
            for kk in range(K):
                m = prog[sub] == str(kk)
                ax.scatter(e[m, 0], e[m, 1], s=1.2, c=cols[kk], linewidths=0, rasterized=True,
                           label=f"program {kk+1}" if (r == 0 and c == 0) else None)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"model seed {s}", fontsize=th.PT["panel"])
            if c == 0:
                ax.set_ylabel(LAB.get(cell, cell), fontsize=th.PT["label"])
            if r == 0 and c == 0:
                ax.legend(markerscale=6, frameon=False, fontsize=th.PT["legend"], loc="best")
    fig.suptitle(f"Learned phi, shared {method} space, coloured by TRUE program   |   "
                 f"{args.sim_base}_seed{args.sim_seed}   |   programs Hungarian-matched",
                 fontsize=th.PT["annot"], y=0.997)
    fig.tight_layout()
    tag = args.sim_base.replace("sim_heart_K5_", "").replace("_frac0.3", "").replace(".", "p")
    th.save(fig, os.path.join(args.out, f"phi_umap_{tag}_seed{args.sim_seed}"))
    print(f"[2] -> phi_umap_{tag}_seed{args.sim_seed}")


if __name__ == "__main__":
    main()
