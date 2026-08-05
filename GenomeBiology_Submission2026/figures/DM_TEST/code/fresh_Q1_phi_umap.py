#!/usr/bin/env python
"""Q1: does phi separate? SIMULATED phi against LEARNED phi, one panel per training seed.

Run: fresh_Q1_phi_umap.py --dir <DM_FRESH> --sim <sim.h5ad> --out <panels>

Leftmost panel is the TRUE phi the simulator drew -- the ceiling, showing the programs are
separable in principle. The remaining panels are what each training seed learned. If the model
works, they reproduce the left panel's structure; if a seed splits or merges programs, the
colours mix within a cluster and that is visible directly rather than inferred from a number.

All panels share ONE embedding, fit on the concatenation of true and learned phi after Hungarian
program matching. Embedding separately would give arbitrary rotations and make identical
structure look different.

Labels 1-indexed; colours from the project palette.
"""
import argparse, glob, os, re, sys
import h5py, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402


def sil(X, labels):
    """Mean silhouette against the TRUE program label.

    Computed on the phi that produced the panel, not on the 2-D embedding: UMAP distorts
    distances non-linearly, so a silhouette measured there scores the embedding rather than the
    representation. The embedding silhouette is reported alongside only as a description of what
    the eye is seeing.
    """
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(X, labels))
sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/_common")
import leaflet_theme as th  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dir", required=True); ap.add_argument("--sim", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--max-cells", type=int, default=4000)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); th.apply_theme()

with h5py.File(a.sim, "r") as f:
    n = f["obs"]["true_program"]
    prog = ((np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])[n["codes"][:]])
            if isinstance(n, h5py.Group) else np.asarray(n[:])).astype(str)
    true_phi = np.asarray(f["obsm"]["true_phi"])
base = os.path.basename(a.sim).replace(".h5ad", "")
fits = {}
for p in sorted(glob.glob(os.path.join(a.dir, f"fit_{base}_dirichletmultinomial_simplex_seed*.npz"))):
    fits[int(re.search(r"seed(\d)\.npz", p).group(1))] = np.load(p, allow_pickle=True)
if not fits:
    print(f"no fits for {base}"); sys.exit(0)
seeds = sorted(fits); K = true_phi.shape[1]
print(f"[1] {base}: true phi + {len(seeds)} learned (seeds {seeds})")

# match each learned phi to the TRUE program order, using psi against nothing -- use phi directly
mats = {"true": true_phi}
for s in seeds:
    ph = fits[s]["phi"]
    C = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            C[i, j] = np.corrcoef(true_phi[:, i], ph[:, j])[0, 1]
    r_, c_ = linear_sum_assignment(-np.nan_to_num(C))
    perm = np.empty(K, int); perm[r_] = c_
    mats[f"seed {s}"] = ph[:, perm]
    print(f"    seed {s} matched, mean r {np.mean([C[i,j] for i,j in zip(r_,c_)]):.4f}")

rng = np.random.default_rng(0)
sub = rng.choice(len(prog), size=min(a.max_cells, len(prog)), replace=False)
keys = list(mats)
stack = np.concatenate([mats[k][sub] for k in keys], 0)
try:
    import umap
    emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(stack); meth = "UMAP"
except Exception:
    x = stack - stack.mean(0); u, sv, _ = np.linalg.svd(x, full_matrices=False)
    emb, meth = u[:, :2] * sv[:2], "PCA"
m = len(sub)
scores = {}
cols = [th.PALETTE[c] for c in ("teal", "indigo", "ochre", "rose", "olive")][:K]
fig, axes = plt.subplots(1, len(keys), figsize=th.figsize("double", h=2.6), squeeze=False)
for i, k in enumerate(keys):
    ax = axes[0][i]; e = emb[i*m:(i+1)*m]
    for kk in range(K):
        msk = prog[sub] == str(kk)
        ax.scatter(e[msk, 0], e[msk, 1], s=1.4, c=cols[kk], linewidths=0, rasterized=True,
                   label=f"program {kk+1}" if i == 0 else None)
    ax.set_xticks([]); ax.set_yticks([])
    s_phi = sil(mats[k][sub], prog[sub])
    s_emb = sil(e, prog[sub])
    scores[k] = (s_phi, s_emb)
    ttl = "SIMULATED $\\phi$" if k == "true" else f"learned, {k}"
    ax.set_title(f"{ttl}\nsilhouette {s_phi:.3f}  (embedding {s_emb:.3f})",
                 fontsize=th.PT["annot"])
    if i == 0:
        ax.legend(markerscale=6, frameon=False, fontsize=th.PT["legend"], loc="best")
        ax.set_ylabel(f"{meth} 2"); th.panel_label(ax, "a")
eff = re.search(r"eff([\d.]+)_", base).group(1)
fig.suptitle(f"Q1  Does $\\phi$ separate?   effect {eff}   |   shared {meth} space, "
             f"coloured by TRUE program, programs matched to truth", fontsize=th.PT["annot"], y=1.0)
fig.tight_layout()
th.save(fig, os.path.join(a.out, f"Q1_phi_umap_eff{eff.replace('.','p')}"))
print(f"[2] -> Q1_phi_umap_eff{eff.replace('.','p')}")
print(f"[3] silhouette vs TRUE program (on phi; embedding in brackets)")
for k, (sp_, se_) in scores.items():
    print(f"      {k:<12} {sp_:>7.4f}  ({se_:.4f})")
import json
with open(os.path.join(a.dir, f"Q1_silhouette_eff{eff.replace('.','p')}.json"), "w") as fh:
    json.dump({k: {"phi": v[0], "embedding": v[1]} for k, v in scores.items()}, fh, indent=2)
