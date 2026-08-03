#!/usr/bin/env python
"""
A32: recover a phi posterior for the released atlas fit, via a frozen-globals E-step.

Why this is needed
------------------
fit.npz for 2026-07-26_atlas_FULL carries psis_loc/psis_scale (the PSI posterior) but only a
POINT estimate for phi -- phi_sd was added in commit 68a3185 on 2026-07-29, after that model
trained. Every downstream DS statistic therefore draws its uncertainty from PSI alone.

That has a specific and damaging consequence for the downsampling panel: since

    prob_de = P(TV > delta)  with  TV built from  dphi . psi_s

and dphi is a fixed vector when phi is a point estimate, the reported FDP is IDENTICAL whether
a group contains 36 cells or 1,737. The uncertainty is structurally blind to n, so the panel
cannot show power loss and the near-constant call count (~2,000 at every rung) is an artifact
rather than a result. Measured: |dphi|_1 = 0.555 +/- 0.005 at n=1,737 versus 0.567 +/- 0.069 at
n=36 -- the point estimate is stable, but nothing in the FDP calculation knows that the second
number is 14x less certain.

What this script does
---------------------
Freezes every global from the released fit (psi, pi, alpha_pi, dir_conc, bb_conc) and re-infers
q(phi) for the requested cells. Because psi is frozen the per-cell problems are independent and
K-dimensional, so inferring phi for a SUBSET gives the same answer as inferring it for all
132,706 cells -- which is what makes this affordable.

Saves the guide's unconstrained (N, K-1) loc/scale so downstream code can draw phi samples and
propagate them into dphi. The posterior sd of a group mean over n cells then scales like
1/sqrt(n), which is exactly the dependence the current statistic lacks.

Usage
-----
    python A32_phi_posterior_estep.py --fit <fit.npz> --h5ad <atlas> \\
        --cell-types Heart_ENDOTHELIAL_CELL Heart_FIBROBLAST --out <dir>
"""

import argparse
import json
import os
import re
import sys
import time

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from leafletfa.minibatch import LeafletFAMinibatch  # noqa: E402

safe = lambda s: re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))


def cat(h5, name, which):
    with h5py.File(h5, "r") as f:
        node = f[which][name]
        if isinstance(node, h5py.Group):
            c = np.array([x.decode() if isinstance(x, bytes) else x
                          for x in node["categories"][:]])
            return pd.Categorical.from_codes(node["codes"][:], c).astype(str)
        return np.array([x.decode() if isinstance(x, bytes) else x for x in node[:]])


def read_rows(h5ad, layer, rows, n_cols, chunk=4000):
    """Dense [len(rows), n_cols] for an arbitrary (unsorted) set of cell indices."""
    order = np.argsort(rows)
    srt = np.asarray(rows)[order]
    out = np.zeros((len(rows), n_cols), dtype=np.float32)
    with h5py.File(h5ad, "r") as f:
        g = f["layers"][layer]
        indptr = g["indptr"]
        for a in range(0, len(srt), chunk):
            b = min(a + chunk, len(srt))
            blk = srt[a:b]
            lo, hi = int(blk[0]), int(blk[-1]) + 1
            ip = indptr[lo:hi + 1]
            s, e = int(ip[0]), int(ip[-1])
            m = sp.csr_matrix((g["data"][s:e], g["indices"][s:e], ip - ip[0]),
                              shape=(hi - lo, n_cols))
            out[order[a:b]] = m[blk - lo].toarray()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell-types", nargs="+", default=None,
                    help="tissue_cell_type values (sanitised); default = from --manifest")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--n-contrasts", type=int, default=6)
    ap.add_argument("--max-cells", type=int, default=30000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    z = np.load(args.fit)
    bad = [k for k in z.files if z[k].dtype.kind == "f" and not np.isfinite(z[k]).all()]
    assert not bad, f"non-finite arrays in fit: {bad}"
    meta = json.load(open(os.path.join(os.path.dirname(args.fit), "result.json")))
    psi = z["psi"]
    K, J = psi.shape
    print(f"[1] fit K={meta['K']} seed={meta['seed']}  psi {psi.shape}", flush=True)

    tct = np.array([safe(x) for x in cat(args.h5ad, "tissue_cell_type", "obs")])

    cts = args.cell_types
    if not cts:
        assert args.manifest, "give --cell-types or --manifest"
        man = pd.read_csv(args.manifest, sep="\t")
        man = man[man.status.isin(["ready", "already_done"])]
        man = man.sort_values("n_junctions", ascending=False).head(args.n_contrasts)
        cts = sorted(set(man.group_1) | set(man.group_2))
    print(f"[2] {len(cts)} cell types: {cts}", flush=True)

    cells = np.flatnonzero(np.isin(tct, cts))
    if len(cells) > args.max_cells:
        # Subsample per cell type so the composition is preserved, not the first N.
        rng = np.random.default_rng(args.seed)
        keep = []
        per = args.max_cells // len(cts)
        for c in cts:
            idx = np.flatnonzero(tct == c)
            keep.append(rng.choice(idx, size=min(per, len(idx)), replace=False))
        cells = np.sort(np.concatenate(keep))
    print(f"[3] {len(cells):,} cells selected", flush=True)

    t0 = time.time()
    junc = read_rows(args.h5ad, "cell_by_junction_matrix", cells, J)
    clus = read_rows(args.h5ad, "cell_by_cluster_matrix", cells, J)
    print(f"[4] layers loaded in {time.time()-t0:.0f}s  ({junc.nbytes/1e9:.1f} GB each)",
          flush=True)

    m = LeafletFAMinibatch(junc, clus, K=K, device="cpu")
    # Freeze EVERY global, not just psi: pi and dir_conc are the Dirichlet prior on phi, so
    # leaving them free would give these cells a prior fitted to themselves.
    m.freeze_globals(psi=psi,
                     pi=z["pi"] if "pi" in z.files else None,
                     alpha_pi=float(z["alpha_pi"]) if "alpha_pi" in z.files else None,
                     dir_conc=float(z["dir_conc"]) if "dir_conc" in z.files else None,
                     bb_conc=float(z["bb_conc"]) if "bb_conc" in z.files else None)
    t0 = time.time()
    phi_new = m.infer_phi(num_steps=args.steps, subsample_size=min(args.batch, len(cells)),
                          lr=args.lr, gamma=args.gamma, seed=args.seed, verbose=False)
    print(f"[5] E-step in {time.time()-t0:.0f}s", flush=True)

    import pyro
    st = pyro.get_param_store()
    loc = st["AutoNormal.locs.phi"].detach().cpu().numpy()
    scale = st["AutoNormal.scales.phi"].detach().cpu().numpy()
    assert np.isfinite(loc).all() and np.isfinite(scale).all(), "non-finite q(phi)"
    sd, top_prob = m.phi_uncertainty(n_samples=64, seed=args.seed)
    sd = np.asarray(sd.cpu().numpy() if hasattr(sd, "cpu") else sd)
    top_prob = np.asarray(top_prob.cpu().numpy() if hasattr(top_prob, "cpu") else top_prob)

    # Sanity: the re-inferred point estimate should track the released one. A large gap means
    # the globals were not frozen correctly, or these cells are not the ones we think.
    corr = float(np.corrcoef(phi_new.ravel(), z["phi"][cells].ravel())[0, 1])
    print(f"[6] re-inferred phi vs released phi: r = {corr:.4f}", flush=True)

    tag = f"K{meta['K']}_seed{meta['seed']}"
    out = os.path.join(args.out, f"phi_posterior_{tag}.npz")
    np.savez_compressed(out, cells=cells, cell_types=np.array(cts),
                        phi_loc=loc, phi_scale=scale, phi_mean=phi_new,
                        phi_sd=sd, phi_top_prob=top_prob,
                        phi_released=z["phi"][cells], corr_with_released=corr)
    print(f"\n=== done ===")
    print(f"  q(phi) loc/scale {loc.shape} (unconstrained, K-1)")
    print(f"  per-cell phi sd: mean {sd.mean():.4f}  median {np.median(sd):.4f}")
    print(f"  top-program posterior prob: median {np.median(top_prob):.3f}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
