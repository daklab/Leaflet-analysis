#!/usr/bin/env python
"""The OTHER marginal: within a cell, do we rank its junctions correctly?

    A41_percell_direction.py <run_dir> <h5mu> <out_csv> [n_cells=6000] [n_hvj=3000] [seed=0]

A3 only ever collapsed the (cell x junction) matrix one way -- per junction, across cells -- and
reported the median over junctions (~0.05). The transposed question was never computed:

    per JUNCTION, across cells   "which cells splice this junction more?"   <- A3, hard, floor 0.000
    per CELL, across junctions   "which junctions does this cell splice?"   <- HERE

READ THE FLOOR, NOT THE VALUE. These two directions are not comparable and the per-cell number will
look far better for a reason that has nothing to do with the model: junctions genuinely differ from
one another, so simply broadcasting each junction's global mean already ranks a cell's junctions
well. That between-junction structure is free. The only meaningful quantity is how far a method gets
ABOVE the junction-mean floor -- which is exactly the trap that made `global_pearson` read 0.794
when its floor was 0.761.

So every method here is reported next to `naive` (the junction mean), and the headline is the
DIFFERENCE, not the correlation.

Cells are subsampled (default 6,000 of 39,812) because the frozen-globals E-step is the expensive
part and a median over 6,000 cells is already precise to ~0.002. Coverage is ATSE-level throughout
(cov = cell_by_cluster_matrix_original > 0), matching A3.
"""
from __future__ import annotations
import sys, os, json
import numpy as np
import pandas as pd
import h5py
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from A3_eval_imputation import (read_sparse_layer, compute_hvj_indices, load_model,
                                load_run_summary, build_chunk_adata, infer_phi_chunk)


def percell_corr(pred, obs, mask, min_junc=20):
    """Pearson and Spearman per CELL across its covered junctions."""
    n = pred.shape[0]
    rp = np.full(n, np.nan, np.float32)
    rs = np.full(n, np.nan, np.float32)
    for i in range(n):
        m = mask[i]
        if m.sum() < min_junc:
            continue
        a, b = pred[i, m].astype(np.float64), obs[i, m].astype(np.float64)
        if a.std() < 1e-9 or b.std() < 1e-9:
            rp[i] = rs[i] = 0.0
            continue
        rp[i] = np.corrcoef(a, b)[0, 1]
        rs[i] = np.corrcoef(rankdata(a), rankdata(b))[0, 1]
    return rp, rs


def main() -> int:
    run_dir, h5mu, out_csv = sys.argv[1:4]
    n_cells = int(sys.argv[4]) if len(sys.argv) > 4 else 6000
    n_hvj   = int(sys.argv[5]) if len(sys.argv) > 5 else 3000
    seed    = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    rng = np.random.default_rng(seed)

    md = load_model(run_dir)
    # junc_specific_prior is NOT in the pickle -- A3 reads it from run_summary.csv (A3:1133)
    junc_prior = bool(load_run_summary(run_dir).get("junc_specific_prior", True))
    psi, K = md["psi"], md["K"]
    hvj = compute_hvj_indices(h5mu, n_hvj)
    psi_hvj = psi[:, hvj]

    with h5py.File(h5mu, "r") as f:
        s = f["mod/splicing"]
        N = int(s["obs/_index"].shape[0])
        idx = np.sort(rng.choice(N, size=min(n_cells, N), replace=False))
        srt = np.sort(hvj); uns = np.argsort(np.argsort(hvj))
        ori = np.asarray(s["layers/junc_ratio_original"][:, srt], np.float32)[:, uns][idx]
        atse = read_sparse_layer(s["layers/cell_by_cluster_matrix_original"])[:, hvj].toarray()[idx]
        junc = read_sparse_layer(s["layers/cell_by_junction_matrix"])[:, hvj].toarray()[idx]
        clus = read_sparse_layer(s["layers/cell_by_cluster_matrix"])[:, hvj].toarray()[idx]
        ctg = s["obs/broad_cell_type"]
        cats = [c.decode() if isinstance(c, bytes) else c for c in ctg["categories"][:]]
        ct = np.array([cats[c] if c >= 0 else "NA" for c in ctg["codes"][:]])[idx]
    cov = atse > 0
    print(f"  {len(idx):,} cells x {len(hvj):,} junctions; covered {int(cov.sum()):,} "
          f"({100*cov.mean():.1f}%); junctions per cell median {int(np.median(cov.sum(1)))}",
          flush=True)

    # frozen-globals E-step, the same call A3 uses
    print("  E-step (all globals frozen)...", flush=True)
    ad = build_chunk_adata(junc.astype(np.float32), clus.astype(np.float32))
    phi = infer_phi_chunk(ad, psi_hvj, K, md["dir_conc"], junc_prior,
                          md["pi"], md["alpha_pi"], md["bb_conc"], num_epochs=500, patience=20)
    phi = np.asarray(phi)
    pred_lf = phi @ psi_hvj

    # baselines, all computed on THESE cells' observed data
    jm = np.where(cov.sum(0) > 0, (ori * cov).sum(0) / np.maximum(cov.sum(0), 1), 0.0)
    pred_naive = np.tile(jm, (len(idx), 1))                       # the floor: junction mean
    pred_ct = np.zeros_like(pred_naive)                            # cell-type mean (uses labels)
    for t in np.unique(ct):
        r = ct == t
        c2 = cov[r]
        pred_ct[r] = np.where(c2.sum(0) > 0,
                              (ori[r] * c2).sum(0) / np.maximum(c2.sum(0), 1), jm)

    rows = []
    for name, pred in (("lf", pred_lf), ("mb_ct", pred_ct), ("naive", pred_naive)):
        rp, rs = percell_corr(pred, ori, cov)
        rows.append(dict(method=name,
                         median_pearson_percell=float(np.nanmedian(rp)),
                         median_spearman_percell=float(np.nanmedian(rs)),
                         mean_pearson_percell=float(np.nanmean(rp)),
                         n_cells_scored=int(np.isfinite(rp).sum())))
        print(f"    {name:<6} per-cell r = {np.nanmedian(rp):.4f}  "
              f"(spearman {np.nanmedian(rs):.4f})", flush=True)

    d = pd.DataFrame(rows)
    base = float(d.loc[d.method == "naive", "median_pearson_percell"].iloc[0])
    d["above_floor"] = d.median_pearson_percell - base
    d.to_csv(out_csv, index=False)

    print(f"\n=== per-CELL direction (across junctions within a cell) ===")
    print(d.to_string(index=False))
    print(f"\n  The junction-mean floor is {base:.4f}. It is high because junctions genuinely differ")
    print(f"  from each other, and that structure is free to any method. Quote `above_floor`.")
    lf = float(d.loc[d.method == "lf", "above_floor"].iloc[0])
    ctv = float(d.loc[d.method == "mb_ct", "above_floor"].iloc[0])
    print(f"  LeafletFA above floor: {lf:+.4f}   cell-type-label oracle above floor: {ctv:+.4f}")
    print(f"\nwrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
