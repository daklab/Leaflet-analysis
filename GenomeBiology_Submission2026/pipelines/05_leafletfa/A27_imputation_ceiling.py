#!/usr/bin/env python
"""How high CAN the per-junction imputation correlation go? Two ceilings, both measured.

    A27_imputation_ceiling.py <h5mu> <out_csv> [n_hvj=3000] [min_cells=20] [bb_conc=1.97]

`median_pearson_lf` on the fixed entry set reads ~0.070. That is only interpretable against a
ceiling, and the metric has two distinct ones stacked on top of each other.

CEILING 1 -- SAMPLING NOISE IN THE TARGET.
    The "truth" is `junc_ratio_original`, an empirical ratio of finite counts (median ATSE depth 8),
    not the latent rate. At 8 reads a PSI of 0.5 carries a binomial SD of ~0.18, so even a predictor
    that knew each cell's true rate exactly could not correlate perfectly with the observed ratio.
    Measured by splitting each cell-ATSE's reads at random into two halves and correlating half A
    against half B across cells, per junction. Spearman-Brown corrects the half-length correlation to
    full length; the square root of that reliability bounds any predictor of the latent rate.

CEILING 2 -- OVERDISPERSION, WHICH IS LOWER AND IS THE BINDING ONE.
    The model predicts p = phi @ psi. Under the Beta-Binomial the cell's realised rate is
    q ~ Beta(p*c, (1-p)*c), so reads inform q, not p, and the two are only loosely tied:

        Var(q | p) = p(1-p) / (c + 1)

    Even a PERFECT p therefore correlates with q at only

        corr(p, q) = sqrt( Var(p) / ( Var(p) + E[p(1-p)] / (c+1) ) )

    computed per junction across cells. At the learned c ~ 1.97 each junction carries about c+1 ~ 3
    reads' worth of information about q regardless of sequencing depth -- which is the same mechanism
    behind the cross-seed phi irreproducibility.

Both are computed on the same junctions, with the same >=min_cells rule, as A3's
`per_junction_correlations`, so the numbers are directly comparable to `median_pearson_lf`.
"""
from __future__ import annotations
import sys, os
import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from A3_eval_imputation import read_sparse_layer, compute_hvj_indices   # identical HVJ selection


def main() -> int:
    h5mu, out_csv = sys.argv[1], sys.argv[2]
    n_hvj = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    min_cells = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    c = float(sys.argv[5]) if len(sys.argv) > 5 else 1.97
    rng = np.random.default_rng(0)

    hvj = compute_hvj_indices(h5mu, n_hvj)
    print(f"\nLoading original-depth counts for {len(hvj):,} junctions...", flush=True)
    with h5py.File(h5mu, "r") as f:
        s = f["mod/splicing"]
        srt = np.sort(hvj); uns = np.argsort(np.argsort(hvj))
        jr = np.asarray(s["layers/junc_ratio_original"][:, srt], dtype=np.float32)[:, uns]
        cl = read_sparse_layer(s["layers/cell_by_cluster_matrix_original"])[:, hvj].toarray()
    cl = cl.astype(np.float32)
    # no cell_by_junction_matrix_original layer exists; reconstruct it the way A4/A5/A7 do
    jc = np.rint(np.nan_to_num(jr) * cl).astype(np.int64)
    cov = cl > 0
    print(f"  covered entries: {int(cov.sum()):,}", flush=True)

    # ---- ceiling 1: split-half reliability of the target -------------------------------------
    # Split each junction's reads binomially; the ATSE denominator is the sum over its junctions,
    # which for this HVJ matrix is approximated per-entry by splitting the cluster total the same
    # way. Both halves are then independent draws at half depth.
    a_j = rng.binomial(jc, 0.5)
    b_j = jc - a_j
    cl_i = np.rint(cl).astype(np.int64)
    a_c = rng.binomial(cl_i, 0.5)
    b_c = cl_i - a_c
    # keep the numerator consistent with its own denominator
    a_j = np.minimum(a_j, a_c)
    b_j = np.minimum(b_j, b_c)
    with np.errstate(invalid="ignore", divide="ignore"):
        psi_a = np.where(a_c > 0, a_j / np.maximum(a_c, 1), np.nan)
        psi_b = np.where(b_c > 0, b_j / np.maximum(b_c, 1), np.nan)

    # ---- ceiling 2: overdispersion, from the observed spread of the per-cell rate -------------
    rows = []
    for j in range(len(hvj)):
        m = cov[:, j]
        if m.sum() < min_cells:
            continue
        rec = dict(junction=int(hvj[j]), n_cells=int(m.sum()))

        aa, bb = psi_a[m, j], psi_b[m, j]
        ok = np.isfinite(aa) & np.isfinite(bb)
        if ok.sum() >= min_cells and np.std(aa[ok]) > 1e-8 and np.std(bb[ok]) > 1e-8:
            r_half = float(np.corrcoef(aa[ok], bb[ok])[0, 1])
            rel = 2 * r_half / (1 + r_half) if r_half > -1 else np.nan   # Spearman-Brown
            rec["r_splithalf"] = r_half
            rec["ceiling_sampling"] = float(np.sqrt(rel)) if rel and rel > 0 else 0.0
        else:
            rec["r_splithalf"] = rec["ceiling_sampling"] = np.nan

        # Var(p) across cells is unknown without the model, so use the observed full-depth spread as
        # its upper bound: Var(psi_obs) >= Var(q) >= Var(p). This makes ceiling 2 GENEROUS.
        obs = jr[m, j].astype(np.float64)
        vp = float(np.var(obs))
        ep = float(np.mean(obs * (1 - obs)))
        rec["ceiling_overdisp"] = float(np.sqrt(vp / (vp + ep / (c + 1)))) if vp > 0 else 0.0
        rows.append(rec)

    d = pd.DataFrame(rows)
    d.to_csv(out_csv, index=False)
    print(f"\n=== ceilings on median per-junction Pearson ({len(d):,} junctions, c={c}) ===")
    print(f"  ceiling 1, sampling noise in the target : {d.ceiling_sampling.median():.3f} "
          f"(IQR {d.ceiling_sampling.quantile(.25):.3f}-{d.ceiling_sampling.quantile(.75):.3f})")
    print(f"  ceiling 2, overdispersion at c={c}      : {d.ceiling_overdisp.median():.3f} "
          f"(IQR {d.ceiling_overdisp.quantile(.25):.3f}-{d.ceiling_overdisp.quantile(.75):.3f})")
    both = np.minimum(d.ceiling_sampling, d.ceiling_overdisp)
    print(f"  BINDING ceiling (min of the two)        : {np.nanmedian(both):.3f}")
    print(f"\n  LeafletFA measured on the fixed entry set: ~0.070")
    print(f"  -> that is {0.070/max(np.nanmedian(both),1e-9):.0%} of the binding ceiling")
    print(f"\nwrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
