#!/usr/bin/env python
"""How high CAN the per-junction imputation correlation go? Two ceilings, both measured.

    A40_imputation_ceiling.py <h5mu> <out_csv> [n_hvj=3000] [min_cells=20] [bb_conc=1.97]

`median_pearson_lf` on the fixed entry set reads ~0.070. That is only interpretable against a
ceiling, and the metric has two distinct ones stacked on top of each other.

CEILING 1 -- SAMPLING NOISE IN THE TARGET.
    The "truth" is `junc_ratio_original`, an empirical ratio of finite counts (median ATSE depth 8),
    not the latent rate. At 8 reads a PSI of 0.5 carries a binomial SD of ~0.18, so even a predictor
    that knew each cell's true rate exactly could not correlate perfectly with the observed ratio.
    Measured by splitting each cell-ATSE's reads at random into two halves and correlating half A
    against half B across cells, per junction. Spearman-Brown corrects the half-length correlation to
    full length; the square root of that reliability bounds any predictor of the latent rate.

CEILING 2 -- BOTH NOISE SOURCES AT ONCE, AND THE ONE TO QUOTE.
    The model predicts p = phi @ psi. Under the Beta-Binomial the cell's realised rate is
    q ~ Beta(p*c, (1-p)*c) and the observation is q thinned by finite reads. Since E[q|p] = p we have
    Cov(p, obs) = Var(p), so the best correlation ANY predictor of p can reach is

        ceiling = sqrt( Var(p) / Var(obs) )        with
        Var(obs) = Var(p)  +  E[p(1-p)]/(c+1)  +  E[q(1-q)/T]
                              ^ overdispersion     ^ finite reads

    Var(p) is recovered by subtracting the two noise terms from the observed variance. An earlier
    version of this script used Var(obs) itself in place of Var(p), which ignores both subtracted
    terms and inflates the ceiling substantially -- do not reintroduce that shortcut.

    At the learned c ~ 1.97 each junction carries about c+1 ~ 3 reads' worth of information about q
    regardless of sequencing depth, the same mechanism behind the cross-seed phi irreproducibility.

Ceiling 2 subsumes ceiling 1 and is the number to quote; ceiling 1 is kept because it is
model-free -- it needs no assumption about c -- and so is a useful independent check.

Note both are computed on HIGH-VARIANCE junctions by construction (the HVJ set is selected on
across-cell PSI variance), which raises the ceiling relative to a random junction. That is the
right comparison here, since it is the same set `median_pearson_lf` is measured on.

READ THIS BEFORE QUOTING THE GAP -- the ceiling is a valid bound but a LOOSE one for imputation.
    Both ceilings are achievable only by a predictor that can see the entry's OWN reads: the
    split-half estimate literally splits those reads, and the variance decomposition asks how much
    of Var(obs) is signal. `lf` = phi @ psi does the opposite -- it predicts entry (c, j) WITHOUT
    using that entry, from the cell's other junctions through phi_c and the global psi_j.

    So "0.070 is 10% of 0.736" mixes two different questions. 0.736 says the target contains
    recoverable per-cell signal, i.e. the metric is not noise-saturated and the low value is not an
    artefact of an unpredictable target. It does NOT say a method that never observes the entry
    should reach 0.736, and the gap should not be read as pure model deficiency.

    The fair comparators are the other methods that also do not see the entry:
        LeafletFA (lf)        0.070
        NMF transfer 0-imp    0.030
        NMF transfer mean-imp 0.064
        naive (junction mean) 0.000   <- exact, by construction
    Against those LeafletFA leads. Decomposing the 0.070-to-0.736 gap into "imputation is
    intrinsically harder than observation" versus "the model is leaving signal behind" would need a
    third quantity -- e.g. an oracle that sees every OTHER junction in the cell at full depth -- and
    is not measured here.

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

        # The quantity that actually bounds the metric. obs = q + sampling noise and E[q|p] = p, so
        # Cov(p, obs) = Var(p) and the best achievable correlation is sqrt(Var(p)/Var(obs)). This
        # subsumes BOTH noise sources rather than treating them separately:
        #
        #     Var(obs) = Var(p)  +  E[p(1-p)]/(c+1)   +  E[q(1-q)/T]
        #                           ^ overdispersion     ^ finite reads
        #
        # Using Var(obs) itself as Var(p) -- as an earlier version did -- ignores both subtracted
        # terms and inflates the ceiling badly. Plugging obs into the two noise terms slightly
        # OVER-estimates them (obs is noisier than p or q), so Var(p) here is if anything an
        # under-estimate and the ceiling is conservative. Reported next to the split-half figure,
        # which bounds sampling noise alone and is therefore always the looser of the two.
        obs = jr[m, j].astype(np.float64)
        T = np.maximum(cl[m, j].astype(np.float64), 1.0)
        v_obs = float(np.var(obs))
        pq = obs * (1 - obs)
        v_over = float(np.mean(pq)) / (c + 1.0)
        v_samp = float(np.mean(pq / T))
        v_p = max(v_obs - v_over - v_samp, 0.0)
        rec["var_obs"], rec["var_p_est"] = v_obs, v_p
        rec["ceiling_overdisp"] = float(np.sqrt(v_p / v_obs)) if v_obs > 0 else 0.0
        rows.append(rec)

    d = pd.DataFrame(rows)
    d.to_csv(out_csv, index=False)
    print(f"\n=== ceilings on median per-junction Pearson ({len(d):,} junctions, c={c}) ===")
    print(f"  ceiling 1, sampling noise in the target : {d.ceiling_sampling.median():.3f} "
          f"(IQR {d.ceiling_sampling.quantile(.25):.3f}-{d.ceiling_sampling.quantile(.75):.3f})")
    print(f"  ceiling 2, sqrt(Var(p)/Var(obs)) at c={c} : {d.ceiling_overdisp.median():.3f} "
          f"(IQR {d.ceiling_overdisp.quantile(.25):.3f}-{d.ceiling_overdisp.quantile(.75):.3f})")
    both = np.minimum(d.ceiling_sampling, d.ceiling_overdisp)
    print(f"  BINDING ceiling (min of the two)        : {np.nanmedian(both):.3f}")
    print(f"\n  LeafletFA measured on the fixed entry set: ~0.070")
    print(f"  -> that is {0.070/max(np.nanmedian(both),1e-9):.0%} of the binding ceiling")
    print(f"\nwrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
