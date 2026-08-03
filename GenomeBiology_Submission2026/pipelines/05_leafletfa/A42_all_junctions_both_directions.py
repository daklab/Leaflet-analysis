#!/usr/bin/env python
"""Is r=0.05 real, or an artefact of evaluating only on high-variance junctions?

    A42_all_junctions_both_directions.py <run_dir> <h5mu> <out_prefix> [n_cells=8000] [seed=0]

Everything reported so far was measured on the HVJ set: the 3,000 junctions with the highest
across-cell variance in observed PSI, expanded to their ATSEs (12,197). That selection is not
neutral. Observed variance = real biological spread + binomial noise, and the noise term is largest
where depth is lowest. Selecting on observed variance therefore preferentially picks junctions whose
spread is noise -- i.e. the least predictable ones -- and we then reported how badly we predict them.

This script removes the selection and measures both marginals:

  PER JUNCTION, across cells   "which cells splice this junction more?"   <- what A3 reports
  PER CELL, across junctions   "which junctions does this cell splice?"   <- never computed

on THREE junction sets fitted in the same E-step, so they are directly comparable:

  hvj     the 3,000-expanded set A3 uses
  random  an equal-sized random sample of junctions passing the same min-cells rule
  all     every junction with enough covered cells

Each is reported next to the junction-mean floor (`naive`). For the per-cell direction the floor is
HIGH -- junctions genuinely differ from one another and that structure is free -- so only the margin
above the floor means anything. For the per-junction direction the floor is exactly 0.

Cells are subsampled because the frozen-globals E-step is the cost; a median over thousands of cells
or junctions is already precise to ~0.002. Coverage is ATSE-level throughout
(cell_by_cluster_matrix_original > 0), matching A3.
"""
from __future__ import annotations
import sys, os, json
import numpy as np
import pandas as pd
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from A3_eval_imputation import (read_sparse_layer, load_model, load_run_summary,
                                build_chunk_adata, infer_phi_chunk)

MIN_CELLS = 50      # per junction, for it to be scorable across cells
MIN_JUNC  = 50      # per cell, for it to be scorable across junctions


def corr_along(pred, obs, mask, axis, min_n):
    """Pearson per row (axis=1) or per column (axis=0), over `mask`, ignoring the rest."""
    if axis == 0:
        pred, obs, mask = pred.T, obs.T, mask.T
    n = pred.shape[0]
    out = np.full(n, np.nan, np.float32)
    for i in range(n):
        m = mask[i]
        if m.sum() < min_n:
            continue
        a, b = pred[i, m].astype(np.float64), obs[i, m].astype(np.float64)
        sa, sb = a.std(), b.std()
        if sb < 1e-9:
            continue                      # target constant -> undefined, not zero
        if sa < 1e-9:
            out[i] = 0.0                  # prediction constant -> no information
            continue
        out[i] = np.corrcoef(a, b)[0, 1]
    return out


def main() -> int:
    run_dir, h5mu, out_prefix = sys.argv[1:4]
    n_cells = int(sys.argv[4]) if len(sys.argv) > 4 else 8000
    seed    = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    rng = np.random.default_rng(seed)

    md = load_model(run_dir)
    # junc_specific_prior is NOT in the pickle -- A3 reads it from run_summary.csv (A3:1133)
    junc_prior = bool(load_run_summary(run_dir).get("junc_specific_prior", True))
    psi, K = md["psi"], md["K"]
    J = psi.shape[1]

    with h5py.File(h5mu, "r") as f:
        s = f["mod/splicing"]
        N = int(s["obs/_index"].shape[0])
        idx = np.sort(rng.choice(N, size=min(n_cells, N), replace=False))
        print(f"  {len(idx):,} cells x {J:,} junctions (ALL junctions, no variance selection)",
              flush=True)
        ori  = np.asarray(s["layers/junc_ratio_original"], np.float32)[idx]
        atse = read_sparse_layer(s["layers/cell_by_cluster_matrix_original"]).tocsr()[idx].toarray()
        junc = read_sparse_layer(s["layers/cell_by_junction_matrix"]).tocsr()[idx].toarray()
        clus = read_sparse_layer(s["layers/cell_by_cluster_matrix"]).tocsr()[idx].toarray()
    cov = atse > 0
    print(f"  covered {int(cov.sum()):,} ({100*cov.mean():.1f}%)", flush=True)

    print("  E-step over ALL junctions (globals frozen)...", flush=True)
    ad = build_chunk_adata(junc.astype(np.float32), clus.astype(np.float32))
    phi = np.asarray(infer_phi_chunk(ad, psi, K, md["dir_conc"], junc_prior,
                                     md["pi"], md["alpha_pi"], md["bb_conc"],
                                     num_epochs=500, patience=20))
    pred = phi @ psi
    jm = np.where(cov.sum(0) > 0, (ori * cov).sum(0) / np.maximum(cov.sum(0), 1), 0.0)
    naive = np.tile(jm, (len(idx), 1))

    # junction sets, all scored from the SAME fit
    n_ok = cov.sum(0)
    ok = np.where(n_ok >= MIN_CELLS)[0]
    hvj_n = 12197
    var_j = np.array([ori[cov[:, j], j].var() if n_ok[j] >= MIN_CELLS else -1.0 for j in range(J)])
    sets = {"hvj":    np.argsort(var_j)[::-1][:hvj_n],
            "random": rng.choice(ok, size=min(hvj_n, len(ok)), replace=False),
            "all":    ok}

    rows = []
    for name, cols in sets.items():
        cols = np.sort(cols)
        for meth, P in (("lf", pred), ("naive", naive)):
            rj = corr_along(P[:, cols], ori[:, cols], cov[:, cols], axis=0, min_n=MIN_CELLS)
            rc = corr_along(P[:, cols], ori[:, cols], cov[:, cols], axis=1, min_n=MIN_JUNC)
            rows.append(dict(junction_set=name, n_junctions=len(cols), method=meth,
                             per_junction_r=float(np.nanmedian(rj)),
                             per_cell_r=float(np.nanmedian(rc)),
                             mean_atse_depth=float(atse[:, cols][cov[:, cols]].mean()),
                             median_cells_per_junction=float(np.median(n_ok[cols]))))
            print(f"    {name:<7} {meth:<6} per-junction {np.nanmedian(rj):+.4f}   "
                  f"per-cell {np.nanmedian(rc):+.4f}", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f"{out_prefix}_summary.csv", index=False)

    print("\n=== both directions, three junction sets, one E-step ===")
    for name in sets:
        a = d[(d.junction_set == name) & (d.method == "lf")].iloc[0]
        b = d[(d.junction_set == name) & (d.method == "naive")].iloc[0]
        print(f"\n  {name} ({int(a.n_junctions):,} junctions, median {int(a.median_cells_per_junction):,} "
              f"cells/junction, mean ATSE depth {a.mean_atse_depth:.1f})")
        print(f"    per-junction (across cells)   lf {a.per_junction_r:+.4f}   "
              f"floor {b.per_junction_r:+.4f}   margin {a.per_junction_r-b.per_junction_r:+.4f}")
        print(f"    per-cell (across junctions)   lf {a.per_cell_r:+.4f}   "
              f"floor {b.per_cell_r:+.4f}   margin {a.per_cell_r-b.per_cell_r:+.4f}")
    print(f"\nwrote {out_prefix}_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
