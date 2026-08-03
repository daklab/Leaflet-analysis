#!/usr/bin/env python
"""Track the SAME ATSEs through every downsampling level. Two methods only.

    A44_atse_across_coverage.py <run_dir> <resampled_dir> <out_csv> [n_cells=2000] [n_hvj=3000]

    lf     = phi @ psi                    the model alone
    lf_bb  = (lf*c + junc_ds) / (lf*c + (1-lf)*c + atse_ds)
                                          the model as a Beta prior, updated with the counts that
                                          survive at this coverage level

Nothing else. No cell-type baselines, no junction means.

One frozen-globals E-step PER LEVEL on the SAME cells and the SAME junctions, so a row can be
followed down the coverage axis and the only thing changing is how many reads the model was given.

Output is one row per (level, cell, junction) with the counts and both predictions, so any ATSE can
be pulled out and read directly. PSI within an ATSE sums to 1 across its junctions, so ATSEs are kept
whole -- printing a single junction hides whether the model got the split right.
"""
from __future__ import annotations
import sys, os, re, glob
import numpy as np
import pandas as pd
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from A3_eval_imputation import (read_sparse_layer, compute_hvj_indices, load_model,
                                load_run_summary, build_chunk_adata, infer_phi_chunk)


def main() -> int:
    run_dir, resampled_dir, out_csv = sys.argv[1:4]
    n_cells = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
    n_hvj   = int(sys.argv[5]) if len(sys.argv) > 5 else 3000

    md = load_model(run_dir)
    junc_prior = bool(load_run_summary(run_dir).get("junc_specific_prior", True))
    psi, K, c = md["psi"], md["K"], float(md["bb_conc"])

    files = sorted(glob.glob(os.path.join(resampled_dir, "RESAMPLED_*_PERCENT_test_*.h5mu")),
                   key=lambda p: int(re.search(r"RESAMPLED_(\d+)_PERCENT", p).group(1)))
    # percent RETAINED; the last level is 99.9% dropped, not 100
    retained = {p: (0.1 if int(re.search(r"RESAMPLED_(\d+)_PERCENT", p).group(1)) >= 100
                    else 100 - int(re.search(r"RESAMPLED_(\d+)_PERCENT", p).group(1))) for p in files}
    print(f"  {len(files)} levels: {sorted(retained.values(), reverse=True)}", flush=True)

    hvj = compute_hvj_indices(files[0], n_hvj)
    psi_h = psi[:, hvj]
    rng = np.random.default_rng(0)

    # fixed across levels: same cells, same junctions, same ATSE annotation
    with h5py.File(files[0], "r") as f:
        s = f["mod/splicing"]
        N = int(s["obs/_index"].shape[0])
        idx = np.sort(rng.choice(N, size=min(n_cells, N), replace=False))
        srt = np.sort(hvj); uns = np.argsort(np.argsort(hvj))
        ori_psi = np.asarray(s["layers/junc_ratio_original"][:, srt], np.float32)[:, uns][idx]
        atse_o = read_sparse_layer(s["layers/cell_by_cluster_matrix_original"])[:, hvj].toarray()[idx]
        ev = s["var/event_id"]
        ecat = [x.decode() if isinstance(x, bytes) else x for x in ev["categories"][:]]
        event = np.array([ecat[i] if i >= 0 else "NA" for i in ev["codes"][:]])[hvj]
        gn = s["var/gene_name"]
        gcat = [x.decode() if isinstance(x, bytes) else x for x in gn["categories"][:]]
        gene = np.array([gcat[i] if i >= 0 else "NA" for i in gn["codes"][:]])[hvj]
    junc_o = np.rint(np.nan_to_num(ori_psi) * atse_o)
    cov = atse_o > 0
    print(f"  {len(idx):,} cells x {len(hvj):,} junctions, {int(cov.sum()):,} covered", flush=True)

    out = []
    for p in files:
        lvl = retained[p]
        with h5py.File(p, "r") as f:
            s = f["mod/splicing"]
            junc_d = read_sparse_layer(s["layers/cell_by_junction_matrix"])[:, hvj].toarray()[idx]
            atse_d = read_sparse_layer(s["layers/cell_by_cluster_matrix"])[:, hvj].toarray()[idx]
        print(f"  E-step at {lvl}% retained...", flush=True)
        ad = build_chunk_adata(junc_d.astype(np.float32), atse_d.astype(np.float32))
        phi = np.asarray(infer_phi_chunk(ad, psi_h, K, md["dir_conc"], junc_prior,
                                         md["pi"], md["alpha_pi"], md["bb_conc"],
                                         num_epochs=500, patience=20))
        lf = phi @ psi_h
        lf_bb = (lf * c + junc_d) / (lf * c + (1 - lf) * c + atse_d)
        ci, ji = np.where(cov)
        out.append(pd.DataFrame(dict(
            retained=lvl, cell=idx[ci], junction=hvj[ji], atse=event[ji], gene=gene[ji],
            junc_orig=junc_o[ci, ji].astype(int), atse_orig=atse_o[ci, ji].astype(int),
            psi_orig=ori_psi[ci, ji],
            junc_ds=junc_d[ci, ji].astype(int), atse_ds=atse_d[ci, ji].astype(int),
            lf=lf[ci, ji], lf_bb=lf_bb[ci, ji])))
        del junc_d, atse_d, phi, lf, lf_bb

    d = pd.concat(out, ignore_index=True)
    d.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}  ({len(d):,} rows)")

    # ---- readable worked examples: whole ATSEs, followed down the coverage axis ----
    full = d[d.retained == 90]
    # pick ATSEs that are deep, multi-junction, and not trivially all-zero
    cand = (full.groupby(["cell", "atse"])
                .agg(n_j=("junction", "size"), depth=("atse_orig", "max"),
                     spread=("psi_orig", lambda x: x.max() - x.min()))
                .query("n_j >= 3 and depth >= 30 and spread > 0.3")
                .sort_values("depth", ascending=False))
    print(f"\n  {len(cand):,} (cell, ATSE) pairs are deep (>=30 reads), multi-junction, informative")
    for (cell, atse) in list(cand.index)[:4]:
        sub = d[(d.cell == cell) & (d.atse == atse)].sort_values(["junction", "retained"],
                                                                 ascending=[True, False])
        g = sub.gene.iloc[0]
        print(f"\n{'='*84}\n  cell {cell}   ATSE {atse}   gene {g}   "
              f"original depth {sub.atse_orig.max()} reads")
        for j in sorted(sub.junction.unique()):
            t = sub[sub.junction == j]
            true = t.psi_orig.iloc[0]
            print(f"\n    junction {j}   TRUE psi = {true:.3f}  ({int(t.junc_orig.iloc[0])}/"
                  f"{int(t.atse_orig.iloc[0])} reads)")
            print(f"      {'retained':>9}{'reads':>10}{'lf':>9}{'lf_bb':>9}"
                  f"{'|lf-true|':>11}{'|lf_bb-true|':>14}")
            for _, r in t.iterrows():
                print(f"      {r.retained:>8}%{int(r.junc_ds):>5}/{int(r.atse_ds):<4}"
                      f"{r.lf:>9.3f}{r.lf_bb:>9.3f}{abs(r.lf-true):>11.3f}{abs(r.lf_bb-true):>14.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
