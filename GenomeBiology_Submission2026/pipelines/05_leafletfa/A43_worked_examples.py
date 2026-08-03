#!/usr/bin/env python
"""Concrete worked examples: what does each method actually predict, entry by entry?

    A43_worked_examples.py <run_dir> <h5mu_downsampled> <out_csv> [n_cells=2500] [n_hvj=3000]

Every number so far has been a median over 12,197 junctions. This prints the underlying rows so the
behaviour is visible rather than summarised:

    original counts (junc / ATSE) and the PSI they imply   <- the TARGET
    the same entry after downsampling                      <- what the model is given
    lf     = phi @ psi
    lf_bb  = (lf*c + junc_ds) / (lf*c + (1-lf)*c + atse_ds)
    mb     = observed PSI where observed, else junction posterior mean  <- PASSTHROUGH when observed

`lf` does NOT ignore this entry's reads. phi_c is inferred by the E-step from ALL of cell c's
downsampled counts, junction j included, so entry (c, j) does influence the prediction -- diluted to
roughly one junction in the ~2,346 that cell covers, but not zero. `lf` differs from `lf_bb` in that
it does not use junc_ds[c, j] DIRECTLY in the posterior update, not in never having seen it.

The one place `lf` is genuinely blind is an entry with no reads at this coverage level: it then
contributes nothing to the E-step, and `lf_bb` collapses to exactly `lf` (verified identical to
machine precision on lost_to_downsampling at all seven levels).

Rows are stratified so the interesting cases are all visible: entries that kept most of their reads,
entries reduced to a single read, and entries that lost every read (where lf_bb collapses to lf).
"""
from __future__ import annotations
import sys, os
import numpy as np
import pandas as pd
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from A3_eval_imputation import (read_sparse_layer, compute_hvj_indices, load_model,
                                load_run_summary, build_chunk_adata, infer_phi_chunk)


def main() -> int:
    run_dir, h5mu, out_csv = sys.argv[1:4]
    n_cells = int(sys.argv[4]) if len(sys.argv) > 4 else 2500
    n_hvj   = int(sys.argv[5]) if len(sys.argv) > 5 else 3000
    rng = np.random.default_rng(0)

    md = load_model(run_dir)
    # junc_specific_prior is NOT in the pickle -- A3 reads it from run_summary.csv (A3:1133)
    junc_prior = bool(load_run_summary(run_dir).get("junc_specific_prior", True))
    psi, K, c = md["psi"], md["K"], float(md["bb_conc"])
    hvj = compute_hvj_indices(h5mu, n_hvj)
    psi_h = psi[:, hvj]

    with h5py.File(h5mu, "r") as f:
        s = f["mod/splicing"]
        N = int(s["obs/_index"].shape[0])
        idx = np.sort(rng.choice(N, size=min(n_cells, N), replace=False))
        srt = np.sort(hvj); uns = np.argsort(np.argsort(hvj))
        ori_psi = np.asarray(s["layers/junc_ratio_original"][:, srt], np.float32)[:, uns][idx]
        atse_o = read_sparse_layer(s["layers/cell_by_cluster_matrix_original"])[:, hvj].toarray()[idx]
        junc_d = read_sparse_layer(s["layers/cell_by_junction_matrix"])[:, hvj].toarray()[idx]
        atse_d = read_sparse_layer(s["layers/cell_by_cluster_matrix"])[:, hvj].toarray()[idx]
        gn = s["var/gene_name"]
        gcat = [x.decode() if isinstance(x, bytes) else x for x in gn["categories"][:]]
        gene = np.array([gcat[i] if i >= 0 else "NA" for i in gn["codes"][:]])[hvj]
    junc_o = np.rint(np.nan_to_num(ori_psi) * atse_o)          # no *_original junction layer exists
    cov = atse_o > 0
    obs = atse_d > 0

    print(f"  E-step on {len(idx):,} cells (globals frozen)...", flush=True)
    ad = build_chunk_adata(junc_d.astype(np.float32), atse_d.astype(np.float32))
    phi = np.asarray(infer_phi_chunk(ad, psi_h, K, md["dir_conc"], junc_prior,
                                     md["pi"], md["alpha_pi"], md["bb_conc"],
                                     num_epochs=500, patience=20))
    lf = phi @ psi_h
    lf_bb = (lf * c + junc_d) / (lf * c + (1 - lf) * c + atse_d)
    ds_psi = np.where(obs, junc_d / np.maximum(atse_d, 1), np.nan)
    # mb: observed PSI passed through unchanged, junction posterior mean elsewhere
    jmean = np.where(obs.sum(0) > 0,
                     (np.nan_to_num(ds_psi) * obs).sum(0) / np.maximum(obs.sum(0), 1), 0.5)
    mb = np.where(obs, np.nan_to_num(ds_psi), jmean[None, :])

    ci, ji = np.where(cov)
    d = pd.DataFrame(dict(
        cell=idx[ci], junction=hvj[ji], gene=gene[ji],
        junc_orig=junc_o[ci, ji].astype(int), atse_orig=atse_o[ci, ji].astype(int),
        psi_orig=ori_psi[ci, ji],
        junc_ds=junc_d[ci, ji].astype(int), atse_ds=atse_d[ci, ji].astype(int),
        psi_ds=ds_psi[ci, ji],
        lf=lf[ci, ji], lf_bb=lf_bb[ci, ji], mb=mb[ci, ji]))
    d["kept_all_reads"] = d.atse_ds >= d.atse_orig
    d["lost_everything"] = d.atse_ds == 0
    d.to_csv(out_csv, index=False)

    strata = [("deep, reads kept (atse_orig>=20, atse_ds>=10)",
               (d.atse_orig >= 20) & (d.atse_ds >= 10)),
              ("shallow, one read left (atse_ds==1)", d.atse_ds == 1),
              ("LOST every read (atse_ds==0)", d.atse_ds == 0)]
    for name, m in strata:
        sub = d[m]
        if not len(sub):
            continue
        print(f"\n### {name}   n={len(sub):,}")
        print(sub.sample(min(8, len(sub)), random_state=1)[
            ["gene", "junc_orig", "atse_orig", "psi_orig", "junc_ds", "atse_ds",
             "psi_ds", "lf", "lf_bb", "mb"]].to_string(index=False,
             float_format=lambda v: f"{v:.3f}"))
        print(f"  mean |err|   lf {np.abs(sub.lf-sub.psi_orig).mean():.4f}   "
              f"lf_bb {np.abs(sub.lf_bb-sub.psi_orig).mean():.4f}   "
              f"mb {np.abs(sub.mb-sub.psi_orig).mean():.4f}")
    print(f"\nwrote {out_csv}  ({len(d):,} covered entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
