#!/usr/bin/env python
"""
Step 04d: NMF imputation baseline evaluation on highly variable junctions.

Trains MiniBatchNMF on training data PSI ratios, projects test cells via
NMF.transform() on downsampled PSI, reconstructs W @ H, and evaluates
per-junction Pearson across cells at each dropout level.

MeanBayes and naive mean baselines are included for direct comparison.

Methods:
  nmf   — W_test @ H, where H is trained on training junc_ratio and
           W_test = nmf.transform(downsampled junc_ratio on test cells)
  mb    — Beta conjugate posterior mean (population statistics only)
  naive — Per-junction mean over observed cells (constant predictor; scores ~0)

Metrics:
  Per-junction Pearson across cells (primary): tests whether the model ranks
  cells correctly for each variable junction. Constant predictors score 0.
  Per-cell Pearson across junctions (secondary): tests per-cell PSI profiles.

Usage:
    python 04d_evaluate_imputation_nmf.py <train_h5ad> <resampled_dir> <output_csv>
        [K=20] [n_hvj=10000] [min_atse=30] [min_cells_per_junc=20]
"""

import sys
import os
import glob
import re
import h5py
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix, issparse
from scipy.stats import rankdata
from sklearn.decomposition import MiniBatchNMF

sys.path.insert(0, os.path.dirname(__file__))
from mean_bayes import MeanBayes


# ── shared helpers ────────────────────────────────────────────────────────────

def read_sparse_layer(grp):
    data    = grp["data"][:]
    indices = grp["indices"][:]
    indptr  = grp["indptr"][:]
    n_rows  = len(indptr) - 1
    n_cols  = int(indices.max()) + 1 if len(indices) else 0
    shape   = tuple(grp.attrs.get("shape", (n_rows, n_cols)))
    return csr_matrix((data, indices, indptr), shape=shape)


def compute_hvj_indices(h5mu_path, n_hvj, expand_to_atse=True):
    """Select top n_hvj junctions by per-cell PSI variance (≥50 cells covered),
    then (default) expand to every junction sharing an ATSE (var/event_id) with a
    selected HVJ — PSI is only meaningful relative to its ATSE. Kept identical to
    04e/04c so LeafletFA and NMF are evaluated on the SAME junction set."""
    print(f"Computing HVJ from: {os.path.basename(h5mu_path)}")
    with h5py.File(h5mu_path, "r") as f:
        ori = np.array(f["mod/splicing/layers/junc_ratio_original"], dtype=np.float32)
        event_codes = f["mod/splicing/var/event_id/codes"][:] if expand_to_atse else None
    covered   = ori > 0
    n_covered = covered.sum(axis=0).astype(np.float32)
    sum_x     = ori.sum(axis=0)
    mean_j    = np.where(n_covered > 0, sum_x / np.maximum(n_covered, 1), 0.0)
    diff_sq   = (ori - mean_j) ** 2 * covered
    var_j     = np.where(n_covered > 1,
                         diff_sq.sum(axis=0) / np.maximum(n_covered - 1, 1),
                         0.0)
    var_j[n_covered < 50] = 0.0
    top_idx = np.sort(np.argsort(var_j)[::-1][:n_hvj])
    print(f"  HVJ variance range: {var_j[top_idx].min():.4f} – {var_j[top_idx].max():.4f}")

    if not expand_to_atse or event_codes is None:
        return top_idx
    hvj_events = np.unique(event_codes[top_idx])
    hvj_events = hvj_events[hvj_events >= 0]
    keep = np.isin(event_codes, hvj_events)
    keep[top_idx] = True
    expanded = np.sort(np.where(keep)[0])
    print(f"  Expanded {len(top_idx):,} HVJ → {len(expanded):,} junctions "
          f"across {len(hvj_events):,} ATSEs")
    return expanded


def per_junction_correlations(imp, ori_hvj, cov_mask, min_cells):
    """Pearson and Spearman r per junction across cells with original coverage."""
    n_hvj      = imp.shape[1]
    r_pearson  = np.full(n_hvj, np.nan, dtype=np.float32)
    r_spearman = np.full(n_hvj, np.nan, dtype=np.float32)
    for j in range(n_hvj):
        mask_j = cov_mask[:, j]
        if mask_j.sum() < min_cells:
            continue
        pred_j = imp[mask_j, j]
        true_j = ori_hvj[mask_j, j]
        if true_j.std() < 1e-8:
            r_pearson[j] = r_spearman[j] = 0.0
            continue
        if pred_j.std() < 1e-8:
            r_pearson[j] = r_spearman[j] = 0.0
            continue
        pc = pred_j - pred_j.mean(); tc = true_j - true_j.mean()
        denom = np.sqrt((pc**2).sum() * (tc**2).sum())
        r_pearson[j] = float((pc * tc).sum() / denom) if denom > 1e-8 else 0.0
        pr = rankdata(pred_j, method="average").astype(np.float32)
        tr = rankdata(true_j, method="average").astype(np.float32)
        pc = pr - pr.mean(); tc = tr - tr.mean()
        denom = np.sqrt((pc**2).sum() * (tc**2).sum())
        r_spearman[j] = float((pc * tc).sum() / denom) if denom > 1e-8 else 0.0
    return r_pearson, r_spearman


def summarise_perjunc(r_pearson, r_spearman, label):
    valid_p = r_pearson[~np.isnan(r_pearson)]
    valid_s = r_spearman[~np.isnan(r_spearman)]
    n = len(valid_p)
    if n == 0:
        return {"n_junc_eval": 0,
                f"median_pearson_{label}":      np.nan,
                f"mean_pearson_{label}":        np.nan,
                f"frac_r_gt_03_{label}":        np.nan,
                f"frac_r_gt_05_{label}":        np.nan,
                f"median_spearman_{label}":     np.nan,
                f"frac_spearman_gt_03_{label}": np.nan}
    return {
        "n_junc_eval":                  n,
        f"median_pearson_{label}":      float(np.median(valid_p)),
        f"mean_pearson_{label}":        float(np.mean(valid_p)),
        f"frac_r_gt_03_{label}":        float((valid_p > 0.3).mean()),
        f"frac_r_gt_05_{label}":        float((valid_p > 0.5).mean()),
        f"median_spearman_{label}":     float(np.median(valid_s)),
        f"frac_spearman_gt_03_{label}": float((valid_s > 0.3).mean()),
    }


def per_cell_pearson(imp, ori_hvj, cov_mask, min_junc=10):
    """Pearson r per cell across its covered HVJ junctions."""
    N      = imp.shape[0]
    r_vals = np.full(N, np.nan, dtype=np.float32)
    n_cov  = cov_mask.sum(axis=1)
    for i in np.where(n_cov >= min_junc)[0]:
        mask_i = cov_mask[i, :]
        pred_i = imp[i, mask_i]; true_i = ori_hvj[i, mask_i]
        pc = pred_i - pred_i.mean(); tc = true_i - true_i.mean()
        denom = np.sqrt((pc**2).sum() * (tc**2).sum())
        r_vals[i] = float((pc * tc).sum() / denom) if denom > 1e-8 else 0.0
    return r_vals


def summarise_percell(r_vals, label):
    valid = r_vals[~np.isnan(r_vals)]
    if len(valid) == 0:
        return {"n_cells_eval": 0,
                f"median_cellpearson_{label}": np.nan,
                f"mean_cellpearson_{label}":   np.nan}
    return {
        "n_cells_eval":                 len(valid),
        f"median_cellpearson_{label}":  float(np.median(valid)),
        f"mean_cellpearson_{label}":    float(np.mean(valid)),
    }


# ── NMF training ──────────────────────────────────────────────────────────────

def train_nmf(train_h5ad, K, batch_size=2048):
    """Train MiniBatchNMF on ALL training junctions (sparse junc_ratio).

    Training on all junctions mirrors how LeafletFA is trained: psi is K × J_full.
    HVJ subsetting is only for evaluation, not model fitting.
    """
    print(f"\n[NMF] Training K={K} on full junction set (all junctions)...")
    adata    = ad.read_h5ad(train_h5ad, backed="r")
    jr_layer = adata.layers.get("junc_ratio", adata.X)

    # Keep sparse — sklearn MiniBatchNMF accepts CSR matrices
    if issparse(jr_layer):
        # Clip in-place via data attribute to avoid densifying the full matrix
        from scipy.sparse import csr_matrix as _csr
        jr_csr = jr_layer.tocsr().astype(np.float32)
        jr_csr.data = np.clip(jr_csr.data, 0.0, 1.0)
        X_train = jr_csr
    else:
        X_train = np.clip(np.asarray(jr_layer, dtype=np.float32), 0.0, 1.0)

    n_cells, n_junc = X_train.shape
    print(f"  X_train: {n_cells} cells × {n_junc} junctions  "
          f"nnz={X_train.nnz if issparse(X_train) else (X_train>0).sum():,}")

    try:
        adata.file.close()
    except Exception:
        pass

    nmf = MiniBatchNMF(
        n_components=K,
        init="nndsvda",
        batch_size=batch_size,
        random_state=42,
        verbose=False,
    )
    W_train = nmf.fit_transform(X_train)
    H       = nmf.components_   # (K × J_full)
    print(f"  Trained. W={W_train.shape}  H={H.shape}")
    del X_train, W_train
    return nmf, H


# ── per-level evaluation ──────────────────────────────────────────────────────

def evaluate_one_level(h5mu_path, nmf, H, hvj_idx,
                       min_atse, min_cells_per_junc, min_junc_per_cell=10):
    n_hvj  = len(hvj_idx)
    J_full = H.shape[1]   # all junctions (H is K × J_full)

    with h5py.File(h5mu_path, "r") as f:
        splice = f["mod/splicing"]
        N = int(splice["obs/_index"].shape[0])

        # Load full junc_ratio (all junctions) for NMF.transform()
        print(f"  Loading junc_ratio full (N={N} × J={J_full}) for NMF projection...")
        jr_full = read_sparse_layer(splice["layers/junc_ratio"])

        print(f"  Loading psi_mask + ground truth...")
        pm_full  = read_sparse_layer(splice["layers/psi_mask"])
        ori_full = np.array(splice["layers/junc_ratio_original"], dtype=np.float32)
        atse_sp  = read_sparse_layer(splice["layers/cell_by_cluster_matrix_original"])

        ds_grp  = splice["obs/dataset"]
        ds_cats = [c.decode() if isinstance(c, bytes) else c
                   for c in ds_grp["categories"][:]]
        ds_codes = ds_grp["codes"][:]
        dataset_labels = np.array([ds_cats[c] for c in ds_codes])

    ori_hvj  = ori_full[:, hvj_idx]; del ori_full
    atse_hvj = atse_sp[:, hvj_idx].toarray().astype(np.float32)
    jr_hvj   = jr_full[:, hvj_idx]    # HVJ subset for MeanBayes / naive
    pm_hvj   = pm_full[:, hvj_idx]

    # Coverage mask: original ATSE > 0 (any cell with reads at the junction).
    # Includes PSI=0 cells (junc=0, atse>0) so the model is tested across the full [0,1] range.
    cov_mask = atse_hvj > 0
    print(f"  Coverage (original ATSE>0): {cov_mask.sum():,} pairs  "
          f"({(cov_mask.sum(axis=0) >= min_cells_per_junc).sum()} evaluable junctions)")

    # ── NMF — project on ALL junctions, reconstruct HVJ subset ───────────────
    # Mirrors LeafletFA: psi trained on all J, phi inferred from all J, evaluate on HVJ
    print(f"\n  [NMF] Projecting test cells via transform() on all {J_full} junctions...")
    jr_full_csr = jr_full.tocsr().astype(np.float32)
    jr_full_csr.data = np.clip(jr_full_csr.data, 0.0, 1.0)
    W_test  = nmf.transform(jr_full_csr)                            # (N × K)
    del jr_full_csr
    imp_nmf = np.clip((W_test @ H[:, hvj_idx]).astype(np.float32),
                      0.0, 1.0)                                      # (N × n_hvj)
    del W_test

    # ── MeanBayes ─────────────────────────────────────────────────────────────
    print(f"\n  [MeanBayes] Fitting Beta prior on HVJ subset...")
    obs_df   = pd.DataFrame(index=[str(i) for i in range(N)])
    var_df   = pd.DataFrame(index=[str(j) for j in range(n_hvj)])
    mb_adata = ad.AnnData(X=jr_hvj.astype(np.float32), obs=obs_df, var=var_df)
    mb_adata.layers["junc_ratio"] = jr_hvj
    mb_adata.layers["psi_mask"]   = pm_hvj
    mb_obj   = MeanBayes(mb_adata, group_by=None)
    imp_mb   = mb_obj.get_imputed_splicing(return_numpy=True).astype(np.float32)

    # ── Naive mean ────────────────────────────────────────────────────────────
    print(f"\n  [Naive] Per-junction mean over observed cells...")
    pm_dense  = pm_hvj.toarray().astype(bool)
    jr_dense  = jr_hvj.toarray().astype(np.float32)
    n_obs_j   = pm_dense.sum(axis=0).astype(float)
    mean_j    = np.where(n_obs_j > 0,
                         (jr_dense * pm_dense).sum(axis=0) / np.maximum(n_obs_j, 1),
                         0.5)
    imp_naive = np.broadcast_to(mean_j[np.newaxis, :], (N, n_hvj)).astype(np.float32)
    del pm_dense, jr_dense

    # ── metrics per cell subset ───────────────────────────────────────────────
    def compute_metrics_for_mask(cell_mask):
        cm    = cov_mask[cell_mask]
        nmf_  = imp_nmf[cell_mask]
        mb_   = imp_mb[cell_mask]
        naive = imp_naive[cell_mask]
        ori   = ori_hvj[cell_mask]

        r_nmf_p,   r_nmf_s   = per_junction_correlations(nmf_,  ori, cm, min_cells_per_junc)
        r_mb_p,    r_mb_s    = per_junction_correlations(mb_,   ori, cm, min_cells_per_junc)
        r_naive_p, r_naive_s = per_junction_correlations(naive, ori, cm, min_cells_per_junc)

        m = {}
        m.update(summarise_perjunc(r_nmf_p,   r_nmf_s,   "nmf"))
        m.update(summarise_perjunc(r_mb_p,    r_mb_s,    "mb"))
        m.update(summarise_perjunc(r_naive_p, r_naive_s, "naive"))

        rc_nmf = per_cell_pearson(nmf_, ori, cm, min_junc=min_junc_per_cell)
        rc_mb  = per_cell_pearson(mb_,  ori, cm, min_junc=min_junc_per_cell)
        m.update(summarise_percell(rc_nmf, "nmf"))
        m.update(summarise_percell(rc_mb,  "mb"))

        flat_orig  = ori[cm]
        flat_nmf   = nmf_[cm]
        flat_mb    = mb_[cm]
        flat_naive = naive[cm]
        if len(flat_orig) > 0 and flat_orig.std() > 1e-8:
            m["global_pearson_nmf"] = float(np.corrcoef(flat_orig, flat_nmf)[0, 1])
            m["global_pearson_mb"]  = float(np.corrcoef(flat_orig, flat_mb)[0, 1])
        else:
            m["global_pearson_nmf"] = np.nan
            m["global_pearson_mb"]  = np.nan

        # Mean absolute error across all covered (cell, junction) pairs
        if len(flat_orig) > 0:
            m["mae_naive"] = float(np.abs(flat_orig - flat_naive).mean())
            m["mae_nmf"]   = float(np.abs(flat_orig - flat_nmf).mean())
            m["mae_mb"]    = float(np.abs(flat_orig - flat_mb).mean())
        else:
            m["mae_naive"] = m["mae_nmf"] = m["mae_mb"] = np.nan

        m["n_cells"] = int(cell_mask.sum())
        return m

    rows = []

    all_mask = np.ones(N, dtype=bool)
    m_all    = compute_metrics_for_mask(all_mask)
    print(f"  [all pearson]   nmf={m_all['median_pearson_nmf']:.4f}  "
          f"mb={m_all['median_pearson_mb']:.4f}  "
          f"naive={m_all['median_pearson_naive']:.4f}  "
          f"n_junc={m_all['n_junc_eval']}")
    print(f"  [all spearman]  nmf={m_all['median_spearman_nmf']:.4f}  "
          f"mb={m_all['median_spearman_mb']:.4f}  "
          f"naive={m_all['median_spearman_naive']:.4f}")
    print(f"  [MAE]           naive={m_all['mae_naive']:.4f}  nmf={m_all['mae_nmf']:.4f}  "
          f"mb={m_all['mae_mb']:.4f}")
    rows.append({"dataset": "all", **m_all})

    for ds in np.unique(dataset_labels):
        ds_mask = dataset_labels == ds
        m_ds    = compute_metrics_for_mask(ds_mask)
        print(f"  [{ds}]  nmf={m_ds['median_pearson_nmf']:.4f}  "
              f"mb={m_ds['median_pearson_mb']:.4f}  "
              f"n_cells={m_ds['n_cells']}  n_junc={m_ds['n_junc_eval']}")
        rows.append({"dataset": ds, **m_ds})

    return rows


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    train_h5ad         = sys.argv[1]
    resampled_dir      = sys.argv[2]
    output_csv         = sys.argv[3]
    K                  = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    n_hvj              = int(sys.argv[5]) if len(sys.argv) > 5 else 10000
    min_atse           = int(sys.argv[6]) if len(sys.argv) > 6 else 30
    min_cells_per_junc = int(sys.argv[7]) if len(sys.argv) > 7 else 20

    if os.path.exists(output_csv):
        print(f"Output CSV already exists — skipping: {output_csv}")
        sys.exit(0)

    print(f"=== NMF Imputation Evaluation ===")
    print(f"K={K}  HVJ={n_hvj}  ATSE>{min_atse}  min_cells_per_junc={min_cells_per_junc}")
    print(f"Train h5ad:    {train_h5ad}")
    print(f"Resampled dir: {resampled_dir}")

    pattern   = os.path.join(resampled_dir, "RESAMPLED_*_PERCENT_test_*_2026-06-17.h5mu")
    h5mu_files = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r"RESAMPLED_(\d+)_PERCENT", p).group(1))
    )
    if not h5mu_files:
        print(f"ERROR: no h5mu files: {pattern}")
        sys.exit(1)
    print(f"Found {len(h5mu_files)} resampled test files\n")

    # HVJ from lowest-dropout file (most reads → most accurate variance estimate)
    # Used for evaluation only — NMF trains on ALL junctions
    hvj_idx = compute_hvj_indices(h5mu_files[0], n_hvj)

    # Train NMF once on ALL training junctions
    nmf, H = train_nmf(train_h5ad, K)

    results = []
    for h5mu_path in h5mu_files:
        fname     = os.path.basename(h5mu_path)
        pct_match = re.search(r"RESAMPLED_(\d+)_PERCENT", fname)
        pct       = int(pct_match.group(1)) if pct_match else -1

        print(f"\n{'='*60}")
        print(f"Downsampling: {pct}% dropped  ({fname})")

        rows = evaluate_one_level(
            h5mu_path, nmf, H, hvj_idx,
            min_atse, min_cells_per_junc
        )

        base = {
            "model_type":       f"nmf_K{K}",
            "K":                K,
            "downsampling_pct": pct,
            "n_hvj":            n_hvj,
            "min_atse_count":   min_atse,
        }
        for row in rows:
            results.append({**base, **row})

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nResults → {output_csv}")
    cols = ["downsampling_pct", "dataset", "median_pearson_nmf",
            "median_pearson_mb", "median_pearson_naive", "n_junc_eval", "n_cells"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
