#!/usr/bin/env python
"""
Step 04c: Per-junction Pearson evaluation with MeanBayes and naive mean baselines.

Motivation:
  Global Pearson is dominated by constitutive junctions where predicting the
  population mean PSI already works well. Per-junction Pearson across cells tests
  whether the model correctly ranks cells for each variable junction — something
  a mean-PSI predictor scores ~0 on.

Methods compared:
  1. LeafletFA  — phi @ psi, where phi is inferred via fixed_psi E-step on downsampled counts
  2. MeanBayes  — Beta conjugate posterior mean per junction, fitted from downsampled junc_ratio
  3. Naive mean — per-junction mean over observed cells in downsampled data

Evaluation:
  - Top n_hvj junctions by per-cell variance in junc_ratio_original (≥50 cells covered)
  - Coverage mask: cell_by_cluster_matrix_original > 0 (any cell with reads at the junction)
    Includes PSI=0 cells (junc=0, atse>0): genuine zero-PSI observations the model must predict.
  - Per-junction Pearson requires ≥ min_cells_per_junc cells in coverage mask
  - Reports median per-junction Pearson, fraction r>0.3, fraction r>0.5

Usage:
    python 04c_evaluate_imputation_perjunc.py <run_dir> <train_h5ad> <resampled_dir> <output_csv>
        [chunk_size=2000] [num_epochs=300] [patience=15]
        [n_hvj=10000] [min_atse=30] [min_cells_per_junc=20]
"""

import sys
import os
import glob
import re
import pickle
import gzip
import h5py
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix, coo_matrix, issparse
from scipy.stats import pearsonr, rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

import torch
torch.set_num_threads(4)

from leafletfa import LeafletFA

# ─────────────────────────────────────────────────────────────────────────────────────────────
# HELD-OUT E-STEP: FREEZE the globals, INFER only phi.
#
# Applying a trained model to held-out cells means posterior inference over the LOCAL variables
# (phi) conditioned on FROZEN globals (psi, pi, alpha, delta, bb_conc). If you merely *initialise*
# them and let SVI keep learning, you are FITTING A NEW MODEL ON THE TEST DATA -- psi and pi adapt
# to the held-out cells, and the "held-out" score is partly a training score. That is leakage and it
# flatters the result.
#
# Two that were wrong here and are now fixed:
#   * pi_init -> pi_fixed. pi_init only SEEDS pi; SVI then re-learned it on the test cells. pi is the
#     PRIOR on phi, so a re-learned pi means test phi is inferred under a different prior than train
#     phi -- any train-vs-test comparison is then confounded.
#   * input_conc_prior=inf forced a BINOMIAL likelihood even when the model was trained
#     Beta-Binomial. phi was being fit under a different model than the one that produced psi.
#     Use the TRAINED bb_conc.
# ─────────────────────────────────────────────────────────────────────────────────────────────

def as_conc_prior(bb_conc):
    """Rebuild the TRAINED likelihood for the E-step.

    get_bb_conc() stores the literal string "infinity" when the model was trained Binomial, and the
    learned scalar when it was trained Beta-Binomial. Either way the E-step must use the SAME
    likelihood the model was trained under.
    """
    if bb_conc is None or isinstance(bb_conc, str):
        return torch.tensor(float("inf"))
    return torch.tensor(float(bb_conc))


import leafletfa.waypoints as wayp

# Local MeanBayes baseline (adapted from Smriti Vaidyanathan's SpliceVI)
sys.path.insert(0, os.path.dirname(__file__))
from mean_bayes import MeanBayes


# ── shared helpers (same as 04b) ─────────────────────────────────────────────

def load_model(run_dir):
    pkl_path = os.path.join(run_dir, "leafletfa_model.pkl.gz")
    data = {}
    with gzip.open(pkl_path, "rb") as f:
        while True:
            try:
                data.update(pickle.load(f))
            except EOFError:
                break
    return data


def load_run_summary(run_dir):
    return pd.read_csv(os.path.join(run_dir, "run_summary.csv")).iloc[0].to_dict()


def get_junction_count(train_h5ad_path):
    adata = ad.read_h5ad(train_h5ad_path, backed="r")
    n = adata.n_vars
    adata.file.close()
    return n


def read_sparse_layer(grp):
    data    = grp["data"][:]
    indices = grp["indices"][:]
    indptr  = grp["indptr"][:]
    n_rows  = len(indptr) - 1
    n_cols  = int(indices.max()) + 1 if len(indices) else 0
    shape   = tuple(grp.attrs.get("shape", (n_rows, n_cols)))
    return csr_matrix((data, indices, indptr), shape=shape)


def compute_hvj_indices(h5mu_path, n_hvj, expand_to_atse=True):
    """Top n_hvj junctions by per-cell PSI variance, then (default) expand to every
    junction sharing an ATSE (var/event_id) with a selected HVJ. Kept identical to
    04e/04d so all imputation evals score the SAME junction set."""
    print(f"Computing HVJ from: {os.path.basename(h5mu_path)}")
    with h5py.File(h5mu_path, "r") as f:
        ori = np.array(f["mod/splicing/layers/junc_ratio_original"], dtype=np.float32)
        event_codes = f["mod/splicing/var/event_id/codes"][:] if expand_to_atse else None
    N, J = ori.shape
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


def build_chunk_adata(junc_chunk, clust_chunk):
    n_cells, n_juncs = junc_chunk.shape
    obs_df = pd.DataFrame(index=[str(i) for i in range(n_cells)])
    var_df = pd.DataFrame(index=[str(j) for j in range(n_juncs)])
    adata  = ad.AnnData(X=junc_chunk.astype(np.float32), obs=obs_df, var=var_df)
    adata.layers["cell_by_junction_matrix"] = junc_chunk
    adata.layers["cell_by_cluster_matrix"]  = clust_chunk
    return adata


def compute_waypoint_init_delta(train_h5ad_path, K, psi_final, phi_final):
    """
    Re-derive the waypoint PSI/phi initialization used during training and compute
    per-junction and per-cell absolute deltas to the final learned values.

    Mirrors the exact logic in run_training.py:
      TruncatedSVD (random_state=42, n_components=20) on junc_ratio_centered or junc_ratio
      → stratified_max_min_sampling (seed=42) by broad_cell_type/cell_type/cell_ontology_class
        or max_min_sampling (seed=42) if no cell type column
      → assign_nearest_cells (num_nearest = min(200, n_cells // K))
      → initialize_psi_phi_with_random_non_zero (epsilon=0.001)

    Returns dict with delta metrics (NaN if reproduction fails).
    """
    nan_result = {
        "delta_psi_median": np.nan, "delta_psi_p95": np.nan,
        "delta_psi_mean": np.nan,
        "delta_phi_median": np.nan, "delta_phi_p95": np.nan,
        "delta_phi_mean": np.nan,
    }
    try:
        print(f"\n  [Waypoint delta] Loading training h5ad for init comparison...")
        adata = ad.read_h5ad(train_h5ad_path)
        n_cells, n_junctions = adata.shape

        _N_WP         = K
        _N_PCA        = min(20, n_cells - 1, n_junctions - 1)
        _N_DIM        = min(20, _N_PCA)
        _METACELL_SIZE = min(200, n_cells // _N_WP)

        _data_pca = (adata.layers["junc_ratio_centered"]
                     if "junc_ratio_centered" in adata.layers
                     else adata.layers.get("junc_ratio", adata.X))

        print(f"  [Waypoint delta] PCA ({n_cells} cells, {_N_PCA} components)...")
        _svd = TruncatedSVD(n_components=_N_PCA, random_state=42)
        _U   = _svd.fit_transform(_data_pca)
        X_pca = _U * _svd.singular_values_

        _ct_col = next((c for c in ["broad_cell_type", "cell_type", "cell_ontology_class"]
                        if c in adata.obs.columns), None)
        if _ct_col:
            _ct_labels = adata.obs[_ct_col].astype(str).tolist()
            waypoint_idx = wayp.stratified_max_min_sampling(
                X_pca, _ct_labels, _N_WP, num_components=_N_DIM, seed=42)
        else:
            waypoint_idx = wayp.max_min_sampling(X_pca, _N_WP, num_components=_N_DIM, seed=42)

        metacell_dict = wayp.assign_nearest_cells(waypoint_idx, X_pca, num_nearest=_METACELL_SIZE)

        _rho = (adata.layers["junc_ratio"]
                if "junc_ratio" in adata.layers
                else adata.X)

        psi_list, phi_list = wayp.generate_initializations(
            _rho, {_N_WP: waypoint_idx}, {_N_WP: metacell_dict}, epsilon=0.001)

        psi_init = psi_list[0].T.astype(np.float32)  # (K × J)
        phi_init = phi_list[0].astype(np.float32)     # (N × K)

        delta_psi = np.abs(psi_final - psi_init)          # (K × J)
        dpj = delta_psi.mean(axis=0)                       # per junction, mean over K
        delta_phi = np.abs(phi_final - phi_init)           # (N × K)
        dpc = delta_phi.mean(axis=1)                       # per cell, mean over K

        print(f"  [Waypoint delta] psi Δ median={np.median(dpj):.4f}  phi Δ median={np.median(dpc):.4f}")
        return {
            "delta_psi_median": float(np.median(dpj)),
            "delta_psi_p95":    float(np.percentile(dpj, 95)),
            "delta_psi_mean":   float(dpj.mean()),
            "delta_phi_median": float(np.median(dpc)),
            "delta_phi_p95":    float(np.percentile(dpc, 95)),
            "delta_phi_mean":   float(dpc.mean()),
        }
    except Exception as e:
        print(f"  [Waypoint delta] Failed: {e}")
        return nan_result


def compute_weighted_junc_means(jc, at, rho=0.1):
    """
    Per-junction beta-binomial WEIGHTED mean PSI — matches exactly the offset used
    to build junc_ratio_centered (leafletfa.waypoints.calculate_centered_psi):

        w_ij       = at_ij / (1 + (at_ij - 1) * rho)      # BB precision weight
        junc_mean_j = sum_i(w_ij * psi_ij) / sum_i(w_ij)  # over observed positions

    jc, at — sparse (N × J) junction and cluster (ATSE) count matrices.
    Returns (J,) float32 weighted mean, 0 where a junction is never observed.
    """
    at_coo = at.tocoo().astype(np.float32)
    nz     = at_coo.data != 0
    rows   = at_coo.row[nz]
    cols   = at_coo.col[nz]
    atd    = at_coo.data[nz]

    jc_csr  = jc.tocsr()
    jc_vals = np.asarray(jc_csr[rows, cols], dtype=np.float32).ravel()
    psi     = jc_vals / atd
    w       = atd / (1.0 + (atd - 1.0) * rho)

    J   = at.shape[1]
    sw  = np.bincount(cols, weights=w,        minlength=J)
    swp = np.bincount(cols, weights=w * psi,  minlength=J)
    return np.where(sw > 0, swp / np.maximum(sw, 1e-12), 0.0).astype(np.float32)


def setup_training_phi_init(train_h5ad_path, train_phi, n_components=20, n_neighbors=5):
    """
    Fit TruncatedSVD on training junc_ratio_centered and build a nearest-neighbor
    index, plus the beta-binomial weighted per-junction mean (junc_means) needed to
    center test cells into the SAME space before projection.

    junc_ratio_centered is SPARSE: only observed positions (at>0) hold psi - mean_j;
    unobserved stay exactly 0. Test cells must be centered identically (observed-only
    subtraction of the TRAINING weighted mean) for the NN search to be meaningful.

    Returns (svd, nn_index, junc_means).
    """
    print(f"\n[phi-init] Loading training data for NN-based phi initialization...")
    adata = ad.read_h5ad(train_h5ad_path)

    if "junc_ratio_centered" not in adata.layers:
        print("  WARNING: junc_ratio_centered layer missing — NN phi-init disabled")
        del adata
        return None, None, None

    # Weighted per-junction mean from the SAME layers used to build junc_ratio_centered
    jc = adata.layers["cell_by_junction_matrix"]
    at = adata.layers["cell_by_cluster_matrix"]
    print(f"[phi-init] Computing beta-binomial weighted junc_means (rho=0.1)...")
    junc_means = compute_weighted_junc_means(jc, at, rho=0.1)

    _data  = adata.layers["junc_ratio_centered"]
    n_comp = min(n_components, adata.n_obs - 1, adata.n_vars - 1)
    print(f"[phi-init] Fitting SVD on junc_ratio_centered ({adata.n_obs} cells × "
          f"{adata.n_vars} junctions, {n_comp} components)...")
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X_train = svd.fit_transform(_data)
    del adata, _data, jc, at

    print(f"[phi-init] Building NN index (n_neighbors={n_neighbors})...")
    nn_index = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=4)
    nn_index.fit(X_train)
    print(f"[phi-init] Ready. train_phi: {train_phi.shape}  junc_means: {junc_means.shape}")
    return svd, nn_index, junc_means


def sample_pi_phi_init(pi_vec, K, n, rng, concentration=None):
    """
    Per-cell phi initialization diversified around the learned population mixing pi.

    Draws each cell's phi_init ~ Dirichlet(pi * concentration) so the MEAN is pi but
    every cell gets a distinct point on the simplex — breaking the symmetry that
    stalls the E-step when all cells start identical. concentration controls spread:
    smaller = more diverse (near vertices), larger = tighter around pi. Default K
    gives average alpha ≈ 1, matching training's Dirichlet(ones(K)) random fallback.
    """
    if pi_vec is None:
        pi_vec = np.full(K, 1.0 / K, dtype=np.float64)
    pi_vec = np.asarray(pi_vec, dtype=np.float64)
    pi_vec = pi_vec / pi_vec.sum()
    if concentration is None:
        concentration = float(K)
    alpha = np.maximum(pi_vec * concentration, 1e-3)
    return rng.dirichlet(alpha, size=n).astype(np.float32)


def get_phi_init_chunk(jc_chunk, at_chunk, svd, nn_index, train_phi, junc_means):
    """
    Build the centered (sparse) test projection exactly as training centered its data:
    at observed positions (at>0) set psi - junc_means[j]; unobserved stay 0. Project
    to the training SVD space, find nearest training cells, average their phi.

    jc_chunk, at_chunk — sparse (N_chunk × J_full) downsampled junction / cluster counts
    Returns phi_init: (N_chunk, K) float32 normalized to sum to 1 per cell.
    """
    n_chunk, J = at_chunk.shape
    at_coo = at_chunk.tocoo().astype(np.float32)
    nz     = at_coo.data != 0
    rows   = at_coo.row[nz]
    cols   = at_coo.col[nz]
    atd    = at_coo.data[nz]

    jc_csr  = jc_chunk.tocsr()
    jc_vals = np.asarray(jc_csr[rows, cols], dtype=np.float32).ravel()
    psi     = jc_vals / atd
    Y_data  = psi - junc_means[cols]                      # observed-only centering
    Y = coo_matrix((Y_data, (rows, cols)), shape=(n_chunk, J)).tocsr()

    X_test = svd.transform(Y)                             # (N_chunk, n_comp)
    _, indices = nn_index.kneighbors(X_test)              # (N_chunk, n_neighbors)
    phi_init = train_phi[indices, :].mean(axis=1).astype(np.float32)
    phi_init /= np.maximum(phi_init.sum(axis=1, keepdims=True), 1e-12)
    return phi_init


def infer_phi_chunk(chunk_adata, psi_tensor, K, dir_conc, bb_conc, junc_specific_prior,
                    pi_init, alpha_pi_init, num_epochs, patience, phi_init=None):
    delta     = torch.tensor(float(dir_conc)) if dir_conc is not None else None
    bb_conc_t = (torch.tensor(float('inf')) if str(bb_conc) == "infinity"
                 else torch.tensor(float(bb_conc))) if bb_conc is not None else None
    pi_t  = torch.tensor(pi_init,       dtype=torch.float32) if pi_init       is not None else None
    ap_t  = torch.tensor(alpha_pi_init, dtype=torch.float32) if alpha_pi_init is not None else None

    model = LeafletFA(
        adata               = chunk_adata,
        K                   = K,
        fixed_psi           = psi_tensor,
        waypoints_use       = False,
        junc_specific_prior = junc_specific_prior,
        input_conc_prior    = bb_conc_t,
        delta_fixed         = delta,
        pi_fixed=pi_t, alpha_pi_fixed=ap_t,
        num_epochs          = num_epochs,
        print_epochs        = num_epochs + 1,
        patience            = patience,
        min_delta           = 0.001,
        ELBO_num_particles  = 1,
        num_samples         = 5,
        log_wandb           = False,
        loss_plot           = False,
        output_dir          = None,
        enable_cpu_optimization = True,
    )
    model.from_anndata()
    phi_init_t = torch.tensor(phi_init, dtype=torch.float32) if phi_init is not None else None
    model.train(num_initializations=1, phi_init=phi_init_t)
    model.get_all_variables()
    phi = model.assign_post
    if isinstance(phi, torch.Tensor):
        phi = phi.detach().cpu().numpy()
    return np.array(phi, dtype=np.float32)


# ── per-junction metric ───────────────────────────────────────────────────────

def per_junction_correlations(imp, ori_hvj, cov_mask, min_cells):
    """
    For each HVJ junction, compute Pearson and Spearman between predicted and
    original PSI across cells with original coverage (cov_mask[:,j] == True).

    Returns (pearson_vals, spearman_vals) arrays of length n_hvj (nan where
    fewer than min_cells covered cells exist or true PSI is constant).
    Spearman uses averaged ranks (scipy.stats.rankdata) for proper tie handling.
    """
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
        r_pearson[j] = float(pearsonr(pred_j, true_j)[0])
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


# ── main evaluation ───────────────────────────────────────────────────────────

def evaluate_one_level(h5mu_path, psi, psi_tensor, hvj_idx, K,
                       dir_conc, bb_conc, junc_specific_prior, pi_init, alpha_pi_init,
                       chunk_size, num_epochs, patience, min_atse, min_cells_per_junc,
                       svd=None, nn_index=None, train_phi=None, junc_means=None,
                       phi_init_mode="atlas", rng=None):

    J_all = psi.shape[1]
    psi_hvj = psi[:, hvj_idx]       # (K × n_hvj)
    n_hvj   = len(hvj_idx)

    with h5py.File(h5mu_path, "r") as f:
        splice = f["mod/splicing"]
        N = int(splice["obs/_index"].shape[0])

        print(f"  Loading count matrices ({N} × {J_all})...")
        junc_full  = read_sparse_layer(splice["layers/cell_by_junction_matrix"])
        clust_full = read_sparse_layer(splice["layers/cell_by_cluster_matrix"])

        print(f"  Loading junc_ratio + psi_mask (for MeanBayes)...")
        jr_full    = read_sparse_layer(splice["layers/junc_ratio"])
        pm_full    = read_sparse_layer(splice["layers/psi_mask"])

        print(f"  Loading ground truth (dense)...")
        ori_full   = np.array(splice["layers/junc_ratio_original"], dtype=np.float32)
        atse_sp    = read_sparse_layer(splice["layers/cell_by_cluster_matrix_original"])

        # Load dataset labels for per-dataset breakdown
        ds_grp  = splice["obs/dataset"]
        ds_cats = [c.decode() if isinstance(c, bytes) else c
                   for c in ds_grp["categories"][:]]
        ds_codes = ds_grp["codes"][:]
        dataset_labels = np.array([ds_cats[c] for c in ds_codes])

    # Subset to HVJ
    ori_hvj  = ori_full[:, hvj_idx]; del ori_full
    atse_hvj = atse_sp[:, hvj_idx].toarray().astype(np.float32)
    jr_hvj   = jr_full[:, hvj_idx]
    pm_hvj   = pm_full[:, hvj_idx]

    # psi structure diagnostic: how much do K programs differ at HVJ junctions?
    psi_hvj_stds = psi_hvj.std(axis=0)   # (n_hvj,) — std across programs per junction
    psi_hvj_mean_std   = float(psi_hvj_stds.mean())
    psi_hvj_median_std = float(np.median(psi_hvj_stds))
    print(f"  psi_hvj std across programs:  mean={psi_hvj_mean_std:.4f}  "
          f"median={psi_hvj_median_std:.4f}  p90={float(np.percentile(psi_hvj_stds, 90)):.4f}")

    # Coverage mask: any cell with original ATSE > 0 (cell had reads at this junction).
    # Includes cells with PSI=0 (junc=0, atse>0) — these are genuine zero-PSI observations
    # that the model must predict correctly. Excluding them would hide predictions at PSI=0.
    cov_mask = atse_hvj > 0
    n_cov    = int(cov_mask.sum())
    print(f"  HVJ coverage (original ATSE>0): {n_cov:,} pairs")
    print(f"  Junctions with ≥{min_cells_per_junc} covered cells: "
          f"{(cov_mask.sum(axis=0) >= min_cells_per_junc).sum()}")

    # ── Step 1: LeafletFA — infer phi in chunks, collect all ─────────────────
    atlas_ready = (svd is not None and nn_index is not None
                   and train_phi is not None and junc_means is not None)
    if phi_init_mode == "atlas" and not atlas_ready:
        print("  WARNING: atlas phi-init requested but setup unavailable — "
              "falling back to diversified pi init")
        phi_init_mode = "pi"
    if rng is None:
        rng = np.random.default_rng(42)
    print(f"\n  [LeafletFA] Inferring phi in {int(np.ceil(N/chunk_size))} chunks "
          f"(phi-init mode: {phi_init_mode})...")
    all_phi = []
    for ci, start in enumerate(range(0, N, chunk_size)):
        end = min(start + chunk_size, N)
        junc_chunk  = junc_full[start:end]
        clust_chunk = clust_full[start:end]
        n_chunk = end - start
        if junc_chunk.nnz == 0:
            # No observed counts anywhere in chunk — diversified pi is best we can do
            all_phi.append(sample_pi_phi_init(pi_init, K, n_chunk, rng))
            continue
        if phi_init_mode == "atlas":
            phi_init_chunk = get_phi_init_chunk(junc_chunk, clust_chunk,
                                                svd, nn_index, train_phi, junc_means)
        else:  # "pi" — diversified Dirichlet centered at learned pi
            phi_init_chunk = sample_pi_phi_init(pi_init, K, n_chunk, rng)
        phi_chunk = infer_phi_chunk(
            build_chunk_adata(junc_chunk, clust_chunk),
            psi_tensor, K, dir_conc, bb_conc, junc_specific_prior,
            pi_init, alpha_pi_init, num_epochs, patience,
            phi_init=phi_init_chunk
        )
        all_phi.append(phi_chunk)
        print(f"    chunk {ci+1}: {start}–{end}  phi={phi_chunk.shape}", flush=True)

    all_phi = np.vstack(all_phi)                   # (N × K)
    imp_lf  = (all_phi @ psi_hvj).astype(np.float32)  # (N × n_hvj)
    print(f"  LeafletFA imp shape: {imp_lf.shape}")

    # ── BB posterior: combine LeafletFA prior with observed counts ────────────
    # posterior_mean = (alpha + obs_junc) / (alpha + beta + obs_atse)
    # where alpha = mu * bb_conc, beta = (1-mu) * bb_conc, mu = phi @ psi
    print(f"\n  [LeafletFA BB] Computing Beta-Binomial posterior mean...")
    bb_val = float('inf') if str(bb_conc) == "infinity" else float(bb_conc)
    if np.isinf(bb_val):
        imp_lf_bb = imp_lf.copy()
    else:
        obs_junc_h = junc_full[:, hvj_idx].toarray().astype(np.float32)
        obs_atse_h = clust_full[:, hvj_idx].toarray().astype(np.float32)
        alpha_arr  = imp_lf * bb_val
        beta_arr   = (1.0 - imp_lf) * bb_val
        imp_lf_bb  = (alpha_arr + obs_junc_h) / (alpha_arr + beta_arr + obs_atse_h)
        del obs_junc_h, obs_atse_h
    print(f"  LeafletFA BB posterior imp shape: {imp_lf_bb.shape}")

    # ── Step 2: MeanBayes baseline ────────────────────────────────────────────
    print(f"\n  [MeanBayes] Fitting Beta prior on HVJ subset ({n_hvj} junctions)...")
    obs_df   = pd.DataFrame(index=[str(i) for i in range(N)])
    var_df   = pd.DataFrame(index=[str(j) for j in range(n_hvj)])
    mb_adata = ad.AnnData(X=jr_hvj.astype(np.float32), obs=obs_df, var=var_df)
    mb_adata.layers["junc_ratio"] = jr_hvj
    mb_adata.layers["psi_mask"]   = pm_hvj
    mb = MeanBayes(mb_adata, group_by=None)
    imp_mb = mb.get_imputed_splicing(return_numpy=True)   # (N × n_hvj) float32
    print(f"  MeanBayes imp shape: {imp_mb.shape}")

    # ── Step 3: Naive mean baseline ───────────────────────────────────────────
    print(f"\n  [Naive mean] Per-junction mean over observed cells...")
    pm_dense = pm_hvj.toarray().astype(bool)
    jr_dense = jr_hvj.toarray().astype(np.float32)
    n_obs_j  = pm_dense.sum(axis=0).astype(float)
    sum_j    = (jr_dense * pm_dense).sum(axis=0)
    mean_j   = np.where(n_obs_j > 0, sum_j / np.maximum(n_obs_j, 1), 0.5)
    imp_naive = np.broadcast_to(mean_j[np.newaxis, :], (N, n_hvj)).astype(np.float32)
    del pm_dense, jr_dense

    # ── Step 4: Per-junction Pearson — global + per-dataset ──────────────────
    print(f"\n  Computing per-junction Pearson ({n_hvj} junctions)...")

    def compute_metrics_for_mask(cell_mask, label_suffix=""):
        """Compute all metrics for a boolean cell subset."""
        cm    = cov_mask[cell_mask]      # (n_subset × n_hvj) coverage mask
        lf    = imp_lf[cell_mask]
        lf_bb = imp_lf_bb[cell_mask]
        mb    = imp_mb[cell_mask]
        naive = imp_naive[cell_mask]
        ori   = ori_hvj[cell_mask]

        r_lf_p,    r_lf_s    = per_junction_correlations(lf,    ori, cm, min_cells_per_junc)
        r_lf_bb_p, r_lf_bb_s = per_junction_correlations(lf_bb, ori, cm, min_cells_per_junc)
        r_mb_p,    r_mb_s    = per_junction_correlations(mb,    ori, cm, min_cells_per_junc)
        r_naive_p, r_naive_s = per_junction_correlations(naive, ori, cm, min_cells_per_junc)

        m = {}
        m.update(summarise_perjunc(r_lf_p,    r_lf_s,    "lf"))
        m.update(summarise_perjunc(r_lf_bb_p, r_lf_bb_s, "lf_bb"))
        m.update(summarise_perjunc(r_mb_p,    r_mb_s,    "mb"))
        m.update(summarise_perjunc(r_naive_p, r_naive_s, "naive"))

        flat_orig  = ori[cm]
        flat_lf    = lf[cm]
        flat_lf_bb = lf_bb[cm]
        flat_mb    = mb[cm]
        flat_naive = naive[cm]
        if len(flat_orig) > 0 and flat_orig.std() > 1e-8:
            m["global_pearson_lf"]    = float(np.corrcoef(flat_orig, flat_lf)[0, 1])
            m["global_pearson_lf_bb"] = float(np.corrcoef(flat_orig, flat_lf_bb)[0, 1])
            m["global_pearson_mb"]    = float(np.corrcoef(flat_orig, flat_mb)[0, 1])
        else:
            m["global_pearson_lf"]    = np.nan
            m["global_pearson_lf_bb"] = np.nan
            m["global_pearson_mb"]    = np.nan

        # Mean absolute error across all covered (cell, junction) pairs
        if len(flat_orig) > 0:
            m["mae_naive"] = float(np.abs(flat_orig - flat_naive).mean())
            m["mae_lf"]    = float(np.abs(flat_orig - flat_lf).mean())
            m["mae_lf_bb"] = float(np.abs(flat_orig - flat_lf_bb).mean())
            m["mae_mb"]    = float(np.abs(flat_orig - flat_mb).mean())
        else:
            m["mae_naive"] = m["mae_lf"] = m["mae_lf_bb"] = m["mae_mb"] = np.nan

        m["n_cells"] = int(cell_mask.sum())
        return m

    rows = []

    # All cells combined
    all_mask = np.ones(N, dtype=bool)
    m_all = compute_metrics_for_mask(all_mask)
    print(f"  [all pearson]   lf={m_all['median_pearson_lf']:.4f}  lf_bb={m_all['median_pearson_lf_bb']:.4f}  "
          f"mb={m_all['median_pearson_mb']:.4f}  naive={m_all['median_pearson_naive']:.4f}  "
          f"n_junc={m_all['n_junc_eval']}")
    print(f"  [all spearman]  lf={m_all['median_spearman_lf']:.4f}  lf_bb={m_all['median_spearman_lf_bb']:.4f}  "
          f"mb={m_all['median_spearman_mb']:.4f}  naive={m_all['median_spearman_naive']:.4f}")
    print(f"  [MAE]           naive={m_all['mae_naive']:.4f}  lf={m_all['mae_lf']:.4f}  "
          f"lf_bb={m_all['mae_lf_bb']:.4f}  mb={m_all['mae_mb']:.4f}")
    rows.append({"dataset": "all",
                 "psi_hvj_mean_std":   psi_hvj_mean_std,
                 "psi_hvj_median_std": psi_hvj_median_std,
                 **m_all})

    # Per dataset
    for ds in np.unique(dataset_labels):
        ds_mask = dataset_labels == ds
        m_ds = compute_metrics_for_mask(ds_mask)
        print(f"  [{ds}]  lf={m_ds['median_pearson_lf']:.4f}  lf_bb={m_ds['median_pearson_lf_bb']:.4f}  "
              f"mb={m_ds['median_pearson_mb']:.4f}  naive={m_ds['median_pearson_naive']:.4f}  "
              f"n_cells={m_ds['n_cells']}  n_junc={m_ds['n_junc_eval']}")
        rows.append({"dataset": ds,
                     "psi_hvj_mean_std":   psi_hvj_mean_std,
                     "psi_hvj_median_std": psi_hvj_median_std,
                     **m_ds})

    return rows


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)

    run_dir            = sys.argv[1]
    train_h5ad         = sys.argv[2]
    resampled_dir      = sys.argv[3]
    output_csv         = sys.argv[4]
    chunk_size         = int(sys.argv[5])  if len(sys.argv) > 5  else 2000
    num_epochs         = int(sys.argv[6])  if len(sys.argv) > 6  else 300
    patience           = int(sys.argv[7])  if len(sys.argv) > 7  else 15
    n_hvj              = int(sys.argv[8])  if len(sys.argv) > 8  else 10000
    min_atse           = int(sys.argv[9])  if len(sys.argv) > 9  else 30
    min_cells_per_junc = int(sys.argv[10]) if len(sys.argv) > 10 else 20
    phi_init_mode      = sys.argv[11]      if len(sys.argv) > 11 else "atlas"

    if phi_init_mode not in ("atlas", "pi"):
        print(f"ERROR: phi_init_mode must be 'atlas' or 'pi', got '{phi_init_mode}'")
        sys.exit(1)

    if os.path.exists(output_csv):
        print(f"Output CSV already exists — skipping: {output_csv}")
        sys.exit(0)

    print(f"=== Per-Junction Imputation Evaluation ===")
    print(f"Run dir:    {run_dir}")
    print(f"HVJ={n_hvj}  ATSE>{min_atse}  min_cells_per_junc={min_cells_per_junc}")
    print(f"Chunk={chunk_size}  epochs={num_epochs}  patience={patience}  phi_init={phi_init_mode}")

    print("\nLoading model...")
    model_data = load_model(run_dir)
    summary    = load_run_summary(run_dir)

    psi        = model_data["psi"].astype(np.float32)
    K          = int(model_data["K"])
    pi         = model_data.get("pi",       None)
    alpha_pi   = model_data.get("alpha_pi", None)
    dir_conc   = model_data.get("dir_conc", None)
    bb_conc    = model_data.get("bb_conc",  None)
    junc_prior = bool(summary.get("junc_specific_prior", True))
    psi_tensor = torch.tensor(psi, dtype=torch.float32)

    print(f"  psi={psi.shape}  K={K}  dir_conc={dir_conc}  bb_conc={bb_conc}")
    if psi.max() < 1e-6:
        print("  WARNING: psi all-zero (collapsed). Skipping.")
        sys.exit(0)

    n_total = get_junction_count(train_h5ad)
    assert n_total == psi.shape[1], f"Junction mismatch: {n_total} vs {psi.shape[1]}"

    # ── Waypoint init vs. final delta (computed once per run) ────────────────
    waypoints_use = bool(summary.get("waypoints_use", False))
    train_phi_arr = model_data.get("assign_post")
    if train_phi_arr is not None:
        train_phi_arr = np.array(train_phi_arr, dtype=np.float32)

    if waypoints_use and train_phi_arr is not None:
        wp_delta = compute_waypoint_init_delta(train_h5ad, K, psi, train_phi_arr)
    else:
        wp_delta = {k: np.nan for k in ["delta_psi_median","delta_psi_p95","delta_psi_mean",
                                         "delta_phi_median","delta_phi_p95","delta_phi_mean"]}

    # ── Atlas phi initialization (only needed for phi_init_mode == "atlas") ──
    # Uses training assign_post as phi starting point for test cells' E-step,
    # breaking the symmetry that causes lf Pearson ≈ 0. The "pi" mode instead
    # diversifies each cell around the learned pi and skips this setup entirely.
    svd_phi = nn_index_phi = junc_means_phi = None
    if phi_init_mode == "atlas":
        if train_phi_arr is not None:
            svd_phi, nn_index_phi, junc_means_phi = setup_training_phi_init(
                train_h5ad, train_phi_arr)
        else:
            print("  WARNING: assign_post not in pkl — atlas init unavailable, "
                  "will fall back to diversified pi init")

    pattern   = os.path.join(resampled_dir, "RESAMPLED_*_PERCENT_test_*_2026-06-17.h5mu")
    h5mu_files = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r"RESAMPLED_(\d+)_PERCENT", p).group(1))
    )
    if not h5mu_files:
        print(f"ERROR: no h5mu files: {pattern}")
        sys.exit(1)
    print(f"\nFound {len(h5mu_files)} resampled test files")

    # Compute HVJ once from the lowest-dropout file
    hvj_idx = compute_hvj_indices(h5mu_files[0], n_hvj)

    # Single rng across all dropout levels for reproducible pi-init draws
    rng = np.random.default_rng(42)

    results = []
    for h5mu_path in h5mu_files:
        fname     = os.path.basename(h5mu_path)
        pct_match = re.search(r"RESAMPLED_(\d+)_PERCENT", fname)
        pct       = int(pct_match.group(1)) if pct_match else -1

        print(f"\n{'='*60}")
        print(f"Downsampling: {pct}% dropped  ({fname})")

        rows = evaluate_one_level(
            h5mu_path, psi, psi_tensor, hvj_idx, K,
            dir_conc, bb_conc, junc_prior, pi, alpha_pi,
            chunk_size, num_epochs, patience, min_atse, min_cells_per_junc,
            svd=svd_phi, nn_index=nn_index_phi,
            train_phi=train_phi_arr, junc_means=junc_means_phi,
            phi_init_mode=phi_init_mode, rng=rng
        )

        base = {
            "run_dir":             os.path.basename(run_dir),
            "param_id":            int(summary["param_id"]),
            "K":                   int(summary["K"]),
            "pruned_K":            int(summary.get("pruned_K", K)),
            "waypoints_use":       waypoints_use,
            "junc_specific_prior": junc_prior,
            "batch_size":          int(summary.get("batch_size", 0)),
            "best_elbo":           float(summary.get("best_elbo", float("nan"))),
            "downsampling_pct":    pct,
            "n_hvj":               n_hvj,
            "min_atse_count":      0,
            "phi_init_mode":       phi_init_mode,
            **wp_delta,
        }
        for row in rows:
            results.append({**base, **row})

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nResults → {output_csv}")
    cols = ["downsampling_pct", "dataset", "median_pearson_lf", "median_pearson_mb",
            "median_pearson_naive", "n_junc_eval", "n_cells"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
