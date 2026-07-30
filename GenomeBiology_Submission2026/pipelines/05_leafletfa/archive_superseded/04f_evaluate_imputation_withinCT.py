#!/usr/bin/env python
"""
Step 04f: Within-cell-type imputation evaluation (pure-prior headline).

Rigorous coverage-robustness eval per IMPUTATION_EVAL_V2_PLAN.md. Key differences
from 04e:
  - HVJ selected on TRAIN (not the test file) — no feature-selection leak.
  - Scores TWO junction sets: (a) HVJ expanded to ATSEs, (b) ALL junctions.
  - Scores TWO masks: (primary) atse_original>0, (sensitivity) atse_original>=10.
  - HEADLINE = PURE priors (no count blend): lf=phi@psi, meanCT (train cell-type
    mean), NMF (W@H), naive. Metric = WITHIN-cell-type per-junction Pearson +
    permutation Δ (matched vs within-CT-permuted) — the artifact-free cell-specific
    signal. Plotted vs median reads/cell.
  - SECONDARY = native count-blended lf_bb/meanCT_bb/NMF_bb using each model's OWN
    learned bb_conc (Binomial models: no blend → equals the pure prior). Interpret
    only at high dropout (read-overlap minimal). Written but flagged secondary.

phi E-step: fixed psi, input_conc_prior=inf, Dirichlet(pi*K) init (same as 04e).

Usage:
    python 04f_evaluate_imputation_withinCT.py <run_dir> <train_h5ad> <resampled_dir> <output_csv>
        [chunk_size=4096] [num_epochs=300] [patience=10]
        [n_hvj=10000] [min_cells_per_junc=20] [sets=hvj,all] [nmf_K=0]

    sets: comma list of {hvj, all} (default both). nmf_K: NMF components (0 = use model K).
Output: long-format CSV, one row per (dropout, junction_set, score_mask, cell_group).
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
from scipy.sparse import csr_matrix, csc_matrix, issparse
from scipy.stats import spearmanr, pearsonr, gaussian_kde
from sklearn.decomposition import MiniBatchNMF, TruncatedSVD
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.set_default_dtype(torch.float32)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"[04f] GPU: {torch.cuda.get_device_name(0)}")
else:
    torch.set_num_threads(4)
    print("[04f] No GPU — running on CPU")

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




# ── loading helpers (verbatim from 04e) ───────────────────────────────────────

def load_model(run_dir):
    data = {}
    with gzip.open(os.path.join(run_dir, "leafletfa_model.pkl.gz"), "rb") as f:
        while True:
            try:
                data.update(pickle.load(f))
            except EOFError:
                break
    return data


def load_run_summary(run_dir):
    return pd.read_csv(os.path.join(run_dir, "run_summary.csv")).iloc[0].to_dict()


def read_sparse_layer(grp):
    data    = grp["data"][:]
    indices = grp["indices"][:]
    indptr  = grp["indptr"][:]
    n_rows  = len(indptr) - 1
    n_cols  = int(indices.max()) + 1 if len(indices) else 0
    shape   = tuple(grp.attrs.get("shape", (n_rows, n_cols)))
    return csr_matrix((data, indices, indptr), shape=shape)


# ── Smriti's imputation reporting layer (method-agnostic: flat obs/pred arrays) ──
def _compute_imputation_metrics(orig_arr, pred_arr):
    """Scalar imputation metrics on flat observed/predicted PSI arrays.
    Mirrors SPLICEVI eval_multivisplice._compute_imputation_metrics."""
    o = np.asarray(orig_arr, dtype=np.float64).ravel()
    p = np.asarray(pred_arr, dtype=np.float64).ravel()
    m = np.isfinite(o) & np.isfinite(p)
    o, p = o[m], p[m]
    n = int(o.size)
    out = dict(n=n, pearson_r=np.nan, spearman_r=np.nan, l1=np.nan, rmse=np.nan)
    if n < 3:
        return out
    out["l1"]   = float(np.abs(o - p).mean())
    out["rmse"] = float(np.sqrt(((o - p) ** 2).mean()))
    if o.std() > 1e-8 and p.std() > 1e-8:
        out["pearson_r"]  = float(pearsonr(o, p)[0])
        out["spearman_r"] = float(spearmanr(o, p)[0])
    return out


def plot_psi_density_scatter(orig_arr, pred_arr, pearson_r, out_png, title=None,
                             max_points=40000, seed=0):
    """2D density scatter: observed PSI (x) vs predicted PSI (y), colored by local
    Gaussian-KDE density, with y=x reference and Pearson annotation.
    Mirrors SPLICEVI eval_multivisplice.plot_psi_density_scatter."""
    o = np.asarray(orig_arr, dtype=np.float64).ravel()
    p = np.asarray(pred_arr, dtype=np.float64).ravel()
    m = np.isfinite(o) & np.isfinite(p)
    o, p = o[m], p[m]
    if o.size < 3:
        return
    rng = np.random.default_rng(seed)
    if o.size > max_points:
        sel = rng.choice(o.size, size=max_points, replace=False)
        o_s, p_s = o[sel], p[sel]
    else:
        o_s, p_s = o, p
    try:
        dens = gaussian_kde(np.vstack([o_s, p_s]))(np.vstack([o_s, p_s]))
        order = dens.argsort()
        o_s, p_s, dens = o_s[order], p_s[order], dens[order]
    except Exception:
        dens = None
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.scatter(o_s, p_s, c=dens, s=4, cmap="viridis", rasterized=True)
    ax.plot([0, 1], [0, 1], "r--", lw=1)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("observed PSI"); ax.set_ylabel("predicted PSI")
    ttl = title or ""
    ax.set_title(f"{ttl}\nPearson r = {pearson_r:.3f}  (n={o.size:,})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── ATSE-level variation across programs (phi-free; psi matrix only) ──────────
def atse_variation_across_programs(psi, event_codes):
    """For each ATSE (junctions sharing an event code), score how differently the K
    programs use it — junctions in an ATSE shift in tandem, so a real AS event shows
    high COHERENT across-program variation. Reports per ATSE:
      atse_var_score — mean over junctions of across-program variance Var_k(psi[k,j]).
                       Simple, but double-counts redundant inclusion junctions and its
                       ceiling depends on n_junc (simplex geometry) → weak cross-ATSE.
      atse_var_max   — max per-junction across-program variance (robust to junction
                       duplication; keys off the biggest-swing junction).
      pc1_var        — leading eigenvalue of the junctions' across-program covariance
                       = magnitude of the dominant coordinated (inclusion↔skip) axis.
      pc1_frac       — pc1_var / total across-program variance ∈ (0,1]. ~1 = coherent
                       cassette (all variation on one axis); low = incoherent/noisy.
                       Comparable across ATSE sizes and rewards tandem coupling.

    psi:          (K, J) program × junction matrix
    event_codes:  (J,) integer ATSE id per junction column (-1 = unassigned)
    """
    K, J = psi.shape
    var_per_junc = psi.var(axis=0)               # (J,) across-program variance per junction
    rows = []
    for ev in np.unique(event_codes):
        if ev < 0:
            continue
        cols = np.where(event_codes == ev)[0]
        if cols.size < 2:                        # need >=2 junctions to be a real ATSE
            continue
        X = psi[:, cols].astype(np.float64)      # (K, n_junc): program profiles for this ATSE
        Xc = X - X.mean(axis=0, keepdims=True)   # center across programs
        C = (Xc.T @ Xc) / max(K - 1, 1)          # (n_junc, n_junc) across-program covariance
        total_var = float(np.trace(C))           # = sum of per-junction across-program variances
        if total_var > 1e-12:
            evals = np.linalg.eigvalsh(C)        # ascending; junctions few → cheap
            pc1_var = float(evals[-1]); pc1_frac = pc1_var / total_var
        else:
            pc1_var = 0.0; pc1_frac = np.nan
        rows.append(dict(event_code=int(ev), n_junc=int(cols.size),
                         atse_var_score=float(var_per_junc[cols].mean()),
                         atse_var_max=float(var_per_junc[cols].max()),
                         pc1_var=pc1_var, pc1_frac=pc1_frac))
    return pd.DataFrame(rows)


def load_event_codes(train_h5ad, n_junctions):
    """Integer ATSE id per junction column, aligned to model/psi junction order."""
    A = ad.read_h5ad(train_h5ad, backed="r")
    if "event_id" in A.var.columns:
        codes = A.var["event_id"].astype("category").cat.codes.values.astype(np.int64)
    else:
        codes = -np.ones(A.n_vars, dtype=np.int64)
    A.file.close()
    assert codes.shape[0] == n_junctions, f"event_id len {codes.shape[0]} vs psi J {n_junctions}"
    return codes


# ── waypoint psi re-derivation (deterministic, matches run_training.py 199-245) ──
def derive_waypoint_psi_init(train_h5ad, K, seed=42):
    """Reproduce the waypoint PSI initialization used at training time so it can be
    used as the 'init' arm of the learned-vs-init ablation. Returns psi_init (K, J)
    or None if the waypoint module / centered layer is unavailable."""
    try:
        import leafletfa.waypoints as wayp
    except Exception as e:
        print(f"  [psi_init] waypoints module unavailable ({e}); skipping init arm.")
        return None
    A = ad.read_h5ad(train_h5ad)
    n_cells, n_junctions = A.shape
    N_WP  = K
    N_PCA = min(20, n_cells - 1, n_junctions - 1)
    N_DIM = min(20, N_PCA)
    METACELL = min(200, n_cells // N_WP)
    if "junc_ratio_centered" in A.layers:
        data_pca = A.layers["junc_ratio_centered"]
    elif "junc_ratio" in A.layers:
        data_pca = A.layers["junc_ratio"]
    else:
        data_pca = A.X
    print(f"  [psi_init] SVD ({n_cells} cells, {N_PCA} comps) for {N_WP} waypoints...")
    svd = TruncatedSVD(n_components=N_PCA, random_state=seed)
    U = svd.fit_transform(data_pca)
    X_pca = U * svd.singular_values_
    ct_col = next((c for c in ["broad_cell_type", "cell_type", "cell_ontology_class"]
                   if c in A.obs.columns), None)
    if ct_col is not None:
        ct_labels = A.obs[ct_col].astype(str).tolist()
        wp_idx = wayp.stratified_max_min_sampling(X_pca, ct_labels, N_WP,
                                                  num_components=N_DIM, seed=seed)
    else:
        wp_idx = wayp.max_min_sampling(X_pca, N_WP, num_components=N_DIM, seed=seed)
    metacell = wayp.assign_nearest_cells(wp_idx, X_pca, num_nearest=METACELL)
    rho = A.layers["junc_ratio"] if "junc_ratio" in A.layers else A.X
    psi_inits, _ = wayp.generate_initializations(rho, {N_WP: wp_idx},
                                                 {N_WP: metacell}, epsilon=0.001)
    psi0 = psi_inits[0]
    if isinstance(psi0, torch.Tensor):
        psi0 = psi0.cpu().numpy()
    psi_init = np.asarray(psi0, dtype=np.float32).T.copy()   # (K, J)
    print(f"  [psi_init] waypoint psi_init shape {psi_init.shape}")
    return psi_init


def build_chunk_adata(junc_chunk, clust_chunk):
    n_cells, n_juncs = junc_chunk.shape
    adata = ad.AnnData(X=junc_chunk.astype(np.float32),
                       obs=pd.DataFrame(index=[str(i) for i in range(n_cells)]),
                       var=pd.DataFrame(index=[str(j) for j in range(n_juncs)]))
    adata.layers["cell_by_junction_matrix"] = junc_chunk
    adata.layers["cell_by_cluster_matrix"]  = clust_chunk
    return adata


def sample_pi_phi_init(pi_vec, K, n, rng, concentration=None):
    if pi_vec is None:
        pi_vec = np.full(K, 1.0 / K, dtype=np.float64)
    pi_vec = np.asarray(pi_vec, dtype=np.float64); pi_vec = pi_vec / pi_vec.sum()
    alpha = np.maximum(pi_vec * (float(K) if concentration is None else concentration), 1e-3)
    return rng.dirichlet(alpha, size=n).astype(np.float32)


def infer_phi_chunk(chunk_adata, psi_tensor, K, dir_conc, junc_specific_prior,
                    pi_init, alpha_pi_init, num_epochs, patience, phi_init=None,
                    num_particles=10, num_samples=100):
    delta = torch.tensor(float(dir_conc)) if dir_conc is not None else None
    pi_t  = torch.tensor(pi_init,       dtype=torch.float32) if pi_init       is not None else None
    ap_t  = torch.tensor(alpha_pi_init, dtype=torch.float32) if alpha_pi_init is not None else None
    on_cpu = DEVICE.type == "cpu"
    model = LeafletFA(
        adata=chunk_adata, K=K, fixed_psi=psi_tensor, waypoints_use=False,
        junc_specific_prior=junc_specific_prior,
        # Use the TRAINED likelihood, not a forced Binomial.
        input_conc_prior=as_conc_prior(bb_conc),
        delta_fixed=delta, pi_fixed=pi_t, alpha_pi_fixed=ap_t,
        num_epochs=num_epochs, print_epochs=num_epochs + 1, patience=patience,
        min_delta=0.001, ELBO_num_particles=num_particles, num_samples=num_samples,
        lr=0.01, gamma=0.05, log_wandb=False, loss_plot=False, output_dir=None,
        enable_cpu_optimization=on_cpu,
    )
    model.from_anndata()
    try:
        model.initialize_triton_mask()
    except Exception:
        pass
    phi_init_t = torch.tensor(phi_init, dtype=torch.float32) if phi_init is not None else None
    model.train(num_initializations=1, phi_init=phi_init_t)
    model.get_all_variables()
    phi = model.assign_post
    if isinstance(phi, torch.Tensor):
        phi = phi.detach().cpu().numpy()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return np.array(phi, dtype=np.float32)


# ── TRAIN-based feature selection + baselines ─────────────────────────────────

def _var_from_ratio_csc(jr_csc):
    """Per-junction per-cell PSI variance over covered (>0) cells, ≥50 covered."""
    J = jr_csc.shape[1]
    var_j = np.zeros(J, dtype=np.float32)
    n_cov = np.diff(jr_csc.indptr)
    for j in range(J):
        s, e = jr_csc.indptr[j], jr_csc.indptr[j + 1]
        if e - s < 50:
            continue
        d = jr_csc.data[s:e]
        var_j[j] = d.var(ddof=1) if len(d) > 1 else 0.0
    return var_j, n_cov


def compute_hvj_train(train_h5ad, n_hvj, expand_to_atse=True):
    """HVJ from TRAIN junc_ratio variance, expanded to full ATSEs via var/event_id."""
    print(f"[HVJ] Selecting on TRAIN: {os.path.basename(train_h5ad)}")
    adata = ad.read_h5ad(train_h5ad, backed="r")
    jr = adata.layers["junc_ratio"]
    jr_csc = csc_matrix(jr).astype(np.float32)
    var_j, _ = _var_from_ratio_csc(jr_csc)
    top_idx = np.sort(np.argsort(var_j)[::-1][:n_hvj])
    ev = None
    if expand_to_atse and "event_id" in adata.var.columns:
        ev = adata.var["event_id"].astype("category").cat.codes.values
    try:
        adata.file.close()
    except Exception:
        pass
    if not expand_to_atse or ev is None:
        print(f"  HVJ = {len(top_idx):,} (no ATSE expansion)")
        return top_idx
    hvj_events = np.unique(ev[top_idx]); hvj_events = hvj_events[hvj_events >= 0]
    keep = np.isin(ev, hvj_events); keep[top_idx] = True
    expanded = np.sort(np.where(keep)[0])
    print(f"  HVJ {len(top_idx):,} → expanded {len(expanded):,} across "
          f"{len(hvj_events):,} ATSEs")
    return expanded


def train_baselines(train_h5ad, celltype_col, nmf_K):
    """Compute on TRAIN: per-junction mean PSI, per-cell-type mean PSI, and NMF.

    Returns (mean_j (J,), ct_means (dict ct->(J,)), ct_global (J,), nmf, H).
    Means use covered (psi>0 OR atse>0) cells; here we use the junc_ratio nonzero
    structure + a full mean over cells with ATSE coverage via psi_mask if present.
    """
    print(f"[baselines] Loading TRAIN for mean/meanCT/NMF...")
    adata = ad.read_h5ad(train_h5ad)
    jr = adata.layers["junc_ratio"]
    jr_csr = csr_matrix(jr).astype(np.float32)
    jr_csr.data = np.clip(jr_csr.data, 0.0, 1.0)
    N, J = jr_csr.shape

    # coverage: use psi_mask if available else (junc_ratio structurally stored)
    if "psi_mask" in adata.layers:
        pm = csr_matrix(adata.layers["psi_mask"]).astype(np.float32)
    else:
        pm = (csr_matrix(adata.layers["cell_by_cluster_matrix"]) > 0).astype(np.float32)

    jr_csc = jr_csr.tocsc(); pm_csc = pm.tocsc()
    sum_j = np.asarray(jr_csc.sum(axis=0)).ravel()
    n_j   = np.asarray(pm_csc.sum(axis=0)).ravel()
    mean_j = np.where(n_j > 0, sum_j / np.maximum(n_j, 1), 0.5).astype(np.float32)

    # per cell-type mean
    ct = adata.obs[celltype_col].astype(str).values if celltype_col in adata.obs else \
         np.array(["NA"] * N)
    ct_means = {}
    for c in np.unique(ct):
        idx = np.where(ct == c)[0]
        sj = np.asarray(jr_csr[idx].sum(axis=0)).ravel()
        nj = np.asarray(pm[idx].sum(axis=0)).ravel()
        ct_means[c] = np.where(nj > 0, sj / np.maximum(nj, 1), mean_j).astype(np.float32)

    # NMF on all train junctions
    print(f"[baselines] Training MiniBatchNMF K={nmf_K} on TRAIN ({N}x{J})...")
    nmf = MiniBatchNMF(n_components=nmf_K, init="nndsvda", batch_size=2048,
                       random_state=42, verbose=False)
    nmf.fit(jr_csr)
    H = nmf.components_.astype(np.float32)
    try:
        adata.file.close()
    except Exception:
        pass
    del adata, jr_csr, jr_csc, pm, pm_csc
    return mean_j, ct_means, mean_j.copy(), nmf, H


# ── vectorized within-cell-type per-junction correlation + permutation ────────

def masked_col_pearson(P, T, M, min_cells):
    """Per-column (junction) Pearson between P and T over rows where M is True.
    P,T: (n,J) float32; M: (n,J) bool. Returns (r (J,), n_used (J,))."""
    Mf = M.astype(np.float32)
    n  = Mf.sum(0)
    sP = (P * Mf).sum(0); sT = (T * Mf).sum(0)
    mP = sP / np.maximum(n, 1); mT = sT / np.maximum(n, 1)
    Pc = (P - mP) * Mf; Tc = (T - mT) * Mf
    cov = (Pc * Tc).sum(0)
    vP  = (Pc * Pc).sum(0); vT = (Tc * Tc).sum(0)
    denom = np.sqrt(vP * vT)
    r = np.where(denom > 1e-12, cov / denom, np.nan).astype(np.float32)
    r[n < min_cells] = np.nan
    return r, n


def within_ct_corr(pred, target, cov_mask, ct_labels, min_cells):
    """Average per-junction Pearson computed WITHIN each cell type.
    Returns (J,) mean over cell types of the within-type correlation (nan-safe)."""
    J = pred.shape[1]
    acc = np.zeros(J, dtype=np.float64); cnt = np.zeros(J, dtype=np.float64)
    for c in np.unique(ct_labels):
        idx = np.where(ct_labels == c)[0]
        if len(idx) < min_cells:
            continue
        r, _ = masked_col_pearson(pred[idx], target[idx], cov_mask[idx], min_cells)
        ok = ~np.isnan(r)
        acc[ok] += r[ok]; cnt[ok] += 1
    out = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype(np.float32)
    return out


def within_ct_permutation(ct_labels, rng):
    """One within-cell-type row permutation, drawn ONCE per dropout level and shared
    across all methods so their permutation-Δ nulls are PAIRED (clean cross-method Δ)."""
    perm = np.arange(len(ct_labels))
    for c in np.unique(ct_labels):
        idx = np.where(ct_labels == c)[0]
        perm[idx] = idx[rng.permutation(len(idx))]
    return perm


def med(x):
    v = x[~np.isnan(x)]
    return float(np.median(v)) if len(v) else np.nan


def frac_gt(x, t):
    v = x[~np.isnan(x)]
    return float((v > t).mean()) if len(v) else np.nan


def demeaned_global_pearson(pred, target, mask):
    """Pooled Pearson over masked pairs, after de-junction-meaning each column."""
    Mf = mask.astype(np.float32)
    n  = Mf.sum(0)
    mP = (pred * Mf).sum(0) / np.maximum(n, 1)
    mT = (target * Mf).sum(0) / np.maximum(n, 1)
    Pr = (pred - mP)[mask]; Tr = (target - mT)[mask]
    if Pr.size < 3 or Pr.std() < 1e-8 or Tr.std() < 1e-8:
        return np.nan
    return float(np.corrcoef(Pr, Tr)[0, 1])


def score_method(name, pred, target, atse_orig, ct_labels, min_cells, perm):
    """All metrics for one prediction, under both masks. Returns list of row dicts.
    `perm` is a shared within-CT row permutation (paired null across methods)."""
    rows = []
    for mask_name, cov in (("atse_gt0", atse_orig > 0), ("atse_ge10", atse_orig >= 10)):
        wr = within_ct_corr(pred, target, cov, ct_labels, min_cells)          # matched
        permuted = within_ct_corr(pred, target[perm], cov[perm], ct_labels, min_cells)
        dl = wr - permuted                                                    # paired Δ
        gp = demeaned_global_pearson(pred, target, cov)
        mae = float(np.abs(pred - target)[cov].mean()) if cov.any() else np.nan
        # Smriti scalar imputation metrics on the pooled covered pairs (raw, not demeaned)
        sm = _compute_imputation_metrics(target[cov], pred[cov]) if cov.any() else \
            dict(pearson_r=np.nan, spearman_r=np.nan, l1=np.nan, rmse=np.nan)
        rows.append(dict(
            method=name, score_mask=mask_name,
            within_ct_median=med(wr), within_ct_frac_gt03=frac_gt(wr, 0.3),
            perm_delta_median=med(dl), global_pearson_demeaned=gp, mae=mae,
            pearson_pooled=sm["pearson_r"], spearman_pooled=sm["spearman_r"],
            rmse_pooled=sm["rmse"],
            n_junc_eval=int((~np.isnan(wr)).sum()), n_pairs=int(cov.sum()),
        ))
    return rows


def evaluate_one_level(h5mu_path, phi, psi, bb_conc, mean_j, ct_means, nmf, H,
                       set_defs, min_cells, rng, phi_init=None, psi_init=None,
                       scatter_dir=None, pct=None):
    """Score every method × junction set × mask at one dropout level.
    If phi_init/psi_init are given, adds the 'lf_init' ablation arm (waypoint psi
    with phi re-inferred under it) and writes lf vs lf_init density scatters."""
    with h5py.File(h5mu_path, "r") as f:
        s = f["mod/splicing"]; N = int(s["obs/_index"].shape[0])
        junc_full  = read_sparse_layer(s["layers/cell_by_junction_matrix"])
        clust_full = read_sparse_layer(s["layers/cell_by_cluster_matrix"])
        jr_full    = read_sparse_layer(s["layers/junc_ratio"])
        ori_full   = np.array(s["layers/junc_ratio_original"], dtype=np.float32)
        atse_sp    = read_sparse_layer(s["layers/cell_by_cluster_matrix_original"])
        ct_grp = s["obs/broad_cell_type"]
        ct_cats = [c.decode() if isinstance(c, bytes) else c for c in ct_grp["categories"][:]]
        ct_codes = ct_grp["codes"][:]
        ct_labels = np.array([ct_cats[c] if c >= 0 else "NA" for c in ct_codes])

    # one within-CT permutation for this level, shared by every method (paired null)
    perm = within_ct_permutation(ct_labels, rng)
    reads_per_cell = float(np.median(np.asarray(junc_full.sum(axis=1)).ravel()))

    # NMF projection on ALL junctions (once), then subset per set
    jr_csr = jr_full.tocsr().astype(np.float32); jr_csr.data = np.clip(jr_csr.data, 0.0, 1.0)
    W_test = nmf.transform(jr_csr); del jr_csr
    bb_val = None if bb_conc is None or str(bb_conc) == "infinity" else float(bb_conc)
    if bb_val is not None and np.isinf(bb_val):
        bb_val = None

    rows = []
    for set_name, idx in set_defs.items():
        n_set = len(idx)
        print(f"    [set={set_name}] {n_set:,} junctions", flush=True)
        # Resident big arrays: target + atse_orig (+ obs_* only if a blend is needed).
        # Predictions are built and scored ONE AT A TIME to keep peak memory bounded.
        target    = ori_full[:, idx]
        atse_orig = atse_sp[:, idx].toarray().astype(np.float32)
        obs_junc  = junc_full[:, idx].toarray().astype(np.float32)  if bb_val is not None else None
        obs_atse  = clust_full[:, idx].toarray().astype(np.float32) if bb_val is not None else None

        def build_prior(name):
            if name == "lf":
                return (phi @ psi[:, idx]).astype(np.float32)
            if name == "lf_init":
                return (phi_init @ psi_init[:, idx]).astype(np.float32)
            if name == "nmf":
                return np.clip((W_test @ H[:, idx]).astype(np.float32), 0, 1)
            if name == "mean":
                return np.broadcast_to(mean_j[idx][None, :], (N, n_set)).astype(np.float32)
            if name == "meanCT":
                mCT = np.empty((N, n_set), dtype=np.float32)
                for c in np.unique(ct_labels):
                    mCT[ct_labels == c] = ct_means.get(c, mean_j)[idx]
                return mCT
            raise ValueError(name)

        def emit(name, pred):
            for r in score_method(name, pred, target, atse_orig, ct_labels, min_cells, perm):
                r.update(dict(junction_set=set_name, n_junc_total=n_set,
                              reads_per_cell_median=reads_per_cell, n_cells=N,
                              bb_conc_used=(bb_val if bb_val is not None else np.inf)))
                rows.append(r)

        methods = ["lf", "meanCT", "nmf", "mean"]
        if psi_init is not None and phi_init is not None:
            methods.insert(1, "lf_init")           # learned-vs-init ablation arm
        first_set = (set_name == list(set_defs)[0])
        for name in methods:
            p = build_prior(name)
            emit(name, p)                          # pure-prior HEADLINE
            # headline density scatter for the learned-vs-init comparison (atse>0, first set)
            if scatter_dir is not None and first_set and name in ("lf", "lf_init"):
                cov = atse_orig > 0
                if cov.any():
                    sm = _compute_imputation_metrics(target[cov], p[cov])
                    os.makedirs(scatter_dir, exist_ok=True)
                    png = os.path.join(scatter_dir,
                        f"scatter_{name}_{set_name}_atse0_drop{pct}.png")
                    plot_psi_density_scatter(target[cov], p[cov], sm["pearson_r"], png,
                        title=f"{name} | {set_name} | {pct}% dropout")
            if bb_val is not None:                 # native count-blend SECONDARY
                pbb = ((p * bb_val + obs_junc) / (bb_val + obs_atse)).astype(np.float32)
                emit(name + "_bb", pbb)
                del pbb
            del p
        del target, atse_orig, obs_junc, obs_atse
    del W_test
    return rows, reads_per_cell


def infer_phi_all(h5mu_path, psi_tensor, K, dir_conc, junc_prior, pi, alpha_pi,
                  chunk_size, num_epochs, patience, rng):
    """Infer phi for every cell at this dropout level (chunked E-step).
    Returns (phi_final, phi_init) so we can measure how far the E-step moved phi
    from its pi-based init (did test counts inform phi, or fall back to pi?)."""
    with h5py.File(h5mu_path, "r") as f:
        s = f["mod/splicing"]; N = int(s["obs/_index"].shape[0])
        junc_full  = read_sparse_layer(s["layers/cell_by_junction_matrix"])
        clust_full = read_sparse_layer(s["layers/cell_by_cluster_matrix"])
    all_phi = []; all_init = []
    for ci, start in enumerate(range(0, N, chunk_size)):
        end = min(start + chunk_size, N); n_chunk = end - start
        jc = junc_full[start:end]; cl = clust_full[start:end]
        phi_init = sample_pi_phi_init(pi, K, n_chunk, rng)
        all_init.append(phi_init)
        if jc.nnz == 0:
            all_phi.append(phi_init); continue     # no reads → phi stays at pi init
        all_phi.append(infer_phi_chunk(build_chunk_adata(jc, cl), psi_tensor, K,
                                       dir_conc, junc_prior, pi, alpha_pi,
                                       num_epochs, patience, phi_init=phi_init))
        print(f"    phi chunk {ci+1}: {start}-{end}", flush=True)
    return np.vstack(all_phi), np.vstack(all_init)


def phi_commitment_stats(phi_final, phi_init, pi, K):
    """Per-cell diagnostics: did phi commit (counts informed it) or stay at pi?
      phi_max          — largest program weight (>>1/K = committed)
      phi_entropy_norm — Shannon entropy / log K  (≈1 = diffuse pi-fallback)
      dist_from_pi_l1  — L1 distance of phi_final from the population prior pi
      l1_from_init     — L1 movement from the (random) pi-based init draw
    Returns (per_cell dict of arrays, summary dict of means)."""
    eps = 1e-12
    pf = np.clip(phi_final.astype(np.float64), eps, 1.0)
    ent = -(pf * np.log(pf)).sum(1) / np.log(K)
    pmax = phi_final.max(1)
    if pi is None:
        pi_vec = np.full(K, 1.0 / K, dtype=np.float64)
    else:
        pi_vec = np.asarray(pi, dtype=np.float64); pi_vec = pi_vec / pi_vec.sum()
    dist_pi = np.abs(phi_final - pi_vec[None, :]).sum(1)
    l1_init = np.abs(phi_final - phi_init).sum(1)
    per_cell = dict(phi_max=pmax, phi_entropy_norm=ent,
                    dist_from_pi_l1=dist_pi, l1_from_init=l1_init)
    summary = {k: float(np.mean(v)) for k, v in per_cell.items()}
    return per_cell, summary


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 5:
        print(__doc__); sys.exit(1)
    run_dir       = sys.argv[1]
    train_h5ad    = sys.argv[2]
    resampled_dir = sys.argv[3]
    output_csv    = sys.argv[4]
    chunk_size    = int(sys.argv[5])  if len(sys.argv) > 5  else 4096
    num_epochs    = int(sys.argv[6])  if len(sys.argv) > 6  else 300
    patience      = int(sys.argv[7])  if len(sys.argv) > 7  else 10
    n_hvj         = int(sys.argv[8])  if len(sys.argv) > 8  else 10000
    min_cells     = int(sys.argv[9])  if len(sys.argv) > 9  else 20
    sets_arg      = sys.argv[10]      if len(sys.argv) > 10 else "hvj,all"
    nmf_K_arg     = int(sys.argv[11]) if len(sys.argv) > 11 else 0

    if os.path.exists(output_csv):
        print(f"Output exists — skipping: {output_csv}"); sys.exit(0)
    want_sets = [x.strip() for x in sets_arg.split(",") if x.strip()]

    print("Loading model...")
    md = load_model(run_dir); summary = load_run_summary(run_dir)
    # Headline psi = DETERMINISTIC posterior median sigmoid(psis_loc) when available
    # (no noisy single draw — that biased the headline and differed pruned vs unpruned).
    # Pruned runs keep nominal-K psis_loc, so fall back to the pruned point estimate md["psi"].
    psi_pt = md["psi"].astype(np.float32)
    K = psi_pt.shape[0]
    if "psis_loc" in md and np.asarray(md["psis_loc"]).shape[0] == K:
        loc = np.asarray(md["psis_loc"], np.float32)
        psi = (1.0 / (1.0 + np.exp(-loc))).astype(np.float32)   # deterministic, consistent
    else:
        psi = psi_pt
    pi = md.get("pi"); alpha_pi = md.get("alpha_pi")
    dir_conc = md.get("dir_conc"); bb_conc = md.get("bb_conc")
    junc_prior = bool(summary.get("junc_specific_prior", True))
    psi_tensor = torch.tensor(psi, dtype=torch.float32)
    nmf_K = nmf_K_arg if nmf_K_arg > 0 else K
    print(f"  K={K}  bb_conc={bb_conc}  junc_prior={junc_prior}  nmf_K={nmf_K}  sets={want_sets}")
    if psi.max() < 1e-6:
        print("  psi collapsed — skipping"); sys.exit(0)

    # TRAIN-based feature selection + baselines (leak-free)
    hvj_idx = compute_hvj_train(train_h5ad, n_hvj) if "hvj" in want_sets else None
    mean_j, ct_means, ct_global, nmf, H = train_baselines(train_h5ad, "broad_cell_type", nmf_K)
    J = H.shape[1]
    assert J == psi.shape[1], f"junction mismatch train {J} vs psi {psi.shape[1]}"

    # ── learned-vs-init ablation setup: re-derive waypoint psi_init (once) ────────
    waypoints_use = bool(summary.get("waypoints_use", False))
    psi_init = None
    if waypoints_use:
        psi_init = derive_waypoint_psi_init(train_h5ad, K)
        if psi_init is not None and psi_init.shape != psi.shape:
            print(f"  [psi_init] shape {psi_init.shape} != learned {psi.shape} "
                  "(pruned K?) — disabling init arm.")
            psi_init = None
    else:
        print("  [psi_init] non-waypoint model — no init arm / ATSE-variation vs init.")
    psi_init_tensor = torch.tensor(psi_init, dtype=torch.float32) if psi_init is not None else None

    # ── ATSE-level variation across programs (phi-free; psi matrix only) ──────────
    event_codes = load_event_codes(train_h5ad, J)
    METRIC_COLS = ["atse_var_score", "atse_var_max", "pc1_var", "pc1_frac"]
    atse_df = atse_variation_across_programs(psi, event_codes).rename(
        columns={c: f"{c}_learned" for c in METRIC_COLS})
    if psi_init is not None:
        atse_init = atse_variation_across_programs(psi_init, event_codes)[
            ["event_code"] + METRIC_COLS].rename(columns={c: f"{c}_init" for c in METRIC_COLS})
        atse_df = atse_df.merge(atse_init, on="event_code", how="left")
        ml = float(np.nanmedian(atse_df["atse_var_score_learned"]))
        mi = float(np.nanmedian(atse_df["atse_var_score_init"]))
        pl = float(np.nanmedian(atse_df["pc1_frac_learned"]))
        pif = float(np.nanmedian(atse_df["pc1_frac_init"]))
        f_var = float((atse_df["atse_var_score_learned"] > atse_df["atse_var_score_init"]).mean())
        f_pc1 = float((atse_df["pc1_var_learned"] > atse_df["pc1_var_init"]).mean())
        print(f"  [ATSE-var] {len(atse_df)} ATSEs | var_score med L={ml:.4g} I={mi:.4g} "
              f"(frac L>I {f_var:.3f}) | pc1_frac med L={pl:.3f} I={pif:.3f} "
              f"| pc1_var frac L>I {f_pc1:.3f}")
    else:
        print(f"  [ATSE-var] {len(atse_df)} ATSEs | var_score med="
              f"{float(np.nanmedian(atse_df['atse_var_score_learned'])):.4g} | pc1_frac med="
              f"{float(np.nanmedian(atse_df['pc1_frac_learned'])):.3f} (no init arm)")
    atse_df.insert(0, "run_dir", os.path.basename(run_dir))
    atse_csv = output_csv.replace(".csv", "_atse_variation.csv")
    atse_df.to_csv(atse_csv, index=False)
    print(f"  [ATSE-var] → {atse_csv}")
    scatter_dir = output_csv.replace(".csv", "_scatter")

    set_defs = {}
    if "hvj" in want_sets: set_defs["hvj"] = hvj_idx
    if "all" in want_sets: set_defs["all"] = np.arange(J)

    # Junction set is versioned by date: the 2026-06-17 resamples (79768 junctions)
    # match the current models; the 2026-06-13 set (83593) is the OLD build and MUST
    # NOT be mixed in — its junctions misalign with psi → Binomial CUDA assert.
    pattern = os.path.join(resampled_dir, "RESAMPLED_*_PERCENT_test_*_2026-06-17.h5mu")
    h5mu_files = sorted(glob.glob(pattern),
                        key=lambda p: int(re.search(r"RESAMPLED_(\d+)_PERCENT", p).group(1)))
    if not h5mu_files:
        print(f"ERROR: no h5mu files: {pattern}"); sys.exit(1)
    print(f"Found {len(h5mu_files)} resampled files")

    rng = np.random.default_rng(42)
    results = []
    PHI_PERCELL = os.environ.get("PHI_PERCELL", "0") == "1"
    percell_frames = []
    for h5mu in h5mu_files:
        pct = int(re.search(r"RESAMPLED_(\d+)_PERCENT", os.path.basename(h5mu)).group(1))
        print(f"\n{'='*60}\nDropout {pct}%  ({os.path.basename(h5mu)})")
        with h5py.File(h5mu, "r") as _f:
            _jshape = _f["mod/splicing/layers/cell_by_junction_matrix"].attrs.get("shape")
            _hjunc = int(_jshape[1]) if _jshape is not None else None
        if _hjunc is not None and _hjunc != J:
            print(f"ERROR: h5mu has {_hjunc} junctions but model/train has {J} — "
                  f"junction-set MISMATCH (wrong resample version?). Aborting."); sys.exit(1)
        phi, phi_init_draw = infer_phi_all(h5mu, psi_tensor, K, dir_conc, junc_prior,
                                           pi, alpha_pi, chunk_size, num_epochs, patience, rng)
        pc_l, sm_l = phi_commitment_stats(phi, phi_init_draw, pi, K)
        print(f"  [phi/learned] max={sm_l['phi_max']:.3f} entropyN={sm_l['phi_entropy_norm']:.3f} "
              f"dist_pi={sm_l['dist_from_pi_l1']:.3f} move={sm_l['l1_from_init']:.3f}")
        if PHI_PERCELL:
            percell_frames.append(pd.DataFrame({"downsampling_pct": pct, "arm": "learned", **pc_l}))
        phi_init = None; sm_i = None
        if psi_init_tensor is not None:
            print("  [lf_init] inferring phi under waypoint psi_init...")
            phi_init, phi_init_draw2 = infer_phi_all(h5mu, psi_init_tensor, K, dir_conc,
                                       junc_prior, pi, alpha_pi, chunk_size, num_epochs, patience, rng)
            pc_i, sm_i = phi_commitment_stats(phi_init, phi_init_draw2, pi, K)
            print(f"  [phi/init]    max={sm_i['phi_max']:.3f} entropyN={sm_i['phi_entropy_norm']:.3f} "
                  f"dist_pi={sm_i['dist_from_pi_l1']:.3f} move={sm_i['l1_from_init']:.3f}")
            if PHI_PERCELL:
                percell_frames.append(pd.DataFrame({"downsampling_pct": pct, "arm": "init", **pc_i}))
        rows, rpc = evaluate_one_level(h5mu, phi, psi, bb_conc, mean_j, ct_means,
                                       nmf, H, set_defs, min_cells, rng,
                                       phi_init=phi_init, psi_init=psi_init,
                                       scatter_dir=scatter_dir, pct=pct)
        base = dict(run_dir=os.path.basename(run_dir), param_id=int(summary["param_id"]),
                    K=int(summary["K"]), waypoints_use=waypoints_use,
                    junc_specific_prior=junc_prior, downsampling_pct=pct)
        base.update({f"phi_learned_{k}": v for k, v in sm_l.items()})
        if sm_i is not None:
            base.update({f"phi_init_{k}": v for k, v in sm_i.items()})
        for r in rows:
            results.append({**base, **r})
        print(f"  reads/cell median={rpc:.1f}")
        # quick headline print
        df0 = pd.DataFrame([r for r in rows if r["junction_set"] == list(set_defs)[0]
                            and r["score_mask"] == "atse_gt0"])
        if len(df0):
            piv = df0.set_index("method")["within_ct_median"]
            print("  [within-CT median r, "+list(set_defs)[0]+", atse>0]: " +
                  "  ".join(f"{m}={piv.get(m, np.nan):.3f}" for m in
                           ["lf","lf_init","meanCT","nmf","mean"] if m in piv.index))

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"\nResults → {output_csv}  ({len(results)} rows)")
    if PHI_PERCELL and percell_frames:
        pc_path = output_csv.replace(".csv", "_phi_percell.parquet")
        try:
            pd.concat(percell_frames, ignore_index=True).to_parquet(pc_path, index=False)
            print(f"Per-cell phi diagnostics → {pc_path}")
        except Exception as e:
            pc_path = pc_path.replace(".parquet", ".csv")
            pd.concat(percell_frames, ignore_index=True).to_csv(pc_path, index=False)
            print(f"Per-cell phi diagnostics (csv fallback) → {pc_path}  ({e})")


if __name__ == "__main__":
    main()
