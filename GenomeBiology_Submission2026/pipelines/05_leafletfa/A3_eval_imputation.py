#!/usr/bin/env python
"""
Step 04e: GPU per-junction imputation evaluation (LeafletFA + baselines).

GPU variant of 04c. Applies the trained model at test time the "strong" way,
mirroring the working LeafletFA_Submission2025/.../05_impute_on_test.py:
  - runs on GPU (falls back to CPU if unavailable)
  - SAMPLES psi ~ sigmoid(psis_loc + psis_scale·eps) instead of the point estimate
  - E-step uses the TRAINED likelihood (input_conc_prior = as_conc_prior(bb_conc)).
    NOTE: this file used to force input_conc_prior = inf (Binomial). That was wrong --
    fitting phi under a different likelihood than the one that produced psi means the
    E-step is not the model. The old rationale ("the tiny learned bb_conc leaves the
    likelihood ~flat") was a symptom of the old bb_conc ~ Gamma(10,10) prior (mean 1),
    which is now Gamma(2, 0.1) (mean 20), so the trained concentration is sensible.
  - ELBO_num_particles=10, num_samples=100 (less noisy VI)
  - batch_size 4096

Methods compared:
  lf     LeafletFA phi@psi (factor model, label-free)
  lf_bb  LeafletFA BB posterior mean w/ observed counts (factor model, label-free)
  nmf0   rank-K NMF, uncovered->0     (factor model, label-free -- fair head-to-head vs lf)
  nmfm   rank-K NMF, uncovered->mean  (factor model, label-free -- fair head-to-head vs lf)
  mb     MeanBayes global per-junction Beta-shrinkage mean (label-free floor)
  naive  per-junction mean over observed cells             (label-free floor)
  mb_ct  MeanBayes stratified by broad_cell_type -- USES CELL-TYPE LABELS, so it is an
         advantaged reference ceiling, NOT a fair competitor to the label-free methods.
NMF matches the LeafletFA run's rank K and mirrors figures/FIGURE2_benchmark/benchmark_sweep.py
so Fig 2 (sim) and Fig 3 (real) report the identical NMF baseline.
Metrics: per-junction Pearson + Spearman (cell ranking), MAE, global Pearson, calibration slope.

Usage:
    python A3_eval_imputation.py <run_dir> <train_h5ad> <resampled_dir> <output_csv>
        [chunk_size=4096] [num_epochs=100] [patience=10]
        [n_hvj=10000] [min_atse=0] [min_cells_per_junc=20] [phi_init_mode=pi]
        [hardest_h5mu]

<output_csv> is the 4th POSITIONAL argument. Omitting it shifts every later argument by one and the
run dies on `int('pi')` -- and if anything is chained after it in the same job the failure is
invisible, so chain with `|| exit`.

[hardest_h5mu] is optional. Given the h5mu at the hardest downsampling, a third entry set
`fixed_lost_at_hardest` is scored alongside `all_covered` and `lost_to_downsampling`: the entries
covered originally but absent at that hardest level, held FIXED across every coverage level so the
curve is a curve rather than a moving population. See the note above `fixed_lost_mask` for the
verified counts. Every method is written for every entry set; the figure does the subsetting.
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
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.neighbors import NearestNeighbors

import torch

# GPU setup (mirrors 05_impute_on_test.py). Falls back to CPU cleanly.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# FAIL FAST on silent CPU fallback. Some GPU nodes (ne1dg7-[001-011]) present no usable CUDA
# device, and torch then quietly returns "cpu" -- the job runs at ~1% speed and burns its whole
# walltime before anyone notices. Exit loudly instead. Set LEAFLET_ALLOW_CPU=1 to override.
if DEVICE.type != "cuda" and os.environ.get("LEAFLET_ALLOW_CPU") != "1":
    sys.exit(f"FATAL: no CUDA device visible (node={os.environ.get('SLURMD_NODENAME','?')}). "
             f"This job would silently run on CPU. Resubmit excluding the bad nodes, or set "
             f"LEAFLET_ALLOW_CPU=1 to run on CPU deliberately.")
print(f"[device] {DEVICE}  node={os.environ.get('SLURMD_NODENAME','?')}", flush=True)
if DEVICE.type == "cuda":
    torch.set_default_dtype(torch.float32)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"[04e] Running on GPU: {torch.cuda.get_device_name(0)}")
else:
    torch.set_num_threads(4)
    print("[04e] No GPU available — running on CPU")

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
    """
    Select the top n_hvj junctions by per-cell PSI variance, then (default)
    EXPAND to every junction sharing an ATSE (var/event_id) with a selected HVJ.

    PSI is only meaningful relative to its ATSE — sibling junctions share the
    cluster denominator — so evaluating a lone high-variance junction while
    ignoring its ATSE partners is a partial view. Expanding returns the full
    set of junctions belonging to any ATSE that contains ≥1 high-variance junction.
    """
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

    # Expand to all junctions in the ATSEs that contain a selected HVJ.
    hvj_events = np.unique(event_codes[top_idx])
    hvj_events = hvj_events[hvj_events >= 0]           # drop NaN-category code (-1)
    keep = np.isin(event_codes, hvj_events)
    keep[top_idx] = True                              # keep any HVJ w/o valid ATSE too
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


def infer_phi_chunk(chunk_adata, psi_tensor, K, dir_conc, junc_specific_prior,
                    pi_init, alpha_pi_init, bb_conc, num_epochs, patience, phi_init=None,
                    num_particles=10, num_samples=100):
    """E-step through the CURRENT (plate minibatch) implementation -- the same code that trained
    the models, rather than the legacy class kept below for reference.

    phi_init is accepted but NOT used: the caller passes a non-informative start. phi is a local
    variable inside the subsampling plate, so seeding it per-cell does not map cleanly onto the
    batched guide, and with the globals frozen the E-step is near-convex and seed-stable
    (mean |delta phi| = 0.004 between seeds on simulated data)."""
    from estep_minibatch import infer_phi_minibatch
    psi_np = psi_tensor.detach().cpu().numpy() if hasattr(psi_tensor, "detach") else np.asarray(psi_tensor)
    phi, _info = infer_phi_minibatch(
        chunk_adata, psi_np, K, dir_conc=dir_conc, pi=pi_init, alpha_pi=alpha_pi_init,
        bb_conc=bb_conc, junc_prior=junc_specific_prior, num_epochs=num_epochs,
        device=("cuda" if DEVICE.type == "cuda" else "cpu"))
    return phi


def _infer_phi_chunk_legacy(chunk_adata, psi_tensor, K, dir_conc, junc_specific_prior,
                    pi_init, alpha_pi_init, bb_conc, num_epochs, patience, phi_init=None,
                    num_particles=10, num_samples=100):
    """
    GPU E-step: infer phi with psi and ALL other globals frozen (fixed_psi, pi_fixed,
    alpha_pi_fixed, delta_fixed) and the TRAINED likelihood via as_conc_prior(bb_conc).
    Only phi is inferred, so this is a genuine projection onto the trained dictionary.
    """
    delta = torch.tensor(float(dir_conc)) if dir_conc is not None else None
    pi_t  = torch.tensor(pi_init,       dtype=torch.float32) if pi_init       is not None else None
    ap_t  = torch.tensor(alpha_pi_init, dtype=torch.float32) if alpha_pi_init is not None else None

    on_cpu = DEVICE.type == "cpu"
    model = LeafletFA(
        adata               = chunk_adata,
        K                   = K,
        fixed_psi           = psi_tensor,
        waypoints_use       = False,
        junc_specific_prior = junc_specific_prior,
        # Use the TRAINED likelihood, not a forced Binomial. Fitting phi under a different
        # likelihood than the one that produced psi means the E-step is not the model.
        input_conc_prior    = as_conc_prior(bb_conc),
        delta_fixed         = delta,
        pi_fixed=pi_t, alpha_pi_fixed=ap_t,
        num_epochs          = num_epochs,
        print_epochs        = num_epochs + 1,
        patience            = patience,
        # ABSOLUTE min_delta against a ~1e7 ELBO is a ~1e-11 tolerance, i.e. inert -- see 07c.
        min_delta           = 0.001,
        min_delta_rel       = 1e-5,
        ELBO_num_particles  = num_particles,
        num_samples         = num_samples,
        # E-step lr 0.01->0.1, gamma 0.05->0.1: the LR sweep (estep_lr_sweep) showed lr=0.01 needs
        # ~1500 epochs to converge phi while lr=0.1 converges by ~300 (corr 0.999 vs the settled phi,
        # perplexity 3.25 == training 3.4), with no instability. The near-convex per-cell problem was
        # just crawling under too-low a learning rate. Phi now actually converges within budget.
        lr                  = 0.1,
        gamma               = 0.1,
        log_wandb           = False,
        loss_plot           = False,
        output_dir          = None,
        enable_cpu_optimization = on_cpu,
    )
    model.from_anndata()
    try:
        model.initialize_triton_mask()   # fast masked matmul on GPU (no-op / skip on CPU)
    except Exception:
        pass
    phi_init_t = torch.tensor(phi_init, dtype=torch.float32) if phi_init is not None else None
    model.train(num_initializations=1, phi_init=phi_init_t)
    model.get_all_variables()
    phi = model.assign_post
    if isinstance(phi, torch.Tensor):
        phi = phi.detach().cpu().numpy()
    # CONVERGENCE: did the E-step converge, or just run out of epochs? min_delta is ABSOLUTE
    # (1e-3) against a large ELBO, so early stopping is near-inert and the cap itself becomes the
    # binding constraint on every imputation number below. Record it rather than assume.
    h = getattr(model, "health", None) or {}
    _E_STEP_CONV.append((int(h.get("epochs_ran", -1)), bool(h.get("stopped_early", False)), num_epochs))
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return np.array(phi, dtype=np.float32)


_E_STEP_CONV = []   # (epochs_ran, stopped_early, cap) per chunk


def report_e_step_convergence(tag=""):
    """Print whether the phi E-step converged or was truncated by the epoch cap."""
    if not _E_STEP_CONV:
        return
    ran   = [r for r, _, _ in _E_STEP_CONV]
    early = sum(1 for _, e, _ in _E_STEP_CONV if e)
    cap   = _E_STEP_CONV[0][2]
    n     = len(_E_STEP_CONV)
    at_cap = sum(1 for r in ran if r >= cap)
    print(f"\n--- E-step convergence {tag} ---", flush=True)
    print(f"  chunks: {n}   epoch cap: {cap}", flush=True)
    print(f"  epochs ran: min={min(ran)} median={int(np.median(ran))} max={max(ran)}", flush=True)
    print(f"  early-stopped: {early}/{n}    hit the cap: {at_cap}/{n}", flush=True)
    if at_cap > n * 0.5:
        print(f"  !! WARNING: {at_cap}/{n} chunks hit the epoch cap without converging. phi is "
              f"TRUNCATED, not converged -- the epoch budget is a live knob on every metric "
              f"below. Raise epochs (or set min_delta_rel) and confirm the numbers are stable.",
              flush=True)


# ── NMF baseline (rank-K matrix factorization, the fair head-to-head) ─────────

def fit_nmf_impute(jr_hvj, pm_hvj, K, max_iter=300, random_state=0):
    """Rank-K NMF imputation baseline — the label-free factor-model competitor to LeafletFA.

    NMF learns K programs and reconstructs the full (cell x junction) PSI matrix, exactly like
    LeafletFA's phi@psi, but WITHOUT the Beta-Binomial likelihood, the coverage model, or any
    cell-type labels. Beating NMF at the SAME rank K is the "our structure is better than vanilla
    matrix factorization" claim. Mirrors figures/FIGURE2_benchmark/benchmark_sweep.py:fit_nmf so
    Fig 2 (sim) and Fig 3 (real) report the identical baseline.

    NMF has no missing-data model, so uncovered entries must be filled before factorizing. Two
    conventions, both reported (the choice is part of the point):
      nmf0  (0-impute)    : uncovered -> 0. The naive default; asserts uncovered == PSI 0, which is
                            FALSE (no coverage != junction skipped). Handicaps NMF.
      nmfm  (mean-impute) : uncovered -> per-junction mean over covered cells. The fairer baseline.

    Returns (imp_nmf0, imp_nmfm), each (N x n_hvj) float32 reconstructions clipped to [0, 1].
    """
    cov = pm_hvj.toarray().astype(bool)                       # (N x n_hvj) 1=observed
    Rc  = np.clip(np.nan_to_num(jr_hvj.toarray().astype(np.float32)), 0.0, 1.0)
    cnt = cov.sum(0)
    cm  = np.where(cnt > 0, (Rc * cov).sum(0) / np.maximum(cnt, 1), 0.0).astype(np.float32)
    out = []
    for tag, X in (("nmf0", np.where(cov, Rc, 0.0).astype(np.float32)),
                   ("nmfm", np.where(cov, Rc, cm[None, :]).astype(np.float32))):
        m = NMF(n_components=K, init="nndsvda", random_state=random_state, max_iter=max_iter)
        W = m.fit_transform(X); H = m.components_
        recon = np.clip((W @ H).astype(np.float32), 0.0, 1.0)
        out.append(recon)
        del W, H, X, m
    del cov, Rc
    return out[0], out[1]


# ── per-junction metric ───────────────────────────────────────────────────────

def calibration(pred, truth, n_bins=10):
    """CALIBRATION, not correlation — Reviewer 1's ask.

    A model can have high Pearson r while being systematically COMPRESSED toward the mean:
    it ranks cells correctly but predicts every PSI closer to average than it really is. That
    is exactly what a shrinkage-heavy latent-variable model does, and correlation cannot see it.

    Returns:
      slope      OLS slope of truth ~ pred.  1.0 = calibrated;  <1 = predictions OVER-dispersed;
                 >1 = predictions COMPRESSED toward the mean (the failure mode we care about).
                 (Regressing truth on pred means slope>1 <=> pred varies too little.)
      intercept  systematic offset in PSI units
      rmse       root mean squared error, in PSI units (interpretable, unlike r)
      mae        mean absolute error, in PSI units
      bias       mean(pred - truth) -- systematic over/under-prediction
      bin_*      observed mean PSI within each decile of PREDICTED PSI, for the calibration
                 curve. Perfect calibration => bin_obs == bin_pred along the diagonal.
    """
    out = {}
    if len(truth) < 10 or np.std(pred) < 1e-8:
        return {"slope": np.nan, "intercept": np.nan, "rmse": np.nan, "mae": np.nan,
                "bias": np.nan, "bin_pred": "", "bin_obs": "", "bin_n": ""}
    slope, intercept = np.polyfit(pred, truth, 1)
    out["slope"]     = float(slope)
    out["intercept"] = float(intercept)
    out["rmse"]      = float(np.sqrt(np.mean((pred - truth) ** 2)))
    out["mae"]       = float(np.mean(np.abs(pred - truth)))
    out["bias"]      = float(np.mean(pred - truth))
    # binned observed-vs-predicted (the calibration curve)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx   = np.clip(np.digitize(pred, edges) - 1, 0, n_bins - 1)
    bp, bo, bn = [], [], []
    for b in range(n_bins):
        m_ = idx == b
        if m_.sum() == 0:
            continue
        bp.append(float(pred[m_].mean())); bo.append(float(truth[m_].mean())); bn.append(int(m_.sum()))
    out["bin_pred"] = ";".join(f"{v:.4f}" for v in bp)
    out["bin_obs"]  = ";".join(f"{v:.4f}" for v in bo)
    out["bin_n"]    = ";".join(str(v) for v in bn)
    return out


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
                       phi_init_mode="atlas", rng=None, train_h5ad=None,
                       hardest_pm=None):

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

        # MEMORY: read ONLY the high-variance junctions instead of densifying all J.
        # `junc_ratio_original` is dense on disk; np.array(...) on the full layer materialises
        # N x J_all float32 (e.g. 40k x 80k = 12.8 GB) just to throw away all but n_hvj columns.
        # h5py fancy-indexes on the column axis directly, so we never hold the full matrix.
        print(f"  Loading ground truth (HVJ columns only)...")
        _hvj_sorted = np.sort(np.asarray(hvj_idx))
        _unsort     = np.argsort(np.argsort(np.asarray(hvj_idx)))   # restore caller's hvj order
        ori_hvj_direct = np.asarray(splice["layers/junc_ratio_original"][:, _hvj_sorted],
                                    dtype=np.float32)[:, _unsort]
        atse_sp    = read_sparse_layer(splice["layers/cell_by_cluster_matrix_original"])

        # Load dataset labels for per-dataset breakdown
        ds_grp  = splice["obs/dataset"]
        ds_cats = [c.decode() if isinstance(c, bytes) else c
                   for c in ds_grp["categories"][:]]
        ds_codes = ds_grp["codes"][:]
        dataset_labels = np.array([ds_cats[c] for c in ds_codes])

        # Load broad_cell_type for the cell-type-stratified MeanBayes baseline
        ct_grp  = splice["obs/broad_cell_type"]
        ct_cats = [c.decode() if isinstance(c, bytes) else c
                   for c in ct_grp["categories"][:]]
        ct_codes = ct_grp["codes"][:]
        celltype_labels = np.array([ct_cats[c] if c >= 0 else "NA" for c in ct_codes])

    # Subset to HVJ  (ori_hvj already read HVJ-only above — no full dense copy)
    ori_hvj  = ori_hvj_direct; del ori_hvj_direct
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
        else:  # "pi" — seed each cell from Dirichlet(pi·K): mean = learned pi,
            # per-cell spread breaks symmetry explicitly (vs. relying on Pyro's
            # default init). Under input_conc_prior=inf this converges to the same
            # phi the observed counts imply, same as 05_impute_on_test.py.
            phi_init_chunk = sample_pi_phi_init(pi_init, K, n_chunk, rng)
        phi_chunk = infer_phi_chunk(
            build_chunk_adata(junc_chunk, clust_chunk),
            psi_tensor, K, dir_conc, junc_specific_prior,
            pi_init, alpha_pi_init, bb_conc, num_epochs, patience,
            phi_init=phi_init_chunk
        )
        all_phi.append(phi_chunk)
        print(f"    chunk {ci+1}: {start}–{end}  phi={phi_chunk.shape}", flush=True)

    report_e_step_convergence("(test phi)")
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

    # ── Step 2b: Cell-type-stratified MeanBayes (mb_ct) ───────────────────────
    # Unobserved cells get their BROAD_CELL_TYPE mean PSI (global fallback for
    # sparse groups) instead of the global population mean. This is the sharpest
    # simple competitor to lf on the correlation metric: it can only rank cells
    # correctly at junctions whose PSI genuinely differs by cell type, so it
    # directly asks "does the factor model beat cell-type averaging?"
    print(f"\n  [MeanBayes-CT] Stratifying by broad_cell_type "
          f"({len(np.unique(celltype_labels))} groups)...")
    mb_ct_adata = ad.AnnData(X=jr_hvj.astype(np.float32), obs=pd.DataFrame(
        {"broad_cell_type": celltype_labels}, index=[str(i) for i in range(N)]),
        var=var_df)
    mb_ct_adata.layers["junc_ratio"] = jr_hvj
    mb_ct_adata.layers["psi_mask"]   = pm_hvj
    mb_ct = MeanBayes(mb_ct_adata, group_by="broad_cell_type")
    imp_mb_ct = mb_ct.get_imputed_splicing(return_numpy=True)
    print(f"  MeanBayes-CT imp shape: {imp_mb_ct.shape}")

    # ── Step 3: Naive mean baseline ───────────────────────────────────────────
    print(f"\n  [Naive mean] Per-junction mean over observed cells...")
    pm_dense = pm_hvj.toarray().astype(bool)
    jr_dense = jr_hvj.toarray().astype(np.float32)
    n_obs_j  = pm_dense.sum(axis=0).astype(float)
    sum_j    = (jr_dense * pm_dense).sum(axis=0)
    mean_j   = np.where(n_obs_j > 0, sum_j / np.maximum(n_obs_j, 1), 0.5)
    imp_naive = np.broadcast_to(mean_j[np.newaxis, :], (N, n_hvj)).astype(np.float32)
    del pm_dense, jr_dense

    # ── Step 3b: NMF baseline — rank-K matrix factorization (fair head-to-head) ──
    # Same rank K as the LeafletFA run so it isolates "does our factorization beat vanilla NMF".
    # Label-free (unlike mb_ct). Both 0-impute (nmf0) and mean-impute (nmfm) reported.
    K_run = int(psi_hvj.shape[0])
    print(f"\n  [NMF] Fitting rank-{K_run} NMF (0-impute + mean-impute) on {n_hvj} junctions...")
    imp_nmf0, imp_nmfm = fit_nmf_impute(jr_hvj, pm_hvj, K_run)      # in-sample (home-field) NMF
    print(f"  NMF imp shapes: nmf0={imp_nmf0.shape}  nmfm={imp_nmfm.shape}")

    # TRANSFER NMF: programs H frozen on TRAIN, loadings projected onto test -- the fair comparator
    # to LeafletFA, which also transfers a frozen Psi and only infers per-cell loadings on test.
    try:
        import nmf_transfer as nt
        # The transfer dictionary H is learned on the TRAINING cells, so it does not depend on how
        # much the TEST set was downsampled. Fitting it inside this per-level function repeated an
        # identical NMF fit (and an identical 15,000-cell read of a 13 GB h5ad) at every coverage
        # level. Both are now cached on the training inputs; only the cheap projection of the test
        # cells runs per level. Same random_state and same training data => identical H.
        tr_jr, tr_cov = nt.load_train_jr_cached(train_h5ad, n_cells=15000, junc_idx=hvj_idx)
        fitted = nt.fit_train_H(tr_jr, tr_cov, K_run,
                                cache_key=(train_h5ad, int(K_run), int(len(hvj_idx))))
        tw = nt.project_test(fitted, jr_hvj.toarray().astype(np.float32),
                             pm_hvj.toarray().astype(bool))
        imp_nmf0_tr = np.clip((tw["nmf0_tr"][0] @ tw["nmf0_tr"][1]).astype(np.float32), 0, 1)
        imp_nmfm_tr = np.clip((tw["nmfm_tr"][0] @ tw["nmfm_tr"][1]).astype(np.float32), 0, 1)
        del tw
        print(f"  NMF-transfer imp shapes: nmf0_tr={imp_nmf0_tr.shape}  nmfm_tr={imp_nmfm_tr.shape}")
    except Exception as e:
        print(f"  [NMF-transfer] skipped: {e}", flush=True)
        imp_nmf0_tr = imp_nmfm_tr = imp_nmf0  # fall back so metric code is unconditional

    # ── Step 4: Per-junction Pearson — global + per-dataset ──────────────────
    print(f"\n  Computing per-junction Pearson ({n_hvj} junctions)...")

    # ENTRY SETS -- which (cell, junction) pairs the metrics are computed over.
    #
    # "all_covered" is every pair covered in the ORIGINAL full-depth data. But MeanBayes returns
    # observed entries UNCHANGED (np.where(mask, psi, post_mean)), and at 10% reads removed 91.6%
    # of these pairs are still covered after downsampling -- so on this entry set mb/mb_ct mostly
    # echo back a lightly-thinned copy of the truth rather than imputing anything. lf (= phi@psi)
    # by contrast replaces EVERY entry with a model prediction and can never pass through. The two
    # are therefore not doing the same task, and cross-group comparisons on this entry set are not
    # apples-to-apples:
    #     pure reconstruction (no passthrough) : lf, nmf0, nmfm
    #     uses the observed value              : lf_bb, mb, mb_ct
    #
    # "lost_to_downsampling" restricts to pairs that WERE covered originally but are NOT covered
    # after downsampling. Nothing can pass through there -- every method must genuinely impute --
    # so it is the honest head-to-head, and it is the number to lead Figure 3 with.
    pm_bool  = pm_hvj.toarray().astype(bool)
    lost_mask = cov_mask & ~pm_bool

    # "fixed_lost" -- the SAME entries at every coverage level.
    #
    # `lost_mask` above changes composition along the coverage axis and so cannot be read as a
    # performance-vs-coverage curve: the shallowest entries lose their last read first, so the lost
    # set starts out enriched for single-read entries and grows steadily deeper as more reads are
    # removed. Two effects then move in opposite directions -- the task gets harder while the target
    # gets cleaner -- and they largely cancel, which is why `lf` looks flat at ~0.05 across a
    # 450-fold depth range.
    #
    # Fixing the population removes the confound: take the entries that are covered originally and
    # absent at the HARDEST downsampling, and score those same entries everywhere. Coverage is
    # ATSE-level by construction (psi_mask is exactly cluster_matrix > 0 -- verified, identical nnz
    # 9,409,850 -- and the cluster matrix is the ATSE total), so an ATSE is either wholly present or
    # wholly absent in a cell and all of its junctions travel together automatically. Verified
    # directly: over 400 ATSEs / 15.9M cell-ATSEs, zero were partially observed.
    #
    # Verified properties of this set on the 70/30 held-out data (n_hvj=3000 -> 12,197 junctions /
    # 2,754 ATSEs / 2,329 genes / 39,812 cells; 95,710,571 covered pairs):
    #   fixed_lost = 93,394,617 entries (97.6% of covered), spanning ALL of those junctions, ATSEs,
    #   genes and cells; median original ATSE depth 8, mean 22.7, 10.7% single-read.
    #   Still observed at each level (% dropped -> % still observed):
    #     10 -> 93.3 | 25 -> 91.4 | 50 -> 86.4 | 75 -> 74.6 | 90 -> 55.2 | 99 -> 14.9 | 100 -> 0.0
    # Note the levels are INDEPENDENT resamples, not a nested thinning: ~1.03M entries (1.1%) lost at
    # 99% dropped are still observed at 100% dropped. So this is "absent at the hardest level", not
    # "absent at every level at or beyond".
    fixed_lost_mask = None
    if hardest_pm is not None:
        fixed_lost_mask = cov_mask & ~hardest_pm
        _still_obs = float((fixed_lost_mask & pm_bool).sum() / max(fixed_lost_mask.sum(), 1))
        print(f"  fixed_lost entries: {int(fixed_lost_mask.sum()):,} "
              f"({100.0*fixed_lost_mask.sum()/max(cov_mask.sum(),1):.1f}% of covered); "
              f"{_still_obs:.1%} of them are STILL OBSERVED at this level")
        print(f"    -> where they are still observed, lf_bb/mb/mb_ct pass the observed value "
              f"through; only lf/nmf0/nmfm are pure prediction everywhere. All methods are still "
              f"written to the CSV; the figure subsets them.")
    del pm_bool
    print(f"\n  Entry sets: all_covered={int(cov_mask.sum()):,}  "
          f"lost_to_downsampling={int(lost_mask.sum()):,} "
          f"({100.0*lost_mask.sum()/max(cov_mask.sum(),1):.1f}% of covered)")

    def compute_metrics_for_mask(cell_mask, label_suffix="", entry_mask=None):
        """Compute all metrics for a boolean cell subset over a given entry set."""
        em    = cov_mask if entry_mask is None else entry_mask
        cm    = em[cell_mask]            # (n_subset × n_hvj) entry mask
        lf    = imp_lf[cell_mask]
        lf_bb = imp_lf_bb[cell_mask]
        mb    = imp_mb[cell_mask]
        mb_ct = imp_mb_ct[cell_mask]
        naive = imp_naive[cell_mask]
        nmf0  = imp_nmf0[cell_mask]
        nmfm  = imp_nmfm[cell_mask]
        nmf0_tr = imp_nmf0_tr[cell_mask]     # transfer NMF (frozen train programs) -- fair vs lf
        nmfm_tr = imp_nmfm_tr[cell_mask]
        ori   = ori_hvj[cell_mask]

        r_lf_p,    r_lf_s    = per_junction_correlations(lf,    ori, cm, min_cells_per_junc)
        r_lf_bb_p, r_lf_bb_s = per_junction_correlations(lf_bb, ori, cm, min_cells_per_junc)
        r_mb_p,    r_mb_s    = per_junction_correlations(mb,    ori, cm, min_cells_per_junc)
        r_mbct_p,  r_mbct_s  = per_junction_correlations(mb_ct, ori, cm, min_cells_per_junc)
        r_naive_p, r_naive_s = per_junction_correlations(naive, ori, cm, min_cells_per_junc)
        r_nmf0_p,  r_nmf0_s  = per_junction_correlations(nmf0,  ori, cm, min_cells_per_junc)
        r_nmfm_p,  r_nmfm_s  = per_junction_correlations(nmfm,  ori, cm, min_cells_per_junc)
        r_n0tr_p,  r_n0tr_s  = per_junction_correlations(nmf0_tr, ori, cm, min_cells_per_junc)
        r_nmtr_p,  r_nmtr_s  = per_junction_correlations(nmfm_tr, ori, cm, min_cells_per_junc)

        m = {}
        m.update(summarise_perjunc(r_lf_p,    r_lf_s,    "lf"))
        m.update(summarise_perjunc(r_lf_bb_p, r_lf_bb_s, "lf_bb"))
        m.update(summarise_perjunc(r_mb_p,    r_mb_s,    "mb"))
        m.update(summarise_perjunc(r_mbct_p,  r_mbct_s,  "mb_ct"))
        m.update(summarise_perjunc(r_naive_p, r_naive_s, "naive"))
        m.update(summarise_perjunc(r_nmf0_p,  r_nmf0_s,  "nmf0"))
        m.update(summarise_perjunc(r_nmfm_p,  r_nmfm_s,  "nmfm"))
        m.update(summarise_perjunc(r_n0tr_p,  r_n0tr_s,  "nmf0_tr"))
        m.update(summarise_perjunc(r_nmtr_p,  r_nmtr_s,  "nmfm_tr"))

        flat_orig  = ori[cm]
        flat_lf    = lf[cm]
        flat_lf_bb = lf_bb[cm]
        flat_mb    = mb[cm]
        flat_mb_ct = mb_ct[cm]
        flat_naive = naive[cm]
        flat_nmf0  = nmf0[cm]
        flat_nmfm  = nmfm[cm]
        flat_n0tr  = nmf0_tr[cm]
        flat_nmtr  = nmfm_tr[cm]
        if len(flat_orig) > 0 and flat_orig.std() > 1e-8:
            m["global_pearson_lf"]    = float(np.corrcoef(flat_orig, flat_lf)[0, 1])
            m["global_pearson_lf_bb"] = float(np.corrcoef(flat_orig, flat_lf_bb)[0, 1])
            m["global_pearson_mb"]    = float(np.corrcoef(flat_orig, flat_mb)[0, 1])
            m["global_pearson_mb_ct"] = float(np.corrcoef(flat_orig, flat_mb_ct)[0, 1])
            m["global_pearson_nmf0"]  = float(np.corrcoef(flat_orig, flat_nmf0)[0, 1])
            m["global_pearson_nmfm"]  = float(np.corrcoef(flat_orig, flat_nmfm)[0, 1])
            m["global_pearson_nmf0_tr"] = float(np.corrcoef(flat_orig, flat_n0tr)[0, 1])
            m["global_pearson_nmfm_tr"] = float(np.corrcoef(flat_orig, flat_nmtr)[0, 1])
        else:
            m["global_pearson_lf"]    = np.nan
            m["global_pearson_lf_bb"] = np.nan
            m["global_pearson_mb"]    = np.nan
            m["global_pearson_mb_ct"] = np.nan
            m["global_pearson_nmf0"]  = np.nan
            m["global_pearson_nmfm"]  = np.nan
            m["global_pearson_nmf0_tr"] = np.nan
            m["global_pearson_nmfm_tr"] = np.nan

        # CALIBRATION (Reviewer 1): slope / RMSE / bias / binned curve, in PSI units.
        # Correlation alone hides compression toward the mean; slope > 1 (truth ~ pred) means
        # the model under-disperses its predictions.
        for _lbl, _pred in (("lf", flat_lf), ("lf_bb", flat_lf_bb), ("mb", flat_mb),
                            ("mb_ct", flat_mb_ct), ("naive", flat_naive),
                            ("nmf0", flat_nmf0), ("nmfm", flat_nmfm),
                            ("nmf0_tr", flat_n0tr), ("nmfm_tr", flat_nmtr)):
            for _k, _v in calibration(_pred, flat_orig).items():
                m[f"calib_{_k}_{_lbl}"] = _v

        # Mean absolute error across all covered (cell, junction) pairs
        if len(flat_orig) > 0:
            m["mae_naive"] = float(np.abs(flat_orig - flat_naive).mean())
            m["mae_lf"]    = float(np.abs(flat_orig - flat_lf).mean())
            m["mae_lf_bb"] = float(np.abs(flat_orig - flat_lf_bb).mean())
            m["mae_mb"]    = float(np.abs(flat_orig - flat_mb).mean())
            m["mae_mb_ct"] = float(np.abs(flat_orig - flat_mb_ct).mean())
            m["mae_nmf0"]  = float(np.abs(flat_orig - flat_nmf0).mean())
            m["mae_nmfm"]  = float(np.abs(flat_orig - flat_nmfm).mean())
            m["mae_nmf0_tr"] = float(np.abs(flat_orig - flat_n0tr).mean())
            m["mae_nmfm_tr"] = float(np.abs(flat_orig - flat_nmtr).mean())
        else:
            m["mae_naive"] = m["mae_lf"] = m["mae_lf_bb"] = m["mae_mb"] = m["mae_mb_ct"] = np.nan
            m["mae_nmf0"]  = m["mae_nmfm"] = m["mae_nmf0_tr"] = m["mae_nmfm_tr"] = np.nan

        m["n_cells"] = int(cell_mask.sum())
        return m

    rows = []

    # All cells combined
    all_mask = np.ones(N, dtype=bool)
    m_all = compute_metrics_for_mask(all_mask)
    print(f"  [all pearson]   lf={m_all['median_pearson_lf']:.4f}  lf_bb={m_all['median_pearson_lf_bb']:.4f}  "
          f"mb={m_all['median_pearson_mb']:.4f}  mb_ct={m_all['median_pearson_mb_ct']:.4f}  "
          f"nmf0={m_all['median_pearson_nmf0']:.4f}  nmfm={m_all['median_pearson_nmfm']:.4f}  "
          f"naive={m_all['median_pearson_naive']:.4f}  n_junc={m_all['n_junc_eval']}")
    print(f"  [all spearman]  lf={m_all['median_spearman_lf']:.4f}  lf_bb={m_all['median_spearman_lf_bb']:.4f}  "
          f"mb={m_all['median_spearman_mb']:.4f}  mb_ct={m_all['median_spearman_mb_ct']:.4f}  "
          f"nmf0={m_all['median_spearman_nmf0']:.4f}  nmfm={m_all['median_spearman_nmfm']:.4f}  "
          f"naive={m_all['median_spearman_naive']:.4f}")
    print(f"  [CALIBRATION]   lf slope={m_all.get('calib_slope_lf', float('nan')):.3f} "
          f"(1.0=calibrated, >1=compressed)  rmse={m_all.get('calib_rmse_lf', float('nan')):.4f}  "
          f"bias={m_all.get('calib_bias_lf', float('nan')):+.4f}")
    print(f"  [MAE]           naive={m_all['mae_naive']:.4f}  lf={m_all['mae_lf']:.4f}  "
          f"lf_bb={m_all['mae_lf_bb']:.4f}  mb={m_all['mae_mb']:.4f}  mb_ct={m_all['mae_mb_ct']:.4f}  "
          f"nmf0={m_all['mae_nmf0']:.4f}  nmfm={m_all['mae_nmfm']:.4f}")
    rows.append({"dataset": "all", "entry_set": "all_covered",
                 "psi_hvj_mean_std":   psi_hvj_mean_std,
                 "psi_hvj_median_std": psi_hvj_median_std,
                 **m_all})

    # THE COMPARABLE-ACROSS-COVERAGE SET: the same entries at every level, so the curve is a curve.
    if fixed_lost_mask is not None and fixed_lost_mask.sum() > 0:
        m_fx = compute_metrics_for_mask(all_mask, entry_mask=fixed_lost_mask)
        print(f"\n  === FIXED-ENTRY-SET imputation (same entries at every coverage level) ===")
        print(f"  [pearson]  lf={m_fx['median_pearson_lf']:.4f}  lf_bb={m_fx['median_pearson_lf_bb']:.4f}  "
              f"nmf0_tr={m_fx['median_pearson_nmf0_tr']:.4f}  nmfm_tr={m_fx['median_pearson_nmfm_tr']:.4f}  "
              f"mb={m_fx['median_pearson_mb']:.4f}  naive={m_fx['median_pearson_naive']:.4f}")
        print(f"  [MAE]      lf={m_fx['mae_lf']:.4f}  lf_bb={m_fx['mae_lf_bb']:.4f}  "
              f"nmf0_tr={m_fx['mae_nmf0_tr']:.4f}  naive={m_fx['mae_naive']:.4f}")
        rows.append({"dataset": "all", "entry_set": "fixed_lost_at_hardest",
                     "psi_hvj_mean_std":   psi_hvj_mean_std,
                     "psi_hvj_median_std": psi_hvj_median_std,
                     **m_fx})
    if lost_mask.sum() > 0:
        m_lost = compute_metrics_for_mask(all_mask, entry_mask=lost_mask)
        print(f"\n  === LOST-ENTRY imputation (no passthrough possible) ===")
        print(f"  [pearson]  lf={m_lost['median_pearson_lf']:.4f}  lf_bb={m_lost['median_pearson_lf_bb']:.4f}  "
              f"nmfm={m_lost['median_pearson_nmfm']:.4f}  nmf0={m_lost['median_pearson_nmf0']:.4f}  "
              f"mb={m_lost['median_pearson_mb']:.4f}  mb_ct={m_lost['median_pearson_mb_ct']:.4f}  "
              f"naive={m_lost['median_pearson_naive']:.4f}")
        print(f"  [MAE]      lf={m_lost['mae_lf']:.4f}  lf_bb={m_lost['mae_lf_bb']:.4f}  "
              f"nmfm={m_lost['mae_nmfm']:.4f}  mb_ct={m_lost['mae_mb_ct']:.4f}  "
              f"naive={m_lost['mae_naive']:.4f}")
        rows.append({"dataset": "all", "entry_set": "lost_to_downsampling",
                     "psi_hvj_mean_std":   psi_hvj_mean_std,
                     "psi_hvj_median_std": psi_hvj_median_std,
                     **m_lost})

    # Per dataset
    for ds in np.unique(dataset_labels):
        ds_mask = dataset_labels == ds
        m_ds = compute_metrics_for_mask(ds_mask)
        print(f"  [{ds}]  lf={m_ds['median_pearson_lf']:.4f}  lf_bb={m_ds['median_pearson_lf_bb']:.4f}  "
              f"mb={m_ds['median_pearson_mb']:.4f}  naive={m_ds['median_pearson_naive']:.4f}  "
              f"n_cells={m_ds['n_cells']}  n_junc={m_ds['n_junc_eval']}")
        rows.append({"dataset": ds, "entry_set": "all_covered",
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
    chunk_size         = int(sys.argv[5])  if len(sys.argv) > 5  else 4096
    num_epochs         = int(sys.argv[6])  if len(sys.argv) > 6  else 100
    patience           = int(sys.argv[7])  if len(sys.argv) > 7  else 10
    n_hvj              = int(sys.argv[8])  if len(sys.argv) > 8  else 10000
    min_atse           = int(sys.argv[9])  if len(sys.argv) > 9  else 0
    min_cells_per_junc = int(sys.argv[10]) if len(sys.argv) > 10 else 20
    phi_init_mode      = sys.argv[11]      if len(sys.argv) > 11 else "pi"
    # optional: h5mu at the HARDEST downsampling, used to fix the lost-entry population
    hardest_pm_path    = sys.argv[12]      if len(sys.argv) > 12 else None

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

    # Sample psi ~ sigmoid(psis_loc + psis_scale·eps) as in 05_impute_on_test.py.
    # Falls back to the point estimate if the posterior params are absent.
    if "psis_loc" in model_data and "psis_scale" in model_data:
        _loc   = np.asarray(model_data["psis_loc"],   dtype=np.float32)
        _scale = np.asarray(model_data["psis_scale"], dtype=np.float32)
        _rng_psi = np.random.default_rng(0)
        psi = (1.0 / (1.0 + np.exp(-(_loc + _scale * _rng_psi.standard_normal(_loc.shape))))
               ).astype(np.float32)
        print(f"  psi SAMPLED from posterior (psis_loc/scale), shape={psi.shape}")
    else:
        psi = model_data["psi"].astype(np.float32)
        print(f"  psi = point estimate (no psis_loc/scale in pkl), shape={psi.shape}")
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

    # ── Waypoint init vs. final delta — heavy DIAGNOSTIC, OFF by default ──────
    # Loads the FULL training h5ad (all layers) + PCA → ~100s of GB, OOM risk on
    # GPU nodes. It is not needed for imputation metrics (only reports how far phi
    # moved from its waypoint init). Enable with COMPUTE_WAYPOINT_DELTA=1.
    waypoints_use = bool(summary.get("waypoints_use", False))
    train_phi_arr = model_data.get("assign_post")
    if train_phi_arr is not None:
        train_phi_arr = np.array(train_phi_arr, dtype=np.float32)

    compute_wp = os.environ.get("COMPUTE_WAYPOINT_DELTA", "0") == "1"
    if compute_wp and waypoints_use and train_phi_arr is not None:
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

    # Read the HARDEST level's psi_mask ONCE, not once per level -- these files are ~38 GB each and
    # the mask does not depend on the level being scored.
    hardest_pm = None
    if hardest_pm_path:
        # Fail fast and loudly. A wrong path here would otherwise either crash 6 h in or, worse,
        # silently produce a fixed set defined at the wrong coverage level.
        if not os.path.exists(hardest_pm_path):
            print(f"ERROR: hardest_h5mu does not exist: {hardest_pm_path}")
            sys.exit(1)
        _hard_pct = max(int(re.search(r"RESAMPLED_(\d+)_PERCENT", p).group(1))
                        for p in h5mu_files)
        _got_pct = re.search(r"RESAMPLED_(\d+)_PERCENT", os.path.basename(hardest_pm_path))
        if not _got_pct or int(_got_pct.group(1)) != _hard_pct:
            print(f"ERROR: hardest_h5mu is not the hardest level. The percentage in the filename is "
                  f"percent DROPPED, so the hardest of the {len(h5mu_files)} levels found is "
                  f"{_hard_pct}%, but got '{os.path.basename(hardest_pm_path)}'.")
            sys.exit(1)
        print(f"\nReading hardest-level psi_mask for the fixed entry set ({_hard_pct}% dropped):\n"
              f"  {os.path.basename(hardest_pm_path)}")
        with h5py.File(hardest_pm_path, "r") as _hf:
            _s = _hf["mod/splicing"] if "mod/splicing" in _hf else _hf
            hardest_pm = read_sparse_layer(
                _s["layers/psi_mask"])[:, hvj_idx].toarray().astype(bool)
        print(f"  observed at hardest level: {int(hardest_pm.sum()):,} of "
              f"{hardest_pm.size:,} (cell, HVJ) pairs")

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
            phi_init_mode=phi_init_mode, rng=rng, train_h5ad=train_h5ad,
            hardest_pm=hardest_pm
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
    cols = ["downsampling_pct", "dataset", "median_pearson_lf", "median_pearson_nmfm",
            "median_pearson_nmf0", "median_pearson_mb", "median_pearson_mb_ct",
            "median_pearson_naive", "n_junc_eval", "n_cells"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
