#!/usr/bin/env python
"""
Step 04b: Evaluate LeafletFA imputation on top-N highly variable junctions (HVJ),
filtered to cell-junction pairs with original ATSE count > min_atse_count.

Motivation:
  Global evaluation on all junctions is dominated by constitutive junctions where
  mean PSI prediction works well regardless of coverage. Focusing on HVJ and
  requiring a minimum ATSE coverage threshold in the ORIGINAL (pre-downsampling)
  data isolates whether the model captures real cell-specific splicing variation.

HVJ selection:
  Per-junction variance of junc_ratio_original, computed only over covered cells
  (junc_ratio_original > 0). Top N junctions by this variance are selected.
  Computed once from the 100%-retained h5mu file; same original PSI in all files.

Filter:
  For each cell-junction pair, require cell_by_cluster_matrix_original > min_atse_count
  in the PRE-downsampling data, so we only evaluate positions we trust the ground truth.

Phi inference:
  Uses ALL junctions (not just HVJ) so phi estimation uses the full signal.
  Only the metric computation is restricted to HVJ + coverage filter.

Usage:
    python 04b_evaluate_imputation_hvj.py <run_dir> <train_h5ad> <resampled_dir> <output_csv>
        [chunk_size=2000] [num_epochs=300] [patience=15]
        [n_hvj=10000] [min_atse_count=30]
"""

import sys
import os
import glob
import pickle
import gzip
import re
import h5py
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr

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




# ── helpers ──────────────────────────────────────────────────────────────────

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


def compute_hvj_indices(h5mu_path, n_hvj):
    """
    Compute top-N HVJ from junc_ratio_original in h5mu.
    Variance computed per junction across cells where junc_ratio_original > 0.
    """
    print(f"Computing HVJ from: {os.path.basename(h5mu_path)}")
    with h5py.File(h5mu_path, "r") as f:
        ori = np.array(f["mod/splicing/layers/junc_ratio_original"], dtype=np.float32)

    N, J = ori.shape
    print(f"  junc_ratio_original shape: {N} x {J}")

    covered = ori > 0
    n_covered = covered.sum(axis=0).astype(np.float32)   # (J,)

    # Mean over covered cells only
    sum_x    = ori.sum(axis=0)                            # (J,) — zeros don't contribute
    mean_j   = np.where(n_covered > 0, sum_x / np.maximum(n_covered, 1), 0.0)

    # Variance over covered cells: E[(x - mean)^2 | covered]
    diff_sq  = (ori - mean_j) ** 2 * covered
    var_j    = np.where(n_covered > 1,
                        diff_sq.sum(axis=0) / np.maximum(n_covered - 1, 1),
                        0.0)

    # Require at least 50 cells covered
    var_j[n_covered < 50] = 0.0

    top_idx = np.argsort(var_j)[::-1][:n_hvj]
    top_idx = np.sort(top_idx)   # keep column order

    print(f"  Top {n_hvj} HVJ variance range: "
          f"{var_j[top_idx].min():.4f} – {var_j[top_idx].max():.4f}")
    print(f"  Median coverage among HVJ: {n_covered[top_idx].median() if hasattr(n_covered[top_idx], 'median') else np.median(n_covered[top_idx]):.0f} cells")

    return top_idx


def build_chunk_adata(junc_chunk, clust_chunk):
    n_cells, n_juncs = junc_chunk.shape
    obs_df = pd.DataFrame(index=[str(i) for i in range(n_cells)])
    var_df = pd.DataFrame(index=[str(j) for j in range(n_juncs)])
    adata  = ad.AnnData(X=junc_chunk.astype(np.float32), obs=obs_df, var=var_df)
    adata.layers["cell_by_junction_matrix"] = junc_chunk
    adata.layers["cell_by_cluster_matrix"]  = clust_chunk
    return adata


def infer_phi_chunk(chunk_adata, psi_tensor, K, dir_conc, bb_conc, junc_specific_prior,
                    pi_init, alpha_pi_init, num_epochs, patience):
    delta    = torch.tensor(float(dir_conc)) if dir_conc is not None else None
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
    model.train(num_initializations=1)
    model.get_all_variables()

    phi = model.assign_post
    if isinstance(phi, torch.Tensor):
        phi = phi.detach().cpu().numpy()
    return np.array(phi, dtype=np.float32)


def compute_metrics(orig_vals, pred_vals):
    if len(orig_vals) == 0:
        return {k: float("nan") for k in ["pearson_r","spearman_r","l1_mean",
                                           "l1_median","l1_p90","rmse","n_eval_pairs"]}
    pearson_r  = float(np.corrcoef(orig_vals, pred_vals)[0, 1])
    spearman_r = float(spearmanr(orig_vals, pred_vals, nan_policy="omit")[0])
    abs_diff   = np.abs(orig_vals - pred_vals)
    return {
        "n_eval_pairs": len(orig_vals),
        "pearson_r":    pearson_r,
        "spearman_r":   spearman_r,
        "l1_mean":      float(np.mean(abs_diff)),
        "l1_median":    float(np.median(abs_diff)),
        "l1_p90":       float(np.quantile(abs_diff, 0.90)),
        "rmse":         float(np.sqrt(np.mean((orig_vals - pred_vals)**2))),
    }


def evaluate_one_level(h5mu_path, psi, psi_tensor, hvj_idx, K,
                       dir_conc, bb_conc, junc_specific_prior, pi_init, alpha_pi_init,
                       chunk_size, num_epochs, patience, min_atse_count):
    J_all = psi.shape[1]
    J_hvj = len(hvj_idx)

    with h5py.File(h5mu_path, "r") as f:
        splice = f["mod/splicing"]
        N = int(splice["obs/_index"].shape[0])

        print(f"  Loading downsampled count matrices ({N} x {J_all})...")
        junc_full  = read_sparse_layer(splice["layers/cell_by_junction_matrix"])
        clust_full = read_sparse_layer(splice["layers/cell_by_cluster_matrix"])

        print(f"  Loading original PSI + ATSE counts...")
        ori_full   = np.array(splice["layers/junc_ratio_original"], dtype=np.float32)
        atse_sparse = read_sparse_layer(splice["layers/cell_by_cluster_matrix_original"])
        atse_full  = atse_sparse.toarray().astype(np.float32)

    # Subset to HVJ for evaluation only (inference uses all junctions)
    ori_hvj  = ori_full[:, hvj_idx]
    atse_hvj = atse_full[:, hvj_idx]
    del ori_full, atse_full

    # Coverage mask: original ATSE count > threshold AND original PSI > 0
    cov_mask = (atse_hvj > min_atse_count) & (ori_hvj > 0)
    n_cov_total = int(cov_mask.sum())
    print(f"  HVJ eval positions with ATSE>{min_atse_count} & PSI>0: {n_cov_total:,}")

    if n_cov_total == 0:
        print("  WARNING: no valid eval positions")
        return compute_metrics(np.array([]), np.array([]))

    # psi subset for reconstruction
    psi_hvj = psi[:, hvj_idx]   # (K x J_hvj)

    n_chunks = int(np.ceil(N / chunk_size))
    all_orig = []
    all_pred = []

    for ci, start in enumerate(range(0, N, chunk_size)):
        end = min(start + chunk_size, N)
        print(f"  Chunk {ci+1}/{n_chunks}: cells {start}–{end} ...", flush=True)

        junc_chunk  = junc_full[start:end]
        clust_chunk = clust_full[start:end]

        if junc_chunk.nnz == 0:
            print(f"    WARNING: all-zero chunk — skipping")
            continue

        chunk_adata = build_chunk_adata(junc_chunk, clust_chunk)
        phi = infer_phi_chunk(
            chunk_adata, psi_tensor, K, dir_conc, bb_conc, junc_specific_prior,
            pi_init, alpha_pi_init, num_epochs, patience
        )

        # Reconstruct on HVJ only
        imp_hvj  = phi @ psi_hvj    # (chunk x J_hvj)
        ori_chunk  = ori_hvj[start:end]
        mask_chunk = cov_mask[start:end]

        all_orig.append(ori_chunk[mask_chunk])
        all_pred.append(imp_hvj[mask_chunk])

        n_so_far = sum(len(x) for x in all_orig)
        print(f"    phi={phi.shape}  hvj eval pairs so far: {n_so_far:,}")

    if not all_orig:
        return compute_metrics(np.array([]), np.array([]))

    orig_all = np.concatenate(all_orig).astype(np.float32)
    pred_all = np.concatenate(all_pred).astype(np.float32)

    metrics = compute_metrics(orig_all, pred_all)
    print(f"  Pearson={metrics['pearson_r']:.4f}  Spearman={metrics['spearman_r']:.4f}  "
          f"L1={metrics['l1_mean']:.4e}  n_pairs={metrics['n_eval_pairs']:,}")
    return metrics


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)

    run_dir       = sys.argv[1]
    train_h5ad    = sys.argv[2]
    resampled_dir = sys.argv[3]
    output_csv    = sys.argv[4]
    chunk_size    = int(sys.argv[5])   if len(sys.argv) > 5 else 2000
    num_epochs    = int(sys.argv[6])   if len(sys.argv) > 6 else 300
    patience      = int(sys.argv[7])   if len(sys.argv) > 7 else 15
    n_hvj         = int(sys.argv[8])   if len(sys.argv) > 8 else 10000
    min_atse      = int(sys.argv[9])   if len(sys.argv) > 9 else 30

    if os.path.exists(output_csv):
        print(f"Output CSV already exists — skipping: {output_csv}")
        sys.exit(0)

    print(f"=== LeafletFA Imputation Evaluation (HVJ={n_hvj}, ATSE>{min_atse}) ===")
    print(f"Run dir:    {run_dir}")
    print(f"Chunk size: {chunk_size}  epochs={num_epochs}  patience={patience}")

    print("\nLoading model...")
    model_data = load_model(run_dir)
    summary    = load_run_summary(run_dir)

    psi        = model_data["psi"].astype(np.float32)   # (K x J)
    K          = int(model_data["K"])
    pi         = model_data.get("pi",       None)
    alpha_pi   = model_data.get("alpha_pi", None)
    dir_conc   = model_data.get("dir_conc", None)
    bb_conc    = model_data.get("bb_conc",  None)
    junc_prior = bool(summary.get("junc_specific_prior", True))

    psi_tensor = torch.tensor(psi, dtype=torch.float32)

    print(f"  psi shape: {psi.shape}  K={K}  dir_conc={dir_conc}  bb_conc={bb_conc}")
    if psi.max() < 1e-6:
        print("  WARNING: psi all-zero (collapsed training). Skipping.")
        sys.exit(0)

    # Verify junction count matches
    n_total = get_junction_count(train_h5ad)
    assert n_total == psi.shape[1], f"Junction mismatch: train={n_total}  psi={psi.shape[1]}"

    # Find resampled files
    pattern   = os.path.join(resampled_dir, "RESAMPLED_*_PERCENT_test_*_2026-06-17.h5mu")
    h5mu_files = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r"RESAMPLED_(\d+)_PERCENT", p).group(1))
    )
    if not h5mu_files:
        print(f"ERROR: no h5mu files found: {pattern}")
        sys.exit(1)
    print(f"\nFound {len(h5mu_files)} resampled test files")

    # Compute HVJ once from the lowest-dropout file (most data = best variance estimate)
    ref_file = h5mu_files[0]   # lowest pct dropped (sorted ascending)
    hvj_idx  = compute_hvj_indices(ref_file, n_hvj)
    print(f"\nHVJ index range: {hvj_idx.min()} – {hvj_idx.max()}")

    results = []
    for h5mu_path in h5mu_files:
        fname     = os.path.basename(h5mu_path)
        pct_match = re.search(r"RESAMPLED_(\d+)_PERCENT", fname)
        pct       = int(pct_match.group(1)) if pct_match else -1

        print(f"\n{'='*60}")
        print(f"Downsampling: {pct}% dropped  ({fname})")

        metrics = evaluate_one_level(
            h5mu_path, psi, psi_tensor, hvj_idx, K,
            dir_conc, bb_conc, junc_prior, pi, alpha_pi,
            chunk_size, num_epochs, patience, min_atse
        )

        row = {
            "run_dir":             os.path.basename(run_dir),
            "param_id":            int(summary["param_id"]),
            "K":                   int(summary["K"]),
            "pruned_K":            int(summary.get("pruned_K", K)),
            "waypoints_use":       summary.get("waypoints_use", False),
            "junc_specific_prior": junc_prior,
            "batch_size":          int(summary.get("batch_size", 0)),
            "best_elbo":           float(summary.get("best_elbo", float("nan"))),
            "downsampling_pct":    pct,
            "n_hvj":               n_hvj,
            "min_atse_count":      min_atse,
            **metrics,
        }
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nResults → {output_csv}")
    print(df[["downsampling_pct","pearson_r","spearman_r","l1_mean","n_eval_pairs"]].to_string(index=False))


if __name__ == "__main__":
    main()
