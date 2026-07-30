#!/usr/bin/env python
"""
Step 04: Evaluate LeafletFA imputation on test cells at varying coverage levels.

Methodology:
  - For each downsampling level (RESAMPLED_X_PERCENT = X% reads dropped):
    1. Load raw downsampled junction counts (cell_by_junction_matrix) and
       cluster totals (cell_by_cluster_matrix) from the h5mu test file
    2. Run LeafletFA inference with psi FIXED to the trained model's psi
       (fixed_psi parameter) — this is a proper E-step that only optimises
       phi (cell factor weights) given the observed (downsampled) counts
    3. Reconstruct PSI = phi @ psi
    4. Compare to junc_ratio_original (pre-downsampling PSI) on all
       originally-covered junctions
  - Cells are processed in chunks to keep CPU memory manageable.

Usage:
    python 04_evaluate_imputation.py <run_dir> <train_h5ad> <resampled_dir> <output_csv>
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
from scipy.sparse import csr_matrix, issparse
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


def get_junction_mask(train_h5ad_path):
    adata = ad.read_h5ad(train_h5ad_path, backed="r")
    col_indices  = np.arange(adata.n_vars)
    junction_ids = adata.var.index.values
    n_total      = adata.n_vars
    adata.file.close()
    return col_indices, junction_ids, n_total


def read_sparse_layer(grp):
    data    = grp["data"][:]
    indices = grp["indices"][:]
    indptr  = grp["indptr"][:]
    n_rows  = len(indptr) - 1
    n_cols  = int(indices.max()) + 1 if len(indices) else 0
    shape   = tuple(grp.attrs.get("shape", (n_rows, n_cols)))
    return csr_matrix((data, indices, indptr), shape=shape)


def build_chunk_adata(junc_chunk, clust_chunk):
    """
    Build a minimal AnnData for one chunk of cells, with the two layers
    that LeafletFA.from_anndata() requires.
    """
    n_cells, n_juncs = junc_chunk.shape
    obs_df = pd.DataFrame(index=[str(i) for i in range(n_cells)])
    var_df = pd.DataFrame(index=[str(j) for j in range(n_juncs)])
    adata  = ad.AnnData(X=junc_chunk.astype(np.float32), obs=obs_df, var=var_df)
    adata.layers["cell_by_junction_matrix"] = junc_chunk
    adata.layers["cell_by_cluster_matrix"]  = clust_chunk
    return adata


def infer_phi_chunk(chunk_adata, psi_tensor, K, dir_conc, bb_conc, junc_specific_prior,
                    pi_init, alpha_pi_init, num_epochs, patience):
    """
    Run LeafletFA with fixed psi on one chunk of cells.
    Returns phi: (n_cells × K) numpy array.
    """
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
        print_epochs        = num_epochs + 1,   # suppress per-epoch output
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


def evaluate_one_level(h5mu_path, psi, psi_tensor, col_indices, K,
                       dir_conc, bb_conc, junc_specific_prior, pi_init, alpha_pi_init,
                       chunk_size, num_epochs, patience):
    """
    Proper imputation evaluation for one downsampling level.

    Uses fixed_psi LeafletFA inference (E-step only) to infer phi from
    downsampled counts, then reconstructs PSI = phi @ psi and compares
    to the original PSI on all originally-covered junctions.
    """
    J = len(col_indices)

    with h5py.File(h5mu_path, "r") as f:
        splice = f["mod/splicing"]
        N = int(splice["obs/_index"].shape[0])

        print(f"  Loading downsampled count matrices ({N} cells × {J} junctions)...")
        junc_full  = read_sparse_layer(splice["layers/cell_by_junction_matrix"])
        clust_full = read_sparse_layer(splice["layers/cell_by_cluster_matrix"])

        junc_sub  = junc_full[:, col_indices]
        clust_sub = clust_full[:, col_indices]

        # junc_ratio_original is stored dense — read fully then subset columns
        print(f"  Loading original PSI (dense)...")
        ori_full = np.array(splice["layers/junc_ratio_original"], dtype=np.float32)
        ori_sub  = ori_full[:, col_indices]
        del ori_full

        n_orig_covered = int((ori_sub > 0).sum())
        print(f"  Original covered (eval set): {n_orig_covered:,}")
        ds_nnz = junc_sub.nnz
        print(f"  Downsampled junction obs: {ds_nnz:,}  ({100*ds_nnz/(N*J):.2f}%)")

    # Process cells in chunks
    n_chunks   = int(np.ceil(N / chunk_size))
    all_orig   = []
    all_pred   = []

    for ci, start in enumerate(range(0, N, chunk_size)):
        end = min(start + chunk_size, N)
        print(f"  Chunk {ci+1}/{n_chunks}: cells {start}–{end} ...", flush=True)

        junc_chunk  = junc_sub[start:end]
        clust_chunk = clust_sub[start:end]

        # Skip all-zero chunks (no reads at this downsampling level)
        if junc_chunk.nnz == 0:
            print(f"    WARNING: all-zero chunk — skipping")
            continue

        chunk_adata = build_chunk_adata(junc_chunk, clust_chunk)
        phi = infer_phi_chunk(
            chunk_adata, psi_tensor, K, dir_conc, bb_conc, junc_specific_prior,
            pi_init, alpha_pi_init, num_epochs, patience
        )

        # Reconstruct PSI from phi @ psi  (n_chunk × J)
        imp = phi @ psi   # both already float32 numpy

        ori_chunk = ori_sub[start:end]
        covered   = ori_chunk > 0
        all_orig.append(ori_chunk[covered])
        all_pred.append(imp[covered])

        n_pairs_so_far = sum(len(x) for x in all_orig)
        print(f"    phi shape={phi.shape}  eval pairs so far: {n_pairs_so_far:,}")

    if not all_orig:
        print("  WARNING: no evaluation pairs — returning NaN")
        return {k: float("nan") for k in ["pearson_r","spearman_r","l1_mean",
                                           "l1_median","l1_p90","smape",
                                           "cosine_sim","minmax_ratio","rmse",
                                           "n_eval_pairs","n_cells"]}

    orig_all = np.concatenate(all_orig).astype(np.float32)
    pred_all = np.concatenate(all_pred).astype(np.float32)
    n_pairs  = len(orig_all)

    pearson_r  = float(np.corrcoef(orig_all, pred_all)[0, 1])
    spearman_r = float(spearmanr(orig_all, pred_all, nan_policy="omit")[0])
    abs_diff   = np.abs(orig_all - pred_all)
    l1_mean    = float(np.mean(abs_diff))
    l1_median  = float(np.median(abs_diff))
    l1_p90     = float(np.quantile(abs_diff, 0.90))
    smape      = float(np.mean(2*abs_diff / (np.abs(orig_all) + np.abs(pred_all) + 1e-8)))
    orig64     = orig_all.astype(np.float64)
    pred64     = pred_all.astype(np.float64)
    denom      = float(np.linalg.norm(orig64) * np.linalg.norm(pred64) + 1e-8)
    cosine_sim = float(np.dot(orig64, pred64) / denom)
    minmax     = float(np.mean(
        np.minimum(np.abs(orig_all), np.abs(pred_all)) /
        (np.maximum(np.abs(orig_all), np.abs(pred_all)) + 1e-8)
    ))
    rmse       = float(np.sqrt(np.mean((orig_all - pred_all)**2)))

    print(f"  Pearson={pearson_r:.4f}  Spearman={spearman_r:.4f}  "
          f"L1={l1_mean:.4e}  n_pairs={n_pairs:,}")

    return {
        "n_cells":       N,
        "n_eval_pairs":  n_pairs,
        "pearson_r":     pearson_r,
        "spearman_r":    spearman_r,
        "l1_mean":       l1_mean,
        "l1_median":     l1_median,
        "l1_p90":        l1_p90,
        "smape":         smape,
        "cosine_sim":    cosine_sim,
        "minmax_ratio":  minmax,
        "rmse":          rmse,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 5:
        print("Usage: python 04_evaluate_imputation.py <run_dir> <train_h5ad> <resampled_dir> <output_csv>")
        sys.exit(1)

    run_dir       = sys.argv[1]
    train_h5ad    = sys.argv[2]
    resampled_dir = sys.argv[3]
    output_csv    = sys.argv[4]
    chunk_size    = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
    num_epochs    = int(sys.argv[6]) if len(sys.argv) > 6 else 300
    patience      = int(sys.argv[7]) if len(sys.argv) > 7 else 15

    if os.path.exists(output_csv):
        print(f"Output CSV already exists — skipping: {output_csv}")
        sys.exit(0)

    print(f"=== LeafletFA Imputation Evaluation (proper E-step inference) ===")
    print(f"Run dir:      {run_dir}")
    print(f"Chunk size:   {chunk_size} cells")
    print(f"Epochs:       {num_epochs}  patience={patience}")

    # Load model
    print("\nLoading model...")
    model_data = load_model(run_dir)
    summary    = load_run_summary(run_dir)

    psi         = model_data["psi"].astype(np.float32)   # (K × J)
    K           = int(model_data["K"])
    pi          = model_data.get("pi",        None)
    alpha_pi    = model_data.get("alpha_pi",  None)
    dir_conc    = model_data.get("dir_conc",  None)
    bb_conc     = model_data.get("bb_conc",   None)
    junc_prior  = bool(summary.get("junc_specific_prior", True))

    psi_tensor  = torch.tensor(psi, dtype=torch.float32)

    print(f"  psi shape: {psi.shape}  K={K}  dir_conc={dir_conc}  bb_conc={bb_conc}")
    if psi.max() < 1e-6:
        print("  WARNING: psi is all-zero — training collapsed. Skipping.")
        sys.exit(0)

    # Junction filter (all junctions — no max_junctions filter in eval)
    print("\nLoading junction order from train adata...")
    col_indices, junction_ids, n_total = get_junction_mask(train_h5ad)
    print(f"  Junctions: {len(col_indices)} / {n_total}")
    assert len(col_indices) == psi.shape[1], \
        f"Mismatch: col_indices={len(col_indices)}  psi.J={psi.shape[1]}"

    # Find resampled test files
    pattern   = os.path.join(resampled_dir, "RESAMPLED_*_PERCENT_test_*_2026-06-17.h5mu")
    h5mu_files = sorted(glob.glob(pattern))
    if not h5mu_files:
        print(f"ERROR: no h5mu files found: {pattern}")
        sys.exit(1)
    print(f"\nFound {len(h5mu_files)} resampled test files")

    results = []
    for h5mu_path in h5mu_files:
        fname     = os.path.basename(h5mu_path)
        pct_match = re.search(r"RESAMPLED_(\d+)_PERCENT", fname)
        pct       = int(pct_match.group(1)) if pct_match else -1

        print(f"\n{'='*60}")
        print(f"Downsampling: {pct}% reads dropped  ({fname})")

        metrics = evaluate_one_level(
            h5mu_path, psi, psi_tensor, col_indices, K,
            dir_conc, bb_conc, junc_prior, pi, alpha_pi,
            chunk_size, num_epochs, patience
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
            **metrics,
        }
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nResults → {output_csv}")
    print(df[["downsampling_pct","pearson_r","spearman_r","l1_mean","n_eval_pairs"]].to_string(index=False))


if __name__ == "__main__":
    main()
