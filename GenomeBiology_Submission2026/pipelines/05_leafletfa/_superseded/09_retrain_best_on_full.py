#!/usr/bin/env python
"""
Step 12: Retrain LeafletFA on the full dataset (train + test) using the best
hyperparameters identified by 08_collect_evaluation.py.

Workflow:
  1. Read best param set (by mean Pearson r at 100% masking) from eval results.
  2. Extract the test-split splice h5ad from the test h5mu (same logic as
     00_extract_train_splice.py — avoids loading dense junc_ratio).
  3. Concatenate train h5ad + test h5ad → full splice h5ad (132,875 cells).
  4. Submit a single SLURM GPU job running run_training.py with the best param_id
     against the full splice h5ad.

Usage:
    python pipelines/05_leafletfa/09_retrain_best_on_full.py \\
        <eval_output_dir>          # from 08_collect_evaluation.py
        <model_dir>                # LEAFLETFA_MODELS/YYYY-MM-DD (source of trained runs)
        <train_h5ad>               # train_splice_model_ready_*.h5ad
        <test_h5mu>                # test_30_70_model_ready_*.h5mu
        <full_model_dir>           # output dir for full-data retrain
        [--K K]                    # which K to use (default: best across all K)
        [--dry-run]
"""

import sys
import os
import argparse
import json
import glob
import gzip
import pickle
import datetime
import subprocess
import h5py
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix, coo_matrix, vstack
from leafletfa.waypoints import calculate_centered_psi


# ── sparse helpers (mirrors 00_extract_train_splice.py) ──────────────────────

def read_sparse_layer(grp):
    data    = grp["data"][:]
    indices = grp["indices"][:]
    indptr  = grp["indptr"][:]
    shape   = tuple(grp.attrs.get("shape", (len(indptr) - 1, indices.max() + 1)))
    return csr_matrix((data, indices, indptr), shape=shape)


def read_obs(grp):
    index_key = "_index"
    idx = grp[index_key][:].astype(str)
    cols = {}
    for key in grp.keys():
        if key == index_key:
            continue
        item = grp[key]
        if isinstance(item, h5py.Dataset):
            vals = item[:]
            cols[key] = vals.astype(str) if vals.dtype.kind in ("S", "O") else vals
        elif isinstance(item, h5py.Group) and "categories" in item:
            cats = item["categories"][:].astype(str)
            codes = item["codes"][:]
            cols[key] = pd.Categorical.from_codes(codes, categories=cats)
    df = pd.DataFrame(cols, index=idx)
    df.index.name = None
    return df


def read_var(grp):
    if "_index" in grp:
        idx = grp["_index"][:].astype(str)
    elif "junction_id" in grp:
        idx = grp["junction_id"][:].astype(str)
    else:
        idx = np.arange(grp["junction_id"].shape[0]).astype(str)
    cols = {}
    for key in grp.keys():
        if key == "_index":
            continue
        item = grp[key]
        if isinstance(item, h5py.Dataset):
            vals = item[:]
            cols[key] = vals.astype(str) if vals.dtype.kind in ("S", "O") else vals
        elif isinstance(item, h5py.Group) and "categories" in item:
            cats = item["categories"][:].astype(str)
            codes = item["codes"][:]
            cols[key] = pd.Categorical.from_codes(codes, categories=cats)
    df = pd.DataFrame(cols, index=idx)
    df.index.name = None
    return df


def compute_psi_layers(jc, at, rho=0.1):
    """
    Compute junc_ratio and junc_ratio_centered using beta-binomial weighted centering.
    Mirrors 00_extract_train_splice.py — uses calculate_centered_psi from leafletfa.
    """
    at_coo = at.tocoo().astype(np.float32)
    rows, cols = at_coo.row, at_coo.col
    jc_vals = np.array(jc.tocsr()[rows, cols], dtype=np.float32).flatten()
    jc_coo_matched = coo_matrix((jc_vals, (rows, cols)), shape=jc.shape)
    junc_ratio = coo_matrix(
        (jc_vals / at_coo.data, (rows, cols)), shape=jc.shape
    ).tocsr().astype(np.float32)
    print(f"  Calling calculate_centered_psi (rho={rho})...")
    Y_sparse = calculate_centered_psi(jc_coo_matched, at_coo, rho=rho)
    return junc_ratio, Y_sparse.tocsr().astype(np.float32)


# ── extract test splice from h5mu ─────────────────────────────────────────────

def extract_test_splice(test_h5mu_path):
    """Extract splicing modality from test h5mu, recompute PSI layers."""
    print(f"\nExtracting test splice from {test_h5mu_path}")
    with h5py.File(test_h5mu_path, "r") as f:
        splice_grp = f["mod/splicing"]
        obs = read_obs(splice_grp["obs"])
        var = read_var(splice_grp["var"])
        print(f"  {len(obs)} cells × {len(var)} junctions")
        jc       = read_sparse_layer(splice_grp["layers/cell_by_junction_matrix"])
        at       = read_sparse_layer(splice_grp["layers/cell_by_cluster_matrix"])
        psi_mask = read_sparse_layer(splice_grp["layers/psi_mask"])

    print("  Computing junc_ratio and junc_ratio_centered (beta-binomial weighted centering)...")
    junc_ratio, junc_ratio_centered = compute_psi_layers(jc, at)

    adata = ad.AnnData(X=jc, obs=obs, var=var)
    adata.layers["cell_by_junction_matrix"] = jc
    adata.layers["cell_by_cluster_matrix"]  = at
    adata.layers["psi_mask"]                = psi_mask
    adata.layers["junc_ratio"]              = junc_ratio
    adata.layers["junc_ratio_centered"]     = junc_ratio_centered
    return adata


# ── main ─────────────────────────────────────────────────────────────────────

PYTHON  = "/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python"
REPO    = "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026"
RUN_SCRIPT = os.path.join(REPO, "pipelines/05_leafletfa/run_training.py")

SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=lfa_full
#SBATCH --output={log_dir}/lfa_full_%j.out
#SBATCH --error={log_dir}/lfa_full_%j.err
#SBATCH --mem=300G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --exclude=ne1dg7-[001-010]

export PYTHONUNBUFFERED=1
{python} -u {run_script} {param_id} {model_dir} {anndata_file} 2>&1
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_output_dir")
    parser.add_argument("model_dir",    help="Source LEAFLETFA_MODELS/YYYY-MM-DD dir")
    parser.add_argument("train_h5ad",   help="train_splice_model_ready_*.h5ad")
    parser.add_argument("test_h5mu",    help="test_30_70_model_ready_*.h5mu")
    parser.add_argument("full_model_dir", help="Output dir for full-data retrain")
    parser.add_argument("--K", type=int, default=None,
                        help="Restrict best-model selection to this K (default: best overall)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # ── 1. Find best param_id from evaluation results ─────────────────────────
    merged_csv = os.path.join(args.eval_output_dir, "all_runs_imputation_results.csv")
    if not os.path.exists(merged_csv):
        print(f"ERROR: {merged_csv} not found — run 08_collect_evaluation.py first")
        sys.exit(1)

    df = pd.read_csv(merged_csv)
    df100 = df[df["downsampling_pct"] == 100]
    if df100.empty:
        print("ERROR: no rows with downsampling_pct=100 in eval results")
        sys.exit(1)

    if args.K is not None:
        df100 = df100[df100["K"] == args.K]
        if df100.empty:
            print(f"ERROR: no rows with K={args.K} in eval results")
            sys.exit(1)

    best_row = df100.loc[df100["mean_pearson_r"].idxmax()]
    best_param_id = int(best_row["param_id"])
    print(f"\nBest model:")
    print(f"  param_id  = {best_param_id}")
    print(f"  K         = {int(best_row['K'])}")
    print(f"  Pearson r = {best_row['mean_pearson_r']:.4f} (at 100% masking)")
    print(f"  waypoints = {best_row['waypoints_use']}")
    print(f"  prior     = {best_row['junc_specific_prior']}")

    # ── 2. Load and copy best param set into full_model_dir ───────────────────
    src_param_file = os.path.join(args.model_dir, "parameter_combinations.json")
    with open(src_param_file) as f:
        param_list = json.load(f)

    best_params = param_list[best_param_id]
    os.makedirs(args.full_model_dir, exist_ok=True)
    log_dir = os.path.join(args.full_model_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Write a single-entry param JSON so run_training.py param_id=0 works
    full_param_file = os.path.join(args.full_model_dir, "parameter_combinations.json")
    with open(full_param_file, "w") as f:
        json.dump([best_params], f, indent=2)
    print(f"\nWrote single-param JSON → {full_param_file}")
    print(f"  Params: {best_params}")

    # ── 3. Build full splice h5ad (train + test) ──────────────────────────────
    full_h5ad_path = os.path.join(
        args.full_model_dir,
        f"full_splice_model_ready_{datetime.date.today()}.h5ad"
    )

    if os.path.exists(full_h5ad_path):
        print(f"\nFull h5ad already exists: {full_h5ad_path} — skipping extraction")
    else:
        print(f"\nLoading train h5ad: {args.train_h5ad}")
        train_adata = ad.read_h5ad(args.train_h5ad)
        print(f"  Train: {train_adata.shape}")

        test_adata = extract_test_splice(args.test_h5mu)
        print(f"  Test:  {test_adata.shape}")

        # Verify junction spaces match
        assert list(train_adata.var_names) == list(test_adata.var_names), \
            "var_names mismatch between train and test — junction spaces differ"

        # Recompute centered PSI over the FULL dataset so col_means reflect all cells
        print("\nConcatenating train + test...")
        full_adata = ad.concat([train_adata, test_adata], join="inner")
        full_adata.var = train_adata.var.copy()
        print(f"  Full: {full_adata.shape}")

        print("\nRecomputing junc_ratio_centered on full dataset (beta-binomial weighted centering)...")
        jc = csr_matrix(full_adata.layers["cell_by_junction_matrix"])
        at = csr_matrix(full_adata.layers["cell_by_cluster_matrix"])
        junc_ratio, junc_ratio_centered = compute_psi_layers(jc, at)
        full_adata.layers["junc_ratio"]          = junc_ratio
        full_adata.layers["junc_ratio_centered"] = junc_ratio_centered

        print(f"\nSaving full splice h5ad → {full_h5ad_path}")
        full_adata.write_h5ad(full_h5ad_path, compression="gzip")
        print(f"Done. Shape: {full_adata.shape}")

    # ── 4. Submit SLURM job ───────────────────────────────────────────────────
    script_content = SLURM_TEMPLATE.format(
        log_dir=log_dir,
        python=PYTHON,
        run_script=RUN_SCRIPT,
        param_id=0,          # single-entry JSON, always index 0
        model_dir=args.full_model_dir,
        anndata_file=full_h5ad_path,
    )

    script_path = os.path.join(args.full_model_dir, "lfa_full_retrain.sh")
    with open(script_path, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    print(f"\nSLURM script → {script_path}")

    if args.dry_run:
        print(f"[dry-run] To submit:\n  sbatch {script_path}")
        return

    result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"sbatch FAILED:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"Submitted: {result.stdout.strip()}")
    print(f"Monitor:   squeue -u $USER | grep lfa_full")


if __name__ == "__main__":
    main()
