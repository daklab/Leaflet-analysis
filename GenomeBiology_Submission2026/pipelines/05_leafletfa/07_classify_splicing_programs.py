#!/usr/bin/env python
"""
09_classify_splicing_programs.py — Evaluate biological signal in LeafletFA cell representations.

For each model run, uses assign_post (N_train × K cell factor weights) — already stored
in the model pkl — to predict cell metadata via:

  - broad_cell_type  classification  → balanced accuracy + macro-F1
  - tissue_cell_type  classification  → balanced accuracy + macro-F1
  - age              ridge regression → R² + Pearson r  (numerical months)
  - dataset          classification  → balanced accuracy + macro-F1 (ideally LOW)

Baselines:
  - scVI linear latent (gene expression, same training cells)
  - shuffled assign_post (column-permuted phi → null)

Classifiers: L2 logistic regression, 5-fold stratified CV.
Regression:  Ridge (alpha=1), 5-fold CV.

Usage:
    python 09_classify_splicing_programs.py \\
        <model_dir> <train_h5ad> <scvi_h5ad> <output_dir>

    model_dir   LEAFLETFA_MODELS/YYYY-MM-DD/
    train_h5ad  train_splice_model_ready_*.h5ad  (for obs metadata + cell order)
    scvi_h5ad   ge_adata_with_scvi_model_*.h5ad  (has obsm X_scVI_linear)
    output_dir  where to write results
"""

import sys
import os
import argparse
import glob
import gzip
import pickle
import re

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import balanced_accuracy_score, f1_score, r2_score
from sklearn.pipeline import Pipeline

try:
    import umap as umap_lib
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False


# Classification targets
CLF_TARGETS = ["broad_cell_type", "tissue_cell_type", "dataset"]
# Regression targets — predicted as numerical values
REG_TARGETS = ["age"]

TARGETS = CLF_TARGETS + REG_TARGETS

# Values to treat as missing for any target — excluded before CV
_MISSING = {"nan", "unknown", "na", "n/a", "none", ""}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(run_dir):
    pkl = os.path.join(run_dir, "leafletfa_model.pkl.gz")
    d = {}
    with gzip.open(pkl, "rb") as f:
        while True:
            try:
                d.update(pickle.load(f))
            except EOFError:
                break
    return d


def parse_age_months(y_raw):
    """
    Convert age strings (e.g. '3m', '6m', '18m') to float months.
    Non-parseable values become NaN.
    """
    out = []
    for v in y_raw:
        s = str(v).strip().lower()
        m = re.match(r"^(\d+(?:\.\d+)?)m?$", s)
        out.append(float(m.group(1)) if m else np.nan)
    return np.array(out, dtype=np.float32)


def run_cv(X, y_str, n_splits=5, seed=42):
    """
    Stratified K-fold LR CV on X (N × dim) predicting y_str (string labels).
    Drops missing/unknown labels and classes with < n_splits examples before fitting.
    Returns unified result dict with score=balanced_accuracy.
    """
    keep = np.array([s.strip().lower() not in _MISSING for s in y_str])
    X_f   = X[keep]
    y_str = y_str[keep]

    le    = LabelEncoder()
    y_enc = le.fit_transform(y_str)

    counts = np.bincount(y_enc)
    valid  = np.where(counts >= n_splits)[0]
    mask   = np.isin(y_enc, valid)
    X_f    = X_f[mask]
    y_f    = LabelEncoder().fit_transform(y_enc[mask])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000, C=1.0, multi_class="multinomial",
            solver="lbfgs", random_state=seed,
        )),
    ])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    ba_vals, f1_vals = [], []
    for tr, va in skf.split(X_f, y_f):
        pipe.fit(X_f[tr], y_f[tr])
        pred = pipe.predict(X_f[va])
        ba_vals.append(balanced_accuracy_score(y_f[va], pred))
        f1_vals.append(f1_score(y_f[va], pred, average="macro", zero_division=0))

    n_classes = int(len(np.unique(y_f)))
    return {
        "metric_name":       "balanced_accuracy",
        "score":             float(np.mean(ba_vals)),
        "balanced_accuracy": float(np.mean(ba_vals)),
        "macro_f1":          float(np.mean(f1_vals)),
        "n_classes":         n_classes,
        "n_cells":           int(mask.sum()),
        "chance_level":      float(1.0 / n_classes),
        "r2":                None,
        "pearson_r":         None,
    }


def run_regression_cv(X, y_num, cov_str=None, n_splits=5, seed=42):
    """
    K-fold Ridge regression CV on X (N × dim) predicting numerical y_num.
    Drops NaN targets before fitting.

    If cov_str is provided (1-D string array of length N), age is residualized
    against one-hot(cov_str) within each fold before fitting Ridge on X.
    This gives incremental R² — what phi explains beyond the covariate alone.

    Returns unified result dict with score=R².
    """
    keep  = ~np.isnan(y_num)
    X_f   = X[keep]
    y_f   = y_num[keep]
    cov_f = cov_str[keep] if cov_str is not None else None

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    Ridge(alpha=1.0)),
    ])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    r2_vals, pr_vals = [], []
    for tr, va in kf.split(X_f):
        y_tr, y_va = y_f[tr], y_f[va]

        if cov_f is not None:
            # Residualize age against tissue_cell_type within fold (no leakage)
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            C_tr = enc.fit_transform(cov_f[tr].reshape(-1, 1))
            C_va = enc.transform(cov_f[va].reshape(-1, 1))
            cov_lr = LinearRegression(fit_intercept=True).fit(C_tr, y_tr)
            y_tr = y_tr - cov_lr.predict(C_tr)
            y_va = y_va - cov_lr.predict(C_va)

        pipe.fit(X_f[tr], y_tr)
        pred = pipe.predict(X_f[va])
        r2_vals.append(r2_score(y_va, pred))
        if y_va.std() > 1e-8 and pred.std() > 1e-8:
            pr_vals.append(float(np.corrcoef(y_va, pred)[0, 1]))
        else:
            pr_vals.append(0.0)

    return {
        "metric_name":       "r2",
        "score":             float(np.mean(r2_vals)),
        "r2":                float(np.mean(r2_vals)),
        "pearson_r":         float(np.mean(pr_vals)),
        "n_cells":           int(keep.sum()),
        "balanced_accuracy": None,
        "macro_f1":          None,
        "n_classes":         None,
        "chance_level":      None,
    }


def run_age_within_groups(X, y_num, groups, min_cells=50, n_splits=5, seed=42):
    """
    Ridge regression for age run separately within each tissue_cell_type group.
    Only includes groups with >= min_cells cells that have valid age values.
    Groups where all cells are the same age (std≈0) are skipped.

    Returns (group_df, summary_dict).
    group_df is a DataFrame with one row per evaluated group.
    summary_dict is a unified result dict with score = mean R² across groups.
    """
    _miss = _MISSING
    group_results = []

    valid_age = ~np.isnan(y_num)
    for g in np.unique(groups):
        if str(g).strip().lower() in _miss:
            continue
        g_mask = (groups == g) & valid_age
        if g_mask.sum() < min_cells:
            continue
        X_g = X[g_mask]
        y_g = y_num[g_mask]
        if y_g.std() < 1e-8:
            continue
        k = min(n_splits, int(g_mask.sum()) // 2)
        if k < 2:
            continue
        pipe = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        r2_v, pr_v = [], []
        for tr, va in kf.split(X_g):
            pipe.fit(X_g[tr], y_g[tr])
            pred = pipe.predict(X_g[va])
            r2_v.append(r2_score(y_g[va], pred))
            if y_g[va].std() > 1e-8 and pred.std() > 1e-8:
                pr_v.append(float(np.corrcoef(y_g[va], pred)[0, 1]))
            else:
                pr_v.append(0.0)
        group_results.append({
            "group": g,
            "n_cells": int(g_mask.sum()),
            "n_folds": k,
            "r2": float(np.mean(r2_v)),
            "pearson_r": float(np.mean(pr_v)),
        })

    if not group_results:
        summary = {
            "metric_name": "r2_within_celltype", "score": float("nan"),
            "r2": float("nan"), "pearson_r": float("nan"),
            "n_cells": 0, "n_classes": 0,
            "balanced_accuracy": None, "macro_f1": None, "chance_level": None,
        }
        return pd.DataFrame(group_results), summary

    gdf = pd.DataFrame(group_results)
    summary = {
        "metric_name":       "r2_within_celltype",
        "score":             float(gdf["r2"].mean()),
        "r2":                float(gdf["r2"].mean()),
        "pearson_r":         float(gdf["pearson_r"].mean()),
        "n_cells":           int(gdf["n_cells"].sum()),
        "n_classes":         len(gdf),   # repurposed as n_groups
        "balanced_accuracy": None,
        "macro_f1":          None,
        "chance_level":      None,
    }
    return gdf, summary


def run_target_cv(X, y_raw, target, cov_str=None, n_splits=5, seed=42):
    """Dispatch to regression or classification based on target.
    cov_str: tissue_cell_type string array, passed as covariate for age regression."""
    if target in REG_TARGETS:
        y_num = parse_age_months(y_raw)
        return run_regression_cv(X, y_num, cov_str=cov_str, n_splits=n_splits, seed=seed)
    return run_cv(X, y_raw.astype(str), n_splits=n_splits, seed=seed)


def make_umap_plot(phi, obs, run_name, output_dir, max_cells=10_000):
    """Compute UMAP on phi (N × K) and plot colored by top 10 broad/tissue cell types.
    Subsamples to max_cells for speed (93K cells × UMAP would take ~30 min)."""
    if not _UMAP_AVAILABLE:
        print("  umap-learn not installed — skipping UMAP")
        return
    N = phi.shape[0]
    if N > max_cells:
        rng_u = np.random.default_rng(0)
        idx   = rng_u.choice(N, size=max_cells, replace=False)
        idx   = np.sort(idx)
        phi_u = phi[idx]
        obs_u = obs.iloc[idx].reset_index(drop=True)
        print(f"  Subsampled {N:,} → {max_cells:,} cells for UMAP")
    else:
        phi_u, obs_u = phi, obs
    print(f"  Computing UMAP on phi ({phi_u.shape[0]} cells × {phi_u.shape[1]} factors)...")
    reducer = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    emb = reducer.fit_transform(phi_u)

    cols = ["broad_cell_type", "tissue_cell_type"]
    titles = ["Top 10 Broad Cell Type", "Top 10 Tissue Cell Type"]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, col, title in zip(axes, cols, titles):
        if col not in obs_u.columns:
            ax.text(0.5, 0.5, f"{col} not in obs", transform=ax.transAxes, ha="center")
            continue

        labels = obs_u[col].astype(object).fillna("Unknown").astype(str).values
        top10  = pd.Series(labels).value_counts().head(10).index.tolist()
        lplot  = np.where(np.isin(labels, top10), labels, "Other")

        ordered = [l for l in pd.Series(labels[np.isin(lplot, top10)]).value_counts().index if l in top10]
        if "Other" in set(lplot):
            ordered = ordered + ["Other"]

        palette  = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
        cmap     = {l: ("lightgray" if l == "Other" else palette[i % len(palette)])
                    for i, l in enumerate(ordered)}

        # plot "Other" first so named types render on top
        other = lplot == "Other"
        ax.scatter(emb[other, 0], emb[other, 1], c="lightgray", s=1, alpha=0.2, rasterized=True)
        for label in [l for l in ordered if l != "Other"]:
            m = lplot == label
            ax.scatter(emb[m, 0], emb[m, 1], c=[cmap[label]], s=2, alpha=0.6,
                       label=f"{label} (n={m.sum():,})", rasterized=True)

        ax.legend(markerscale=5, fontsize=7, loc="best", title=col,
                  framealpha=0.7, ncol=1)
        ax.set_title(f"{run_name} — {title}", fontsize=10)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.axis("off")

    plt.tight_layout()
    out = os.path.join(output_dir, f"{run_name}_umap_phi.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  UMAP → {out}")

    # Copy to tests/umaps/ for quick inspection
    import shutil
    tests_umap_dir = "/gpfs/commons/home/kisaev/Leaflet-analysis/tests/umaps"
    os.makedirs(tests_umap_dir, exist_ok=True)
    dest = os.path.join(tests_umap_dir, os.path.basename(out))
    shutil.copy2(out, dest)
    print(f"  UMAP copy → {dest}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("train_h5ad")
    ap.add_argument("output_dir")
    ap.add_argument("--scvi-h5ad", default=None,
                    help="Optional scVI latent h5ad for the scVI_linear baseline. "
                         "Omit to skip (run the scVI baseline separately).")
    ap.add_argument("--run-dir", default=None,
                    help="If set, evaluate only this single run_N directory (for SLURM array use).")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    single_run_mode = args.run_dir is not None

    # ── Training cell metadata ────────────────────────────────────────────────
    print("Loading train h5ad metadata...")
    adata = ad.read_h5ad(args.train_h5ad, backed="r")
    available_targets = [t for t in TARGETS if t in adata.obs.columns]
    obs = adata.obs[["cell_id"] + available_targets].copy().reset_index(drop=True)
    n_train = len(obs)
    adata.file.close()
    print(f"  {n_train} training cells | targets available: {available_targets}")
    for t in available_targets:
        if t in REG_TARGETS:
            y_num = parse_age_months(obs[t].astype(str).values)
            print(f"    {t}: {int((~np.isnan(y_num)).sum())} valid cells, "
                  f"unique values={sorted(set(obs[t].dropna().astype(str).unique()))}")
        else:
            print(f"    {t}: {obs[t].nunique()} classes")

    # ── scVI baseline (OPTIONAL — decoupled; run separately on matched cells) ──
    # Pass --scvi-h5ad to include it. Omitted by default so classification depends
    # only on LeafletFA phi + null baseline (needed to rank runs for model selection).
    scvi_ok  = False
    X_scvi   = np.zeros((0, 0), dtype=np.float32)
    obs_scvi = obs.iloc[:0]
    if args.scvi_h5ad and str(args.scvi_h5ad).lower() not in ("none", "skip", ""):
        print("\nLoading scVI latent...")
        scvi_ad = ad.read_h5ad(args.scvi_h5ad, backed="r")
        scvi_cell_ids = scvi_ad.obs["cell_id"].values if "cell_id" in scvi_ad.obs.columns \
                        else scvi_ad.obs.index.values

        # Normalize cell IDs: train IDs carry a plate lane suffix
        # (e.g. "D7_B001382_S271_L004") while scVI IDs are bare ("D7_B001382").
        def _norm_id(x):
            return re.sub(r"_S\d+_L\d+$", "", str(x))

        scvi_id_to_row = {}
        for i, cid in enumerate(scvi_cell_ids):
            scvi_id_to_row.setdefault(_norm_id(cid), i)

        train_cell_ids = obs["cell_id"].values
        scvi_rows = np.array([scvi_id_to_row.get(_norm_id(cid), -1) for cid in train_cell_ids])
        has_scvi  = scvi_rows >= 0
        print(f"  {has_scvi.sum()}/{n_train} training cells have scVI latent "
              f"(after ID normalization)")

        X_scvi_all = np.array(scvi_ad.obsm["X_scVI_linear"], dtype=np.float32)
        scvi_ad.file.close()

        X_scvi   = X_scvi_all[scvi_rows[has_scvi]]
        obs_scvi = obs[has_scvi].reset_index(drop=True)
        scvi_ok  = int(has_scvi.sum()) >= 50
        if not scvi_ok:
            print(f"  WARNING: only {has_scvi.sum()} cells matched scVI — skipping baseline")
    else:
        print("\nscVI baseline: not provided (--scvi-h5ad omitted) — skipping.")

    all_results = []

    # Groups for within-celltype age regression
    age_groups      = obs["tissue_cell_type"].astype(str).values \
                      if "tissue_cell_type" in obs.columns else None
    age_groups_scvi = obs_scvi["tissue_cell_type"].astype(str).values \
                      if "tissue_cell_type" in obs_scvi.columns else None

    # ── Permuted-label null (computed once — independent of model) ────────────
    print("\n--- Null: permuted labels / values ---")
    rng_label = np.random.default_rng(0)
    X_dummy   = rng_label.standard_normal((n_train, 10)).astype(np.float32)
    for target in available_targets:
        y_raw  = obs[target].astype(str).values
        y_perm = rng_label.permutation(y_raw)
        if target in REG_TARGETS:
            y_num_perm = parse_age_months(y_perm)
            _, res = run_age_within_groups(X_dummy, y_num_perm, age_groups)
            print(f"  {target}: R²={res['r2']:.3f}  pearson_r={res['pearson_r']:.3f}  "
                  f"n_groups={res['n_classes']}")
        else:
            res = run_cv(X_dummy, y_perm.astype(str))
            print(f"  {target}: bal_acc={res['balanced_accuracy']:.3f}  "
                  f"chance={res['chance_level']:.3f}")
        all_results.append({
            "model_type": "label_permuted", "run_name": "label_permuted",
            "param_id": -2, "K": -1, "effective_k_90": None,
            "waypoints_use": False, "junc_specific_prior": False,
            "batch_size": -1, "best_elbo": None,
            "target": target, **res,
        })

    print("\n--- Baseline: scVI linear latent ---")
    for target in (available_targets if scvi_ok else []):
        y_raw = obs_scvi[target].astype(str).values
        if target in REG_TARGETS:
            y_num_scvi = parse_age_months(y_raw)
            _, res = run_age_within_groups(X_scvi, y_num_scvi, age_groups_scvi)
            print(f"  {target}: R²={res['r2']:.3f}  pearson_r={res['pearson_r']:.3f}  "
                  f"n_groups={res['n_classes']}  n_cells={res['n_cells']:,}")
        else:
            res = run_cv(X_scvi, y_raw.astype(str))
            print(f"  {target}: bal_acc={res['balanced_accuracy']:.3f}  "
                  f"f1={res['macro_f1']:.3f}  n_classes={res['n_classes']}  "
                  f"chance={res['chance_level']:.3f}")
        all_results.append({
            "model_type": "scVI_linear", "run_name": "scVI_linear",
            "param_id": -1, "K": X_scvi.shape[1], "effective_k_90": None,
            "waypoints_use": False, "junc_specific_prior": False,
            "batch_size": -1, "best_elbo": None,
            "target": target, **res,
        })

    # ── LeafletFA runs ────────────────────────────────────────────────────────
    if single_run_mode:
        run_dirs_to_eval = [args.run_dir]
    else:
        run_dirs_to_eval = sorted(glob.glob(os.path.join(args.model_dir, "run_*")))

    summaries = []
    for run_dir in run_dirs_to_eval:
        csv = os.path.join(run_dir, "run_summary.csv")
        if not os.path.exists(csv):
            continue
        row = pd.read_csv(csv).iloc[0].to_dict()
        row["run_dir"] = run_dir
        summaries.append(row)
    print(f"\nFound {len(summaries)} LeafletFA run(s) to evaluate")

    for row in summaries:
        run_name = os.path.basename(row["run_dir"])
        print(f"\n--- {run_name}: K={row['K']}, waypoints={row['waypoints_use']}, "
              f"prior={row['junc_specific_prior']}, batch={row['batch_size']} ---")

        m   = load_model(row["run_dir"])
        phi = m["assign_post"].astype(np.float32)
        assert phi.shape[0] == n_train, \
            f"assign_post rows ({phi.shape[0]}) != train cells ({n_train})"

        # How many factors account for 90% of variance in cell assignments?
        col_var   = phi.var(axis=0)
        sorted_var = np.sort(col_var)[::-1]
        cumvar     = np.cumsum(sorted_var) / sorted_var.sum()
        effective_k_90 = int(np.searchsorted(cumvar, 0.90) + 1)
        print(f"  effective_k_90={effective_k_90} (of K={int(row['K'])})")

        for target in available_targets:
            y_raw = obs[target].astype(str).values
            if target in REG_TARGETS:
                y_num = parse_age_months(y_raw)
                group_df, res = run_age_within_groups(phi, y_num, age_groups)
                print(f"  {target}: R²={res['r2']:.3f}  pearson_r={res['pearson_r']:.3f}  "
                      f"n_groups={res['n_classes']}  (within tissue_cell_type, ≥50 cells)")
                # Save per-group breakdown alongside the per-run CSV
                if single_run_mode and not group_df.empty:
                    grp_csv = os.path.join(
                        args.output_dir, f"{run_name}_age_by_celltype.csv"
                    )
                    group_df.insert(0, "run_name", run_name)
                    group_df.to_csv(grp_csv, index=False)
                    print(f"    per-group age results → {grp_csv}")
            else:
                res = run_cv(phi, y_raw.astype(str))
                print(f"  {target}: bal_acc={res['balanced_accuracy']:.3f}  "
                      f"f1={res['macro_f1']:.3f}  n_classes={res['n_classes']}")
            all_results.append({
                "model_type": "LeafletFA", "run_name": run_name,
                "param_id": int(row["param_id"]), "K": int(row["K"]),
                "effective_k_90": effective_k_90,
                "waypoints_use": bool(row["waypoints_use"]),
                "junc_specific_prior": bool(row["junc_specific_prior"]),
                "batch_size": int(row["batch_size"]),
                "best_elbo": float(row["best_elbo"]),
                "target": target, **res,
            })

        # Null: column-permuted phi
        rng = np.random.default_rng(42)
        phi_null = phi.copy()
        for col in range(phi_null.shape[1]):
            phi_null[:, col] = rng.permutation(phi_null[:, col])

        for target in available_targets:
            y_raw = obs[target].astype(str).values
            if target in REG_TARGETS:
                y_num = parse_age_months(y_raw)
                _, res = run_age_within_groups(phi_null, y_num, age_groups)
            else:
                res = run_cv(phi_null, y_raw.astype(str))
            all_results.append({
                "model_type": "shuffled", "run_name": f"{run_name}_shuffled",
                "param_id": int(row["param_id"]), "K": int(row["K"]),
                "effective_k_90": effective_k_90,
                "waypoints_use": bool(row["waypoints_use"]),
                "junc_specific_prior": bool(row["junc_specific_prior"]),
                "batch_size": int(row["batch_size"]),
                "best_elbo": float(row["best_elbo"]),
                "target": target, **res,
            })

        # ── Save CSV first (before UMAP — so results are preserved if UMAP is slow) ──
        if single_run_mode:
            _df_tmp = pd.DataFrame(all_results)
            _out_csv_tmp = os.path.join(args.output_dir, f"{run_name}_classifier_results.csv")
            _df_tmp.to_csv(_out_csv_tmp, index=False)
            print(f"\nResults (pre-UMAP) → {_out_csv_tmp}")

        # ── UMAP of phi space (subsample to 10K for speed) ───────────────────
        make_umap_plot(phi, obs, run_name, args.output_dir)

    # ── Save ─────────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    if single_run_mode:
        run_name = os.path.basename(args.run_dir.rstrip("/"))
        out_csv  = os.path.join(args.output_dir, f"{run_name}_classifier_results.csv")
        df.to_csv(out_csv, index=False)
        print(f"\nResults → {out_csv}")
        return   # skip plots in single-run mode — 08b_collect_classification.py does those

    out_csv = os.path.join(args.output_dir, "classifier_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nResults → {out_csv}")

    # ── Plot: score per target × K (best ELBO model per K) ───────────────────
    lfa     = df[df["model_type"] == "LeafletFA"]
    scvi_df = df[df["model_type"] == "scVI_linear"]
    null_df = df[df["model_type"] == "shuffled"].groupby("target")["score"].mean()

    best_per_K = (lfa.sort_values("best_elbo", ascending=True)
                     .groupby(["K", "target"])
                     .first()
                     .reset_index())

    fig, axes = plt.subplots(1, len(available_targets),
                             figsize=(4 * len(available_targets), 5), sharey=False)
    if len(available_targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, available_targets):
        sub = best_per_K[best_per_K["target"] == target].sort_values("K")
        if sub.empty:
            ax.set_visible(False)
            continue

        metric_name = sub["metric_name"].iloc[0]
        K_labels    = sub["K"].astype(str).values
        ax.bar(K_labels, sub["score"], color="steelblue", alpha=0.8, label="LeafletFA")

        scvi_val = scvi_df[scvi_df["target"] == target]["score"]
        if not scvi_val.empty:
            ax.axhline(scvi_val.values[0], ls="--", c="orange", lw=1.5, label="scVI")

        if target in null_df.index:
            ax.axhline(null_df[target], ls=":", c="gray", lw=1.5, label="null (shuffled φ)")

        if target not in REG_TARGETS and not sub.empty:
            ax.axhline(sub["chance_level"].iloc[0], ls="-.", c="red", lw=1,
                       alpha=0.6, label="chance")

        ax.set_title(target, fontsize=10)
        ax.set_xlabel("K")
        ax.set_ylabel(metric_name)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)

    fig.suptitle("LeafletFA vs scVI — classifier/regression evaluation (5-fold CV)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "classifier_balanced_accuracy.png"), dpi=150)
    plt.close()

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n=== Best-per-K score (balanced_accuracy or R²) ===")
    pivot = best_per_K.pivot(index="K", columns="target", values="score")
    print(pivot.round(3).to_string())
    print("\n=== scVI baseline ===")
    print(scvi_df[["target", "metric_name", "score", "macro_f1", "n_classes", "pearson_r"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
