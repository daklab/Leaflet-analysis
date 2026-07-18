#!/usr/bin/env python
"""
Step 07c: Cell-type + aging decoding on HELD-OUT INFERRED phi.

Fixes the memorization confound in 07 (which decoded TRAINING assign_post — phi fit
WITH those cells). Here phi for the held-out 30% test cells is inferred by the
frozen-psi test-time E-step (same machinery as 04f), so decodability measures the
representation quality that TRANSFERS — the property the released model will have.

Also computed for comparison: the SAME decode on training phi → train−heldout gap
= memorization estimate (Claude Science's suggestion).

Improvements over 07:
  - held-out inferred phi (not training phi)
  - age: within-cell-type SPEARMAN rho (robust) + R² with RidgeCV (capacity-matched
    alpha via inner CV → no −5.39 blow-up), reported PER cell type + pooled
  - within-DATASET age control (Tabula Muris vs Senis confounded with age)
  - effective_k_90 reported; cell-type also gets macro-F1 + permuted/shuffled nulls

Usage:
    python 07c_classify_heldout_phi.py <run_dir> <train_h5ad> <test_h5mu> <output_csv>
        [chunk=4096] [epochs=300] [patience=10]
"""

import sys, os, re, pickle, gzip, h5py
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix, coo_matrix
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import balanced_accuracy_score, f1_score, r2_score

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.set_default_dtype(torch.float32)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"[07c] GPU: {torch.cuda.get_device_name(0)}")
else:
    torch.set_num_threads(4); print("[07c] CPU")
from leafletfa import LeafletFA

_MISSING = {"", "nan", "na", "none", "unknown", "unassigned"}
CLS_TARGETS = ["broad_cell_type", "tissue_cell_type", "dataset"]


# ── model loading + E-step (from 04f) ─────────────────────────────────────────

def load_model(run_dir):
    d = {}
    with gzip.open(os.path.join(run_dir, "leafletfa_model.pkl.gz"), "rb") as f:
        while True:
            try: d.update(pickle.load(f))
            except EOFError: break
    return d


def read_sparse_layer(grp):
    data, ind, indptr = grp["data"][:], grp["indices"][:], grp["indptr"][:]
    n_rows = len(indptr) - 1
    n_cols = int(ind.max()) + 1 if len(ind) else 0
    shape = tuple(grp.attrs.get("shape", (n_rows, n_cols)))
    return csr_matrix((data, ind, indptr), shape=shape)


def build_chunk_adata(jc, cl):
    a = ad.AnnData(X=jc.astype(np.float32),
                   obs=pd.DataFrame(index=[str(i) for i in range(jc.shape[0])]),
                   var=pd.DataFrame(index=[str(j) for j in range(jc.shape[1])]))
    a.layers["cell_by_junction_matrix"] = jc
    a.layers["cell_by_cluster_matrix"]  = cl
    return a


def sample_pi_phi_init(pi_vec, K, n, rng):
    if pi_vec is None: pi_vec = np.full(K, 1.0 / K)
    pi_vec = np.asarray(pi_vec, float); pi_vec /= pi_vec.sum()
    return rng.dirichlet(np.maximum(pi_vec * float(K), 1e-3), size=n).astype(np.float32)


def as_conc_prior(bb_conc):
    """Turn a saved bb_conc into an input_conc_prior for the E-step.

    get_bb_conc() stores the literal string "infinity" when the model was trained with a Binomial
    likelihood, and the learned scalar when it was trained Beta-Binomial. Either way the E-step must
    use the SAME likelihood the model was trained under: passing inf when training was
    Beta-Binomial treats every read as more informative than the model believes, which sharpens
    q(phi), pulls it onto the data and away from the prior, and inflates held-out decodability.
    """
    if bb_conc is None or isinstance(bb_conc, str):     # "infinity" -> Binomial
        return torch.tensor(float("inf"))
    return torch.tensor(float(bb_conc))                 # learned concentration -> Beta-Binomial


def infer_phi_chunk(chunk_adata, psi_tensor, K, dir_conc, junc_prior, pi, alpha_pi, bb_conc,
                    epochs, patience, phi_init):
    delta = torch.tensor(float(dir_conc)) if dir_conc is not None else None
    model = LeafletFA(
        adata=chunk_adata, K=K, fixed_psi=psi_tensor, waypoints_use=False,
        junc_specific_prior=junc_prior,
        input_conc_prior=as_conc_prior(bb_conc),
        delta_fixed=delta,
        # pi_fixed, NOT pi_init: pi_init only *initialises* a pi that SVI then re-learns on the test
        # cells. pi is the prior over phi, so a re-learned pi means held-out phi is inferred under a
        # different prior than training phi -- which confounds the very train-vs-heldout gap this
        # script measures. pi_fixed freezes it, so both sides share the source prior.
        pi_fixed=torch.tensor(pi, dtype=torch.float32) if pi is not None else None,
        alpha_pi_fixed=torch.tensor(alpha_pi, dtype=torch.float32) if alpha_pi is not None else None,
        num_epochs=epochs, print_epochs=epochs + 1, patience=patience, min_delta=0.001,
        ELBO_num_particles=10, num_samples=100, lr=0.01, gamma=0.05,
        log_wandb=False, loss_plot=False, output_dir=None,
        enable_cpu_optimization=(DEVICE.type == "cpu"))
    model.from_anndata()
    model.train(num_initializations=1,
                phi_init=torch.tensor(phi_init, dtype=torch.float32))
    model.get_all_variables()
    phi = model.assign_post
    if isinstance(phi, torch.Tensor): phi = phi.detach().cpu().numpy()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    return np.array(phi, dtype=np.float32)


def infer_phi_fullcov(test_h5mu, psi_tensor, K, dir_conc, junc_prior, pi, alpha_pi, bb_conc,
                      chunk, epochs, patience, rng):
    """Infer phi for held-out test cells from FULL-COVERAGE counts.
    junc_orig reconstructed = round(junc_ratio_original * atse_original)."""
    with h5py.File(test_h5mu, "r") as f:
        s = f["mod/splicing"]; N = int(s["obs/_index"].shape[0])
        jr_ori = np.array(s["layers/junc_ratio_original"], dtype=np.float32)
        atse   = read_sparse_layer(s["layers/cell_by_cluster_matrix_original"]).tocsr().astype(np.float32)
        labels = {}
        for col in ["broad_cell_type", "tissue_cell_type", "dataset", "age", "mouse.id", "sex"]:
            if col in s["obs"]:
                g = s[f"obs/{col}"]
                if isinstance(g, h5py.Group):
                    cats = [c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]]
                    labels[col] = np.array([cats[c] if c >= 0 else "NA" for c in g["codes"][:]])
                else:
                    labels[col] = np.array(g[:])
    atse_d = atse.toarray()
    junc_orig = np.rint(jr_ori * atse_d).astype(np.float32)   # full-coverage junction counts
    del jr_ori
    junc_sp = csr_matrix(junc_orig); atse_sp = csr_matrix(atse_d)
    del junc_orig, atse_d

    print(f"[E-step] full-coverage test phi: {N} cells, {int(np.ceil(N/chunk))} chunks")
    all_phi = []
    for start in range(0, N, chunk):
        end = min(start + chunk, N); n = end - start
        jc = junc_sp[start:end]; cl = atse_sp[start:end]
        if jc.nnz == 0:
            all_phi.append(sample_pi_phi_init(pi, K, n, rng)); continue
        all_phi.append(infer_phi_chunk(build_chunk_adata(jc, cl), psi_tensor, K,
                       dir_conc, junc_prior, pi, alpha_pi, bb_conc, epochs, patience,
                       sample_pi_phi_init(pi, K, n, rng)))
        print(f"    phi {start}-{end}", flush=True)
    return np.vstack(all_phi), labels


# ── metrics ───────────────────────────────────────────────────────────────────

def effective_k_90(phi):
    m = np.sort(phi.mean(0))[::-1]
    if m.sum() <= 0: return 0
    return int(np.searchsorted(np.cumsum(m) / m.sum(), 0.90) + 1)


def parse_age(y):
    out = []
    for v in y:
        mm = re.match(r"^(\d+(?:\.\d+)?)m?$", str(v).strip().lower())
        out.append(float(mm.group(1)) if mm else np.nan)
    return np.array(out, float)


def cls_cv(X, y_str, n_splits=5, seed=42):
    keep = np.array([str(s).strip().lower() not in _MISSING for s in y_str])
    X, y_str = X[keep], y_str[keep]
    y = LabelEncoder().fit_transform(y_str)
    valid = np.where(np.bincount(y) >= n_splits)[0]
    m = np.isin(y, valid); X, y = X[m], LabelEncoder().fit_transform(y[m])
    if len(np.unique(y)) < 2:
        return dict(balanced_accuracy=np.nan, macro_f1=np.nan, n_classes=int(len(np.unique(y))), n_cells=int(m.sum()))
    pipe = Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=seed))])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    ba, f1 = [], []
    for tr, va in skf.split(X, y):
        pipe.fit(X[tr], y[tr]); p = pipe.predict(X[va])
        ba.append(balanced_accuracy_score(y[va], p)); f1.append(f1_score(y[va], p, average="macro", zero_division=0))
    nc = int(len(np.unique(y)))
    return dict(balanced_accuracy=float(np.mean(ba)), macro_f1=float(np.mean(f1)),
                n_classes=nc, n_cells=int(m.sum()), chance=1.0 / nc)


def _partial_out_sex(Xg, yg, sexg):
    """Remove additive SEX effects from BOTH phi and age, then model the residuals.

    Sex confounds age analyses: if sex differs in both splicing (phi) and age composition, part of
    an apparent age signal is really a sex difference. Frisch-Waugh-Lovell: regress each of X and y
    on sex dummies and keep the residuals -- the ridge then measures how much of the age variation
    phi explains BEYOND sex. Preferred over adding sex as a feature, which would let sex itself
    predict age and inflate R².

    No-ops when sex is missing or the group is single-sex (nothing to remove).
    """
    if sexg is None:
        return Xg, yg
    sv = np.array([str(v).strip().lower() for v in sexg])
    keep_levels = [u for u in np.unique(sv) if u not in _MISSING]
    if len(keep_levels) < 2:
        return Xg, yg                      # single-sex group: nothing to partial out
    D = np.column_stack([(sv == u).astype(float) for u in keep_levels])   # sex dummies
    D = np.column_stack([D, np.ones(len(D))])                            # + intercept
    # least-squares residualisation (pinv handles the collinear intercept)
    B  = np.linalg.pinv(D) @ np.column_stack([Xg, yg[:, None]])
    R  = np.column_stack([Xg, yg[:, None]]) - D @ B
    return R[:, :-1], R[:, -1]


def age_within_ct(X, age, ct, sex=None, min_cells=50, n_splits=5, seed=42,
                  alphas=(0.1, 1, 10, 100, 1000)):
    """Within-tissue-cell-type age: RidgeCV (capacity-matched alpha), R² + Spearman per group.

    Grouping is tissue_cell_type ONLY -- deliberately not pooled to broad_cell_type, because
    tissues age differentially and pooling across tissue within a broad type would mix aging rates.

    Groups with < min_cells are EXCLUDED (not down-weighted): a handful of cells gives a noisy
    per-group estimate that distorts any summary. The caller reports the MEDIAN and RANGE across
    the surviving groups rather than the mean, for the same reason.

    If `sex` is given, additive sex effects are partialled out of phi and age within each group.
    """
    rows = []
    valid = ~np.isnan(age)
    for g in np.unique(ct):
        if str(g).strip().lower() in _MISSING: continue
        gm = (ct == g) & valid
        if gm.sum() < min_cells: continue
        Xg, yg = X[gm], age[gm]
        Xg, yg = _partial_out_sex(Xg, yg, None if sex is None else np.asarray(sex)[gm])
        if yg.std() < 1e-8: continue
        k = min(n_splits, int(gm.sum()) // 2)
        if k < 2: continue
        pipe = Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=alphas))])
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        r2v, spv = [], []
        for tr, va in kf.split(Xg):
            pipe.fit(Xg[tr], yg[tr]); p = pipe.predict(Xg[va])
            r2v.append(r2_score(yg[va], p))
            spv.append(spearmanr(yg[va], p).correlation if yg[va].std() > 1e-8 else 0.0)
        rows.append(dict(group=g, n_cells=int(gm.sum()),
                         r2=float(np.mean(r2v)), spearman=float(np.nanmean(spv))))
    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def decode_block(X, labels, phi_source, base, seed=42):
    """Cell-type + age decode for one phi matrix. Returns list of row dicts."""
    rows = []
    effk = effective_k_90(X)
    # cell type + dataset
    for tgt in CLS_TARGETS:
        if tgt not in labels: continue
        r = cls_cv(X, labels[tgt].astype(str), seed=seed)
        rows.append({**base, "phi_source": phi_source, "effective_k_90": effk,
                     "target": tgt, "scope": "all", "metric": "balanced_accuracy",
                     "score": r["balanced_accuracy"], "macro_f1": r.get("macro_f1"),
                     "n_classes": r["n_classes"], "n_cells": r["n_cells"]})
    # shuffled-phi null (cell type)
    if "broad_cell_type" in labels:
        rng = np.random.default_rng(0)
        Xs = np.column_stack([rng.permutation(X[:, k]) for k in range(X.shape[1])])
        r = cls_cv(Xs, labels["broad_cell_type"].astype(str), seed=seed)
        rows.append({**base, "phi_source": phi_source, "effective_k_90": effk,
                     "target": "broad_cell_type", "scope": "shuffled_phi_null",
                     "metric": "balanced_accuracy", "score": r["balanced_accuracy"],
                     "macro_f1": r.get("macro_f1"), "n_classes": r["n_classes"], "n_cells": r["n_cells"]})
    # age — within cell type, all datasets + within each dataset
    if "age" in labels and "tissue_cell_type" in labels:
        age = parse_age(labels["age"]); ct = labels["tissue_cell_type"].astype(str)
        ds = labels["dataset"].astype(str) if "dataset" in labels else np.array(["all"] * len(age))
        scopes = [("all", np.ones(len(age), bool))] + \
                 [(f"dataset={d}", ds == d) for d in np.unique(ds)]
        sex = labels["sex"] if "sex" in labels else None
        for sc_name, sc in scopes:
            gdf = age_within_ct(X[sc], age[sc], ct[sc],
                                sex=None if sex is None else sex[sc])
            if len(gdf) == 0:
                continue
            # MEDIAN + RANGE across the surviving tissue-cell-types, not the mean. Small groups are
            # already excluded (min_cells); the mean is still dragged by the remaining marginal
            # ones, so the median is the honest central estimate and the range shows the spread.
            rows.append({**base, "phi_source": phi_source, "effective_k_90": effk,
                         "target": "age", "scope": sc_name, "metric": "spearman_within_ct",
                         "score":            float(gdf["spearman"].median()),
                         "spearman_median":  float(gdf["spearman"].median()),
                         "spearman_min":     float(gdf["spearman"].min()),
                         "spearman_max":     float(gdf["spearman"].max()),
                         "spearman_q25":     float(gdf["spearman"].quantile(0.25)),
                         "spearman_q75":     float(gdf["spearman"].quantile(0.75)),
                         "spearman_mean":    float(gdf["spearman"].mean()),   # kept for comparison
                         "r2_within_ct":     float(gdf["r2"].median()),
                         "r2_min":           float(gdf["r2"].min()),
                         "r2_max":           float(gdf["r2"].max()),
                         "sex_partialled":   bool(sex is not None),
                         "n_classes": len(gdf), "n_cells": int(gdf["n_cells"].sum())})
    return rows


def main():
    if len(sys.argv) < 5:
        print(__doc__); sys.exit(1)
    run_dir, train_h5ad, test_h5mu, out_csv = sys.argv[1:5]
    chunk    = int(sys.argv[5]) if len(sys.argv) > 5 else 4096
    epochs   = int(sys.argv[6]) if len(sys.argv) > 6 else 300
    patience = int(sys.argv[7]) if len(sys.argv) > 7 else 10
    if os.path.exists(out_csv):
        print(f"exists — skip {out_csv}"); sys.exit(0)

    md = load_model(run_dir)
    summary = pd.read_csv(os.path.join(run_dir, "run_summary.csv")).iloc[0].to_dict()
    # Point-estimate psi is PRUNED to the effective K (authoritative); psis_loc/scale
    # may retain the nominal K, so only sample when the shapes match (else use point).
    psi_pt = md["psi"].astype(np.float32)
    K = psi_pt.shape[0]
    if ("psis_loc" in md and "psis_scale" in md
            and np.asarray(md["psis_loc"]).shape[0] == K):
        loc = np.asarray(md["psis_loc"], np.float32); scl = np.asarray(md["psis_scale"], np.float32)
        psi = (1 / (1 + np.exp(-(loc + scl * np.random.default_rng(0).standard_normal(loc.shape))))).astype(np.float32)
        print("  psi SAMPLED from posterior")
    else:
        psi = psi_pt
        print(f"  psi = point estimate (pruned K={K} or no posterior params)")
    pi = md.get("pi"); alpha_pi = md.get("alpha_pi"); dir_conc = md.get("dir_conc")
    bb_conc = md.get("bb_conc")   # freeze the trained likelihood concentration for the E-step
    junc_prior = bool(summary.get("junc_specific_prior", True))
    print(f"  E-step likelihood: {'Binomial' if (bb_conc is None or isinstance(bb_conc, str)) else f'Beta-Binomial (conc={float(bb_conc):.3g})'}")
    psi_tensor = torch.tensor(psi, dtype=torch.float32)
    base = dict(run_dir=os.path.basename(run_dir), param_id=int(summary["param_id"]),
                K=int(summary["K"]), waypoints_use=bool(summary.get("waypoints_use", False)),
                junc_specific_prior=junc_prior, train_effk=int(summary.get("effective_k_90", -1)))
    print(f"K={K} junc_prior={junc_prior}")

    rng = np.random.default_rng(42)
    results = []

    # HELD-OUT inferred phi (the fix)
    phi_test, test_labels = infer_phi_fullcov(test_h5mu, psi_tensor, K, dir_conc,
                                              junc_prior, pi, alpha_pi, bb_conc,
                                              chunk, epochs, patience, rng)
    results += decode_block(phi_test, test_labels, "heldout_inferred", base)

    # TRAINING phi (assign_post) for the memorization gap
    if "assign_post" in md:
        train_phi = np.asarray(md["assign_post"], np.float32)
        To = ad.read_h5ad(train_h5ad, backed="r")
        train_labels = {c: To.obs[c].astype(str).values for c in
                        ["broad_cell_type", "tissue_cell_type", "dataset", "age"] if c in To.obs}
        try: To.file.close()
        except Exception: pass
        if train_phi.shape[0] == len(next(iter(train_labels.values()))):
            results += decode_block(train_phi, train_labels, "train_assign_post", base)
        else:
            print(f"  WARN: train_phi rows {train_phi.shape[0]} != labels — skip train block")

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"\nResults → {out_csv} ({len(df)} rows)")
    # quick print: heldout vs train cell-type + age
    for src in ["heldout_inferred", "train_assign_post"]:
        sub = df[(df.phi_source == src)]
        if len(sub):
            bc = sub[(sub.target == "broad_cell_type") & (sub.scope == "all")]["score"]
            ag = sub[(sub.target == "age") & (sub.scope == "all")]["score"]
            print(f"  [{src}] broad_bacc={bc.iloc[0] if len(bc) else np.nan:.3f}  "
                  f"age_spearman={ag.iloc[0] if len(ag) else np.nan:.3f}")


if __name__ == "__main__":
    main()
