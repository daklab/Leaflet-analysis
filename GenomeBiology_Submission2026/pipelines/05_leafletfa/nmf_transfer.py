"""NMF-TRANSFER baseline: freeze the NMF programs (H) on TRAIN, project TEST loadings onto them.

The in-sample NMF baselines (fit_nmf_W / fit_nmf_impute) fit BOTH programs and loadings on the test
cells -- home-field advantage. LeafletFA instead transfers a frozen dictionary (Psi learned on train)
and infers only the per-cell loadings (phi) on test. This module matches NMF to that setup:

  1. fit NMF on the TRAINING matrix  -> H_train (K programs)
  2. project TEST onto frozen H_train via NMF.transform -> W_test (loadings only)

Test entries are filled with TRAIN's per-junction mean (not the test mean) so no test statistic
leaks into the projection -- exactly like Psi/pi being frozen from train. Returns W_test (for 07c
structure) and H_train (for 04e reconstruction).
"""
import numpy as np
import h5py
import scipy.sparse as sp
from sklearn.decomposition import NMF


def load_train_jr(train_h5ad, n_cells=15000, junc_idx=None):
    """Training junc_ratio (PSI) + coverage for the first n_cells; optional junction subset."""
    with h5py.File(train_h5ad, "r") as h:
        def rows(g, n):
            ip = g["indptr"][:n + 1]
            a, b = int(ip[0]), int(ip[-1])
            return sp.csr_matrix((g["data"][a:b], g["indices"][a:b], ip - ip[0]),
                                 shape=(n, int(g.attrs["shape"][1]))).toarray().astype(np.float32)
        N = min(n_cells, h["layers/junc_ratio"]["indptr"].shape[0] - 1)
        jr = rows(h["layers/junc_ratio"], N)
        cl = rows(h["layers/cell_by_cluster_matrix"], N)
    cov = cl > 0
    if junc_idx is not None:
        jr = jr[:, junc_idx]
        cov = cov[:, junc_idx]
    return jr, cov


def _fill(jr, cov, cm):
    Rc = np.clip(np.nan_to_num(jr), 0.0, 1.0).astype(np.float32)
    return np.where(cov, Rc, 0.0 if cm is None else cm[None, :]).astype(np.float32)


def fit_nmf_transfer(train_jr, train_cov, test_jr, test_cov, K, max_iter=300, random_state=0):
    """Fit H on train, project test. Returns {"nmf0_tr": (W_test, H), "nmfm_tr": (W_test, H)}.

    W_test is (n_test, K); H is (K, n_junc). 0-impute and mean-impute variants, with the mean taken
    on TRAIN so the projection uses no test-derived statistic."""
    Rc_tr = np.clip(np.nan_to_num(train_jr), 0.0, 1.0).astype(np.float32)
    cnt = train_cov.sum(0)
    cm_tr = np.where(cnt > 0, (Rc_tr * train_cov).sum(0) / np.maximum(cnt, 1), 0.0).astype(np.float32)
    out = {}
    for tag, cm in (("nmf0_tr", None), ("nmfm_tr", cm_tr)):
        Xtr = _fill(train_jr, train_cov, cm)
        m = NMF(n_components=K, init="nndsvda", random_state=random_state, max_iter=max_iter)
        m.fit(Xtr)                                   # programs H learned on TRAIN
        Xte = _fill(test_jr, test_cov, cm)           # test filled with TRAIN mean -> no leakage
        W_test = m.transform(Xte).astype(np.float32) # loadings only, projected onto frozen H
        out[tag] = (W_test, m.components_.astype(np.float32))
        print(f"[NMF-transfer] {tag}: fit H on {Xtr.shape[0]} train cells -> W_test {W_test.shape}",
              flush=True)
        del Xtr, Xte, m
    return out

# ── cached transfer: H depends only on the TRAINING cells ────────────────────────────────────────
# `fit_nmf_transfer` fits H on train and then projects the test cells. When it is called once per
# downsampling level, the train fit (and the train data load) is repeated identically every time --
# downsampling the TEST set cannot change a dictionary learned on TRAIN. On the atlas eval that was
# 7 identical NMF fits on 15,000 x 12,197 plus 7 reads of a 13 GB h5ad per model. These two helpers
# split the work so the train side is done once. Results are unchanged: same random_state, same
# training input, therefore the same H.
_TRAIN_CACHE = {}
_H_CACHE = {}


def load_train_jr_cached(train_h5ad, n_cells=15000, junc_idx=None):
    key = (train_h5ad, int(n_cells), None if junc_idx is None else hash(np.asarray(junc_idx).tobytes()))
    if key not in _TRAIN_CACHE:
        _TRAIN_CACHE[key] = load_train_jr(train_h5ad, n_cells=n_cells, junc_idx=junc_idx)
    return _TRAIN_CACHE[key]


def fit_train_H(train_jr, train_cov, K, max_iter=300, random_state=0, cache_key=None):
    """Fit the NMF dictionary on the training cells once; return {tag: (model, train_col_mean)}."""
    if cache_key is not None and cache_key in _H_CACHE:
        return _H_CACHE[cache_key]
    cnt = train_cov.sum(0)
    cm = np.where(cnt > 0, (np.clip(np.nan_to_num(train_jr), 0, 1) * train_cov).sum(0)
                  / np.maximum(cnt, 1), 0.0)
    out = {}
    for tag, X in (("nmf0_tr", _fill(train_jr, train_cov, None)),
                   ("nmfm_tr", _fill(train_jr, train_cov, cm))):
        m = NMF(n_components=K, init="nndsvda", random_state=random_state, max_iter=max_iter)
        m.fit(X)
        out[tag] = (m, cm)
        print(f"[NMF-transfer] {tag}: fit H on {X.shape[0]} train cells (cached for reuse)",
              flush=True)
    if cache_key is not None:
        _H_CACHE[cache_key] = out
    return out


def project_test(fitted, test_jr, test_cov):
    """Project test cells onto a frozen, already-fitted dictionary. Cheap; run per coverage level."""
    out = {}
    for tag, (m, cm) in fitted.items():
        Xte = _fill(test_jr, test_cov, None if tag == "nmf0_tr" else cm)
        W = m.transform(Xte).astype(np.float32)
        out[tag] = (W, m.components_)
    return out
