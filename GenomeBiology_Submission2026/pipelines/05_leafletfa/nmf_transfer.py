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
