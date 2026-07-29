"""Deterministic, label-free psi initialisation for the atlas.

Same construction as `A15_seed_consistency_sim.deterministic_psi_init` (NNDSVD, hard assignment,
coverage-weighted group means), but NNDSVD on 132,706 x 79,768 is not affordable directly, so the
assignment is learned on a FIXED subsample of cells and the profiles are then computed from those
cells. The subsample index is fixed, so the result is still bit-identical on every run -- which is the
whole point. Profiles are per-junction group means, so they transfer to the full matrix unchanged.
"""
import numpy as np, h5py, scipy.sparse as sp


def build(train_h5ad, K, n_sub=20000, min_group=200, seed=0):
    import anndata as ad
    a = ad.read_h5ad(train_h5ad, backed="r")
    N = a.shape[0]
    idx = np.sort(np.random.default_rng(seed).choice(N, size=min(n_sub, N), replace=False))
    sub = a[idx].to_memory()
    g = lambda k: np.asarray(sub.layers[k].todense() if sp.issparse(sub.layers[k])
                             else sub.layers[k], dtype=np.float32)
    junc, clus = g("cell_by_junction_matrix"), g("cell_by_cluster_matrix")
    cov = clus > 0
    psi_obs = np.where(cov, junc / np.maximum(clus, 1.0), 0.0).astype(np.float32)
    from sklearn.decomposition import _nmf
    W, _ = _nmf._initialize_nmf(psi_obs, K, init="nndsvda", random_state=0)
    lab = np.maximum(W, 0).argmax(1)
    covf = cov.astype(np.float32)
    glob = (psi_obs * cov).sum(0) / np.maximum(covf.sum(0), 1e-8)
    out = []
    for k in range(K):
        m = lab == k
        out.append(glob if m.sum() < min_group
                   else (psi_obs[m] * cov[m]).sum(0) / np.maximum(covf[m].sum(0), 1e-8))
    P = np.clip(np.vstack(out), 1e-3, 1 - 1e-3)
    print(f"[det-init] {len(idx):,} cells, group sizes {np.bincount(lab, minlength=K)}", flush=True)
    print(f"[det-init] contrast {P.std(0).mean():.4f}  prog-corr "
          f"{np.corrcoef(P)[~np.eye(K,dtype=bool)].mean():+.3f}", flush=True)
    return P
