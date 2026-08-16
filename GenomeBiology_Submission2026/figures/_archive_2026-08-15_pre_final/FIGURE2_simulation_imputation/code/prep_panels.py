"""Precompute the panel data that needs the full simulation h5ad (slow) so the figure script
can stay fast and re-runnable.

Writes `panel_data.npz` with

    ATSE illustration
        psi_diff   (K, 3)   true junction usage of ONE differential cassette, per program
        psi_null   (K, 3)   the same for a non-differential cassette
        delta_diff, delta_null   effect sizes (max-min skip usage across programs)
    Embeddings (cells x 2, all on the same simulation)
        emb_true   from the TRUE phi           -- the structure that was simulated
        emb_lfa    from LeafletFA's phi        -- what the model recovered
        emb_nmf0   from NMF W (0-impute)       -- what the baseline recovered
        emb_nmfm   from NMF W (mean-impute)
        true_prog  (cells,) the true program label used to colour every embedding

Run once:  python prep_panels.py
"""
from __future__ import annotations
import os
import sys
import numpy as np
import anndata as ad
import scipy.sparse as sp
from sklearn.decomposition import NMF

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = ("/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION/"
       "sim_heart_K5_binom_eff1.0_frac0.5_seed0.h5ad")   # regenerated 2026-07-26
K = 5


def _cuda():
    import torch
    return torch.cuda.is_available()


def _dense(x):
    return np.asarray(x.todense() if sp.issparse(x) else x)


def embed(mat: np.ndarray, seed: int = 0) -> np.ndarray:
    """2-D UMAP of a cells x factors matrix; PCA fallback if umap-learn is missing."""
    mat = np.nan_to_num(np.asarray(mat, float))
    try:
        import umap
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(mat)
    except Exception as e:
        print(f"  (umap unavailable -> PCA fallback: {type(e).__name__})")
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(mat)


def main():
    print(f"reading {os.path.basename(SIM)}")
    a = ad.read_h5ad(SIM)
    psi_skip = np.asarray(a.uns["true_psi_skip"])          # (K, n_ATSE) P(skip) per program
    delta = np.asarray(a.uns["atse_delta_psi"])            # (n_ATSE,) max-min across programs

    # ── ATSE illustration: one clearly differential cassette, one flat one ────────────────
    i_diff = int(np.argmax(delta))                         # the strongest differential event
    i_null = int(np.argmin(delta))                         # a cassette that does not differ
    # Cassette junction usage: skip = psi, the two inclusion junctions split the remainder.
    def junc_usage(i):
        s = psi_skip[:, i]
        return np.column_stack([(1 - s) / 2, (1 - s) / 2, s])   # J1, J2, J3(skip)
    psi_diff, psi_null = junc_usage(i_diff), junc_usage(i_null)
    print(f"  differential ATSE #{i_diff}: delta={delta[i_diff]:.3f}  skip usage={psi_skip[:, i_diff].round(2)}")
    print(f"  null ATSE        #{i_null}: delta={delta[i_null]:.3f}  skip usage={psi_skip[:, i_null].round(2)}")

    # ── embeddings ───────────────────────────────────────────────────────────────────────
    true_phi = np.asarray(a.obsm["true_phi"], float)
    true_prog = true_phi.argmax(1)

    R = np.clip(np.nan_to_num(_dense(a.layers["junc_ratio"]).astype(float)), 0, 1)
    cov = _dense(a.layers["cell_by_cluster_matrix"]) > 0
    cnt = cov.sum(0)
    cmean = np.where(cnt > 0, (R * cov).sum(0) / np.maximum(cnt, 1), 0.0)

    # Silhouettes are computed on the LATENT representation (true phi, LeafletFA phi, NMF W), not on
    # the 2-D UMAP. UMAP exaggerates separation, so scoring the embedding would flatter every method
    # and would mostly measure UMAP's behaviour rather than the representation's.
    from sklearn.metrics import silhouette_score

    def sil(mat):
        return float(silhouette_score(np.nan_to_num(np.asarray(mat, float)), true_prog))

    embs = {"emb_true": embed(true_phi), "sil_true": sil(true_phi)}
    print(f"    true phi silhouette = {embs['sil_true']:.3f}")
    print("  fitting NMF baselines (same construction as benchmark_sweep.fit_nmf)")
    for tag, X in (("nmf0", np.where(cov, R, 0.0)),
                   ("nmfm", np.where(cov, R, cmean[None, :]))):
        m = NMF(n_components=K, init="nndsvda", random_state=0, max_iter=300)
        W = m.fit_transform(X)
        embs[f"emb_{tag}"] = embed(W)
        embs[f"sil_{tag}"] = sil(W)
        print(f"    {tag}: W={W.shape}  silhouette={embs[f'sil_{tag}']:.3f}")

    # Fit LeafletFA with the SHIPPED trainer (plate minibatch + early stopping) rather than reading
    # a previously-saved fit: the panel then always reflects the current implementation, and cannot
    # silently disappear if an external .npz is moved.
    print("  fitting LeafletFA (plate minibatch trainer)")
    sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
    import pyro
    from leafletfa.minibatch import LeafletFAMinibatch
    junc = _dense(a.layers["cell_by_junction_matrix"]).astype(np.float32)
    clus = _dense(a.layers["cell_by_cluster_matrix"]).astype(np.float32)
    pyro.clear_param_store()
    lfa = LeafletFAMinibatch(junc, clus, K=K, junc_specific_prior=True,
                             device="cuda" if _cuda() else "cpu", likelihood="betabinomial")
    lfa.fit(num_steps=5000, subsample_size=min(2048, lfa.N), lr=0.05, gamma=0.1, seed=0,
            verbose=False)
    phi_lfa = lfa.phi_mean().cpu().numpy()
    embs["emb_lfa"] = embed(phi_lfa)
    embs["sil_lfa"] = sil(phi_lfa)
    np.savez_compressed(os.path.join(HERE, "representative_fit.npz"),
                        phi=phi_lfa, psi=lfa.psi_mean().cpu().numpy(), true_prog=true_prog)
    print(f"    emb_lfa: phi={phi_lfa.shape}  silhouette={embs['sil_lfa']:.3f}  "
          f"(steps used {getattr(lfa, 'stopped_step', '?')})")

    out = os.path.join(HERE, "panel_data.npz")
    np.savez_compressed(out, psi_diff=psi_diff, psi_null=psi_null,
                        delta_diff=delta[i_diff], delta_null=delta[i_null],
                        true_prog=true_prog, **embs)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
