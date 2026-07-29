"""Are the held-out Leiden clusters real structure, or fragmentation?

07c finds that LeafletFA's held-out loadings give the best-separated clusters (highest silhouette)
but the weakest agreement with annotated cell types (lowest ARI), and roughly 50 clusters where NMF
finds 21-37. Two readings fit that equally well:

  (a) phi captures splicing structure that does not reduce to cell identity -- cell-type labels come
      from gene expression, so partial alignment is expected and is the interesting result;
  (b) phi is over-fragmenting, and 50 tight clusters of nothing are not a latent space worth having.

Silhouette and ARI cannot separate these. Reproducibility can: structure driven by biology should
reappear when the model is retrained from a different random initialisation, whereas fragmentation
should not. This script therefore asks

  1. SEED REPRODUCIBILITY  cluster the held-out cells under each of the three same-K models and
     compare the partitions to each other (ARI). High agreement => the clusters are a property of
     the data, not of one fit.
  2. WHAT DO THEY TRACK    compare each partition to annotated cell type, to tissue, and to
     sequencing depth. A partition that tracks depth is a coverage artefact; one that tracks tissue
     or cell type is biological; one that tracks nothing needs explaining.
  3. NMF AS A YARDSTICK    the same two measurements for the NMF baseline, so "reproducible" has a
     scale.

Cells are subsampled (default 12,000 of 39,812) purely for runtime; the E-step is per-cell
independent, so this does not change any cell's loadings.

Usage: A5_cluster_reproducibility.py <model_dir> <train_h5ad> <test_h5mu> <out_csv> [K] [n_cells]
"""
from __future__ import annotations
import sys, os, json, gzip, pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")


def read_sparse_layer(grp):
    return sp.csr_matrix((np.array(grp["data"]), np.array(grp["indices"]), np.array(grp["indptr"])),
                         shape=tuple(grp.attrs["shape"]))


def leiden_of(mat, seed=0, resolution=1.0):
    import scanpy as sc, anndata as ad
    a = ad.AnnData(np.nan_to_num(np.asarray(mat, np.float32)))
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X", random_state=seed)
    sc.tl.leiden(a, resolution=resolution, random_state=seed, key_added="leiden",
                 flavor="igraph", n_iterations=2, directed=False)
    return a.obs["leiden"].astype(str).values


def main():
    from sklearn.metrics import adjusted_rand_score as ari
    model_dir, train_h5ad, test_h5mu, out_csv = sys.argv[1:5]
    K = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    n_cells = int(sys.argv[6]) if len(sys.argv) > 6 else 12000
    from estep_minibatch import infer_phi_minibatch
    import anndata as ad

    tasks = {10: (0, 1, 2), 20: (3, 4, 5), 30: (6, 7, 8)}[K]
    print(f"=== cluster reproducibility, K={K}, tasks {tasks} ===", flush=True)

    with h5py.File(test_h5mu, "r") as f:
        s = f["mod/splicing"] if "mod/splicing" in f else f
        # There is no `cell_by_junction_matrix_original` layer: the full-coverage junction counts are
        # reconstructed the same way A4 does it, as round(junc_ratio_original * atse_original).
        cl = read_sparse_layer(s["layers/cell_by_cluster_matrix_original"]).tocsr().astype(np.float32)
        jr_ori = np.array(s["layers/junc_ratio_original"], dtype=np.float32)
        jc = sp.csr_matrix(np.rint(np.nan_to_num(jr_ori) * cl.toarray()).astype(np.float32))
        obs_names = [x.decode() if isinstance(x, bytes) else str(x)
                     for x in np.array(s["obs/_index"])]
        meta = {}
        for key in ("broad_cell_type", "tissue", "batch", "dataset"):
            if f"obs/{key}" in s:
                g = s[f"obs/{key}"]
                if isinstance(g, h5py.Group):        # categorical
                    cats = [c.decode() if isinstance(c, bytes) else str(c)
                            for c in np.array(g["categories"])]
                    meta[key] = np.array(cats, dtype=object)[np.array(g["codes"])]
                else:
                    meta[key] = np.array([x.decode() if isinstance(x, bytes) else str(x)
                                          for x in np.array(g)])
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(jc.shape[0], size=min(n_cells, jc.shape[0]), replace=False))
    jc, cl = jc[idx], cl[idx]
    meta = {k: v[idx] for k, v in meta.items()}
    depth = np.asarray(cl.sum(1)).ravel()                    # total reads per cell
    depth_bin = pd.qcut(depth, 5, labels=False, duplicates="drop").astype(str)
    print(f"  {jc.shape[0]:,} cells x {jc.shape[1]:,} junctions; "
          f"metadata: {list(meta)}", flush=True)

    # The E-step is full-batch over the cells it is given, and 12,000 x 79,768 exceeds the card
    # (~4 GB per temporary, several of them in the Beta-Binomial backward). Split into the same
    # 4,096-cell chunks A3 uses: cells are conditionally independent given frozen globals, so
    # chunking is exact rather than an approximation.
    CH = 4096

    def phi_for(model, K):
        out = []
        for a0 in range(0, jc.shape[0], CH):
            a1 = min(a0 + CH, jc.shape[0])
            ck = ad.AnnData(sp.csr_matrix((a1 - a0, jc.shape[1]), dtype=np.float32))
            ck.layers["cell_by_junction_matrix"] = jc[a0:a1]
            ck.layers["cell_by_cluster_matrix"] = cl[a0:a1]
            bb = model.get("bb_conc")
            p, info = infer_phi_minibatch(
                ck, np.asarray(model["psi"], np.float32), K,
                dir_conc=model.get("dir_conc"), pi=model.get("pi"),
                alpha_pi=model.get("alpha_pi"),
                bb_conc=(None if bb in (None, "infinity") else bb),
                junc_prior=True, num_epochs=500)
            out.append(p)
        return np.vstack(out)

    parts, rows = {}, []
    cache_f = out_csv.replace(".csv", "_phi_cache.npz")
    cache = dict(np.load(cache_f)) if os.path.exists(cache_f) else {}
    for t in tasks:
        key = f"phi_{t}"
        if key in cache:
            phi = cache[key]
            print(f"  task {t}: phi from cache {phi.shape}", flush=True)
        else:
            with gzip.open(f"{model_dir}/eval_grouped/task_{t}/model/leafletfa_model.pkl.gz", "rb") as fh:
                m = pickle.load(fh)
            phi = phi_for(m, K)
            cache[key] = phi
            np.savez_compressed(cache_f, **cache)   # the E-steps are the expensive part; keep them
        parts[f"lfa_seed{t}"] = leiden_of(phi)
        print(f"  task {t}: phi {phi.shape}, {len(set(parts[f'lfa_seed{t}']))} Leiden clusters",
              flush=True)

    # NMF yardstick: same rank, refit per seed on the same cells
    from sklearn.decomposition import NMF
    clD = cl.toarray().astype(np.float32)
    R = np.clip(np.nan_to_num(jc.toarray().astype(np.float32) / np.maximum(clD, 1.0)), 0, 1)
    cov = clD > 0
    del clD
    for sd in (0, 1, 2):
        W = NMF(n_components=K, init="nndsvda", random_state=sd, max_iter=300).fit_transform(
            np.where(cov, R, 0.0))
        parts[f"nmf_seed{sd}"] = leiden_of(W)
        print(f"  NMF seed {sd}: {len(set(parts[f'nmf_seed{sd}']))} Leiden clusters", flush=True)

    def group(pref):
        return [k for k in parts if k.startswith(pref)]

    print("\n=== 1. seed reproducibility (ARI between partitions from different fits) ===", flush=True)
    for pref, lab in (("lfa", "LeafletFA"), ("nmf", "NMF")):
        ks = group(pref)
        vals = [ari(parts[a], parts[b]) for i, a in enumerate(ks) for b in ks[i + 1:]]
        print(f"  {lab:<10} pairwise ARI = {np.round(vals,3).tolist()}  mean={np.mean(vals):.3f}", flush=True)
        rows.append(dict(K=K, method=lab, measure="seed_reproducibility_ARI", value=float(np.mean(vals))))

    print("\n=== 2. what do the clusters track? (ARI vs each annotation) ===", flush=True)
    ann = {"broad_cell_type": meta.get("broad_cell_type"), "tissue": meta.get("tissue"),
           "batch": meta.get("batch"), "dataset": meta.get("dataset"),
           "depth_quintile": depth_bin}
    for pref, lab in (("lfa", "LeafletFA"), ("nmf", "NMF")):
        for aname, av in ann.items():
            if av is None:
                continue
            vals = [ari(parts[k], av) for k in group(pref)]
            print(f"  {lab:<10} vs {aname:<16} ARI = {np.mean(vals):.3f}", flush=True)
            rows.append(dict(K=K, method=lab, measure=f"ARI_vs_{aname}", value=float(np.mean(vals))))
        n = np.mean([len(set(parts[k])) for k in group(pref)])
        rows.append(dict(K=K, method=lab, measure="n_clusters", value=float(n)))

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
