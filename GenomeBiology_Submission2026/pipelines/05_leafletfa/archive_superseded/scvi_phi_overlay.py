"""Panel E: gene-expression (scVI) UMAP with LeafletFA phi program activity overlaid.

Trains scVI on the held-out cells' RNA raw counts (mod/rna X) -> latent -> UMAP (falls back to
scanpy PCA if scvi is unavailable). Computes phi for the same cells via the frozen-Psi E-step (07c
machinery), then draws: (1) the GE UMAP coloured by broad_cell_type, and (2) one small panel per
program coloured by phi_k activity -- showing where each splicing program is active RELATIVE to GE
structure (does it mark a cell type, or cut across several).

Usage: scvi_phi_overlay.py <run_dir> <test_h5mu> <out_png> [max_cells=20000] [epochs=400]
"""
import sys, os, importlib.util
import numpy as np, h5py, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import scanpy as sc, anndata as ad, torch

P = "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa"
spec = importlib.util.spec_from_file_location("h07c", f"{P}/07c_heldout_structure.py")
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)

run_dir, test_h5mu, out_png = sys.argv[1:4]
MAXC = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
EP   = int(sys.argv[5]) if len(sys.argv) > 5 else 400

# ---- RNA raw counts + labels ----
with h5py.File(test_h5mu, "r") as f:
    r = f["mod/rna"]; N = r["obs/_index"].shape[0]
    idx = np.sort(np.random.default_rng(0).choice(N, min(MAXC, N), replace=False))
    Xg = h.read_sparse_layer(r["X"]).tocsr()[idx] if isinstance(r["X"], h5py.Group) else \
         sp.csr_matrix(np.array(r["X"])[idx])
    s = f["mod/splicing"]
    g = s["obs/broad_cell_type"]
    if isinstance(g, h5py.Group):
        cats = [c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]]
        labels = np.array([cats[c] if c >= 0 else "NA" for c in g["codes"][:]])[idx]
    else:
        labels = np.array(g[:])[idx]
print(f"RNA {Xg.shape} on {len(idx)} cells", flush=True)

a = ad.AnnData(Xg.astype(np.float32)); a.obs["cell_type"] = labels
a.layers["counts"] = a.X.copy()
sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
sc.pp.highly_variable_genes(a, n_top_genes=2000, subset=True)

emb_name = "scVI"
try:
    import scvi
    a.layers["counts"] = a.layers["counts"][:, a.var_names.isin(a.var_names)]  # keep HVG counts
    av = a.copy(); av.X = av.layers["counts"]
    scvi.model.SCVI.setup_anndata(av, layer=None)
    m = scvi.model.SCVI(av, n_latent=30)
    m.train(max_epochs=EP, early_stopping=True)
    a.obsm["X_emb"] = m.get_latent_representation()
    print("scVI latent done", flush=True)
except Exception as e:
    print(f"scVI unavailable ({e}); falling back to PCA", flush=True)
    sc.pp.scale(a, max_value=10); sc.tl.pca(a, n_comps=30); a.obsm["X_emb"] = a.obsm["X_pca"]; emb_name = "PCA"

sc.pp.neighbors(a, use_rep="X_emb", random_state=0); sc.tl.umap(a, random_state=0)
um = a.obsm["X_umap"]

# ---- phi for the same cells (frozen-Psi E-step) ----
m = h.load_model(run_dir); psi = np.asarray(m["psi"], np.float32); K = psi.shape[0]
psi_t = torch.tensor(psi, dtype=torch.float32, device=h.DEVICE)
with h5py.File(test_h5mu, "r") as f:
    s = f["mod/splicing"]
    jr = np.array(s["layers/junc_ratio_original"])[idx].astype(np.float32)
    atse = h.read_sparse_layer(s["layers/cell_by_cluster_matrix_original"]).tocsr()[idx].astype(np.float32)
atse_d = atse.toarray(); junc = np.rint(jr * atse_d).astype(np.float32); del jr
rng = np.random.default_rng(0); pi = np.asarray(m["pi"], float)
phis = []
for st in range(0, len(idx), 4096):
    en = min(st + 4096, len(idx))
    jc = sp.csr_matrix(junc[st:en]); cl = sp.csr_matrix(atse_d[st:en])
    phis.append(h.infer_phi_chunk(h.build_chunk_adata(jc, cl), psi_t, K, m.get("dir_conc"),
                bool(m.get("junc_specific_prior", False)), pi, np.asarray(m["alpha_pi"], float),
                m.get("bb_conc"), 500, 20, h.sample_pi_phi_init(pi, K, en-st, rng)))
phi = np.vstack(phis); phi = phi / phi.sum(1, keepdims=True)
print(f"phi {phi.shape}", flush=True)

# ---- plot: GE UMAP by cell type + per-program phi overlay ----
ncol = 6; nrow = 1 + int(np.ceil(K / ncol))
fig, ax = plt.subplots(nrow, ncol, figsize=(ncol*2.6, nrow*2.6))
ax = ax.ravel()
uniq = sorted(set(labels)); cmap = plt.colormaps.get_cmap("tab20")
for i, ct in enumerate(uniq):
    mk = labels == ct
    ax[0].scatter(um[mk,0], um[mk,1], s=2, alpha=0.5, color=cmap(i/max(len(uniq)-1,1)))
ax[0].set_title(f"{emb_name} (GE) by cell type", fontsize=9); ax[0].axis("off")
for k in range(1, ncol): ax[k].axis("off")
for k in range(K):
    axk = ax[ncol + k]
    sctr = axk.scatter(um[:,0], um[:,1], s=2, c=phi[:,k], cmap="viridis", vmin=0, vmax=np.percentile(phi[:,k],99))
    axk.set_title(f"program {k}", fontsize=8); axk.axis("off")
for k in range(ncol + K, len(ax)): ax[k].axis("off")
fig.suptitle(f"Splicing programs (phi) on the gene-expression map — {emb_name}, {len(idx)} cells, K={K}", fontsize=11)
fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(out_png, dpi=120)
print(f"wrote {out_png}", flush=True)
