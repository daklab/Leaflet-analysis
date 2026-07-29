"""Do held-out cells use the splicing programs the way training cells do?

A descriptive alternative to scoring a discrete clustering of phi against cell-type labels. Instead
of forcing a partition and reporting one agreement number, this reports the mean loading of each
program within each annotated cell type -- once for the cells the model was trained on, and once for
held-out cells whose loadings come from the frozen-dictionary E-step.

Because both maps come from the SAME model, the programs are identical by construction and need no
matching: the comparison measures directly whether the representation transfers, i.e. whether the
cell types that load on a program during training load on the same program when the model is applied
to cells it has never seen. That is a stronger and more interpretable statement than an ARI against
gene-expression-derived labels, which penalises any splicing structure that does not coincide with
cell identity.

The held-out phi is read from <phi_cache.npz> if that file already holds this task (A5 writes one
while doing its own E-steps, so reuse it rather than paying twice); otherwise it is computed here and
written to that path. Give each task its own cache file when running as an array -- concurrent writes
to one npz would race.

Usage: A7_program_celltype_map.py <model_dir> <task_id> <train_h5ad> <test_h5mu> <phi_cache.npz>
                                  <out_prefix> [n_test_cells]
"""
from __future__ import annotations
import sys, json, os, gzip, pickle
import numpy as np, pandas as pd, h5py, scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")

model_dir, task_id, train_h5ad, test_h5mu, phi_cache, out_prefix = sys.argv[1:7]
n_test = int(sys.argv[7]) if len(sys.argv) > 7 else 12000


def celltypes(path, key="broad_cell_type"):
    with h5py.File(path, "r") as f:
        s = f["mod/splicing"] if "mod/splicing" in f else f
        g = s[f"obs/{key}"]
        if isinstance(g, h5py.Group):
            cats = [c.decode() if isinstance(c, bytes) else str(c) for c in np.array(g["categories"])]
            return np.array(cats, dtype=object)[np.array(g["codes"])], s["obs/_index"].shape[0]
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in np.array(g)]), \
               s["obs/_index"].shape[0]


# training phi comes straight out of the fit; test phi from the frozen-dictionary E-step
d = np.load(f"{model_dir}/mb_run_{task_id}/fit.npz")
r = json.load(open(f"{model_dir}/mb_run_{task_id}/result.json"))
phi_tr, K = np.asarray(d["phi"]), int(r["K"])
ct_tr, n_tr = celltypes(train_h5ad)
assert phi_tr.shape[0] == n_tr, f"train phi {phi_tr.shape[0]} vs {n_tr} cells"

ct_te_all, n_te_total = celltypes(test_h5mu)
# same rng and n_test as A5, so a cached phi corresponds to exactly these cells
idx = np.sort(np.random.default_rng(0).choice(n_te_total, size=min(n_test, n_te_total), replace=False))
ct_te = ct_te_all[idx]


def read_sparse_layer(grp):
    return sp.csr_matrix((np.array(grp["data"]), np.array(grp["indices"]), np.array(grp["indptr"])),
                         shape=tuple(grp.attrs["shape"]))


def heldout_phi(K):
    """Frozen-globals E-step for the held-out cells: only phi is inferred, everything else is fixed.

    Chunked at 4,096 cells because the E-step is full-batch over the cells it is given and the
    Beta-Binomial backward needs several temporaries of that size. Cells are conditionally independent
    given the frozen globals, so chunking is exact rather than an approximation.
    """
    from estep_minibatch import infer_phi_minibatch
    import anndata as ad
    with h5py.File(test_h5mu, "r") as f:
        sm = f["mod/splicing"] if "mod/splicing" in f else f
        cl = read_sparse_layer(sm["layers/cell_by_cluster_matrix_original"]).tocsr().astype(np.float32)
        jr = np.array(sm["layers/junc_ratio_original"], dtype=np.float32)
        # there is no cell_by_junction_matrix_original layer; reconstruct it the way A4/A5 do
        jc = sp.csr_matrix(np.rint(np.nan_to_num(jr) * cl.toarray()).astype(np.float32))
    jc, cl = jc[idx], cl[idx]
    with gzip.open(f"{model_dir}/eval_grouped/task_{task_id}/model/leafletfa_model.pkl.gz", "rb") as fh:
        m = pickle.load(fh)
    bb = m.get("bb_conc")
    out, CH = [], 4096
    for a0 in range(0, jc.shape[0], CH):
        a1 = min(a0 + CH, jc.shape[0])
        ck = ad.AnnData(sp.csr_matrix((a1 - a0, jc.shape[1]), dtype=np.float32))
        ck.layers["cell_by_junction_matrix"] = jc[a0:a1]
        ck.layers["cell_by_cluster_matrix"] = cl[a0:a1]
        p, _ = infer_phi_minibatch(ck, np.asarray(m["psi"], np.float32), K,
                                   dir_conc=m.get("dir_conc"), pi=m.get("pi"),
                                   alpha_pi=m.get("alpha_pi"),
                                   bb_conc=(None if bb in (None, "infinity") else bb),
                                   junc_prior=True, num_epochs=500)
        out.append(p)
        print(f"    E-step {a1:,}/{jc.shape[0]:,} cells", flush=True)
    return np.vstack(out)


cache = dict(np.load(phi_cache)) if os.path.exists(phi_cache) else {}
key = f"phi_{task_id}"
if key in cache:
    phi_te = cache[key]
    print(f"  held-out phi from cache {phi_cache}", flush=True)
else:
    print(f"  no cached phi for task {task_id} -- running the frozen-globals E-step", flush=True)
    phi_te = heldout_phi(K)
    cache[key] = phi_te
    np.savez_compressed(phi_cache, **cache)
assert phi_te.shape[0] == len(ct_te), f"test phi {phi_te.shape[0]} vs {len(ct_te)} labels"
assert phi_te.shape[1] == K, f"test phi has {phi_te.shape[1]} programs, model has K={K}"

# only cell types represented in both, with enough cells to average over
shared = [t for t in pd.Series(ct_tr).value_counts().index
          if (ct_tr == t).sum() >= 50 and (ct_te == t).sum() >= 50]
print(f"K={K} (task {task_id}, seed {r['seed']}): {phi_tr.shape[0]:,} train / {phi_te.shape[0]:,} test "
      f"cells, {len(shared)} shared cell types", flush=True)

M_tr = np.vstack([phi_tr[ct_tr == t].mean(0) for t in shared])
M_te = np.vstack([phi_te[ct_te == t].mean(0) for t in shared])
cols = [f"P{i+1}" for i in range(K)]
pd.DataFrame(M_tr, index=shared, columns=cols).to_csv(f"{out_prefix}_train.csv")
pd.DataFrame(M_te, index=shared, columns=cols).to_csv(f"{out_prefix}_test.csv")

print(f"\n  overall train-vs-test map correlation: r = {np.corrcoef(M_tr.ravel(), M_te.ravel())[0,1]:.3f}",
      flush=True)
per_prog = [np.corrcoef(M_tr[:, k], M_te[:, k])[0, 1] for k in range(K)]
print(f"  per-program (across cell types): median r = {np.nanmedian(per_prog):.3f}, "
      f"{int(np.nansum(np.array(per_prog) > 0.9))}/{K} above 0.9", flush=True)
per_type = [np.corrcoef(M_tr[i], M_te[i])[0, 1] for i in range(len(shared))]
print(f"  per-cell-type (across programs): median r = {np.nanmedian(per_type):.3f}, "
      f"{int(np.nansum(np.array(per_type) > 0.9))}/{len(shared)} above 0.9", flush=True)
# does each cell type's top program agree between train and held-out?
agree = np.mean(M_tr.argmax(1) == M_te.argmax(1))
print(f"  cell types whose TOP program is the same in train and held-out: {agree:.1%}", flush=True)

np.savez_compressed(f"{out_prefix}.npz", types=np.array(shared, dtype=object),
                    train=M_tr, test=M_te, K=K)
print(f"\nwrote {out_prefix}_{{train,test}}.csv and {out_prefix}.npz", flush=True)
