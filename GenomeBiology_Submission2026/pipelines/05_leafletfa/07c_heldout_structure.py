#!/usr/bin/env python
"""
Step 07c: HELD-OUT latent-space structure of phi (UNSUPERVISED).

WHAT CHANGED AND WHY
--------------------
Earlier 07c decoded broad_cell_type from phi and used the accuracy as a quality metric. That is the
wrong test: cell-type labels are defined from GENE EXPRESSION, so phi failing to reproduce them is
NOT necessarily a weak latent space -- splicing may carve cells differently than expression does,
which is the interesting result, and supervised decoding penalises exactly that case.

So this version measures the STRUCTURE of the held-out phi directly, without treating cell type as
ground truth:

  1. Leiden clustering of phi        -- clusters from SPLICING ALONE, no labels used
  2. silhouette of phi BY CELL TYPE  -- do GE-defined cell types separate in splicing space?
                                        (low is EXPECTED and interesting, not a failure)
  3. ARI(Leiden, cell type)          -- does unsupervised splicing structure recover cell types?
  4. per-cluster purity + contingency-- which Leiden clusters map to which cell types

The CONTRAST between (2/internal separability) and (3/cell-type alignment) is the point:
high internal structure + only partial cell-type alignment => phi captures real, splicing-specific
structure that is not merely cell identity.

phi is the frozen-psi test-time E-step on the held-out 30% cells (same machinery as before), so it
reflects the representation the released model gives on cells it never trained on. The SAME analysis
is run on training phi for the memorization contrast.

NOTE: aging is NOT here. It belongs on the full-data model (Figure 4 case study, Tabula Muris Senis),
not on a 70%/K-sweep held-out set.

Usage:
    python 07c_heldout_structure.py <run_dir> <train_h5ad> <test_h5mu> <output_csv>
        [chunk=4096] [epochs=300] [patience=10]
"""

import sys, os, pickle, gzip, h5py
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.sparse import csr_matrix
from sklearn.metrics import silhouette_score, adjusted_rand_score

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.set_default_dtype(torch.float32)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"[07c] GPU: {torch.cuda.get_device_name(0)}", flush=True)
else:
    torch.set_num_threads(4); print("[07c] CPU", flush=True)
from leafletfa import LeafletFA

_MISSING = {"", "nan", "na", "none", "unknown", "unassigned", "NA"}


# ── model loading + frozen-psi E-step (unchanged core) ────────────────────────
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

# ── unsupervised structure of phi ─────────────────────────────────────────────

def _clean(labels, key):
    """Return (values, mask) with missing labels dropped."""
    v = np.asarray(labels[key]).astype(str)
    mask = np.array([x.strip().lower() not in _MISSING for x in v])
    return v, mask


def structure_metrics(phi, labels, phi_source, base, seed=0):
    """Leiden on phi (no labels), silhouette by cell type, ARI vs cell type, per-cluster purity."""
    rows = []
    K = phi.shape[1]
    # scanpy pipeline on the phi embedding
    a = ad.AnnData(np.asarray(phi, dtype=np.float32))
    a.obsm["X_phi"] = a.X.copy()
    sc.pp.neighbors(a, use_rep="X_phi", n_neighbors=15, random_state=seed)
    sc.tl.leiden(a, resolution=1.0, random_state=seed, flavor="igraph", n_iterations=2, directed=False)
    leiden = a.obs["leiden"].astype(str).values
    n_leiden = len(np.unique(leiden))

    # effective K used per cell (perplexity) -- context for how many programs are active
    p = phi / (phi.sum(1, keepdims=True) + 1e-12)
    perp = float(np.median(np.exp(-(p * np.log(p + 1e-12)).sum(1))))

    rows.append({**base, "phi_source": phi_source, "metric": "n_leiden_clusters",
                 "value": n_leiden, "n_cells": len(leiden)})
    rows.append({**base, "phi_source": phi_source, "metric": "phi_perplexity_median",
                 "value": perp, "n_cells": len(leiden)})

    # INTRINSIC separability of the Leiden clustering (mildly circular -- report but don't lead)
    if n_leiden > 1:
        rows.append({**base, "phi_source": phi_source, "metric": "silhouette_by_leiden",
                     "value": float(silhouette_score(phi, leiden)), "n_cells": len(leiden)})

    for key in ("broad_cell_type", "tissue_cell_type"):
        if key not in labels:
            continue
        v, mask = _clean(labels, key)
        if mask.sum() < 50 or len(np.unique(v[mask])) < 2:
            continue
        # silhouette of phi BY CELL TYPE: does GE-biology separate in splicing space?
        rows.append({**base, "phi_source": phi_source, "metric": f"silhouette_by_{key}",
                     "value": float(silhouette_score(phi[mask], v[mask])),
                     "n_classes": len(np.unique(v[mask])), "n_cells": int(mask.sum())})
        # ARI: does UNSUPERVISED splicing structure recover the cell types?
        rows.append({**base, "phi_source": phi_source, "metric": f"ari_leiden_vs_{key}",
                     "value": float(adjusted_rand_score(v[mask], leiden[mask])),
                     "n_classes": len(np.unique(v[mask])), "n_cells": int(mask.sum())})
        # per-Leiden-cluster purity: max cell-type fraction within each cluster, cell-weighted mean
        if key == "broad_cell_type":
            pur = []
            for cl in np.unique(leiden[mask]):
                sub = v[mask][leiden[mask] == cl]
                pur.append((len(sub), pd.Series(sub).value_counts(normalize=True).iloc[0]))
            wpur = float(np.average([p for _, p in pur], weights=[n for n, _ in pur]))
            rows.append({**base, "phi_source": phi_source, "metric": "leiden_purity_broad_ct",
                         "value": wpur, "n_classes": n_leiden, "n_cells": int(mask.sum())})
    return rows, leiden


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    run_dir, train_h5ad, test_h5mu, out_csv = sys.argv[1:5]
    chunk    = int(sys.argv[5]) if len(sys.argv) > 5 else 4096
    epochs   = int(sys.argv[6]) if len(sys.argv) > 6 else 300
    patience = int(sys.argv[7]) if len(sys.argv) > 7 else 10
    rng = np.random.default_rng(0)

    m = load_model(run_dir)
    K = int(np.asarray(m["psi"]).shape[0])
    psi = torch.tensor(np.asarray(m["psi"], dtype=np.float32), device=DEVICE)
    dir_conc = m.get("dir_conc"); junc_prior = bool(m.get("junc_specific_prior", True))
    pi = np.asarray(m["pi"], float) if m.get("pi") is not None else None
    alpha_pi = np.asarray(m["alpha_pi"], float) if m.get("alpha_pi") is not None else None
    bb_conc = m.get("bb_conc")
    base = {"run_dir": os.path.basename(run_dir.rstrip("/")), "K": K,
            "junc_prior": junc_prior,
            "bb_conc": (np.nan if isinstance(bb_conc, str) else float(np.asarray(bb_conc)))}
    print(f"[07c] {base['run_dir']}  K={K}  bb_conc={base['bb_conc']}", flush=True)

    all_rows = []

    # HELD-OUT: frozen-psi E-step on the 30% test cells, then unsupervised structure
    phi_test, labels = infer_phi_fullcov(test_h5mu, psi, K, dir_conc, junc_prior,
                                         pi, alpha_pi, bb_conc, chunk, epochs, patience, rng)
    print(f"[07c] held-out phi {phi_test.shape}; running Leiden + silhouette", flush=True)
    rows, _ = structure_metrics(phi_test, labels, "heldout_inferred", base, seed=0)
    all_rows += rows

    # TRAINING phi (from the saved model) for the memorization contrast, if labels are available
    tr_phi = np.asarray(m.get("assign_post")) if m.get("assign_post") is not None else None
    if tr_phi is not None and os.path.exists(train_h5ad):
        try:
            ta = ad.read_h5ad(train_h5ad, backed="r")
            tl = {k: ta.obs[k].astype(str).values for k in
                  ("broad_cell_type", "tissue_cell_type") if k in ta.obs}
            if tl and tr_phi.shape[0] == ta.n_obs:
                print(f"[07c] training phi {tr_phi.shape}; structure for memorization contrast",
                      flush=True)
                rows, _ = structure_metrics(tr_phi, tl, "train_assign_post", base, seed=0)
                all_rows += rows
        except Exception as e:
            print(f"[07c] training-phi structure skipped: {e}", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)

    # readable summary + the memorization contrast
    print("\n=== HELD-OUT phi structure ===", flush=True)
    show = df[df.phi_source == "heldout_inferred"][["metric", "value", "n_classes", "n_cells"]]
    print(show.to_string(index=False), flush=True)
    piv = df.pivot_table(index="metric", columns="phi_source", values="value", aggfunc="first")
    for key in ("silhouette_by_broad_cell_type", "ari_leiden_vs_broad_cell_type"):
        if key in piv.index and {"heldout_inferred", "train_assign_post"} <= set(piv.columns):
            h, t = piv.loc[key, "heldout_inferred"], piv.loc[key, "train_assign_post"]
            print(f"\n  {key}:  held-out {h:.3f}  vs  train {t:.3f}  (gap {t-h:+.3f})", flush=True)
    print(f"\nwrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
