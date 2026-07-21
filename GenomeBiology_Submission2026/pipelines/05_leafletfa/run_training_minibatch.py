"""Array trainer for the 70% atlas split using TRUE minibatch SVI (leafletfa/minibatch.py).

Replaces the chunked run_training.py, which rebuilt model+guide per chunk and left psi mislocated
(mean 0.67 vs an observed rate of 0.295 -- see LeafletFA/METHODS_MINIBATCH.md). This runner keeps
one model, one guide and one optimiser alive for the whole run with a real pyro.plate over cells, so
the ELBO is a single continuous curve and psi lands at the data.

It reads the SAME parameter_combinations.json the chunked array used, so task ids line up, but only
the knobs that still apply under minibatch are honoured:
    K, seed, junc_specific_prior
The chunked-only knobs (num_passes, num_epochs_first/later, metacell_size, batch_size=8192, ...) are
ignored; minibatch uses num_steps / subsample_size / lr / gamma passed on the command line.

Initialisation is random (AutoNormal default). Init-independence was verified on this model: the
waypoint and random arms converge to the same psi (0.6927 vs 0.6931 under the old code, and identical
recovery under the plate model), because a persistent guide accumulates evidence regardless of where
it starts. The `arm`/`waypoints_use` grid column is retained in the output for bookkeeping only.

Usage:
    run_training_minibatch.py <task_id> <model_dir> <train_h5ad>
        [num_steps=1500] [subsample_size=2048] [lr=0.05] [gamma=0.1] [learn_ab_hyperprior=0]

Writes into <model_dir>/mb_run_<task_id>/: fit.npz, result.json, loss_curve.png.
"""
import sys, os, json, time
import numpy as np
import h5py
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")


def load_counts(train_h5ad, n_cells=0, n_junc=0):
    """junc + clus for the 70% training matrix, coverage from clus>0 (never junc>0).

    n_cells / n_junc > 0 take a subset (for CPU smoke tests): the first n_cells cells and the
    best-covered n_junc junctions, so the subset is informative rather than arbitrary.
    """
    with h5py.File(train_h5ad, "r") as h:
        def rows(g, n):
            ip = g["indptr"][:n + 1] if n else g["indptr"][:]
            a, b = int(ip[0]), int(ip[-1])
            return sp.csr_matrix((g["data"][a:b], g["indices"][a:b], ip - ip[0]),
                                 shape=((n if n else len(ip) - 1),
                                        int(g.attrs["shape"][1]))).toarray().astype(np.float32)
        junc = rows(h["layers/cell_by_junction_matrix"], n_cells)
        clus = rows(h["layers/cell_by_cluster_matrix"], n_cells)
    if n_junc and n_junc < junc.shape[1]:
        keep = np.argsort(-(clus > 0).sum(0))[:n_junc]; keep.sort()
        junc, clus = junc[:, keep], clus[:, keep]
    return junc, clus


def load_cell_labels(train_h5ad, n_cells=0):
    """broad_cell_type per cell, for colouring the phi UMAP. Returns None if unavailable."""
    try:
        with h5py.File(train_h5ad, "r") as h:
            col = next((c for c in ("broad_cell_type", "cell_type", "cell_ontology_class")
                        if c in h["obs"]), None)
            if col is None:
                return None
            g = h[f"obs/{col}"]
            if isinstance(g, h5py.Group):        # categorical: codes + categories
                cats = [c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]]
                codes = g["codes"][:] if not n_cells else g["codes"][:n_cells]
                return np.array([cats[c] if c >= 0 else "NA" for c in codes])
            arr = g[:] if not n_cells else g[:n_cells]
            return np.array([x.decode() if isinstance(x, bytes) else x for x in arr])
    except Exception:
        return None


def main():
    task_id = int(sys.argv[1])
    model_dir = sys.argv[2]
    train_h5ad = sys.argv[3]
    num_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
    batch = int(sys.argv[5]) if len(sys.argv) > 5 else 2048
    lr = float(sys.argv[6]) if len(sys.argv) > 6 else 0.05
    gamma = float(sys.argv[7]) if len(sys.argv) > 7 else 0.1
    learn_ab = bool(int(sys.argv[8])) if len(sys.argv) > 8 else False
    n_cells  = int(sys.argv[9]) if len(sys.argv) > 9 else 0    # subset: smoke only
    n_junc   = int(sys.argv[10]) if len(sys.argv) > 10 else 0

    params = json.load(open(os.path.join(model_dir, "parameter_combinations.json")))[task_id]
    K = int(params["K"])
    seed = int(params.get("seed", 0))
    junc_prior = bool(params.get("junc_specific_prior", False))
    out_dir = os.path.join(model_dir, f"mb_run_{task_id}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== minibatch task {task_id}: K={K} seed={seed} junc_prior={junc_prior} "
          f"arm={params.get('arm')} ===", flush=True)

    try:
        import wandb
        wandb.init(project="LeafletFA-mouse-atlas-70pct-train", group="minibatch_rerun",
                   name=f"mb_run_{task_id}", config={**params, "trainer": "minibatch",
                   "num_steps": num_steps, "subsample_size": batch, "lr": lr, "gamma": gamma,
                   "learn_ab_hyperprior": learn_ab},
                   tags=[f"K{K}", f"seed{seed}",
                         ("juncprior" if junc_prior else "globalprior"), "minibatch"],
                   dir=out_dir, reinit=True)
        _wandb = True
    except Exception as e:
        print(f"  wandb off: {e}", flush=True); _wandb = False

    junc, clus = load_counts(train_h5ad, n_cells, n_junc)
    cov = clus > 0
    obs_rate = float(junc.sum() / max(clus.sum(), 1))
    print(f"  data {junc.shape}  coverage={cov.mean()*100:.2f}%  observed_rate={obs_rate:.4f}",
          flush=True)

    from leafletfa.minibatch import LeafletFAMinibatch
    m = LeafletFAMinibatch(junc, clus, K=K, junc_specific_prior=junc_prior,
                           learn_ab_hyperprior=learn_ab)
    t0 = time.time()
    losses = m.fit(num_steps=num_steps, subsample_size=batch, lr=lr, gamma=gamma, seed=seed,
                   log_every=50)
    dt = time.time() - t0

    psi = m.psi_mean().cpu().numpy()
    phi = m.phi_mean().cpu().numpy()
    p = np.clip(phi, 1e-12, None)
    perp = float(np.median(np.exp(-(p * np.log(p)).sum(1))))
    g = m.globals_()
    res = {
        "task_id": task_id, "K": K, "seed": seed, "junc_prior": junc_prior,
        "arm": params.get("arm"), "trainer": "minibatch",
        "num_steps": num_steps, "subsample_size": batch, "lr": lr, "gamma": gamma,
        "n_cells": int(junc.shape[0]), "n_junctions": int(junc.shape[1]),
        "observed_rate": obs_rate,
        "psi_mean": float(psi.mean()), "psi_median": float(np.median(psi)),
        "psi_location_err": float(abs(psi.mean() - obs_rate)),
        "psi_frac_below_005": float((psi < 0.05).mean()),
        "psi_frac_above_095": float((psi > 0.95).mean()),
        "psi_contrast": float(psi.std(axis=0).mean()),
        "pred_mean": float((phi @ psi).mean()),
        "phi_perplexity": perp, "K_max_perplexity": K,
        "final_loss": float(losses[-1]) if np.isfinite(losses[-1]) else None,
        "n_finite_steps": int(np.isfinite(losses).sum()), "total_steps": len(losses),
        "seconds": round(dt, 1),
        "bb_conc": float(np.mean(g.get("bb_conc", np.nan))),
        "dir_conc": float(np.mean(g.get("dir_conc", np.nan))),
    }
    np.savez_compressed(os.path.join(out_dir, "fit.npz"), psi=psi, phi=phi,
                        losses=np.asarray(losses), **{k: v for k, v in g.items()})
    json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=2)

    # THE loss curve -- now a single continuous ELBO, not 96 stitched per-chunk segments.
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    L = np.asarray(losses, dtype=float)
    ax[0].plot(L); ax[0].set_xlabel("SVI step"); ax[0].set_ylabel("-ELBO"); ax[0].set_yscale("log")
    ax[0].set_title(f"loss (task {task_id}, K={K})")
    # psi location is the headline diagnostic; annotate the target
    ax[1].axhline(obs_rate, ls="--", c="k", lw=1, label=f"observed rate {obs_rate:.3f}")
    ax[1].axhline(res["psi_mean"], c="C1", label=f"psi mean {res['psi_mean']:.3f}")
    ax[1].set_ylim(0, 1); ax[1].legend(); ax[1].set_title("psi location vs data")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=110)

    # PHI UMAP coloured by cell type -- brought back from the chunked runner. Purely diagnostic:
    # does the splicing latent space carry cell-type structure? Wrapped so a UMAP failure can NEVER
    # kill a training run (the model is already saved above).
    umap_png = None
    try:
        import scanpy as sc, anndata as ad
        labels = load_cell_labels(train_h5ad, n_cells)
        a = ad.AnnData(phi.astype(np.float32))
        a.obsm["X_phi"] = a.X.copy()
        sc.pp.neighbors(a, use_rep="X_phi", n_neighbors=30, random_state=42)
        sc.tl.umap(a, random_state=42)
        um = a.obsm["X_umap"]
        figu, axu = plt.subplots(figsize=(7, 7))
        if labels is not None and len(labels) == phi.shape[0]:
            uniq = sorted(set(labels))
            cmap = plt.colormaps.get_cmap("tab20" if len(uniq) > 10 else "tab10")
            for i, ct in enumerate(uniq):
                mk = labels == ct
                axu.scatter(um[mk, 0], um[mk, 1], s=3, alpha=0.4,
                            color=cmap(i / max(len(uniq) - 1, 1)), label=str(ct))
            axu.legend(markerscale=3, fontsize=5, ncol=2, loc="best")
        else:
            axu.scatter(um[:, 0], um[:, 1], s=3, alpha=0.4)
        axu.set_xlabel("UMAP 1"); axu.set_ylabel("UMAP 2")
        axu.set_title(f"phi UMAP | task {task_id}, K={K}, {phi.shape[0]:,} cells")
        figu.tight_layout()
        umap_png = os.path.join(out_dir, "phi_umap.png")
        figu.savefig(umap_png, dpi=110)
        print(f"  phi UMAP -> {umap_png}", flush=True)
    except Exception as e:
        print(f"  phi UMAP skipped: {e}", flush=True)

    print("\n================ RESULT ================", flush=True)
    for k in ("observed_rate", "psi_mean", "psi_location_err", "pred_mean", "psi_contrast",
              "phi_perplexity", "n_finite_steps", "total_steps", "seconds"):
        print(f"  {k:<20} {res[k]}", flush=True)
    verdict = ("FIXED" if res["psi_location_err"] < 0.05 else
               "MISLOCATED" if res["psi_location_err"] > 0.15 else "PARTIAL")
    print(f"  VERDICT: {verdict}   (chunked models: psi 0.67 vs rate 0.295)", flush=True)

    if _wandb:
        import wandb
        # the whole loss curve as a wandb line series, plus final health scalars
        for i, l in enumerate(L):
            if i % 10 == 0 and np.isfinite(l):
                wandb.log({"neg_elbo": float(l)}, step=i)
        wandb.log({k: v for k, v in res.items() if isinstance(v, (int, float))})
        wandb.log({"loss_curve": wandb.Image(os.path.join(out_dir, "loss_curve.png"))})
        if umap_png:
            wandb.log({"phi_umap": wandb.Image(umap_png)})
        wandb.finish()
    print(f"\nwrote {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
