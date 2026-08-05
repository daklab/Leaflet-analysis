#!/usr/bin/env python
"""A44: train DM + simplex on the full mouse atlas (132,706 cells x 79,768 junctions).

Why a new trainer rather than A1_train_atlas.py
-----------------------------------------------
A1 trains the RELEASED model (BetaBinomial + independent-Beta psi) and reads its grid from
parameter_combinations.json. It has no way to pass `event_id`, which the DM requires -- without
the ATSE index there is no way to group junctions into events, and the DM is nothing but that
grouping. Bolting a likelihood switch onto A1 would put the released model and an unvalidated one
behind the same entry point, and a released-model rerun could silently pick up the new arm. Two
scripts, two models, no way to run the wrong one by accident.

What it does
------------
Loads the full training matrix densely (42 GB per layer; `data_on_cpu` in LeafletFAMinibatch
auto-engages past 4 GB so the matrices stay in host RAM and only the minibatch crosses to the GPU),
fits DM + simplex, and saves psi/phi/loc/scale plus the loss curve.

Safety for an unattended overnight run
--------------------------------------
Every atlas fit attempted so far has hit a non-finite loss (simplex at steps 1161/1433/1663,
baseline at 2113/2249). `fit()` already carries a NaN-guard: it snapshots the lowest-EMA parameter
state on a fixed interval and, on a non-finite loss, restores that snapshot and stops. So a NaN
costs the steps since the last checkpoint, not the run -- which is what makes launching these
unattended defensible rather than reckless. `stopped_step` is recorded in result.json so a fit that
died early is visible in the summary rather than being read as a converged one.

Usage:
    A44_train_atlas_dm.py --K 20 --seed 0 --steps 12000 --batch 8192 --out <dir>
"""

import argparse, json, os, sys, time

import h5py
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import pyro   # noqa: E402

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from leafletfa.minibatch import LeafletFAMinibatch          # noqa: E402
from leafletfa.dirichlet_multinomial import build_atse_index  # noqa: E402

WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
H5 = f"{WD}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad"


def cat(name, which, h5):
    with h5py.File(h5, "r") as f:
        n = f[which][name]
        if isinstance(n, h5py.Group):
            c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
            return c[n["codes"][:]]
        return np.array([x.decode() if isinstance(x, bytes) else x for x in n[:]])


def read_all(layer, h5, n_rows, n_cols, chunk=4000):
    """Dense read in row blocks. 42 GB at atlas shape, so never materialise the CSR whole."""
    out = np.zeros((n_rows, n_cols), dtype=np.float32)
    with h5py.File(h5, "r") as f:
        g = f["layers"][layer]
        ip_all = g["indptr"]
        for a in range(0, n_rows, chunk):
            b = min(a + chunk, n_rows)
            ip = ip_all[a:b + 1]
            s, e = int(ip[0]), int(ip[-1])
            out[a:b] = sp.csr_matrix((g["data"][s:e], g["indices"][s:e], ip - ip[0]),
                                     shape=(b - a, n_cols)).toarray()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--h5", default=H5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", type=int, default=0, help="0 = all; >0 subsets for a smoke test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--psi-logit-prior-sd", type=float, default=1.0)
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--wandb-project", default="LeafletFA-mouse-atlas")
    args = ap.parse_args()

    run = f"dm_simplex_K{args.K}_seed{args.seed}_b{args.batch}"
    out_dir = os.path.join(args.out, run)
    os.makedirs(out_dir, exist_ok=True)

    ev = cat("event_id", "var", args.h5)
    J = len(ev)
    with h5py.File(args.h5, "r") as f:
        N_all = int(f["layers"]["cell_by_junction_matrix"].attrs["shape"][0])
    N = min(args.cells, N_all) if args.cells else N_all
    _, _, sizes = build_atse_index(ev)
    n_atse = len(sizes)
    print(f"[1] {run}: {N:,} cells x {J:,} junctions, {n_atse:,} ATSEs "
          f"(m=2 for {int((sizes == 2).sum()):,})", flush=True)

    wb = None
    try:
        import wandb
        wb = wandb.init(project=os.environ.get("LFA_WANDB_PROJECT", args.wandb_project),
                        group="dm_simplex_atlas", name=run, dir=out_dir, reinit=True,
                        config=dict(K=args.K, seed=args.seed, steps=args.steps, batch=args.batch,
                                    lr=args.lr, gamma=args.gamma, likelihood="dirichletmultinomial",
                                    psi_parameterization="simplex", n_cells=N, n_junctions=J,
                                    n_atse=n_atse, psi_logit_prior_sd=args.psi_logit_prior_sd),
                        tags=[f"K{args.K}", f"seed{args.seed}", "dirichletmultinomial", "simplex"])
        print("[1b] wandb on", flush=True)
    except Exception as e:
        print(f"[1b] wandb off: {e}", flush=True)

    t0 = time.time()
    junc = read_all("cell_by_junction_matrix", args.h5, N, J)
    clus = read_all("cell_by_cluster_matrix", args.h5, N, J)
    print(f"[2] loaded in {time.time()-t0:.0f}s ({junc.nbytes/1e9:.1f} GB each), "
          f"coverage {100*(clus>0).mean():.2f}%, observed rate {junc.sum()/max(clus.sum(),1):.4f}",
          flush=True)

    pyro.clear_param_store()
    t0 = time.time()
    m = LeafletFAMinibatch(junc, clus, K=args.K, device=args.device,
                           likelihood="dirichletmultinomial", psi_parameterization="simplex",
                           psi_logit_prior_sd=args.psi_logit_prior_sd, event_id=ev)
    L = m.fit(num_steps=args.steps, subsample_size=min(args.batch, N), lr=args.lr,
              gamma=args.gamma, seed=args.seed, verbose=True, log_every=100,
              early_stop=not args.no_early_stop, patience=12, min_delta_rel=1e-5)
    train_s = time.time() - t0
    finished = int(getattr(m, "stopped_step", args.steps))
    nan_hit = finished < args.steps and not getattr(m, "stopped_early", False)
    print(f"[3] {finished:,}/{args.steps:,} steps in {train_s/60:.1f} min "
          f"({1000*train_s/max(finished,1):.0f} ms/step)"
          f"{'  -- NaN-GUARD FIRED' if nan_hit else ''}", flush=True)

    psi, phi = m.psi_mean(), m.phi_mean()
    loc, scale = m.psi_loc_scale()
    def to_np(x):
        """psi_loc_scale returns tensors under some paths and arrays under others."""
        if x is None:
            return np.zeros(0, np.float32)
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return np.asarray(x, np.float32)
    try:
        med = m._median()
        conc = float(med["bb_conc"]) if "bb_conc" in med else float("nan")
    except Exception:
        conc = float("nan")

    # within-ATSE sums: under the simplex these are 1 by construction, so a deviation means the
    # constraint is not actually active and every downstream TV is on an arbitrary scale.
    psi, phi = to_np(psi), to_np(phi)
    uq, inv = np.unique(ev, return_inverse=True)
    S = np.zeros((args.K, len(uq)), np.float32)
    np.add.at(S.T, inv, psi.T)
    frac_valid = float(((S >= 0.9) & (S <= 1.1)).mean())
    print(f"[4] psi within-ATSE sum: median {np.median(S):.6f}, max dev "
          f"{np.abs(S-1).max():.2e}, valid {100*frac_valid:.2f}%  |  conc {conc:.3f}", flush=True)

    np.savez_compressed(os.path.join(out_dir, "fit.npz"),
                        psi_mean=psi, phi=phi, psi_loc=to_np(loc), psi_scale=to_np(scale),
                        event_id=np.asarray(uq, dtype=object), inv_ev=inv.astype(np.int32),
                        losses=np.asarray(L, np.float64), bb_conc=conc, frac_valid=frac_valid,
                        K=args.K, seed=args.seed, steps_run=finished)
    res = dict(run=run, K=args.K, seed=args.seed, steps_requested=args.steps,
               steps_run=finished, nan_guard_fired=bool(nan_hit),
               early_stopped=bool(getattr(m, "stopped_early", False)),
               train_minutes=round(train_s / 60, 1), ms_per_step=round(1000*train_s/max(finished,1)),
               final_loss=float(L[-1]) if L else None, bb_conc=conc, frac_valid=frac_valid,
               n_cells=N, n_junctions=J, n_atse=int(n_atse), batch=args.batch, lr=args.lr)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(res, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(L, lw=0.8)
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("-ELBO")
    ax.set_title(f"{run}  ({finished:,} steps{'  NaN-guard' if nan_hit else ''})", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=140)

    if wb is not None:
        import wandb
        for i, l in enumerate(L):
            if i % 25 == 0 and np.isfinite(l):
                wandb.log({"neg_elbo": float(l)}, step=i)
        wandb.log({k: v for k, v in res.items() if isinstance(v, (int, float, bool))})
        wandb.log({"loss_curve": wandb.Image(os.path.join(out_dir, "loss_curve.png"))})
        wandb.finish()
    print(f"[5] wrote {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
