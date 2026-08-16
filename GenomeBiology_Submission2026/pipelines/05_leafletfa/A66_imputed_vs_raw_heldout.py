#!/usr/bin/env python
"""A66: how far is imputed PSI from raw PSI, on HELD-OUT cells, at full and reduced coverage.

The sanity check behind both the imputation evaluation and the DS test. Two imputed quantities per
(cell, junction):
  pure prior        rho_cj = phi_c . psi_:,j                       (no reads from that cell-event)
  DM posterior mean (gamma_e rho_cj + y_cj) / (gamma_e + N_ce)     (what one would use in practice:
                                                                    the model's prior updated by
                                                                    the cell's own reads)
against the observed junction ratio at FULL coverage (junc_ratio_original) for entries covered
there. Reads are removed at 10/50/90%; phi is re-inferred by the E-step at each level with psi,
pi and gamma frozen from the 70/30 fit, so the cells were never seen in training.

Writes one npz per level with the covered entries (cell, junction, N_orig, N_level, obs_full,
pure, post) plus phi; plotting is separate (plot_A66.py) so it iterates without the E-step.
"""
import argparse, os, sys, time, glob
import numpy as np, h5py, scipy.sparse as sp, anndata as ad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from estep_minibatch import infer_phi_minibatch

M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"


def cat(grp):
    c = np.array([x.decode() if isinstance(x, bytes) else x for x in grp["categories"][:]])
    return c[grp["codes"][:]]


def rows_csr(g, rows, J):
    ip = g["indptr"][:]; d, ix, pt = [], [], [0]
    for r in rows:
        s, e = int(ip[r]), int(ip[r + 1]); d.append(g["data"][s:e]); ix.append(g["indices"][s:e]); pt.append(pt[-1] + e - s)
    return sp.csr_matrix((np.concatenate(d).astype(np.float32), np.concatenate(ix), np.array(pt)), shape=(len(rows), J))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default=f"{M}/ATLAS_DM/runs_heldout70/dm_simplex_K20_seed0_b8192/fit.npz")
    ap.add_argument("--h5mu-dir", default=f"{M}/IMPUTATION_DM/resampled_combined")
    ap.add_argument("--train-h5ad", default=f"{M}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad")
    ap.add_argument("--levels", default="0,10,50,90", help="percent of reads DROPPED; 0 = original counts")
    ap.add_argument("--n-cells", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=f"{M}/IMPUTATION_DM/a66_imputed_vs_raw")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    z = np.load(args.fit, allow_pickle=True)
    K = int(z["K"]); psi = np.asarray(z["psi_mean"], np.float32); conc = np.asarray(z["bb_conc_vec"], np.float32)
    pi = np.asarray(z["trace_pi"])[-1]
    with h5py.File(args.train_h5ad, "r") as f:
        ev = cat(f["var"]["event_id"])
    J = len(ev); assert psi.shape[1] == J
    uev, inv = np.unique(ev, return_inverse=True)
    # conc is per fit-event in fit order (event_id in z); map onto uev order
    fev = np.asarray(z["event_id"]).astype(str); pos = {e: i for i, e in enumerate(fev)}
    conc_j = np.array([conc[pos[e]] for e in ev], np.float32)          # per junction, its event's gamma

    files = {int(os.path.basename(p).split("_")[1]): p for p in glob.glob(f"{args.h5mu_dir}/RESAMPLED_*_PERCENT_test_*.h5mu")}
    ref = files[min(files)]
    with h5py.File(ref, "r") as f:
        s = f["mod/splicing"]; N = s["obs/_index"].shape[0]
        bct = cat(s["obs/broad_cell_type"]); ds = cat(s["obs/dataset"])
    rng = np.random.default_rng(args.seed)
    # stratified: up to n_cells / n_types per broad type, top up at random
    types, cnt = np.unique(bct, return_counts=True)
    per = max(5, args.n_cells // len(types)); pick = []
    for t in types:
        idx = np.flatnonzero(bct == t); pick.extend(rng.choice(idx, min(per, len(idx)), replace=False))
    rest = np.setdiff1d(np.arange(N), pick)
    if len(pick) < args.n_cells:
        pick.extend(rng.choice(rest, args.n_cells - len(pick), replace=False))
    cells = np.sort(np.array(pick)); print(f"{len(cells)} held-out cells, {len(types)} broad types", flush=True)
    np.save(f"{args.out}/cells.npy", cells); np.save(f"{args.out}/cells_bct.npy", bct[cells]); np.save(f"{args.out}/cells_dataset.npy", ds[cells])

    # full-coverage truth from the reference file (identical *_original layers in every file)
    with h5py.File(ref, "r") as f:
        s = f["mod/splicing"]
        Corig = rows_csr(s["layers/cell_by_cluster_matrix_original"], cells, J)
        jr = np.asarray(s["layers/junc_ratio_original"][cells, :], np.float32)       # dense rows
    Cd = np.asarray(Corig.todense()); obs_full = np.where(Cd > 0, jr, np.nan)
    Yorig = np.rint(np.nan_to_num(jr) * Cd).astype(np.float32)
    covered = Cd > 0
    print(f"covered entries at full coverage: {covered.sum():,}  (median reads {np.median(Cd[covered]):.0f})", flush=True)

    for lev in [int(x) for x in args.levels.split(",")]:
        t0 = time.time()
        if lev == 0:
            Ym, Cm = sp.csr_matrix(Yorig), sp.csr_matrix(Cd)
        else:
            with h5py.File(files[lev], "r") as f:
                s = f["mod/splicing"]
                Ym = rows_csr(s["layers/cell_by_junction_matrix"], cells, J)
                Cm = rows_csr(s["layers/cell_by_cluster_matrix"], cells, J)
        a = ad.AnnData(X=Ym.copy()); a.layers["cell_by_junction_matrix"] = Ym; a.layers["cell_by_cluster_matrix"] = Cm
        phi, info = infer_phi_minibatch(a, psi, K, dir_conc=float(K), pi=pi, bb_conc=conc, num_epochs=args.epochs,
                                        device=args.device, likelihood="dirichletmultinomial", event_id=ev, conc_min=2.0)
        pure = (phi @ psi).astype(np.float32)
        Yd = np.asarray(Ym.todense()); Nd = np.asarray(Cm.todense())
        post = (conc_j[None, :] * pure + Yd) / (conc_j[None, :] + Nd)
        r, c = np.nonzero(covered)
        np.savez_compressed(f"{args.out}/level_{lev:03d}.npz", cell=r.astype(np.int32), junc=c.astype(np.int32),
                            N_orig=Cd[r, c].astype(np.float32), N_level=Nd[r, c].astype(np.float32),
                            obs_full=obs_full[r, c], pure=pure[r, c], post=post[r, c].astype(np.float32),
                            phi=phi.astype(np.float32), level=lev)
        keep = Cd[r, c] >= 5
        pr = np.corrcoef(obs_full[r, c][keep], pure[r, c][keep])[0, 1]; po = np.corrcoef(obs_full[r, c][keep], post[r, c][keep])[0, 1]
        print(f"level {lev:3d}% dropped: E-step {time.time()-t0:.0f}s {info}; entries {len(r):,}; on N_orig>=5: "
              f"pearson pure {pr:.3f}  posterior-mean {po:.3f}; frac entries with 0 reads at this level "
              f"{np.mean(Nd[r, c] == 0):.2f}", flush=True)


if __name__ == "__main__":
    main()
