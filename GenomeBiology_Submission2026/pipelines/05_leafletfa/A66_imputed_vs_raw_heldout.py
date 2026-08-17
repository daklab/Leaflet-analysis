#!/usr/bin/env python
"""A66 (v2, ALL held-out cells): how far is imputed PSI from raw PSI, at full and reduced coverage.

All 39,812 held-out cells (30% split, never in training) through the 70/30 fit's E-step at each
coverage level, in chunks. Junction set = A3's HVJ set (top 10,000 by variance, expanded to whole
ATSEs, ~31k junctions) so LeafletFA, NMF and the junction-mean baseline are scored on the same
entries as Figure 2's imputation panel. Entries = covered at full coverage with >= 5 reads.

Three predictors per (cell, junction):
  post   DM posterior mean (gamma_e phi.psi + y)/(gamma_e + N)  -- LeafletFA in practice
  pure   phi.psi                                                 -- LeafletFA prior only
  jmean  the junction's mean observed PSI over covered cells at THIS level -- the trivial baseline
         that gets every junction's level right and nothing else
Sufficient statistics are accumulated per junction (for within-junction r), per cell (within-cell
r), globally (all-entry r), and separately on entries whose reads were all removed; a 1% subsample
of entries is kept for the hexbins. NMF is added by A66b on the same cells/junctions/levels.
"""
import argparse, os, sys, time, glob
import numpy as np, h5py, scipy.sparse as sp, anndata as ad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from estep_minibatch import infer_phi_minibatch
from A3_eval_imputation import compute_hvj_indices

M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"


def cat(grp):
    c = np.array([x.decode() if isinstance(x, bytes) else x for x in grp["categories"][:]]); return c[grp["codes"][:]]


def rows_csr(g, r0, r1, J):
    ip = g["indptr"][:]; a, b = int(ip[r0]), int(ip[r1])
    return sp.csr_matrix((g["data"][a:b].astype(np.float32), g["indices"][a:b], ip[r0:r1 + 1] - a), shape=(r1 - r0, J))


class Acc:
    """per-junction and per-cell sufficient stats for Pearson, plus global."""
    def __init__(self, J, N):
        self.J = np.zeros((J, 6)); self.C = np.zeros((N, 6)); self.G = np.zeros(6)
    def add(self, cell, junc, x, y):
        v = np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], 1)
        np.add.at(self.J, junc, v); np.add.at(self.C, cell, v); self.G += v.sum(0)
    @staticmethod
    def r(S):
        n, sx, sy, sxx, syy, sxy = S.T if S.ndim == 2 else S
        cov = sxy / n - (sx / n) * (sy / n); vx = sxx / n - (sx / n) ** 2; vy = syy / n - (sy / n) ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            return cov / np.sqrt(vx * vy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default=f"{M}/ATLAS_DM/runs_heldout70/dm_simplex_K20_seed0_b8192/fit.npz")
    ap.add_argument("--h5mu-dir", default=f"{M}/IMPUTATION_DM/resampled_combined")
    ap.add_argument("--train-h5ad", default=f"{M}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad")
    ap.add_argument("--levels", default="0,10,50,90")
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-hvj", type=int, default=10000)
    ap.add_argument("--min-reads", type=int, default=5)
    ap.add_argument("--sub", type=float, default=0.01, help="fraction of entries kept for hexbins")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=f"{M}/IMPUTATION_DM/a66_all_cells")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    z = np.load(args.fit, allow_pickle=True)
    K = int(z["K"]); psi = np.asarray(z["psi_mean"], np.float32); conc = np.asarray(z["bb_conc_vec"], np.float32); pi = np.asarray(z["trace_pi"])[-1]
    with h5py.File(args.train_h5ad, "r") as f:
        ev = cat(f["var"]["event_id"])
    J = len(ev); assert psi.shape[1] == J
    fev = np.asarray(z["event_id"]).astype(str); pos = {e: i for i, e in enumerate(fev)}
    conc_j = np.array([conc[pos[e]] for e in ev], np.float32)
    files = {int(os.path.basename(p).split("_")[1]): p for p in glob.glob(f"{args.h5mu_dir}/RESAMPLED_*_PERCENT_test_*.h5mu")}
    ref = files[min(files)]
    hvj = np.sort(np.asarray(compute_hvj_indices(ref, args.n_hvj)))
    print(f"HVJ set: {len(hvj):,} junctions", flush=True)
    np.save(f"{args.out}/hvj.npy", hvj)
    with h5py.File(ref, "r") as f:
        s = f["mod/splicing"]; N = s["obs/_index"].shape[0]; bct = cat(s["obs/broad_cell_type"])
    np.save(f"{args.out}/cells_bct.npy", bct)
    types, tcode = np.unique(bct, return_inverse=True); T = len(types); np.save(f"{args.out}/types.npy", types)
    rng = np.random.default_rng(0)
    for lev in [int(x) for x in args.levels.split(",")]:
        t0 = time.time()
        acc = {k: Acc(len(hvj), N) for k in ("post", "pure", "jmean")}
        acc_lost = {k: Acc(len(hvj), N) for k in ("post", "pure", "jmean")}
        # cell-TYPE level: per (type, junction): n cells, sum obs, sum reads, sum obs*reads (pooled), sum pred per method
        TL = {k: np.zeros((T, len(hvj))) for k in ("n", "obs", "reads", "obs_w", "post", "pure", "jmean")}
        sub = {k: [] for k in ("post", "pure", "jmean", "obs", "cell", "junc", "N_orig", "N_level")}
        # pass 1: junction means at this level (over covered cells)
        jsum = np.zeros(J); jcnt = np.zeros(J)
        with h5py.File(files.get(lev, ref), "r") as f:
            s = f["mod/splicing"]
            for r0 in range(0, N, args.chunk):
                r1 = min(N, r0 + args.chunk)
                if lev == 0:
                    Cm = rows_csr(s["layers/cell_by_cluster_matrix_original"], r0, r1, J).toarray()
                    R = np.nan_to_num(np.asarray(s["layers/junc_ratio_original"][r0:r1, :], np.float32)) * (Cm > 0)
                else:
                    Ym = rows_csr(s["layers/cell_by_junction_matrix"], r0, r1, J).toarray(); Cm = rows_csr(s["layers/cell_by_cluster_matrix"], r0, r1, J).toarray()
                    R = np.where(Cm > 0, Ym / np.maximum(Cm, 1), 0.0)
                jsum += R.sum(0); jcnt += (Cm > 0).sum(0)
        jmean_all = np.where(jcnt > 0, jsum / np.maximum(jcnt, 1), np.nan).astype(np.float32)
        # pass 2: E-step per chunk and accumulate
        with h5py.File(ref, "r") as fr, h5py.File(files.get(lev, ref), "r") as fl:
            sr = fr["mod/splicing"]; sl = fl["mod/splicing"]
            for r0 in range(0, N, args.chunk):
                r1 = min(N, r0 + args.chunk)
                Corig = rows_csr(sr["layers/cell_by_cluster_matrix_original"], r0, r1, J).toarray()
                jr = np.asarray(sr["layers/junc_ratio_original"][r0:r1, :], np.float32)
                if lev == 0:
                    Yd = np.rint(np.nan_to_num(jr) * Corig).astype(np.float32); Nd = Corig
                    Ym, Cm = sp.csr_matrix(Yd), sp.csr_matrix(Nd)
                else:
                    Ym = rows_csr(sl["layers/cell_by_junction_matrix"], r0, r1, J); Cm = rows_csr(sl["layers/cell_by_cluster_matrix"], r0, r1, J)
                    Yd = Ym.toarray(); Nd = Cm.toarray()
                a = ad.AnnData(X=Ym.copy()); a.layers["cell_by_junction_matrix"] = Ym; a.layers["cell_by_cluster_matrix"] = Cm
                phi, _ = infer_phi_minibatch(a, psi, K, dir_conc=float(K), pi=pi, bb_conc=conc, num_epochs=args.epochs,
                                             device=args.device, likelihood="dirichletmultinomial", event_id=ev, conc_min=2.0)
                pure = (phi @ psi[:, hvj]).astype(np.float32)
                Yh, Nh = Yd[:, hvj], Nd[:, hvj]
                post = (conc_j[hvj][None, :] * pure + Yh) / (conc_j[hvj][None, :] + Nh)
                obs = jr[:, hvj]; Co = Corig[:, hvj]
                keep = Co >= args.min_reads
                r, c = np.nonzero(keep)
                x = obs[r, c]; preds = {"post": post[r, c], "pure": pure[r, c], "jmean": jmean_all[hvj][c]}
                okj = np.isfinite(preds["jmean"])
                for k, y in preds.items():
                    m = okj if k == "jmean" else np.ones(len(x), bool)
                    acc[k].add(r0 + r[m], c[m], x[m], y[m])
                lost = Nh[r, c] == 0
                if lost.any():
                    for k, y in preds.items():
                        m = lost & (okj if k == "jmean" else True)
                        acc_lost[k].add(r0 + r[m], c[m], x[m], y[m])
                tt = tcode[r0 + r]; w = Co[r, c]
                np.add.at(TL["n"], (tt, c), 1); np.add.at(TL["obs"], (tt, c), x); np.add.at(TL["reads"], (tt, c), w); np.add.at(TL["obs_w"], (tt, c), x * w)
                for k, y in preds.items(): np.add.at(TL[k], (tt, c), np.nan_to_num(y))
                pick = rng.random(len(x)) < args.sub
                for k, y in preds.items(): sub[k].append(y[pick])
                sub["obs"].append(x[pick]); sub["cell"].append(r0 + r[pick]); sub["junc"].append(c[pick]); sub["N_orig"].append(Co[r, c][pick]); sub["N_level"].append(Nh[r, c][pick])
                print(f"  level {lev}: cells {r1:,}/{N:,}  {time.time()-t0:.0f}s", flush=True)
        out = dict(level=lev, hvj=hvj, **{f"sub_{k}": np.concatenate(v) for k, v in sub.items()}, **{f"TL_{k}": v for k, v in TL.items()})
        for k in acc:
            out[f"J_{k}"] = acc[k].J; out[f"C_{k}"] = acc[k].C; out[f"G_{k}"] = acc[k].G
            out[f"Jlost_{k}"] = acc_lost[k].J; out[f"Clost_{k}"] = acc_lost[k].C; out[f"Glost_{k}"] = acc_lost[k].G
        np.savez_compressed(f"{args.out}/level_{lev:03d}.npz", **out)
        # cell-type-level: per junction, Pearson across cell types (>= 10 cells each) of pooled observed vs mean predicted
        okT = TL["n"] >= 10; obsT = np.where(okT, TL["obs_w"] / np.maximum(TL["reads"], 1), np.nan)
        for k in ("post", "pure", "jmean"):
            predT = np.where(okT, TL[k] / np.maximum(TL["n"], 1), np.nan); rs = []
            for j in range(len(hvj)):
                m = okT[:, j]
                if m.sum() >= 5 and np.nanstd(obsT[m, j]) > 1e-6 and np.nanstd(predT[m, j]) > 1e-6:
                    rs.append(np.corrcoef(obsT[m, j], predT[m, j])[0, 1])
            print(f"level {lev:3d}% {k:6s}: CELL-TYPE-level within-junction r (across types, >=5 types with >=10 cells): median {np.median(rs):.3f} over {len(rs):,} junctions", flush=True)
        for k in acc:
            wj = Acc.r(acc[k].J[acc[k].J[:, 0] >= 30]); wc = Acc.r(acc[k].C[acc[k].C[:, 0] >= 30]); g = Acc.r(acc[k].G)
            wl = Acc.r(acc_lost[k].J[acc_lost[k].J[:, 0] >= 30]) if acc_lost[k].G[0] > 0 else np.array([np.nan])
            print(f"level {lev:3d}% {k:6s}: all-entry r {g:.3f}; within-junction median r {np.nanmedian(wj):.3f}; within-cell median r {np.nanmedian(wc):.3f}; "
                  f"no-reads-left within-junction {np.nanmedian(wl):.3f} (n entries {int(acc_lost[k].G[0]):,})", flush=True)


if __name__ == "__main__":
    main()
