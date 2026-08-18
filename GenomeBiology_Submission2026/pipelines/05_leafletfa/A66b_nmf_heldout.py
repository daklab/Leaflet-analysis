#!/usr/bin/env python
"""A66b: NMF imputation on the same held-out cells and levels as A66, for the comparison row.

NMF cannot represent missingness, so the input is the junction-ratio matrix at that coverage level
with uncovered entries mean-imputed (per junction, the gauge that scored best in A3), rank K = 20 to
match the LeafletFA fit; the reconstruction W H clipped to [0, 1] is the imputed PSI. Saved as
nmf_level_XXX.npz aligned to A66's (cell, junc) entries.
"""
import argparse, os, glob, time
import numpy as np, h5py, scipy.sparse as sp
from sklearn.decomposition import NMF

M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"


def rows_csr(g, rows, J):
    ip = g["indptr"][:]; d, ix, pt = [], [], [0]
    for r in rows:
        s, e = int(ip[r]), int(ip[r + 1]); d.append(g["data"][s:e]); ix.append(g["indices"][s:e]); pt.append(pt[-1] + e - s)
    return sp.csr_matrix((np.concatenate(d).astype(np.float32), np.concatenate(ix), np.array(pt)), shape=(len(rows), J))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=f"{M}/IMPUTATION_DM/a66_all_cells")
    ap.add_argument("--all-cells", action="store_true", help="v2 layout: all cells, HVJ junction subset (hvj.npy), sufficient stats + subsample")
    ap.add_argument("--h5mu-dir", default=f"{M}/IMPUTATION_DM/resampled_combined")
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--max-iter", type=int, default=600)
    args = ap.parse_args()
    if args.all_cells:
        return main_all(args)
    cells = np.load(f"{args.dir}/cells.npy")
    files = {int(os.path.basename(p).split("_")[1]): p for p in glob.glob(f"{args.h5mu_dir}/RESAMPLED_*_PERCENT_test_*.h5mu")}
    ref = files[min(files)]
    for lev_file in sorted(glob.glob(f"{args.dir}/level_*.npz")):
        z = np.load(lev_file); lev = int(z["level"]); t0 = time.time()
        with h5py.File(files.get(lev, ref), "r") as f:
            s = f["mod/splicing"]; J = int(s["layers/cell_by_junction_matrix"].attrs["shape"][1])
            if lev == 0:
                R = np.asarray(s["layers/junc_ratio_original"][cells, :], np.float32)
                C = rows_csr(s["layers/cell_by_cluster_matrix_original"], cells, J).toarray()
            else:
                Y = rows_csr(s["layers/cell_by_junction_matrix"], cells, J).toarray()
                C = rows_csr(s["layers/cell_by_cluster_matrix"], cells, J).toarray()
                R = np.where(C > 0, Y / np.maximum(C, 1), 0.0).astype(np.float32)
        cov = C > 0
        R = np.clip(np.nan_to_num(R), 0, 1)
        cnt = cov.sum(0); cm = np.where(cnt > 0, (R * cov).sum(0) / np.maximum(cnt, 1), 0.0)
        X = np.where(cov, R, cm[None, :]).astype(np.float32)
        model = NMF(n_components=args.K, init="nndsvda", max_iter=args.max_iter, random_state=0)
        Wm = model.fit_transform(X); H = model.components_
        rec = np.clip(Wm @ H, 0, 1).astype(np.float32)
        r, c = z["cell"], z["junc"]
        np.savez_compressed(f"{args.dir}/nmf_level_{lev:03d}.npz", nmf=rec[r, c], level=lev, n_iter=model.n_iter_)
        keep = z["N_orig"] >= 5
        print(f"level {lev:3d}%: NMF {time.time()-t0:.0f}s, {model.n_iter_} iters; pearson on N_orig>=5: {np.corrcoef(z['obs_full'][keep], rec[r, c][keep])[0,1]:.3f}", flush=True)


def main_all(args):
    """all held-out cells x HVJ junctions: NMF (rank K, mean-imputed) at each level; per-junction /
    per-cell sufficient stats on the same >= 5-read entries as A66 v2, plus values on A66's subsample."""
    sys_path = os.path.dirname(os.path.abspath(__file__))
    import sys; sys.path.insert(0, sys_path)
    from A66_imputed_vs_raw_heldout import Acc
    hvj = np.load(f"{args.dir}/hvj.npy")
    bct = np.load(f"{args.dir}/cells_bct.npy", allow_pickle=True); types, tcode = np.unique(bct, return_inverse=True); T = len(types)
    files = {int(os.path.basename(p).split("_")[1]): p for p in glob.glob(f"{args.h5mu_dir}/RESAMPLED_*_PERCENT_test_*.h5mu")}
    ref = files[min(files)]
    with h5py.File(ref, "r") as f:
        s = f["mod/splicing"]; N = s["obs/_index"].shape[0]; J = int(s["layers/cell_by_junction_matrix"].attrs["shape"][1])
        Corig = rows_csr(s["layers/cell_by_cluster_matrix_original"], np.arange(N), J)[:, hvj].toarray()
        jr = np.asarray(s["layers/junc_ratio_original"][:, hvj], np.float32)
    keep = Corig >= 5
    for lev_file in sorted(glob.glob(f"{args.dir}/level_*.npz")):
        z = np.load(lev_file); lev = int(z["level"]); t0 = time.time()
        with h5py.File(files.get(lev, ref), "r") as f:
            s = f["mod/splicing"]
            if lev == 0:
                C = Corig; R = np.nan_to_num(jr) * (C > 0)
            else:
                Y = rows_csr(s["layers/cell_by_junction_matrix"], np.arange(N), J)[:, hvj].toarray()
                C = rows_csr(s["layers/cell_by_cluster_matrix"], np.arange(N), J)[:, hvj].toarray()
                R = np.where(C > 0, Y / np.maximum(C, 1), 0.0).astype(np.float32)
        cov = C > 0; R = np.clip(np.nan_to_num(R), 0, 1)
        cnt = cov.sum(0); cm = np.where(cnt > 0, (R * cov).sum(0) / np.maximum(cnt, 1), 0.0)
        X = np.where(cov, R, cm[None, :]).astype(np.float32)
        model = NMF(n_components=args.K, init="nndsvda", max_iter=args.max_iter, random_state=0)
        Wm = model.fit_transform(X); H = model.components_; rec = np.clip(Wm @ H, 0, 1).astype(np.float32); del X
        acc = Acc(len(hvj), N); acc_l = Acc(len(hvj), N)
        r, c = np.nonzero(keep); x = jr[r, c]; y = rec[r, c]
        acc.add(r, c, x, y); lost = C[r, c] == 0
        if lost.any(): acc_l.add(r[lost], c[lost], x[lost], y[lost])
        sub_y = rec[z["sub_cell"], z["sub_junc"]]
        TLn = np.zeros((T, len(hvj))); TLp = np.zeros((T, len(hvj)))
        np.add.at(TLn, (tcode[r], c), 1); np.add.at(TLp, (tcode[r], c), y.astype(np.float64))
        np.savez_compressed(f"{args.dir}/nmf_level_{lev:03d}.npz", level=lev, J_nmf=acc.J, C_nmf=acc.C, G_nmf=acc.G,
                            Jlost_nmf=acc_l.J, Clost_nmf=acc_l.C, Glost_nmf=acc_l.G, sub_nmf=sub_y, n_iter=model.n_iter_, TL_n=TLn, TL_nmf=TLp)
        wj = Acc.r(acc.J[acc.J[:, 0] >= 30]); wl = Acc.r(acc_l.J[acc_l.J[:, 0] >= 30]) if acc_l.G[0] > 0 else np.array([np.nan])
        print(f"level {lev:3d}% nmf: {time.time()-t0:.0f}s {model.n_iter_} it; all-entry r {Acc.r(acc.G):.3f}; within-junction median r {np.nanmedian(wj):.3f}; "
              f"no-reads-left within-junction {np.nanmedian(wl):.3f}", flush=True)


if __name__ == "__main__":
    main()
