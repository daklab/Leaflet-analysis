#!/usr/bin/env python
"""A65: does a posterior DS call agree with what the cells in that contrast actually do?

Motivated by Gna13 ATSE1 (heart endothelial vs fibroblast): the posterior test gives TV 0.33-0.41 in
all three seeds, but the covered cells sit at per-cell-mean PSI 0.42 vs 0.36 (pooled 0.55 vs 0.46),
and LeafCutter's pseudobulk TV is 0.07. The model's group means (0.67 vs 0.23) are ~4-5 SE from the
data in BOTH arms. psi is a global program property; programs 7 (endothelial) and 10 (mesenchymal)
are polarised at that event by brain cell types that genuinely differ (microglia 0.18, neurons 0.88),
and 366 heart cells at ~2 effective molecules each cannot outvote them. TV = sum_k dphi_k psi_kj is
then a statement about the programs, not about heart cells.

This script measures how often that happens. For every (contrast, ATSE) in the gated table:
  obs_tv_pooled    0.5 * sum_j |PSI1_j - PSI2_j| with PSI = pooled counts (LeafCutter's quantity)
  obs_tv_cellmean  same with PSI = mean over covered cells of y_j / N_e (each cell one vote, which
                   is closer to what conc=2 makes the DM see)
  model_tv         0.5 * sum_j |(phibar1 - phibar2) . psi_:,j| at the posterior mean (the statistic)
  misfit_z         max over arms and junctions of |model PSI - cellmean PSI| / SE(cellmean PSI)
Then summarises by call category: LeafletFA-only / both / LeafCutter-only / neither.
"""
import argparse, os, sys
import numpy as np, pandas as pd, h5py, torch, scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from _phi_rows import phi_row_of
from leafletfa.dirichlet_multinomial import segment_softmax

M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"


def obs_cat(f, name):
    n = f["obs"][name]
    c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
    return c[n["codes"][:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=f"{M}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad")
    ap.add_argument("--fit", default=f"{M}/ATLAS_DM/runs_final_deltaonly/dm_simplex_K20_seed0_b8192/fit.npz")
    ap.add_argument("--gated", default=f"{M}/LEAFCUTTER_CMP_DELTAONLY/analysis_matched/posterior_consensus_gated.tsv.gz")
    ap.add_argument("--out", default=f"{M}/LEAFCUTTER_CMP_DELTAONLY/analysis_consistency")
    ap.add_argument("--block", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    z = np.load(args.fit, allow_pickle=True)
    fev = np.asarray(z["event_id"]).astype(str); inv = np.asarray(z["inv_ev"]).astype(np.int64)
    n_atse = len(fev); J = len(inv)
    phi = np.asarray(z["phi"], np.float32); K = phi.shape[1]
    psi = segment_softmax(torch.from_numpy(np.asarray(z["psi_loc"], np.float32)),
                          torch.from_numpy(inv), n_atse).numpy()                  # [K, J]
    g = pd.read_csv(args.gated, sep="\t")
    contrasts = sorted(g.contrast.unique())
    types = sorted(set(t for c in contrasts for t in c.split("__vs__")))
    print(f"{len(g):,} rows, {len(contrasts)} contrasts, {len(types)} cell types", flush=True)

    with h5py.File(args.h5, "r") as f:
        lab = obs_cat(f, "tissue_cell_type")
        lab_key = np.array([l.replace(" ", "_") for l in lab])
        assert f["var"]["_index"].shape[0] == J, "fit J != h5 J"
        prow = phi_row_of(z, len(lab))
        tidx = {t: i for i, t in enumerate(types)}
        cell_t = np.array([tidx.get(l, -1) for l in lab_key])
        T = len(types)
        S = np.zeros((T, J), np.float64)          # pooled counts
        R = np.zeros((T, J), np.float64)          # sum of per-cell ratios y_j / N_e (covered cells)
        R2 = np.zeros((T, J), np.float64)         # sum of squares, for the SE
        C = np.zeros((T, n_atse), np.float64)     # covered-cell count per (type, event)
        PH = np.zeros((T, K), np.float64); NPH = np.zeros(T)
        gl = f["layers"]["cell_by_junction_matrix"]; ip = gl["indptr"][:]; N = len(ip) - 1
        one_hot = sp.csr_matrix((np.ones(J), (inv, np.arange(J))), shape=(n_atse, J))  # event x junction
        for s0 in range(0, N, args.block):
            s1 = min(N, s0 + args.block)
            rows = np.arange(s0, s1); keep = cell_t[rows] >= 0
            if not keep.any():
                continue
            a, b = int(ip[s0]), int(ip[s1])
            X = sp.csr_matrix((gl["data"][a:b].astype(np.float64), gl["indices"][a:b], ip[s0:s1 + 1] - a),
                              shape=(s1 - s0, J))[keep]
            tt = cell_t[rows[keep]]
            Ne = X @ one_hot.T                                    # [n, n_atse] cell x event totals
            # per-cell ratio: y_j / N_e(j)
            Nj = Ne @ one_hot                                     # broadcast totals back to junctions
            Nj = Nj.tocsr(); Xc = X.tocsr()
            ratio = Xc.multiply(Nj.power(-1, dtype=np.float64).multiply(Nj > 0)) if False else None
            # simpler: elementwise on coo
            Xc = Xc.tocoo()
            tot = np.asarray(Nj[Xc.row, Xc.col]).ravel()
            r = np.zeros_like(Xc.data); nz = tot > 0; r[nz] = Xc.data[nz] / tot[nz]
            Rm = sp.csr_matrix((r, (Xc.row, Xc.col)), shape=Xc.shape)
            G = sp.csr_matrix((np.ones(len(tt)), (tt, np.arange(len(tt)))), shape=(T, len(tt)))
            S += (G @ Xc).toarray(); R += (G @ Rm).toarray(); R2 += (G @ Rm.power(2)).toarray()
            C += (G @ (Ne > 0).astype(np.float64)).toarray()
            pr = prow[rows[keep]]; ok = pr >= 0
            PH += (G[:, ok] @ phi[pr[ok]]); NPH += np.asarray(G[:, ok].sum(1)).ravel()
            print(f"  {s1:,}/{N:,}", flush=True)
    PH /= np.maximum(NPH, 1)[:, None]
    model_psi = PH @ psi                                          # [T, J]
    Sev = S @ one_hot.T                                           # [T, n_atse]
    pooled = np.zeros_like(S); Ce = np.asarray(C @ one_hot).astype(float)  # covered count per junction col
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = S / np.asarray(Sev @ one_hot)
        cellmean = R / Ce
        var = R2 / Ce - cellmean ** 2
        se = np.sqrt(np.maximum(var, 0) / np.maximum(Ce, 1))
    ev_of = {e: i for i, e in enumerate(fev)}
    out = []
    for c in contrasts:
        t1, t2 = c.split("__vs__"); i1, i2 = tidx[t1], tidx[t2]
        sub = g[g.contrast == c]
        eidx = np.array([ev_of.get(x, -1) for x in sub.cluster.astype(str)])
        ok = eidx >= 0
        d_pool = np.abs(pooled[i1] - pooled[i2]); d_cm = np.abs(cellmean[i1] - cellmean[i2])
        d_mod = np.abs(model_psi[i1] - model_psi[i2])
        z1 = np.abs(model_psi[i1] - cellmean[i1]) / np.maximum(se[i1], 1e-3)
        z2 = np.abs(model_psi[i2] - cellmean[i2]) / np.maximum(se[i2], 1e-3)
        tv_pool = 0.5 * np.bincount(inv, weights=np.nan_to_num(d_pool), minlength=n_atse)
        tv_cm = 0.5 * np.bincount(inv, weights=np.nan_to_num(d_cm), minlength=n_atse)
        tv_mod = 0.5 * np.bincount(inv, weights=d_mod, minlength=n_atse)
        zmax = np.maximum(np.bincount(inv, weights=np.nan_to_num(z1), minlength=n_atse) * 0,  # placeholder
                          0)
        # max z over junctions in the event, per arm
        zm1 = np.full(n_atse, 0.0); np.maximum.at(zm1, inv, np.nan_to_num(z1))
        zm2 = np.full(n_atse, 0.0); np.maximum.at(zm2, inv, np.nan_to_num(z2))
        df = sub.loc[ok, ["contrast", "cluster", "gene", "m", "tv", "votes", "lfa_cons", "lc_sig",
                          "tv_lc", "ncov1", "ncov2", "mincov", "eligible"]].copy()
        e = eidx[ok]
        df["obs_tv_pooled"] = tv_pool[e]; df["obs_tv_cellmean"] = tv_cm[e]; df["model_tv"] = tv_mod[e]
        df["misfit_z1"] = zm1[e]; df["misfit_z2"] = zm2[e]
        df["cov1"] = C[i1, e]; df["cov2"] = C[i2, e]
        out.append(df)
    res = pd.concat(out, ignore_index=True)
    res.to_csv(f"{args.out}/data_consistency.tsv.gz", sep="\t", index=False)
    print("saved", len(res))
    el = res[res.eligible]
    cat = np.where(el.lfa_cons & el.lc_sig, "both", np.where(el.lfa_cons, "LFA only",
                   np.where(el.lc_sig, "LC only", "neither")))
    el = el.assign(cat=cat)
    pd.set_option("display.width", 200)
    summ = el.groupby("cat").agg(n=("tv", "size"),
                                 model_tv_med=("model_tv", "median"),
                                 obs_pooled_med=("obs_tv_pooled", "median"),
                                 obs_cellmean_med=("obs_tv_cellmean", "median"),
                                 frac_obs_pooled_lt_0p15=("obs_tv_pooled", lambda x: np.mean(x < 0.15)),
                                 frac_obs_cellmean_lt_0p15=("obs_tv_cellmean", lambda x: np.mean(x < 0.15)),
                                 frac_misfit_z_gt3_both_arms=("misfit_z1", lambda x: np.nan))
    for c_ in summ.index:
        s = el[el.cat == c_]
        summ.loc[c_, "frac_misfit_z_gt3_both_arms"] = np.mean((s.misfit_z1 > 3) & (s.misfit_z2 > 3))
        summ.loc[c_, "frac_misfit_z_gt3_any_arm"] = np.mean((s.misfit_z1 > 3) | (s.misfit_z2 > 3))
    print(summ.round(3).to_string())
    summ.to_csv(f"{args.out}/summary_by_category.tsv", sep="\t")
    for c_ in ["LFA only", "both"]:
        s = el[el.cat == c_]
        print(f"\n{c_}: corr(model_tv, obs_tv_pooled) = {np.corrcoef(s.model_tv, s.obs_tv_pooled)[0,1]:.3f}; "
              f"corr(model_tv, obs_tv_cellmean) = {np.corrcoef(s.model_tv, s.obs_tv_cellmean)[0,1]:.3f}")
        print("  quantiles of obs_tv_pooled:", np.quantile(s.obs_tv_pooled, [.1, .25, .5, .75, .9]).round(3))


if __name__ == "__main__":
    main()
