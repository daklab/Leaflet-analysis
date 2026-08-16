#!/usr/bin/env python
"""Prep for Figure 2 v2: the SETUP panels (what was simulated) and the program-recovery UMAPs.

One simulation (heart, K=5, DM emission conc 50, effect 0.5, 30% of ATSEs differential, real
per-cell coverage from the heart cells) supplies:
  - true phi UMAP coloured by true program           (a: what the cells look like)
  - two example cassette exons: one differential, one not: true skip-PSI per program, and the
    observed per-cell skip PSI grouped by the cell's true program (b/c)
  - learned phi UMAP (LeafletFA, shipped recipe, seed 0) and NMF W UMAP (mean-impute), each
    coloured by TRUE program, with silhouette on the latent space (d/e)
Saved to data/fig2_v2_prep.npz.
"""
import os, sys
import numpy as np, anndata as ad, h5py
from sklearn.metrics import silhouette_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim_benchmark as SB
W = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD"
SIM = f"{W}/SIMULATION/sim_heart_K5_dm50_eff0.5_frac0.3_seed0.h5ad"
FIT = f"{W}/MOUSE_SPLICING_FOUNDATION/DM_SHIPPED_SIM/fit_sim_heart_K5_dm50_eff0.5_frac0.3_seed0_dirichletmultinomial_simplex_seed0.npz"
OUT = os.path.join(HERE, "..", "data", "fig2_v2_prep.npz")


def umap_of(X, seed=0):
    import umap
    return umap.UMAP(n_components=2, random_state=seed, n_neighbors=30, min_dist=0.3).fit_transform(X)


def main():
    sim = ad.read_h5ad(SIM)
    T = SB.truth_of(sim)
    prog = np.asarray(sim.obs["true_program"]).astype(int)
    tphi = T["phi"]
    z = np.load(FIT, allow_pickle=True)
    phi = np.asarray(z["phi"])
    # NMF (mean-impute, the gauge that scored best), W = cells x K
    nmf = SB.fit_nmf(sim, 5)
    H, Wn = nmf["mean-impute"][0], nmf["mean-impute"][1]
    Wn = np.asarray(Wn); Wn = Wn / np.maximum(Wn.sum(1, keepdims=True), 1e-12)
    embs = {k: umap_of(v) for k, v in [("true", tphi), ("lfa", phi), ("nmf", Wn)]}
    sil = {k: silhouette_score(v, prog) for k, v in [("true", tphi), ("lfa", phi), ("nmf", Wn)]}
    print("silhouette by true program:", {k: round(float(v), 3) for k, v in sil.items()})

    # example ATSEs -- observed skip PSI per cell: junction J3 count / ATSE count
    ev = sim.var["event_id"].astype(str).values; role = sim.var["junction_role"].values
    ids = np.asarray(sim.uns["atse_ids"]).astype(str); tps = np.asarray(sim.uns["true_psi_skip"])
    isd = T["is_diff"]; td = T["true_delta"]
    Y = SB._dense(sim.layers["cell_by_junction_matrix"]); C = SB._dense(sim.layers["cell_by_cluster_matrix"])
    cov_n = (C > 0).sum(0)
    def pick(mask):
        # well covered, and for the differential one a large delta with programs spread out
        cand = [i for i in np.flatnonzero(mask)]
        cols = {e: np.flatnonzero((ev == e) & (role == "J3"))[0] for e in ids[cand]}
        cov = np.array([cov_n[cols[ids[i]]] for i in cand])
        return cand, cov, cols
    dc, dcov, dcols = pick(isd & (td > 0.4))
    order = np.argsort(-dcov); i_ds = dc[order[0]]
    nc, ncov, ncols = pick(~isd)
    order = np.argsort(-ncov); i_null = nc[order[0]]
    ex = {}
    for tag, i, cols in [("ds", i_ds, dcols), ("null", i_null, ncols)]:
        j = cols[ids[i]]
        n = C[:, j]; y = Y[:, j]; m = n > 0
        ex[tag] = dict(id=ids[i], true_psi=tps[:, i], obs=np.where(m, y / np.maximum(n, 1), np.nan),
                       n=n, delta=td[i])
        print(tag, ids[i], "true psi per program", np.round(tps[:, i], 2), "covered", int(m.sum()), "delta", round(float(td[i]), 3))
    np.savez_compressed(OUT, prog=prog, emb_true=embs["true"], emb_lfa=embs["lfa"], emb_nmf=embs["nmf"],
                        sil_true=sil["true"], sil_lfa=sil["lfa"], sil_nmf=sil["nmf"],
                        ds_id=ex["ds"]["id"], ds_true=ex["ds"]["true_psi"], ds_obs=ex["ds"]["obs"], ds_n=ex["ds"]["n"], ds_delta=ex["ds"]["delta"],
                        null_id=ex["null"]["id"], null_true=ex["null"]["true_psi"], null_obs=ex["null"]["obs"], null_n=ex["null"]["n"], null_delta=ex["null"]["delta"],
                        n_cells=len(prog), n_atse=len(ids), frac_diff=float(isd.mean()), phi_r=float(np.nan))
    print("->", OUT)


if __name__ == "__main__":
    main()
