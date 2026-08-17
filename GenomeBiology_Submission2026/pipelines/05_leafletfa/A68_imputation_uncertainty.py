#!/usr/bin/env python
"""A68: is the model's per-cell imputation uncertainty calibrated against seed-to-seed variation?

For a random sample of ATSEs (all their junctions) and a random sample of cells, compute per
(cell, junction):
  post_sd   sd of rho_cj = phi_c . psi_:,j over S joint draws from q(phi) q(psi) of the SEED-0 fit
  post_mean seed-0 posterior mean of rho_cj, and the 5-95% interval
  seed_i    posterior-mean rho_cj from the other seeds (their phi and psi at the posterior mean)
  cross_sd  sd of the three seeds' means
Outputs a table for plotting: cross_sd vs post_sd, and the fraction of other-seed values that fall
inside seed-0's 90% interval, by post_sd bin (should be ~0.9 if the intervals mean what they say).
Also joins broad cell type and per-(cell, event) read count for the coverage story.
"""
import argparse, os, sys
import numpy as np, pandas as pd, h5py, torch, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from A51_group_ds_posterior import draw_psi
from _phi_rows import phi_row_of
from leafletfa.dirichlet_multinomial import segment_softmax
M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"


def cat(g):
    c = np.array([x.decode() if isinstance(x, bytes) else x for x in g["categories"][:]]); return c[g["codes"][:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", nargs="+", default=[f"{M}/ATLAS_DM/runs_final_deltaonly/dm_simplex_K20_seed{s}_b8192/fit.npz" for s in (0, 1, 2)])
    ap.add_argument("--h5", default=f"{M}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad")
    ap.add_argument("--n-atse", type=int, default=400); ap.add_argument("--n-cells", type=int, default=6000)
    ap.add_argument("--samples", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{M}/IMPUTATION_DM/a68_uncertainty.tsv.gz")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    Z = [np.load(f, allow_pickle=True) for f in args.fits]
    z0 = Z[0]; fev = np.asarray(z0["event_id"]).astype(str); inv = np.asarray(z0["inv_ev"]).astype(np.int64)
    n_atse = len(fev); K = int(z0["phi"].shape[1])
    with h5py.File(args.h5, "r") as f:
        lab = cat(f["obs"]["tissue_cell_type"]); bct = cat(f["obs"]["broad_cell_type"]); N = len(lab)
    ev_pick = np.sort(rng.choice(n_atse, args.n_atse, replace=False))
    cols = np.flatnonzero(np.isin(inv, ev_pick)); print(f"{len(ev_pick)} ATSEs -> {len(cols):,} junctions", flush=True)
    prow0 = phi_row_of(z0, N); cells = np.sort(rng.choice(np.flatnonzero(prow0 >= 0), args.n_cells, replace=False))
    # reads per (cell, junction) for coverage
    with h5py.File(args.h5, "r") as f:
        g = f["layers"]["cell_by_cluster_matrix"]; ip = g["indptr"][:]; J = len(inv)
        Ncnt = np.zeros((len(cells), len(cols)), np.float32)
        for i, c in enumerate(cells):
            s, e = int(ip[c]), int(ip[c + 1]); idx = g["indices"][s:e]; dat = g["data"][s:e]
            m = np.isin(idx, cols)
            if m.any():
                pos = np.searchsorted(cols, idx[m]); Ncnt[i, pos] = dat[m]
    # seed 0: joint draws
    inv_t = torch.from_numpy(inv)
    psi_s = draw_psi(z0, inv_t, n_atse, args.samples, seed=args.seed)[:, :, cols]          # [S,K,j]
    sb = torch.distributions.transforms.StickBreakingTransform()
    loc = torch.from_numpy(np.asarray(z0["phi_loc"], np.float32)[prow0[cells]]); sc = torch.from_numpy(np.asarray(z0["phi_scale"], np.float32)[prow0[cells]])
    gen = torch.Generator().manual_seed(args.seed)
    acc = torch.empty(args.samples, len(cells), len(cols))
    for s_ in range(args.samples):
        ph = sb(loc + sc * torch.randn(loc.shape, generator=gen)); acc[s_] = ph @ psi_s[s_]
    post_mean = acc.mean(0).numpy(); post_sd = acc.std(0).numpy()
    lo = torch.quantile(acc, 0.05, dim=0).numpy(); hi = torch.quantile(acc, 0.95, dim=0).numpy()
    del acc
    # other seeds: posterior-mean rho
    means = [post_mean]
    for z in Z[1:]:
        psi = segment_softmax(torch.from_numpy(np.asarray(z["psi_loc"], np.float32)), inv_t, n_atse).numpy()[:, cols]
        prow = phi_row_of(z, N); means.append(np.asarray(z["phi"])[prow[cells]] @ psi)
    means = np.stack(means)                                                                    # [3, cells, j]
    cross_sd = means.std(0); inside = ((means[1:] >= lo[None]) & (means[1:] <= hi[None])).mean(0)
    r, c = np.meshgrid(np.arange(len(cells)), np.arange(len(cols)), indexing="ij")
    df = pd.DataFrame(dict(cell=cells[r.ravel()], junc=cols[c.ravel()], event=inv[cols][c.ravel()],
                           bct=bct[cells][r.ravel()], reads=Ncnt.ravel(), post_mean=post_mean.ravel(), post_sd=post_sd.ravel(),
                           cross_sd=cross_sd.ravel(), inside90=inside.ravel(),
                           seed1=means[1].ravel(), seed2=means[2].ravel()))
    df.to_csv(args.out, sep="\t", index=False)
    from scipy.stats import spearmanr
    print(f"entries {len(df):,}: median post_sd {df.post_sd.median():.3f}, median cross_sd {df.cross_sd.median():.3f}, "
          f"Spearman {spearmanr(df.post_sd, df.cross_sd).correlation:.3f}, frac other-seed inside seed-0 90% CI {df.inside90.mean():.3f}")
    df["bin"] = pd.qcut(df.post_sd, 10, labels=False, duplicates="drop")
    print(df.groupby("bin").agg(post_sd=("post_sd", "median"), cross_sd=("cross_sd", "median"), inside90=("inside90", "mean"), n=("cell", "size")).round(3).to_string())


if __name__ == "__main__":
    main()
