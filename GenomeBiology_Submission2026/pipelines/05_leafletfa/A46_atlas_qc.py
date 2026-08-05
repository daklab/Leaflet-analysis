#!/usr/bin/env python
"""A46: do the atlas DM+simplex fits make sense, and do independent seeds agree?

Four questions, in order of what would invalidate the most if it failed.

1. STRUCTURAL -- is the model's own promise kept?
   Under the simplex every program's psi must sum to 1 within each ATSE, exactly, by construction.
   If it does not, the constraint is not active and every downstream TV and LRT is computed on an
   arbitrary per-event scale. This is the check that catches a silently broken parameterisation:
   the released beta-psi fits sit at ~86% valid, and 14% invalid was doing real damage before
   anyone was looking for it.

2. IMPUTATION -- do the reconstructed usages sum to 1 per (cell, ATSE)?
   mu = phi @ psi. Since phi rows are on the simplex and psi sums to 1 within an event, mu must
   too. It follows algebraically, which is exactly why it is worth checking numerically: if it
   fails, one of the two factors is not what its name says.

3. LATENTS -- are the programs actually used, and do they mean anything?
   Program occupancy and perplexity catch the failure where K programs are fitted but only a few
   carry cells (the K=50 random-init collapse). Agreement with held-out `broad_cell_type` labels
   is reported as a sanity signal ONLY: the model never sees labels, and programs are not supposed
   to BE cell types, so low agreement is not a failure -- but near-zero agreement across every K
   would mean the latents carry no biology.

4. SEEDS -- do independent random initialisations find the same thing?
   Programs come out in arbitrary order, so seeds are matched with the Hungarian algorithm on psi
   and the SAME permutation is applied to phi. Getting that wrong is not hypothetical: it made
   cross-seed TV agreement read 0.47-0.56 earlier in this project when the true value was 0.88-0.99.
   Reported per K: matched psi correlation, phi correlation, top-N junction overlap, and the
   fraction of programs with a confident match -- the last being the honest headline, since only
   ~50% of programs reproduced across seeds in the released fits.
"""

import argparse, glob, json, os, sys

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
H5 = f"{WD}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad"


def obs_cat(name, h5=H5):
    with h5py.File(h5, "r") as f:
        if name not in f["obs"]:
            return None
        n = f["obs"][name]
        if isinstance(n, h5py.Group):
            c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
            return c[n["codes"][:]]
        return np.array([x.decode() if isinstance(x, bytes) else x for x in n[:]])


def read_rows(layer, rows, n_cols, h5=H5, chunk=2000):
    order = np.argsort(rows); srt = np.asarray(rows)[order]
    out = np.zeros((len(rows), n_cols), np.float32)
    with h5py.File(h5, "r") as f:
        g = f["layers"][layer]; ip_all = g["indptr"]
        for a in range(0, len(srt), chunk):
            blk = srt[a:a + chunk]
            lo, hi = int(blk[0]), int(blk[-1]) + 1
            ip = ip_all[lo:hi + 1]; s, e = int(ip[0]), int(ip[-1])
            m = sp.csr_matrix((g["data"][s:e], g["indices"][s:e], ip - ip[0]), shape=(hi - lo, n_cols))
            out[order[a:a + len(blk)]] = m[blk - lo].toarray()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=f"{WD}/ATLAS_DM/runs")
    ap.add_argument("--out", default=f"{WD}/ATLAS_DM/qc")
    ap.add_argument("--h5", default=H5)
    ap.add_argument("--check-cells", type=int, default=4000,
                    help="cells sampled for the imputation and label checks")
    ap.add_argument("--top-n", type=int, default=2000)
    ap.add_argument("--match-tol", type=float, default=0.5,
                    help="matched psi correlation above which a program counts as reproduced")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    fits = sorted(glob.glob(os.path.join(args.runs, "*", "fit.npz")))
    if not fits:
        print(f"no fits under {args.runs}"); return
    print(f"[0] {len(fits)} fits\n")

    ev = None
    with h5py.File(args.h5, "r") as f:
        n = f["var"]["event_id"]
        if isinstance(n, h5py.Group):
            c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
            ev = c[n["codes"][:]]
        else:
            ev = np.array([x.decode() if isinstance(x, bytes) else x for x in n[:]])
    ev = ev.astype(str)
    uq, inv = np.unique(ev, return_inverse=True)
    n_atse, J = len(uq), len(ev)
    rng = np.random.default_rng(0)
    with h5py.File(args.h5, "r") as f:
        N_all = int(f["layers"]["cell_by_junction_matrix"].attrs["shape"][0])
    cells = np.sort(rng.choice(N_all, size=min(args.check_cells, N_all), replace=False))
    junc = read_rows("cell_by_junction_matrix", cells, J, args.h5)
    clus = read_rows("cell_by_cluster_matrix", cells, J, args.h5)
    cov = clus > 0
    ctype = obs_cat("broad_cell_type", args.h5)
    print(f"[0b] {n_atse:,} ATSEs, {J:,} junctions; QC on {len(cells):,} sampled cells\n")

    rows, store = [], {}
    for fp in fits:
        z = np.load(fp, allow_pickle=True)
        run = os.path.basename(os.path.dirname(fp))
        K, seed = int(z["K"]), int(z["seed"])
        psi, phi = np.asarray(z["psi_mean"]), np.asarray(z["phi"])
        rj = json.load(open(os.path.join(os.path.dirname(fp), "result.json")))

        # 1. structural
        S = np.zeros((K, n_atse), np.float32)
        np.add.at(S.T, inv, psi.T)
        frac_valid = float(((S >= 0.9) & (S <= 1.1)).mean())
        max_dev = float(np.abs(S - 1).max())

        # 2. imputation: mu must sum to 1 per (cell, ATSE)
        ph = phi[cells]
        mu = ph @ psi                                  # [n_cells, J]
        M = np.zeros((len(cells), n_atse), np.float32)
        np.add.at(M.T, inv, mu.T)
        imp_dev = float(np.abs(M - 1).max())
        # and does it track the data at all?
        obs = np.where(cov, junc / np.maximum(clus, 1), np.nan)
        ok = cov & np.isfinite(obs)
        r_imp = float(np.corrcoef(mu[ok], obs[ok])[0, 1]) if ok.sum() > 100 else np.nan

        # 3. latents
        dom = ph.argmax(1)
        occ = np.bincount(dom, minlength=K) / len(cells)
        p = np.clip(ph, 1e-12, None)
        perp = float(np.exp(-(p * np.log(p)).sum(1)).mean())
        n_live = int((occ >= 0.01).sum())
        ari = np.nan
        if ctype is not None:
            try:
                from sklearn.metrics import adjusted_rand_score
                ari = float(adjusted_rand_score(ctype[cells], dom))
            except Exception:
                pass

        rows.append(dict(run=run, K=K, seed=seed, batch=rj.get("batch"),
                         steps_run=rj.get("steps_run"), nan_guard=rj.get("nan_guard_fired"),
                         early_stop=rj.get("early_stopped"), minutes=rj.get("train_minutes"),
                         conc=rj.get("bb_conc"), psi_valid=frac_valid, psi_max_dev=max_dev,
                         impute_max_dev=imp_dev, impute_r=r_imp, programs_live=n_live,
                         perplexity=perp, occ_min=float(occ.min()), occ_max=float(occ.max()),
                         ari_celltype=ari))
        store[(K, seed, rj.get("batch"))] = (psi, phi)
        print(f"[1] {run}: psi sums valid {100*frac_valid:.2f}% (max dev {max_dev:.2e}), "
              f"impute dev {imp_dev:.2e}, impute r {r_imp:.3f}, live {n_live}/{K}, "
              f"perp {perp:.2f}, ARI {ari:.3f}", flush=True)

    Q = pd.DataFrame(rows).sort_values(["K", "batch", "seed"])
    Q.to_csv(os.path.join(args.out, "atlas_qc.tsv"), sep="\t", index=False)
    print("\n=== per-fit QC ===")
    print(Q[["run", "steps_run", "nan_guard", "conc", "psi_valid", "psi_max_dev",
             "impute_max_dev", "impute_r", "programs_live", "perplexity",
             "ari_celltype"]].to_string(index=False))

    # 4. seed consistency, within each (K, batch)
    print("\n=== seed consistency (Hungarian on psi; the SAME permutation applied to phi) ===")
    srows = []
    keys = sorted({(k, b) for (k, s_, b) in store})
    for (K, b) in keys:
        seeds = sorted(s_ for (k, s_, bb) in store if k == K and bb == b)
        if len(seeds) < 2:
            print(f"  K={K} batch={b}: only seed(s) {seeds} -- skipped")
            continue
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                (pA, fA), (pB, fB) = store[(K, seeds[i], b)], store[(K, seeds[j], b)]
                C = np.corrcoef(pA, pB)[:K, K:]
                r_, c_ = linear_sum_assignment(-np.nan_to_num(C))
                matched = np.array([C[a, bb] for a, bb in zip(r_, c_)])
                perm = np.empty(K, int); perm[r_] = c_
                pBm, fBm = pB[perm], fB[:, perm]        # psi AND phi, same permutation
                phi_r = float(np.corrcoef(fA[cells].ravel(), fBm[cells].ravel())[0, 1])
                # do the two fits pick the same junctions as most program-variable?
                vA = pA.max(0) - pA.min(0); vB = pBm.max(0) - pBm.min(0)
                tA = set(np.argsort(-vA)[:args.top_n]); tB = set(np.argsort(-vB)[:args.top_n])
                srows.append(dict(K=K, batch=b, pair=f"{seeds[i]}v{seeds[j]}",
                                  psi_r_mean=float(matched.mean()),
                                  psi_r_min=float(matched.min()),
                                  frac_reproduced=float((matched >= args.match_tol).mean()),
                                  phi_r=phi_r, spread_r=float(np.corrcoef(vA, vB)[0, 1]),
                                  top_overlap=len(tA & tB) / args.top_n))
    if srows:
        S = pd.DataFrame(srows)
        S.to_csv(os.path.join(args.out, "atlas_seed_consistency.tsv"), sep="\t", index=False)
        print(S.to_string(index=False))
        print("\n  frac_reproduced is the headline: the share of programs a second seed recovers "
              f"at matched r >= {args.match_tol}.")
        print("  psi_r_min shows the WORST-matched program, which mean correlation hides.")
    print(f"\n-> {args.out}/")


if __name__ == "__main__":
    main()
