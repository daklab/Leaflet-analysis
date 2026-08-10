#!/usr/bin/env python
"""A56: is phi reproducible in the ATLAS's regime, on data where truth is known?

The atlas failure, stated precisely
-----------------------------------
The same command with the same seed, run twice, gave Fbln2 skip PSI 0.65/0.03 on one run and
0.42/0.55 on the next. Program recovery (psi) was fine in both. So the failure is specifically in
the per-cell encoding and in the GROUP CONTRAST computed from it, and it has never been reproduced
anywhere truth is known -- every simulation to date ran at concentration 10-200 and a median 33
reads per observed entry, against the atlas's fitted 0.52 and median 6.

This scores the 2x2 that closes that gap:

              depth 1.0 (sim)     depth 0.12 (atlas-like)
    conc 50   A control            B sparsity alone
    conc  2   C low conc alone     D both = the atlas regime

What is measured, and why each one
----------------------------------
psi_r            recovered vs TRUE psi, Hungarian-matched. The control: if this degrades the
                 problem is program recovery, which is NOT what the atlas shows.
phi_ari          ARI of argmax(phi) against the true program label. Does phi carry the structure.
phi_r_seed       across-seed agreement of phi after matching programs. The atlas's weak direction.
contrast_l1      the DIRECT analogue of the Fbln2 panel. Take the two true programs, form each
                 one's predicted junction composition, and measure how far apart the two groups
                 are: ||mean_psi(group 1) - mean_psi(group 2)||_1 over the differential ATSEs.
contrast_sd      the ACROSS-SEED spread of that same quantity. This is the number that matters --
                 on the atlas it swung 0.65/0.03 to 0.42/0.55 between identical runs, so a healthy
                 regime must show contrast_sd << contrast_l1.
contrast_r       across-seed correlation of the per-ATSE contrast vector. Stability of WHICH events
                 differ, not just how much.

Programs are permuted freely by the model, so everything cross-fit goes through Hungarian matching
on |corr(psi)|. Matching psi alone and then applying the SAME permutation to phi is deliberate:
matching phi separately would let the two disagree and manufacture agreement that is not there.
"""

import argparse, glob, json, os, re, sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
from sklearn.metrics import adjusted_rand_score


def match_programs(psi_a, psi_b):
    """Permutation of b's programs onto a's, by |correlation| of the psi rows."""
    Ka, Kb = psi_a.shape[0], psi_b.shape[0]
    C = np.zeros((Ka, Kb))
    for i in range(Ka):
        for j in range(Kb):
            va, vb = psi_a[i], psi_b[j]
            if va.std() < 1e-12 or vb.std() < 1e-12:
                C[i, j] = 0.0
            else:
                C[i, j] = abs(np.corrcoef(va, vb)[0, 1])
    r, c = linear_sum_assignment(-C)
    return c, C[r, c]


def group_composition(phi, psi, labels, g):
    """Predicted junction composition averaged over the cells of one true program."""
    m = labels == g
    if m.sum() == 0:
        return None
    return phi[m].mean(0) @ psi


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True, help="label=dir_of_fits")
    ap.add_argument("--sim-dir", default="/gpfs/commons/groups/knowles_lab/Karin/"
                                         "Leaflet-analysis-WD/SIMULATION")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for spec in args.arms:
        label, d = spec.split("=", 1)
        fits = sorted(glob.glob(os.path.join(d, "fit_*.npz")))
        if len(fits) < 2:
            print(f"[{label}] only {len(fits)} fits -- skipped", flush=True)
            continue
        # the simulation these fits came from, recovered from the fit filename
        # the captured tag ALREADY starts with "sim_", so prepending another gave sim_sim_heart_...
        tag = re.search(r"fit_(.+?)_dirichletmultinomial", os.path.basename(fits[0]))
        base = tag.group(1) if tag else None
        h5 = None
        if base:
            name = base if base.startswith("sim_") else f"sim_{base}"
            h5 = os.path.join(args.sim_dir, f"{name}.h5ad")
        if not (h5 and os.path.exists(h5)):
            print(f"[{label}] cannot locate the source sim for {os.path.basename(fits[0])}",
                  flush=True)
            continue
        a = ad.read_h5ad(h5)
        true_prog = np.asarray(a.obs["true_program"]).astype(int)
        is_diff = np.asarray(a.var["is_differential"]).astype(bool)
        true_phi = np.asarray(a.obsm["true_phi"], float)

        g0, g1 = 0, 1                  # two true programs; the analogue of endo vs fibro

        # --- the TRUE junction composition, so nothing is judged against a proxy -------------
        # cassettes: the skip junction carries psi, the two inclusion junctions split the rest.
        true_skip = np.asarray(a.uns["true_psi_skip"], float)          # [K_true, n_ATSE]
        role = np.asarray(a.var["junction_role"]).astype(str)
        ev_j = np.asarray(a.var["event_id"]).astype(str)
        atse_ids = np.asarray(a.uns["atse_ids"]).astype(str)
        pos = {e: i for i, e in enumerate(atse_ids)}
        keep = np.array([e in pos for e in ev_j])
        col_a = np.array([pos.get(e, -1) for e in ev_j])
        true_psi_j = np.zeros((true_skip.shape[0], len(ev_j)))
        for k in range(true_skip.shape[0]):
            v = np.where(role == "J3", true_skip[k, col_a], (1 - true_skip[k, col_a]) / 2.0)
            true_psi_j[k] = np.where(keep, v, 0.0)
        c0t = true_phi[true_prog == g0].mean(0) @ true_psi_j
        c1t = true_phi[true_prog == g1].mean(0) @ true_psi_j
        true_contrast = np.abs(c0t - c1t)[is_diff]

        Z = [np.load(f, allow_pickle=True) for f in fits]
        psis = [np.asarray(z["psi_mean"], float) for z in Z]
        phis = [np.asarray(z["phi"], float) for z in Z]

        # --- per-fit quality, and the contrast each fit predicts -------------------------
        contrasts, aris, concs = [], [], []
        for psi, phi in zip(psis, phis):
            aris.append(adjusted_rand_score(true_prog, phi.argmax(1)))
            c0 = group_composition(phi, psi, true_prog, g0)
            c1 = group_composition(phi, psi, true_prog, g1)
            contrasts.append(np.abs(c0 - c1)[is_diff])
        for z in Z:
            cv = np.atleast_1d(np.asarray(z["bb_conc"]).ravel())
            concs.append(float(np.median(cv)))
        contrasts = np.array(contrasts)                       # [n_fits, n_diff_junctions]

        # --- psi recovery against TRUTH, matched -----------------------------------------
        # true psi is per (program, ATSE) skip usage; compare on the shared junction axis via the
        # model's own psi is not possible directly, so recovery is scored fit-vs-fit and
        # fit-vs-true-phi structure instead. psi_r here is CROSS-SEED, which is the atlas question.
        psi_rs, phi_rs = [], []
        for i in range(len(Z)):
            for j in range(i + 1, len(Z)):
                perm, qual = match_programs(psis[i], psis[j])
                pj = psis[j][perm]
                psi_rs.append(np.corrcoef(psis[i].ravel(), pj.ravel())[0, 1])
                # the SAME permutation applied to phi -- never matched separately
                phi_rs.append(np.corrcoef(phis[i].ravel(), phis[j][:, perm].ravel())[0, 1])

        cl1 = contrasts.sum(1)
        # across-seed correlation of WHICH junctions differ
        crs = [np.corrcoef(contrasts[i], contrasts[j])[0, 1]
               for i in range(len(contrasts)) for j in range(i + 1, len(contrasts))]

        # accuracy against the TRUTH, which is the whole reason to be on simulation
        tl1 = float(true_contrast.sum())
        mag = float(cl1.mean() / max(tl1, 1e-12))            # 1.0 = right size
        acc = [float(np.corrcoef(c, true_contrast)[0, 1]) for c in contrasts]
        phi_true_r = []
        for psi, phi in zip(psis, phis):
            kk = min(phi.shape[1], true_phi.shape[1])
            perm, _ = match_programs(true_psi_j[:kk], psi[:kk])
            phi_true_r.append(float(np.corrcoef(true_phi[:, :kk].ravel(),
                                                phi[:, perm[:kk]].ravel())[0, 1]))

        rows.append(dict(arm=label, n_fits=len(Z),
                         true_contrast_l1=tl1,
                         contrast_mag_vs_true=mag,
                         contrast_acc=float(np.mean(acc)),
                         phi_r_true=float(np.mean(phi_true_r)),
                         conc_hat=float(np.median(concs)),
                         phi_ari=float(np.mean(aris)),
                         psi_r_seed=float(np.mean(psi_rs)),
                         phi_r_seed=float(np.mean(phi_rs)),
                         contrast_l1=float(cl1.mean()),
                         contrast_sd=float(cl1.std()),
                         contrast_cv=float(cl1.std() / max(cl1.mean(), 1e-12)),
                         contrast_r=float(np.mean(crs))))
        print(f"[{label}] {len(Z)} fits, conc_hat {np.median(concs):.2f}, "
              f"contrast {cl1.mean():.3f} +/- {cl1.std():.3f}", flush=True)

    if not rows:
        print("nothing scored"); return
    R = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    R.to_csv(args.out, sep="\t", index=False)
    print()
    print("=== phi reproducibility by regime ===")
    print(R.to_string(index=False))
    print()
    print("contrast_cv is the headline: the across-seed spread of the group contrast, relative to")
    print("its size. The atlas swung 0.65/0.03 -> 0.42/0.55 between IDENTICAL runs, so a regime")
    print("that reproduces the failure will show a large cv here and a healthy one will not.")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
