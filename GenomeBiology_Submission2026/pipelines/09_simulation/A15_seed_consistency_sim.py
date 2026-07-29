"""How consistent is phi across random initialisations, and what would make it more so?

The atlas shows only ~30-50% of phi directions reproducing between seeds, and A13 traced the ceiling to
the assay's overdispersion rather than to anything fixable. But "overdispersion sets a ceiling" does not
mean the current setup reaches that ceiling. This script asks, on simulated data where the truth is
known, how much of the seed-to-seed variation comes from each of two separable sources:

  INITIALISATION   which basin the optimiser starts in. Removed by seeding psi deterministically from
                   the data itself.
  MINIBATCH        which cells each SVI step happens to see. Present even with identical init, and not
                   removable without full-batch training.

Comparing random init against a deterministic one, holding everything else fixed, splits the total.
If most of the variance is initialisation, a deterministic init buys consistency cheaply. If most is
minibatch noise, it does not, and the honest answer is consensus over replicates.

THE DETERMINISTIC INITIALISER uses no cell-type labels and no metadata -- only the observed splicing
matrix. It runs NNDSVD (the deterministic SVD-based initialiser standard in NMF; sklearn's default),
assigns each cell to its strongest component, and sets each program's profile to the coverage-weighted
mean observed PSI within that group. This is worth distinguishing from waypoint initialisation,
which reads cell-type annotations and so cannot support a claim of finding structure "without prior
cell-type labels". NNDSVD reads only the data matrix.

Worth being precise about what a deterministic init would and would not prove. It removes seed variance
BY CONSTRUCTION rather than by making the solution identifiable: every run starts in the same place, so
of course they agree. That is reproducibility, not uniqueness, and reporting it as though the posterior
were well determined would be misleading. Its honest use is engineering -- a released model that others
can reproduce exactly -- with identifiability reported separately via the consensus analysis.

Also reported at every condition: recovery against the KNOWN phi, so a setting that produces agreeing
but wrong fits is not mistaken for a good one. That failure mode is real: a deterministic init makes
every run agree even when all of them are equally wrong.

Outputs a tidy CSV for the supplementary figure: one row per (dispersion, init, seed-pair).

Usage: A15_seed_consistency_sim.py <out_csv> [n_seeds] [device]
"""
from __future__ import annotations
import sys, os, itertools
import numpy as np, pandas as pd, anndata as ad
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
SIM = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION"

LEVELS = [("binom", "no overdispersion"), ("bb20", "mild (c=20)"), ("bb2", "atlas-like (c=2)")]
DATA_SEED = 0


def cca(A, B):
    A = np.asarray(A, float); B = np.asarray(B, float)
    A = A - A.mean(0); B = B - B.mean(0)
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    return np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0, 1)


def matched_r(A, B):
    K = A.shape[0]
    C = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            a, b = A[i], B[j]
            C[i, j] = 0.0 if (a.std() < 1e-12 or b.std() < 1e-12) else abs(np.corrcoef(a, b)[0, 1])
    r, c = linear_sum_assignment(-np.nan_to_num(C))
    return float(np.median(C[r, c]))


def deterministic_psi_init(junc, clus, K, min_group=50):
    """Data-driven, label-free, and identical on every run.

    NNDSVD on the observed PSI gives deterministic soft cell weights (no random state anywhere). Each
    cell is then assigned to its strongest component and each program's profile is the coverage-weighted
    mean observed PSI within that group, which puts the result on the [0, 1] scale psi actually lives on.

    The HARD assignment matters. Reweighting by the soft NNDSVD weights instead gives programs that are
    nearly identical to one another (measured contrast 0.017, program-program correlation +0.991),
    because NNDSVD's first component captures the overall mean and dominates every weight vector.
    Starting all K programs on top of each other is precisely the collapse the initialisation is meant
    to avoid. Hard assignment gives contrast 0.111 and correlation +0.683, against fitted values of
    ~0.08 and ~+0.87 -- i.e. it starts the programs slightly more separated than they end up.

    k-means on the same matrix gives marginally better numbers but produced a one-cell cluster here,
    whose profile would be noise; groups below `min_group` therefore fall back to the global mean.
    """
    from sklearn.decomposition import _nmf
    cov = clus > 0
    psi_obs = np.where(cov, junc / np.maximum(clus, 1.0), 0.0).astype(np.float32)
    W, _ = _nmf._initialize_nmf(psi_obs, K, init="nndsvda", random_state=0)
    lab = np.maximum(W, 0).argmax(1)
    covf = cov.astype(np.float32)
    glob = (psi_obs * cov).sum(0) / np.maximum(covf.sum(0), 1e-8)
    out = []
    for k in range(K):
        m = lab == k
        if m.sum() < min_group:
            out.append(glob)                      # too few cells to estimate a profile
            continue
        out.append((psi_obs[m] * cov[m]).sum(0) / np.maximum(covf[m].sum(0), 1e-8))
    return np.clip(np.vstack(out), 1e-3, 1 - 1e-3)


def main():
    out_csv = sys.argv[1]
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    device = sys.argv[3] if len(sys.argv) > 3 else "cuda"
    from leafletfa.minibatch import LeafletFAMinibatch

    rows = []
    for tag, label in LEVELS:
        f = f"{SIM}/sim_heart_K5_{tag}_eff0.25_frac0.3_seed{DATA_SEED}.h5ad"
        if not os.path.exists(f):
            print(f"{label}: missing -- skipped", flush=True)
            continue
        a = ad.read_h5ad(f)
        g = lambda k: np.asarray(a.layers[k].todense() if hasattr(a.layers[k], "todense")
                                 else a.layers[k], dtype=np.float32)
        junc, clus = g("cell_by_junction_matrix"), g("cell_by_cluster_matrix")
        phi_true = np.asarray(a.obsm["true_phi"], float)
        K = phi_true.shape[1]
        psi0 = deterministic_psi_init(junc, clus, K)
        print(f"\n=== {label} ===  {junc.shape}, K={K}", flush=True)

        for init_name in ("random", "deterministic"):
            fits = {}
            for s in range(n_seeds):
                m = LeafletFAMinibatch(junc, clus, K=K, junc_specific_prior=True,
                                       likelihood="betabinomial",
                                       psi_init=(psi0 if init_name == "deterministic" else None))
                m.fit(num_steps=5000, subsample_size=2048, lr=0.05, gamma=0.1, seed=s,
                      log_every=2000, verbose=False, patience=25, min_delta_rel=1e-6)
                fits[s] = m.phi_mean().cpu().numpy()
            rec = [matched_r(phi_true.T, fits[s].T) for s in fits]
            for x, y in itertools.combinations(sorted(fits), 2):
                sp = cca(fits[x], fits[y])
                rows.append(dict(dispersion=tag, dispersion_label=label, init=init_name,
                                 K=K, seed_a=x, seed_b=y,
                                 mean_canon_corr=float(sp.mean()),
                                 effective_K=int((sp > 0.9).sum()),
                                 frac=float((sp > 0.9).sum()) / (K - 1),
                                 phi_recovery=float(np.mean(rec))))
            sub = [r for r in rows if r["dispersion"] == tag and r["init"] == init_name]
            print(f"  {init_name:>14}: mean canonical corr "
                  f"{np.mean([r['mean_canon_corr'] for r in sub]):.3f}   "
                  f"effective K {np.mean([r['effective_K'] for r in sub]):.1f}/{K}   "
                  f"phi recovery {np.mean(rec):.3f}", flush=True)

    if not rows:
        sys.exit("no simulated datasets found")
    d = pd.DataFrame(rows)
    d.to_csv(out_csv, index=False)

    print("\n=== how much of the seed variance is initialisation? ===", flush=True)
    piv = d.groupby(["dispersion_label", "init"])[["mean_canon_corr", "phi_recovery"]].mean()
    print(piv.round(3).to_string(), flush=True)
    for lab in d.dispersion_label.unique():
        s = d[d.dispersion_label == lab]
        r_ = s[s.init == "random"]["mean_canon_corr"].mean()
        det = s[s.init == "deterministic"]["mean_canon_corr"].mean()
        acc_r = s[s.init == "random"]["phi_recovery"].mean()
        acc_d = s[s.init == "deterministic"]["phi_recovery"].mean()
        print(f"\n  {lab}: consistency {r_:.3f} -> {det:.3f} ({det-r_:+.3f}); "
              f"accuracy {acc_r:.3f} -> {acc_d:.3f} ({acc_d-acc_r:+.3f})", flush=True)
        if det - r_ > 0.05 and acc_d >= acc_r - 0.02:
            print("    deterministic init buys consistency at no accuracy cost", flush=True)
        elif det - r_ > 0.05:
            print("    deterministic init buys consistency BUT costs accuracy -- agreeing on a worse "
                  "answer is not an improvement", flush=True)
        else:
            print("    initialisation is not the main source; the residual is minibatch sampling "
                  "and the information ceiling", flush=True)
    print(f"\nwrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
