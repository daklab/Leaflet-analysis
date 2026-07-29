"""Is the seed-to-seed irreproducibility of phi an information limit, or a model defect?

On the atlas only ~40-50% of phi directions agree between independently initialised fits. Two causes
were tested and rejected (sparse coverage -- agreement is flat across a 450-fold depth range; program
collinearity -- centred programs are as decorrelated as centering permits, exactly -1/(K-1)). The
surviving explanation is the learned overdispersion: bb_conc ~ 1.97, and for a Beta-Binomial the
precision of a rate estimated from n reads saturates at c+1, so each junction yields at most ~3 reads'
worth of information however deep the cell is sequenced. If that is right, per-cell phi is
underdetermined for reasons no modelling change can fix.

That claim is testable, because the simulator controls overdispersion directly and knows the truth.
Fitting the SAME simulated design at three dispersion levels and asking how much independently seeded
fits agree separates the two possibilities:

  bb-conc = inf (Binomial)  no overdispersion  -> if phi is reproducible here, the machinery is fine
  bb-conc = 20              mild
  bb-conc = 2               matches the atlas   -> if phi collapses to ~50% here, the atlas number is
                                                   an information limit, not a defect

Simulated data is also free of model misspecification, which matters because bb_conc is LEARNED on the
atlas: a low value there could mean genuine overdispersion or the Beta-Binomial soaking up
unmodelled variation. Here the generating dispersion is known.

Reported per level: agreement between seeds (effective K and the mean canonical correlation, both
permutation- and rotation-invariant), and recovery against the known truth, so a level where phi is
irreproducible can still be checked for whether Psi was recovered.

Usage: A13_overdispersion_vs_reproducibility.py <out_csv> [n_seeds] [device]
"""
from __future__ import annotations
import sys, os, itertools
import numpy as np, pandas as pd, anndata as ad
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
SIM = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION"

# same effect size and differential fraction at every level; only the dispersion changes
LEVELS = [("binom", "no overdispersion (Binomial)"),
          ("bb20", "mild (c = 20)"),
          ("bb2", "atlas-like (c = 2)")]
DATA_SEED = 0          # one simulated dataset per level; MODEL seeds are what vary


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
    ri, ci = linear_sum_assignment(-np.nan_to_num(C))
    return float(np.median(C[ri, ci]))


def main():
    out_csv = sys.argv[1]
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    device = sys.argv[3] if len(sys.argv) > 3 else "cuda"
    from leafletfa.minibatch import LeafletFAMinibatch

    rows = []
    for tag, label in LEVELS:
        f = f"{SIM}/sim_heart_K5_{tag}_eff0.25_frac0.3_seed{DATA_SEED}.h5ad"
        if not os.path.exists(f):
            print(f"{label}: {f} missing -- skipped", flush=True)
            continue
        a = ad.read_h5ad(f)
        junc = np.asarray(a.layers["cell_by_junction_matrix"].todense()
                          if hasattr(a.layers["cell_by_junction_matrix"], "todense")
                          else a.layers["cell_by_junction_matrix"], dtype=np.float32)
        clus = np.asarray(a.layers["cell_by_cluster_matrix"].todense()
                          if hasattr(a.layers["cell_by_cluster_matrix"], "todense")
                          else a.layers["cell_by_cluster_matrix"], dtype=np.float32)
        # truth: true_phi is (cells, K) and directly comparable to the fitted phi, which is what this
        # test is about. true_psi_skip is per-ATSE (4,268) while fitted Psi is per-junction (12,804),
        # so it is not compared here.
        phi_true = np.asarray(a.obsm["true_phi"], float)
        K = phi_true.shape[1]
        print(f"\n=== {label} === {f.split('/')[-1]}  {junc.shape}, K={K}", flush=True)

        fits = {}
        for s in range(n_seeds):
            m = LeafletFAMinibatch(junc, clus, K=K, junc_specific_prior=True,
                                   likelihood="betabinomial")
            m.fit(num_steps=5000, subsample_size=2048, lr=0.05, gamma=0.1, seed=s,
                  log_every=1000, verbose=False, patience=25, min_delta_rel=1e-6)
            fits[s] = dict(phi=m.phi_mean().cpu().numpy(), psi=m.psi_mean().cpu().numpy())
            # accuracy against the KNOWN phi, after matching programs: distinguishes "seeds disagree
            # because neither is right" from "seeds disagree but both recover the truth"
            rec = matched_r(phi_true.T, fits[s]["phi"].T)
            print(f"  seed {s}: phi recovery vs truth = {rec:.3f}", flush=True)
            fits[s]["recovery"] = rec

        eff, mn = [], []
        for x, y in itertools.combinations(sorted(fits), 2):
            s_ = cca(fits[x]["phi"], fits[y]["phi"])
            eff.append(int((s_ > 0.9).sum())); mn.append(float(s_.mean()))
        rows.append(dict(level=tag, label=label, K=K, n_seeds=n_seeds,
                         effective_K=float(np.mean(eff)), frac=float(np.mean(eff)) / K,
                         mean_canon_corr=float(np.mean(mn)),
                         phi_recovery=float(np.nanmean([v["recovery"] for v in fits.values()]))))
        print(f"  -> effective K = {np.mean(eff):.1f}/{K} ({np.mean(eff)/K:.0%}), "
              f"mean canonical corr = {np.mean(mn):.3f}", flush=True)

    if not rows:
        sys.exit("no simulated datasets found")
    d = pd.DataFrame(rows)
    d.to_csv(out_csv, index=False)
    print("\n=== dose-response: does overdispersion cost phi reproducibility? ===", flush=True)
    print(d[["label", "effective_K", "frac", "mean_canon_corr", "phi_recovery"]]
          .round(3).to_string(index=False), flush=True)
    if len(d) >= 2:
        hi = d[d.level == "binom"]; lo = d[d.level == "bb2"]
        if len(hi) and len(lo):
            drop = hi.frac.iloc[0] - lo.frac.iloc[0]
            print(f"\n  Binomial -> c=2 changes the reproducible fraction by {-drop:+.0%}.", flush=True)
            if drop > 0.2:
                print("  => overdispersion is the limit. The atlas figure of ~50% is an information"
                      "\n     ceiling set by the data, not a defect of the model or the optimiser,"
                      "\n     and no modelling change will recover it.", flush=True)
            else:
                print("  => overdispersion does NOT explain it. Something in the model or the"
                      "\n     optimiser is losing information, and it is worth hunting.", flush=True)
    print(f"\nwrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
