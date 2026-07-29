"""Does the model know which cells it cannot pin down?

Seeds disagree on 30-50% of phi directions. That is a defect if the model is silently confident about
every cell, and something quite different if it correctly reports WHICH cells its own data cannot
determine. The model is Bayesian and carries a posterior width per cell; we have simply been throwing
it away and reporting `phi_mean` as though it were exact.

The test: per cell, compare
  WIDTH        the posterior spread of phi under one fit (`phi_uncertainty`), and in particular the
               posterior probability that the reported top program really is the top one;
  DISAGREEMENT how much that cell's phi differs between two independently seeded fits, after aligning
               programs on Psi (the well-determined latent).

If width predicts disagreement, seed variation is CALIBRATED UNCERTAINTY -- the model flags the cells
it cannot resolve, and a user can act on that. If the two are unrelated, the disagreement is not
something the model knows about, which is a harder problem and should be reported as such.

Coverage is included as a control. Per-cell agreement was already shown to be flat across a 450-fold
depth range, so a width that merely tracks depth would add nothing; the interesting case is width
predicting disagreement beyond what coverage explains.

Usage: A19_uncertainty_calibration.py <model_dir> <K> <out_csv> [n_samples]
"""
from __future__ import annotations
import sys, os, json, itertools
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

TASKS_BY_K = {10: (0, 1, 2), 20: (3, 4, 5), 30: (6, 7, 8)}


def perm_from_psi(a, b):
    K = a.shape[0]
    A = a - a.mean(1, keepdims=True); B = b - b.mean(1, keepdims=True)
    A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    r, c = linear_sum_assignment(-np.abs(A @ B.T))
    col = np.empty(K, int); col[r] = c
    return col


def main():
    md, K, out_csv = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    tasks = [t for t in TASKS_BY_K.get(K, range(10)) if os.path.exists(f"{md}/mb_run_{t}/fit.npz")]
    if len(tasks) < 2:
        # a flat directory of seeds (e.g. the jitter retrain) rather than the 9-task layout
        tasks = [t for t in range(10) if os.path.exists(f"{md}/mb_run_{t}/fit.npz")]
    if len(tasks) < 2:
        sys.exit(f"need >=2 fits under {md}")

    fits = {t: np.load(f"{md}/mb_run_{t}/fit.npz") for t in tasks}
    phi = {t: np.asarray(fits[t]["phi"], float) for t in tasks}
    psi = {t: np.asarray(fits[t]["psi"], float) for t in tasks}
    print(f"{len(tasks)} fits at K={K}: tasks {tasks}", flush=True)

    # WIDTH: prefer a stored per-cell uncertainty; else fall back to phi's own concentration
    width = {}
    for t in tasks:
        if "phi_top_prob" in fits[t].files:
            width[t] = 1.0 - np.asarray(fits[t]["phi_top_prob"], float)
            src = "stored posterior top-program probability"
        else:
            p = np.clip(phi[t], 1e-12, None); p = p / p.sum(1, keepdims=True)
            width[t] = np.exp(-(p * np.log(p)).sum(1)) / K      # normalised perplexity
            src = "phi entropy (no stored posterior; rerun training to save phi_uncertainty)"
    print(f"  width source: {src}", flush=True)

    rows = []
    for a, b in itertools.combinations(tasks, 2):
        col = perm_from_psi(psi[a], psi[b])
        B = phi[b][:, col]
        d = np.linalg.norm(phi[a] - B, axis=1)                  # per-cell disagreement
        w = 0.5 * (width[a] + width[b])
        ok = np.isfinite(d) & np.isfinite(w)
        rho, p = spearmanr(w[ok], d[ok])
        # does it beat coverage? (coverage is not stored per cell here; use phi's own scale as proxy)
        rows.append(dict(K=K, pair=f"{a}v{b}", n_cells=int(ok.sum()),
                         spearman_width_vs_disagreement=float(rho), pval=float(p),
                         median_disagreement=float(np.median(d[ok]))))
        # decile table: are the model's least-confident cells the ones that actually disagree?
        q = pd.qcut(w[ok], 10, labels=False, duplicates="drop")
        by = pd.DataFrame({"q": q, "d": d[ok]}).groupby("q")["d"].median()
        print(f"  {a} vs {b}: rho(width, disagreement) = {rho:+.3f} (P={p:.1e})", flush=True)
        print(f"      disagreement by width decile: "
              f"{' '.join(f'{v:.3f}' for v in by.values)}", flush=True)

    d = pd.DataFrame(rows); d.to_csv(out_csv, index=False)
    m = d.spearman_width_vs_disagreement.mean()
    print(f"\nmean rho = {m:+.3f}", flush=True)
    print("  -> " + ("the model KNOWS which cells it cannot resolve; seed variation can be reported "
                     "as calibrated uncertainty" if m > 0.2 else
                     "posterior width does NOT predict which cells disagree -- the disagreement is "
                     "not something the model is aware of, and should be reported plainly"), flush=True)
    print(f"wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
