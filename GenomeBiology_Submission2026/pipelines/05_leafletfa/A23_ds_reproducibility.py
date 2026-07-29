"""Do independently seeded fits call the SAME differential splicing?

Every reproducibility number so far has been about latent variables -- phi correlations, canonical
directions, effective K. None of those is what the paper reports. The paper reports differentially
spliced junctions, so that is what should be shown to reproduce, and in the units a reader cares about.

For each fit, sample the Psi posterior (the guide's loc/scale, saved as `psis_loc`/`psis_scale`),
compute the per-(program, junction) effect size

    beta_kj = psi_kj - mean over the other programs of psi_k'j

which is the unweighted contrast `compute_psi_effect_size` actually implements, and take the posterior
probability that |beta| exceeds a threshold. Calls are then compared BETWEEN fits after matching
programs on Psi -- the well-determined latent, so the correspondence is reliable.

Reported per pair of seeds:

  JACCARD    overlap of the called sets. The headline: do two runs report the same junctions?
  RECALL     of one fit's calls, what fraction does the other also call? Asymmetric, so both directions
             are shown -- a fit that simply calls more will look good on one and bad on the other.
  SLOPE      of beta_b on beta_a, over junctions called by either. NOT the correlation: correlation is
             scale-invariant and would read ~1.0 even if one fit systematically shrank every effect.
             The slope is what detects attenuation.
  TOP-J      overlap restricted to each fit's strongest calls, where agreement should be easiest and a
             failure would be most damaging.

Union-of-either is used for the slope rather than intersection, so the comparison is not restricted to
the junctions where the two fits already agree.

Usage: A23_ds_reproducibility.py <model_dir> <tasks csv, e.g. 6,7,8> <out_csv>
                                 [n_samples] [delta] [prob_threshold]
"""
from __future__ import annotations
import sys, os, json, itertools
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment


def sample_psi(loc, scale, n, rng, delta, chunk=4000):
    """Posterior P(|beta| > delta) and mean beta per (program, junction).

    Streams over junctions: the full sample array would be n x K x J (~1 GB at n=100, K=30,
    J=79,768), and only two summaries of it are needed.
    """
    K, J = loc.shape
    P = np.zeros((K, J), dtype=np.float32)      # posterior P(|beta| > delta)
    B = np.zeros((K, J), dtype=np.float32)      # posterior mean beta
    for a in range(0, J, chunk):
        b = min(a + chunk, J)
        z = rng.standard_normal((n, K, b - a)).astype(np.float32)
        psi = 1.0 / (1.0 + np.exp(-(loc[:, a:b][None] + scale[:, a:b][None] * z)))
        beta = psi - (psi.sum(1, keepdims=True) - psi) / (K - 1)
        P[:, a:b] = np.mean(np.abs(beta) > delta, axis=0)
        B[:, a:b] = beta.mean(0)
    return P, B


def perm_from_psi(a, b):
    K = a.shape[0]
    A = a - a.mean(1, keepdims=True); Bm = b - b.mean(1, keepdims=True)
    A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    Bm /= np.maximum(np.linalg.norm(Bm, axis=1, keepdims=True), 1e-12)
    r, c = linear_sum_assignment(-np.abs(A @ Bm.T))
    col = np.empty(K, int); col[r] = c
    return col


def main():
    md, tasks_s, out_csv = sys.argv[1:4]
    tasks = [int(x) for x in tasks_s.split(",")]
    n_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    delta = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1
    pthr = float(sys.argv[6]) if len(sys.argv) > 6 else 0.9

    fits = {}
    for t in tasks:
        f = f"{md}/mb_run_{t}/fit.npz"
        if not os.path.exists(f):
            print(f"task {t}: missing"); continue
        d = np.load(f)
        if "psis_loc" not in d.files:
            print(f"task {t}: no psis_loc -- the Psi posterior was not saved"); continue
        rng = np.random.default_rng(0)
        P, B = sample_psi(np.asarray(d["psis_loc"], np.float32),
                          np.asarray(d["psis_scale"], np.float32), n_samples, rng, delta)
        fits[t] = dict(P=P, B=B, psi=np.asarray(d["psi"], np.float64),
                       seed=json.load(open(f"{md}/mb_run_{t}/result.json"))["seed"])
        print(f"  seed {fits[t]['seed']}: {int((P > pthr).sum()):,} calls at "
              f"P(|beta|>{delta}) > {pthr}", flush=True)

    if len(fits) < 2:
        sys.exit("need >=2 fits with a saved Psi posterior")

    rows = []
    for a, b in itertools.combinations(sorted(fits), 2):
        col = perm_from_psi(fits[a]["psi"], fits[b]["psi"])
        Pa, Pb = fits[a]["P"], fits[b]["P"][col]
        Ba, Bb = fits[a]["B"], fits[b]["B"][col]
        Sa, Sb = Pa > pthr, Pb > pthr
        inter, union = (Sa & Sb).sum(), (Sa | Sb).sum()
        m = Sa | Sb
        slope = float(np.polyfit(Ba[m], Bb[m], 1)[0]) if m.sum() > 10 else np.nan
        r = float(np.corrcoef(Ba[m], Bb[m])[0, 1]) if m.sum() > 10 else np.nan
        # strongest calls only
        def top(P, B, n=2000):
            idx = np.argsort(-np.abs(B).ravel())[:n]
            return set(idx.tolist())
        ta, tb = top(Pa, Ba), top(Pb, Bb)
        rows.append(dict(pair=f"seed{fits[a]['seed']}_vs_seed{fits[b]['seed']}",
                         n_a=int(Sa.sum()), n_b=int(Sb.sum()),
                         jaccard=float(inter / max(union, 1)),
                         recall_a_in_b=float((Sa & Sb).sum() / max(Sa.sum(), 1)),
                         recall_b_in_a=float((Sa & Sb).sum() / max(Sb.sum(), 1)),
                         beta_slope=slope, beta_r=r,
                         top2000_overlap=float(len(ta & tb) / 2000)))
        print(f"  {rows[-1]['pair']}: Jaccard {rows[-1]['jaccard']:.3f}  "
              f"slope {slope:.3f} (r {r:.3f})  top-2000 overlap "
              f"{rows[-1]['top2000_overlap']:.1%}", flush=True)

    d = pd.DataFrame(rows); d.to_csv(out_csv, index=False)
    print("\n=== summary ===", flush=True)
    print(f"  calls per fit: {d.n_a.tolist() + d.n_b.tolist()}", flush=True)
    print(f"  Jaccard {d.jaccard.mean():.3f}   effect-size slope {d.beta_slope.mean():.3f} "
          f"(1.0 = no attenuation)   top-2000 overlap {d.top2000_overlap.mean():.1%}", flush=True)
    print("\n  Slope, not correlation, is the attenuation test: correlation is scale-invariant and "
          "would read ~1.0\n  even if one fit shrank every effect systematically.", flush=True)
    print(f"\nwrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
