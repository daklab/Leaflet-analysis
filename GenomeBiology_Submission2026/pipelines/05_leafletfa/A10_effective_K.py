"""How many of the K latent dimensions are actually determined by the data?

A6 reports that per-cell loadings agree only at r = 0.65-0.87 between seeds while the program profiles
Psi agree at 0.97, and attributes the gap to sparse per-cell evidence. That explanation turns out to
be wrong, or at least incomplete: per-cell agreement does not improve with sequencing depth (median r
by depth quintile is flat), so it is not that individual cells lack reads.

The actual structure is visible only with a rotation-invariant comparison. Matching programs one-to-one
assumes each program in one fit corresponds to exactly one program in another; if two fits divide the
same variation differently, that assumption fails and the disagreement it reports is an artefact of
the matching. Canonical correlations between the two phi matrices ask the question the matching cannot:
do these fits span the same subspace, however they happen to have divided it up?

At K=10 the answer has a sharp cliff -- roughly the first five or six canonical correlations are ~1.0
and the rest fall to near-orthogonal. So about half the latent dimensions are reproducible and the
rest are seed-specific: the model is over-parameterised relative to what the atlas determines. The
count above a threshold is therefore a usable EFFECTIVE K, and it is the honest number to report
alongside the nominal one.

This requires phi for the SAME cells under different seeds, which is what the held-out E-step caches
provide.

Usage: A10_effective_K.py <model_dir> <out_csv> [threshold]
"""
from __future__ import annotations
import sys, os, json, itertools
import numpy as np, pandas as pd

TASKS_BY_K = {10: (0, 1, 2), 20: (3, 4, 5), 30: (6, 7, 8)}


def phi_of(model_dir, task):
    for p in (f"{model_dir}/A7_phi_task{task}.npz", f"{model_dir}/A5_repro_K10_phi_cache.npz",
              f"{model_dir}/A7full_phi_task{task}.npz"):
        if os.path.exists(p):
            z = np.load(p)
            if f"phi_{task}" in z.files:
                return z[f"phi_{task}"]
    return None


def canonical_corrs(A, B):
    """Singular values of Qa'Qb -- the cosines of the principal angles between the two column spaces.

    Invariant to any invertible linear map of either side, so it is blind to both the ordering of the
    programs and to any redivision of the same variation among them.

    phi rows lie on the simplex, so K programs span only K-1 directions -- the K-th singular value of
    the centred phi is exactly zero. The achievable maximum is therefore K-1, and a fraction reported
    against K understates it (e.g. 4 of 5 at zero overdispersion is 100% reproduction, not 80%).
    """
    A = np.asarray(A, float) - np.asarray(A, float).mean(0)
    B = np.asarray(B, float) - np.asarray(B, float).mean(0)
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    return np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0, 1)


def main():
    model_dir, out_csv = sys.argv[1:3]
    thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
    rows = []
    for K, tasks in TASKS_BY_K.items():
        P = {t: phi_of(model_dir, t) for t in tasks}
        P = {t: v for t, v in P.items() if v is not None}
        if len(P) < 2:
            print(f"K={K}: fewer than two phi caches available", flush=True)
            continue
        print(f"\n=== K = {K} ===", flush=True)
        for a, b in itertools.combinations(sorted(P), 2):
            if P[a].shape[0] != P[b].shape[0]:
                print(f"  seed pair {a},{b}: phi on different cell sets -- skipped", flush=True)
                continue
            s = canonical_corrs(P[a], P[b])
            n = int((s > thr).sum())
            sa = json.load(open(f"{model_dir}/mb_run_{a}/result.json"))["seed"]
            sb = json.load(open(f"{model_dir}/mb_run_{b}/result.json"))["seed"]
            rows.append(dict(K=K, seed_a=sa, seed_b=sb, effective_K=n, frac=n / (K - 1),
                             **{f"cc{i+1}": float(v) for i, v in enumerate(s)}))
            print(f"  seed{sa} vs seed{sb}: {np.round(s, 3)}", flush=True)
            print(f"{'':16} effective K = {n}/{K - 1} achievable above {thr}", flush=True)

    if not rows:
        sys.exit("no comparable phi caches found")
    d = pd.DataFrame(rows)
    d.to_csv(out_csv, index=False)
    print("\n=== effective vs nominal K ===", flush=True)
    for K, g in d.groupby("K"):
        print(f"  K = {K:>2}: effective {g.effective_K.mean():.1f} "
              f"({g.frac.mean():.0%} of the programs reproduce across seeds)", flush=True)
    print(f"\nwrote {out_csv}", flush=True)
    print("A dimension with a low canonical correlation is not noise in the usual sense -- it is a "
          "direction the data does not determine, so each fit picks its own. Reporting effective K "
          "alongside nominal K is the honest summary.", flush=True)


if __name__ == "__main__":
    main()
