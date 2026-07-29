"""Merge several seeds into one consensus phi, instead of discarding what disagrees.

Only ~30-50% of phi directions clear a 0.9 canonical-correlation threshold between independently
seeded fits, and that ceiling is set by the assay's overdispersion rather than by anything fixable
(A13: same simulated design at c=inf/20/2 gives 80%/67%/33% reproducible, with recovery against KNOWN
truth degrading in step; A11: doubling the training budget and dropping the junction-specific prior
both did nothing).

The first instinct is to keep the directions above the threshold and drop the rest. That is wrong, and
measurably so. The discarded directions are not noise -- at K=30 the ones below the cut still correlate
at 0.87, 0.83, 0.82, 0.80, 0.74, 0.72, 0.68, 0.62, 0.55 between seeds. Only the last handful are near
zero. Dropping them throws away real shared structure.

MERGING RECOVERS IT. If each fit is (shared structure + seed-specific error), averaging aligned fits
reinforces the shared part and cancels the error -- the ordinary posterior-mean argument, and the
degeneracy is exactly the regime where it pays. Measured by leave-one-out on the atlas, averaging just
TWO aligned fits predicts a held-out third better than either fit alone, at every K:

    K=10   single 0.6620 -> merged 0.6717   (+0.0097)
    K=20   single 0.6701 -> merged 0.6922   (+0.0221)
    K=30   single 0.6521 -> merged 0.6679   (+0.0158)

With independent per-seed error the gain should scale as sqrt(N), so more seeds is the cheap lever;
this script takes as many as it is given and reports the leave-one-out gain so the benefit is measured
rather than assumed.

Programs are exchangeable, so fits must be ALIGNED before averaging: fit A's program 7 may be fit B's
program 22, and averaging column 7 with column 7 would blend unrelated programs.

The permutation is found by Hungarian matching on PSI, not phi. Psi is the well-determined latent
(reproduces at r=0.97 between seeds) while phi is the badly-determined one (0.65-0.87), so matching on
Psi gives a far more reliable correspondence. The SAME permutation is then applied to every
program-indexed latent -- Psi, phi, pi, and the junction-specific a/b if present -- so the merged model
stays internally consistent. Globals without a program axis (bb_conc, dir_conc) are averaged directly.

The output is a complete consensus MODEL, not just a consensus representation: the calibration analysis
needs a merged Psi (the dictionary) at least as much as a merged phi.

The threshold-based consensus dimension is still reported, as a DIAGNOSTIC of how much of the
representation is strongly determined -- not as a filter to apply.

Usage: A14_consensus_subspace.py <model_dir> <K> <out_npz> [threshold]
"""
from __future__ import annotations
import sys, os, json, itertools
import numpy as np
from scipy.optimize import linear_sum_assignment

TASKS_BY_K = {10: (0, 1, 2), 20: (3, 4, 5), 30: (6, 7, 8)}


def cca_full(A, B):
    """Canonical correlations plus the directions, in each input's own coordinates."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    Qa, Ra = np.linalg.qr(Ac)
    Qb, Rb = np.linalg.qr(Bc)
    U, s, Vt = np.linalg.svd(Qa.T @ Qb)
    # columns of Wa map A's centred coordinates onto the canonical directions
    Wa = np.linalg.solve(Ra, U)
    Wb = np.linalg.solve(Rb, Vt.T)
    return np.clip(s, 0, 1), Wa, Wb


def permutation_from_psi(psi_ref, psi_other):
    """Program correspondence between two fits, from PSI.

    Psi is the well-determined latent (r~0.97 between seeds); phi is not (0.65-0.87). Matching on the
    reliable one and applying the result to everything is more trustworthy than matching on phi.
    Returns `col` such that other[..., col] is aligned to ref's program order.
    """
    K = psi_ref.shape[0]
    C = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            a, b = psi_ref[i], psi_other[j]
            C[i, j] = 0.0 if (a.std() < 1e-12 or b.std() < 1e-12) else abs(np.corrcoef(a, b)[0, 1])
    r, c = linear_sum_assignment(-np.nan_to_num(C))
    col = np.empty(K, dtype=int)
    col[r] = c
    return col, float(np.median(C[r, c]))


def align_to(ref, other):
    """Legacy phi-column matching, kept only for the leave-one-out diagnostic."""
    K = ref.shape[1]
    C = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            a, b = ref[:, i], other[:, j]
            C[i, j] = 0.0 if (a.std() < 1e-12 or b.std() < 1e-12) else abs(np.corrcoef(a, b)[0, 1])
    r, c = linear_sum_assignment(-np.nan_to_num(C))
    out = np.zeros_like(other)
    out[:, r] = other[:, c]
    return out


def subspace_agree(A, B):
    A = np.asarray(A, float) - np.asarray(A, float).mean(0)
    B = np.asarray(B, float) - np.asarray(B, float).mean(0)
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    return np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0, 1)


def main():
    md, K, out_npz = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    thr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.9
    tasks = [t for t in TASKS_BY_K[K] if os.path.exists(f"{md}/mb_run_{t}/fit.npz")]
    if len(tasks) < 3:
        sys.exit(f"need 3 fits at K={K}, found {len(tasks)}")

    phi = {t: np.asarray(np.load(f"{md}/mb_run_{t}/fit.npz")["phi"], float) for t in tasks}
    seed = {t: json.load(open(f"{md}/mb_run_{t}/result.json"))["seed"] for t in tasks}

    # medoid: the least idiosyncratic fit, by mean subspace agreement with the others
    cen = {t: float(np.mean([cca_full(phi[t], phi[o])[0].mean() for o in tasks if o != t]))
           for t in tasks}
    ref = max(cen, key=cen.get)
    others = [t for t in tasks if t != ref]
    print(f"K={K}: medoid is seed {seed[ref]} (task {ref}, centrality {cen[ref]:.3f}); "
          f"comparing against seeds {[seed[o] for o in others]}\n", flush=True)

    # a direction counts as consensus only if it clears the threshold against EVERY other seed
    keep_mask, Ws = None, []
    for o in others:
        s, Wa, _ = cca_full(phi[ref], phi[o])
        print(f"  vs seed {seed[o]}: canonical corrs {np.round(s, 3)}", flush=True)
        print(f"{'':14} {int((s > thr).sum())}/{K} above {thr}", flush=True)
        m = s > thr
        keep_mask = m if keep_mask is None else (keep_mask & m)
        Ws.append(Wa)
    d = int(keep_mask.sum())
    print(f"\n  consensus dimension: {d}/{K} directions agree with EVERY seed "
          f"({d/K:.0%})", flush=True)
    if d == 0:
        sys.exit("no directions survive -- nothing to ship at this threshold")

    print(f"  (reported as a diagnostic of how much is strongly determined -- NOT used to filter)",
          flush=True)

    # ---- the deliverable: merge, don't discard ------------------------------------------------
    # Programs are exchangeable, so align every fit into the medoid's frame before averaging.
    # Averaging unaligned fits would blur distinct programs together.
    print("\n=== consensus by merging (all K programs retained) ===", flush=True)
    # one permutation per fit, found on Psi, applied to EVERY program-indexed latent
    fits = {t: dict(np.load(f"{md}/mb_run_{t}/fit.npz")) for t in tasks}
    psi_ref = np.asarray(fits[ref]["psi"], float)
    perms = {ref: (np.arange(K), 1.0)}
    for o in others:
        perms[o] = permutation_from_psi(psi_ref, np.asarray(fits[o]["psi"], float))
        print(f"  seed {seed[o]} -> medoid frame: median matched Psi r = {perms[o][1]:.3f}", flush=True)

    def stack(key, axis):
        """Permute `key` onto the medoid's program order and average. None if it has no program axis.

        Not every saved latent is program-indexed -- alpha_pi is a scalar, a/b are per-junction -- so
        the program axis is verified rather than assumed before permuting.
        """
        out = []
        for t in tasks:
            if key not in fits[t]:
                return None
            v = np.asarray(fits[t][key], float)
            if v.ndim <= axis or v.shape[axis] != K:
                return None                      # no program axis here; handled by direct averaging
            col = perms[t][0]
            out.append(np.take(v, col, axis=axis))
        return np.mean(out, axis=0)

    merged = {}
    merged["psi"] = stack("psi", 0)                    # (K, J)
    merged["phi"] = stack("phi", 1)                    # (N, K)
    merged["pi"] = stack("pi", 0)                      # (K,)
    # anything without a program axis -- alpha_pi (scalar), junction-specific a/b, the globals --
    # is averaged directly, with no permutation
    for k in ("alpha_pi", "a", "b", "dir_conc", "bb_conc"):
        vs = [np.asarray(fits[t][k], float) for t in tasks if k in fits[t]]
        if vs and all(v.shape == vs[0].shape for v in vs):
            merged[k] = np.mean(vs, axis=0)

    consensus = np.clip(merged["phi"], 1e-12, None)
    consensus = consensus / consensus.sum(1, keepdims=True)      # numerical hygiene after clipping
    merged["phi"] = consensus
    print(f"  merged latents: {', '.join(f'{k}{tuple(np.shape(v))}' for k, v in merged.items())}",
          flush=True)

    # leave-one-out: is merging actually better than a single fit here? measured, not assumed
    singles, merges = [], []
    for held in tasks:
        rest = [t for t in tasks if t != held]
        r0 = rest[0]
        avg = np.mean([phi[r0]] + [align_to(phi[r0], phi[t]) for t in rest[1:]], axis=0)
        singles.append(subspace_agree(phi[r0], phi[held]).mean())
        merges.append(subspace_agree(avg, phi[held]).mean())
    gain = float(np.mean(merges) - np.mean(singles))
    print(f"  leave-one-out agreement with a held-out seed:", flush=True)
    print(f"    single fit  {np.mean(singles):.4f}", flush=True)
    print(f"    merged      {np.mean(merges):.4f}   ({gain:+.4f})", flush=True)
    print(f"  -> merging {'HELPS' if gain > 0 else 'does not help'} with {len(tasks)} seeds; "
          f"expect roughly sqrt(N) scaling, so more seeds is the cheap lever", flush=True)

    Z = consensus

    np.savez_compressed(out_npz, **{f"consensus_{k}": v for k, v in merged.items()},
                        keep_mask=keep_mask, K=K,
                        strongly_determined_dim=d, n_seeds=len(tasks),
                        loo_single=float(np.mean(singles)), loo_merged=float(np.mean(merges)),
                        ref_task=ref, ref_seed=seed[ref],
                        tasks=np.array(tasks), seeds=np.array([seed[t] for t in tasks]),
                        centrality=np.array([cen[t] for t in tasks]), threshold=thr)
    print(f"\nwrote {out_npz}", flush=True)
    print(f"  a COMPLETE consensus model, aligned to seed {seed[ref]}'s frame and averaged over "
          f"{len(tasks)} fits: consensus_psi, consensus_phi, consensus_pi and the globals.", flush=True)
    print(f"  Use it in place of any single fit's phi. Report K={K} with the caveat that "
          f"{d} directions are strongly determined ({d/K:.0%}).", flush=True)


if __name__ == "__main__":
    main()
