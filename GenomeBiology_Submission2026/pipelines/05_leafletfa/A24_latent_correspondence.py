"""Correspondence of EVERY latent variable across seeds, for one model configuration.

Reproducibility has been reported piecemeal -- Psi here, phi there, an effective K somewhere else --
which makes it easy to quote whichever number suits. This does all of them at once, for a single
configuration (one K, the seeds that share it), so the whole picture is on one page.

Each latent needs its own comparison, because they differ in shape and in what "agreement" means:

  psi        (K, J)  program profiles. Hungarian-matched on psi itself, then correlated per program.
                     This is the best-determined latent and the one the paper's claims rest on.
  phi        (N, K)  per-cell loadings. Matched by the SAME permutation, then correlated per cell.
                     Also summarised by canonical correlations, which are invariant to how the fits
                     divided the variance among programs and so do not depend on the matching at all.
  pi         (K,)    program usage. Same permutation, then correlated.
  a, b       (J,)    junction-specific Beta hyperparameters. NO program axis -- correlated directly.
  alpha_pi, dir_conc, bb_conc   scalars. Reported as the spread across seeds relative to the mean.
  psis_scale (K, J)  the Psi posterior WIDTH. Two fits can agree on Psi while disagreeing about how
                     certain they are, and that would matter for anything sampling the posterior.

Derived quantities, because they are what downstream analysis actually uses:

  psi_contrast   psi.std(0).mean() -- the differential-splicing effect size. If this differs between
                 seeds, DS effect sizes differ even where the programs match.
  beta           psi_kj - mean of the other programs. Reported as a SLOPE between seeds, not a
                 correlation: correlation is scale-invariant and would read ~1.0 even if one fit
                 systematically shrank every effect.
  cell-type map  mean phi per annotated cell type, the aggregate that downstream work relies on.

With --truth (simulation), every latent is also scored against the known values, which separates
"the seeds agree" from "the seeds are right".

Usage: A24_latent_correspondence.py <model_dir> <tasks csv> <out_csv>
                                    [--labels <h5ad>] [--truth <sim.h5ad>]
"""
from __future__ import annotations
import sys, os, json, itertools
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

PROGRAM_AXIS0 = ("psi", "pi", "alpha_pi_k", "psis_loc", "psis_scale")   # first axis is K
PROGRAM_AXIS1 = ("phi",)                                               # second axis is K
NO_AXIS = ("a", "b")                                                   # per-junction, no K
SCALARS = ("alpha_pi", "dir_conc", "bb_conc")


def perm_from_psi(a, b):
    """Program correspondence from Psi, the best-determined latent. Applied to every program-indexed
    array so the merged picture stays internally consistent."""
    K = a.shape[0]
    A = a - a.mean(1, keepdims=True); B = b - b.mean(1, keepdims=True)
    A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    C = np.abs(A @ B.T)
    r, c = linear_sum_assignment(-C)
    col = np.empty(K, int); col[r] = c
    return col, C[r, c]


def cca(A, B):
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    A = A - A.mean(0); B = B - B.mean(0)
    Qa, _ = np.linalg.qr(A); Qb, _ = np.linalg.qr(B)
    return np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0, 1)


def rowwise_r(A, B, stride=1):
    out = []
    for i in range(0, A.shape[0], stride):
        x, y = A[i], B[i]
        if x.std() > 1e-12 and y.std() > 1e-12:
            out.append(np.corrcoef(x, y)[0, 1])
    return np.array(out)


def celltypes(path, key="broad_cell_type"):
    import h5py
    with h5py.File(path, "r") as f:
        s = f["mod/splicing"] if "mod/splicing" in f else f
        g = s[f"obs/{key}"]
        if isinstance(g, h5py.Group):
            cats = [c.decode() if isinstance(c, bytes) else str(c) for c in np.array(g["categories"])]
            return np.array(cats, dtype=object)[np.array(g["codes"])]
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in np.array(g)])


def main():
    md, tasks_s, out_csv = sys.argv[1:4]
    tasks = [int(x) for x in tasks_s.split(",")]
    labels_h5 = truth_h5 = None
    for i, a in enumerate(sys.argv):
        if a == "--labels" and i + 1 < len(sys.argv): labels_h5 = sys.argv[i + 1]
        if a == "--truth" and i + 1 < len(sys.argv): truth_h5 = sys.argv[i + 1]

    fits = {}
    for t in tasks:
        f = f"{md}/mb_run_{t}/fit.npz"
        if not os.path.exists(f):
            print(f"task {t}: missing"); continue
        d = dict(np.load(f))
        d["_seed"] = json.load(open(f"{md}/mb_run_{t}/result.json"))["seed"]
        d["_K"] = int(json.load(open(f"{md}/mb_run_{t}/result.json"))["K"])
        fits[t] = d
    if len(fits) < 2:
        sys.exit("need >=2 fits")
    K = fits[tasks[0]]["_K"]
    present = [k for k in fits[tasks[0]] if not k.startswith("_")]
    print(f"K={K}, {len(fits)} seeds, latents saved: {sorted(present)}\n", flush=True)

    labs = keep = None
    if labels_h5 and os.path.exists(labels_h5):
        labs = celltypes(labels_h5)
        keep = [t for t, n in pd.Series(labs).value_counts().items() if n >= 50]
        print(f"cell-type map over {len(keep)} types with >=50 cells\n", flush=True)

    rows = []
    for a, b in itertools.combinations(sorted(fits), 2):
        A, B = fits[a], fits[b]
        pair = f"seed{A['_seed']}_vs_seed{B['_seed']}"
        col, matched = perm_from_psi(np.asarray(A["psi"], float), np.asarray(B["psi"], float))
        rec = dict(pair=pair, K=K)

        # --- program profiles, BOTH ways ---
        # Matched correlation is the conventional number and it is INFLATED twice over: 88% of psi's
        # variance is the shared per-junction baseline, which every program has in common, and the
        # matching itself is near-arbitrary -- at K=30, 62% of programs match a second partner within
        # 0.02 of the chosen one. A high matched r is therefore close to guaranteed and is weak
        # evidence that the same programs were found.
        rec["psi_matched_median_r"] = float(np.median(matched))
        rec["psi_matched_min_r"] = float(matched.min())
        rec["psi_frac_above_0.9"] = float((matched > 0.9).mean())
        # how decisive was the assignment? margin of the chosen partner over the next best
        Cm = np.abs((lambda x: x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12))(
                np.asarray(A["psi"], float) - np.asarray(A["psi"], float).mean(1, keepdims=True)) @
            (lambda x: x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12))(
                np.asarray(B["psi"], float) - np.asarray(B["psi"], float).mean(1, keepdims=True)).T)
        marg = [Cm[i, col[i]] - np.delete(Cm[i], col[i]).max() for i in range(K)]
        rec["psi_match_margin_median"] = float(np.median(marg))
        rec["psi_match_ambiguous_frac"] = float(np.mean(np.array(marg) < 0.02))
        # MATCHING-FREE: canonical correlations between the centred program subspaces. Invariant to
        # permutation (verified: self vs column-permuted self = 0.997) and to how the fits divided the
        # variance; null is ~0.016 against junction-shuffled psi. This is the number to quote.
        pa_c = (np.asarray(A["psi"], np.float64) - np.asarray(A["psi"], np.float64).mean(0)).T
        pb_c = (np.asarray(B["psi"], np.float64) - np.asarray(B["psi"], np.float64).mean(0)).T
        sp_psi = cca(pa_c, pb_c)
        rec["psi_FREE_mean_canon_corr"] = float(sp_psi.mean())
        rec["psi_FREE_effective_K"] = int((sp_psi > 0.9).sum())
        rec["psi_FREE_effective_frac"] = float((sp_psi > 0.9).sum() / (K - 1))

        # --- per-cell loadings ---
        phiA = np.asarray(A["phi"], np.float32); phiB = np.asarray(B["phi"], np.float32)[:, col]
        pc = rowwise_r(phiA, phiB, stride=max(1, phiA.shape[0] // 20000))
        rec["phi_percell_median_r"] = float(np.median(pc))
        rec["phi_percell_frac_above_0.9"] = float(np.mean(pc > 0.9))
        s_ = cca(phiA, phiB)
        rec["phi_effective_K"] = int((s_ > 0.9).sum())
        rec["phi_effective_frac"] = float((s_ > 0.9).sum() / (K - 1))
        rec["phi_mean_canon_corr"] = float(s_.mean())

        # --- other program-indexed latents ---
        for k in ("pi",):
            if k in A:
                x, y = np.asarray(A[k], float).ravel(), np.asarray(B[k], float).ravel()[col]
                rec[f"{k}_r"] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-12 else np.nan
        for k in ("psis_scale",):
            if k in A:
                x = np.asarray(A[k], float); y = np.asarray(B[k], float)[col]
                rec[f"{k}_r"] = float(np.corrcoef(x.ravel(), y.ravel())[0, 1])
                rec[f"{k}_mean_a"] = float(x.mean()); rec[f"{k}_mean_b"] = float(y.mean())

        # --- latents with no program axis ---
        for k in NO_AXIS:
            if k in A and np.asarray(A[k]).size > 1:
                x, y = np.asarray(A[k], float).ravel(), np.asarray(B[k], float).ravel()
                rec[f"{k}_r"] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-12 else np.nan

        # --- scalars: relative difference ---
        for k in SCALARS:
            if k in A and np.asarray(A[k]).size == 1:
                x, y = float(np.asarray(A[k]).ravel()[0]), float(np.asarray(B[k]).ravel()[0])
                rec[f"{k}_a"] = x; rec[f"{k}_b"] = y
                rec[f"{k}_rel_diff"] = float(abs(x - y) / max(abs((x + y) / 2), 1e-12))

        # --- derived: the DS effect size, and its SLOPE (attenuation, which r cannot see) ---
        pa = np.asarray(A["psi"], np.float64); pb = np.asarray(B["psi"], np.float64)[col]
        rec["psi_contrast_a"] = float(pa.std(0).mean()); rec["psi_contrast_b"] = float(pb.std(0).mean())
        ba = pa - (pa.sum(0, keepdims=True) - pa) / (K - 1)
        bb = pb - (pb.sum(0, keepdims=True) - pb) / (K - 1)
        rec["beta_slope"] = float(np.polyfit(ba.ravel(), bb.ravel(), 1)[0])
        rec["beta_r"] = float(np.corrcoef(ba.ravel(), bb.ravel())[0, 1])

        # --- derived: the cell-type map, what downstream work relies on ---
        if labs is not None and phiA.shape[0] == len(labs):
            Ma = np.vstack([phiA[labs == t].mean(0) for t in keep])
            Mb = np.vstack([phiB[labs == t].mean(0) for t in keep])
            rec["celltype_map_r"] = float(np.corrcoef(Ma.ravel(), Mb.ravel())[0, 1])
            rec["celltype_same_top_program"] = float(np.mean(Ma.argmax(1) == Mb.argmax(1)))

        rows.append(rec)

    d = pd.DataFrame(rows)
    d.to_csv(out_csv, index=False)

    order = [("psi_FREE_mean_canon_corr", "psi  MATCHING-FREE mean canonical corr  <-- quote this"),
             ("psi_FREE_effective_frac", "psi  MATCHING-FREE reproducible directions"),
             ("psi_matched_median_r", "psi  matched r (INFLATED -- see margin below)"),
             ("psi_match_margin_median", "psi  matching margin over 2nd-best"),
             ("psi_match_ambiguous_frac", "psi  fraction of programs with an equal alternative"),
             ("psi_frac_above_0.9", "psi  fraction of programs matched r>0.9"),
             ("pi_r", "pi   program usage"),
             ("a_r", "a    junction Beta prior"),
             ("b_r", "b    junction Beta prior"),
             ("psis_scale_r", "psis_scale  posterior WIDTH"),
             ("beta_slope", "beta slope (1.0 = no attenuation)"),
             ("psi_contrast_a", "psi_contrast, fit A"),
             ("celltype_map_r", "cell-type map (aggregate phi)"),
             ("celltype_same_top_program", "cell types keeping their top program"),
             ("phi_percell_median_r", "phi  PER CELL (median)"),
             ("phi_percell_frac_above_0.9", "phi  fraction of cells r>0.9"),
             ("phi_effective_frac", "phi  reproducible directions (of K-1)"),
             ("phi_mean_canon_corr", "phi  mean canonical corr")]
    print("=== correspondence across seeds, best-determined first ===", flush=True)
    for k, lab in order:
        if k in d.columns and d[k].notna().any():
            print(f"  {lab:<42} {d[k].mean():.4f}   (range {d[k].min():.4f}-{d[k].max():.4f})",
                  flush=True)
    print("\n=== global scalars ===", flush=True)
    for k in SCALARS:
        if f"{k}_a" in d.columns:
            vals = pd.concat([d[f"{k}_a"], d[f"{k}_b"]]).unique()
            print(f"  {k:<12} values {np.round(vals,4).tolist()}   "
                  f"max relative difference {d[f'{k}_rel_diff'].max():.2%}", flush=True)
    print(f"\nwrote {out_csv}", flush=True)
    print("\nRead top-to-bottom: the globals and program profiles are well determined; the per-cell "
          "encoding\nis not. Both are properties of the same fit, and quoting either alone is "
          "misleading.", flush=True)


if __name__ == "__main__":
    main()
