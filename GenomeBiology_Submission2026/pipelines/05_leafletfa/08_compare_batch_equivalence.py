#!/usr/bin/env python
"""
08_compare_batch_equivalence.py — do full-batch and minibatch LeafletFA agree?

Compares two run_* directories that differ ONLY in batch_size (the atlas equivalence
test: run_0 full-batch vs run_1 minibatch). LeafletFA programs are identifiable only up
to permutation, so we Hungarian-match programs by psi correlation, then measure agreement
of the matched programs (psi) and the per-cell loadings (phi). Handles the two runs
pruning to different final K.

Metrics:
  - psi program matching: K_a×K_b correlation → optimal assignment; report matched-pair
    psi Pearson r (median / min / per-program) = "are the learned splicing programs the same?"
  - phi cell-loading agreement: reorder run B's phi columns to A's, then per-cell cosine +
    per-program Pearson over cells = "do cells get the same program mix?"
  - top-program argmax agreement: fraction of cells whose dominant program matches.
  - pi (global program weights) and best_elbo comparison.

Usage:
    python 08_compare_batch_equivalence.py <run_dir_A> <run_dir_B> [--out-csv <path>]
        [--label-a full] [--label-b mini]
"""

import argparse
import gzip
import os
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr


def load_model(run_dir):
    data = {}
    with gzip.open(os.path.join(run_dir, "leafletfa_model.pkl.gz"), "rb") as f:
        while True:
            try:
                data.update(pickle.load(f))
            except EOFError:
                break
    return data


def get_psi(md):
    """Deterministic psi (K×J): sigmoid(psis_loc) when available (matches eval), else md['psi']."""
    psi_pt = np.asarray(md["psi"], dtype=np.float64)
    K = psi_pt.shape[0]
    if "psis_loc" in md and np.asarray(md["psis_loc"]).shape[0] == K:
        loc = np.asarray(md["psis_loc"], dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-loc))
    return psi_pt


def corr_matrix(A, B):
    """Row-wise Pearson correlation matrix between A (Ka×J) and B (Kb×J)."""
    Az = (A - A.mean(1, keepdims=True))
    Bz = (B - B.mean(1, keepdims=True))
    Az /= (np.linalg.norm(Az, axis=1, keepdims=True) + 1e-12)
    Bz /= (np.linalg.norm(Bz, axis=1, keepdims=True) + 1e-12)
    return Az @ Bz.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir_a")
    ap.add_argument("run_dir_b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    ma, mb = load_model(args.run_dir_a), load_model(args.run_dir_b)
    psi_a, psi_b = get_psi(ma), get_psi(mb)
    Ka, Kb = psi_a.shape[0], psi_b.shape[0]
    J = psi_a.shape[1]
    assert psi_b.shape[1] == J, f"junction dim mismatch {J} vs {psi_b.shape[1]}"
    label_a, label_b = args.label_a, args.label_b

    print(f"{'='*70}")
    print(f"  {args.label_a}={os.path.basename(args.run_dir_a.rstrip('/'))}  "
          f"K={Ka}  batch={ma.get('_meta',{}).get('batch_size','?')}  "
          f"ELBO={ma.get('best_elbo', np.nan):.1f}")
    print(f"  {args.label_b}={os.path.basename(args.run_dir_b.rstrip('/'))}  "
          f"K={Kb}  batch={mb.get('_meta',{}).get('batch_size','?')}  "
          f"ELBO={mb.get('best_elbo', np.nan):.1f}")
    print(f"  junctions J={J:,}")
    if Ka != Kb:
        print(f"  ⚠ pruned to different K ({Ka} vs {Kb}) — matching min({Ka},{Kb})={min(Ka,Kb)} programs")
    print(f"{'='*70}")

    # ── program matching by psi correlation (Hungarian) ────────────────────────
    C = corr_matrix(psi_a, psi_b)                       # Ka×Kb
    row, col = linear_sum_assignment(-C)                # maximize total correlation
    matched_r = np.array([C[i, j] for i, j in zip(row, col)])
    order = np.argsort(-matched_r)
    print("\n[psi] matched-program Pearson r (sorted):")
    print("  " + "  ".join(f"{matched_r[o]:.3f}" for o in order))
    print(f"  median={np.median(matched_r):.3f}  min={matched_r.min():.3f}  "
          f"mean={matched_r.mean():.3f}  n_matched={len(matched_r)}")
    print(f"  programs with r>0.95: {(matched_r>0.95).sum()}/{len(matched_r)}  "
          f"| r>0.99: {(matched_r>0.99).sum()}/{len(matched_r)}")

    rows = [dict(metric="psi_match_r", program=int(a_i), matched_to=int(b_j), value=float(C[a_i, b_j]))
            for a_i, b_j in zip(row, col)]

    # ── EFFECT-SIZE SCALE: the metric the correlations above are blind to ──────
    # Pearson r, cosine and argmax are all SCALE-INVARIANT: they compare the *shape* of each
    # program across junctions. If minibatching shrinks every factor toward a common per-junction
    # mean, each program's shape survives (r ~ 0.99) while the factor-to-factor CONTRAST — which is
    # exactly the differential-splicing effect size beta — collapses. "Programs agree" is what you
    # would see even under 4x attenuation, so r cannot answer "are the effect sizes the right size?"
    #
    # beta_kj = psi_kj - mean_{k' != k} psi_k'j   (the DS effect size in differential_splicing.py)
    # Regress beta_B on beta_A and report the SLOPE:
    #     slope ~ 1  -> no attenuation; batching is fine
    #     slope << 1 -> run B's effect sizes are shrunk; its DS calls / FDR are not comparable
    def beta(psi):
        K_ = psi.shape[0]
        return psi - (psi.sum(0, keepdims=True) - psi) / (K_ - 1)

    bA = beta(psi_a[row]).ravel()
    bB = beta(psi_b[col]).ravel()
    slope = float(np.polyfit(bA, bB, 1)[0])
    r_beta = float(np.corrcoef(bA, bB)[0, 1])
    contrast_a = float(psi_a[row].std(axis=0).mean())
    contrast_b = float(psi_b[col].std(axis=0).mean())

    print(f"\n[EFFECT SIZE] beta = psi_k - mean(psi_other).  Scale-SENSITIVE, unlike r above.")
    print(f"  slope of beta_{label_b} on beta_{label_a} : {slope:.3f}   <-- the number that matters")
    print(f"  correlation of the two betas             : {r_beta:.3f}")
    print(f"  factor contrast  {label_a}: {contrast_a:.4f}   {label_b}: {contrast_b:.4f}"
          f"   ({contrast_a / max(contrast_b, 1e-12):.2f}x)")
    if slope < 0.8:
        print(f"  *** {label_b} ATTENUATES effect sizes by {1/max(slope,1e-12):.1f}x — its DS/FDR "
              f"results are NOT comparable to {label_a}. ***")
    elif slope > 1.25:
        print(f"  *** {label_b} INFLATES effect sizes by {slope:.1f}x. ***")
    else:
        print(f"  effect sizes agree (0.8 <= slope <= 1.25) — batching is not distorting beta.")

    rows += [dict(metric="beta_slope",      program=-1, matched_to=-1, value=slope),
             dict(metric="beta_r",          program=-1, matched_to=-1, value=r_beta),
             dict(metric=f"psi_contrast_{label_a}", program=-1, matched_to=-1, value=contrast_a),
             dict(metric=f"psi_contrast_{label_b}", program=-1, matched_to=-1, value=contrast_b)]

    # ── posterior width: is q(psi) calibrated to the full dataset or to one batch? ──
    sa, sb = ma.get("psis_scale"), mb.get("psis_scale")
    if sa is not None and sb is not None:
        wa, wb = float(np.asarray(sa).mean()), float(np.asarray(sb).mean())
        print(f"\n[POSTERIOR WIDTH] mean psis_scale  {label_a}: {wa:.4f}   {label_b}: {wb:.4f}"
              f"   ({wb/max(wa,1e-12):.2f}x)")
        print(f"  This is the posterior the DS/FDR analysis samples from. A chunked run whose "
              f"q(psi)\n  is re-initialised every chunk never accumulates evidence across chunks.")
        rows += [dict(metric=f"psis_scale_{label_a}", program=-1, matched_to=-1, value=wa),
                 dict(metric=f"psis_scale_{label_b}", program=-1, matched_to=-1, value=wb)]

    # ── phi cell-loading agreement (reorder B's programs to A's) ────────────────
    phi_a = np.asarray(ma.get("assign_post"), dtype=np.float64) if ma.get("assign_post") is not None else None
    phi_b = np.asarray(mb.get("assign_post"), dtype=np.float64) if mb.get("assign_post") is not None else None
    if phi_a is not None and phi_b is not None and phi_a.shape[0] == phi_b.shape[0]:
        N = phi_a.shape[0]
        phi_b_al = phi_b[:, col]           # B columns aligned to A's matched rows
        phi_a_al = phi_a[:, row]
        # per-cell cosine similarity between the two aligned loading vectors
        num = (phi_a_al * phi_b_al).sum(1)
        den = np.linalg.norm(phi_a_al, axis=1) * np.linalg.norm(phi_b_al, axis=1) + 1e-12
        cos = num / den
        # per-program Pearson across cells
        prog_r = [pearsonr(phi_a_al[:, k], phi_b_al[:, k])[0]
                  if phi_a_al[:, k].std() > 1e-9 and phi_b_al[:, k].std() > 1e-9 else np.nan
                  for k in range(phi_a_al.shape[1])]
        prog_r = np.array(prog_r, dtype=float)
        argmax_agree = (phi_a.argmax(1) == col[phi_b.argmax(1)] if Ka == Kb
                        else np.nan)  # only clean when same K & full permutation
        print(f"\n[phi] cells N={N:,}")
        print(f"  per-cell cosine sim:  median={np.nanmedian(cos):.3f}  "
              f"p5={np.nanpercentile(cos,5):.3f}  min={np.nanmin(cos):.3f}")
        print(f"  per-program Pearson over cells: median={np.nanmedian(prog_r):.3f}  "
              f"min={np.nanmin(prog_r):.3f}")
        if isinstance(argmax_agree, np.ndarray):
            print(f"  dominant-program agreement: {100*argmax_agree.mean():.1f}% of cells")
        for k, r in enumerate(prog_r):
            rows.append(dict(metric="phi_program_r", program=k, matched_to=int(col[k]), value=float(r)))
        rows.append(dict(metric="phi_cell_cosine_median", program=-1, matched_to=-1,
                         value=float(np.nanmedian(cos))))
    else:
        print("\n[phi] skipped (assign_post missing or cell-count mismatch)")

    # ── pi comparison ──────────────────────────────────────────────────────────
    pi_a, pi_b = ma.get("pi"), mb.get("pi")
    if pi_a is not None and pi_b is not None and Ka == Kb:
        pi_a = np.asarray(pi_a, float).ravel(); pi_b = np.asarray(pi_b, float).ravel()
        pi_b_al = pi_b[col]
        print(f"\n[pi] L1 distance (aligned) = {np.abs(pi_a - pi_b_al).sum():.4f}  "
              f"(0=identical, 2=disjoint)")

    print(f"\n{'='*70}")
    verdict = ("STRONG equivalence" if np.median(matched_r) > 0.95 else
               "PARTIAL — inspect low-r programs" if np.median(matched_r) > 0.8 else
               "WEAK — batching changed the learned programs")
    print(f"VERDICT: {verdict}  (median psi match r = {np.median(matched_r):.3f})")
    print(f"{'='*70}")

    if args.out_csv:
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print(f"→ {args.out_csv}")


if __name__ == "__main__":
    main()
