#!/usr/bin/env python
"""
Step 07d: is the frozen-Psi phi E-step CONVERGING, or just hitting its epoch cap?

WHY THIS EXISTS
---------------
Both 04e (imputation) and 07c (structure) infer held-out phi with the same frozen-globals E-step,
run for a fixed epoch budget (04e default 100, 07c default 300) with early stopping on `patience`.
But `min_delta` is ABSOLUTE (1e-3) against an ELBO of large magnitude, so the early-stopping test is
effectively inert and degenerates to "stop after `patience` epochs with no new global best". An
observed 04e run reported `epochs ran: 100 / 100` -- it never early-stopped, it ran out of budget.

If phi is truncated rather than converged, the epoch budget is a free knob sitting underneath EVERY
04e and 07c number, and any comparison against NMF (which is run to its own convergence) is unfair
in our disfavour. This probe settles it empirically instead of by assumption.

WHAT IT DOES
------------
Takes ONE chunk of held-out cells and re-runs the identical E-step at several epoch budgets, then
measures how much phi still moves between consecutive budgets:

  mean_L1      mean over cells of sum_k |phi_b(c,k) - phi_ref(c,k)|   (0 = identical; 2 = disjoint)
  max_L1       worst cell
  corr         flattened Pearson vs the largest budget
  argmax_agree fraction of cells whose dominant program is unchanged vs the largest budget
  perplexity   median effective number of programs per cell -- drifts if phi is still sharpening

Read it as: phi has converged at budget B if going from B to the next budget barely moves it. If
mean_L1 between the two largest budgets is still large, even the largest budget is too small.

Usage:
    python 07d_estep_convergence_probe.py <run_dir> <test_h5mu> <out_csv>
        [n_cells=4096] [budgets=50,100,200,400,800] [patience=10]
"""

import sys, os, importlib.util
import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_07c():
    """Import 07c's helpers -- the filename starts with a digit so plain import won't work."""
    p = os.path.join(_HERE, "07c_heldout_structure.py")
    spec = importlib.util.spec_from_file_location("h07c", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    run_dir, test_h5mu, out_csv = sys.argv[1:4]
    n_cells  = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
    budgets  = [int(x) for x in (sys.argv[5].split(",") if len(sys.argv) > 5
                                 else "50,100,200,400,800".split(","))]
    patience = int(sys.argv[6]) if len(sys.argv) > 6 else 10
    budgets  = sorted(budgets)

    h = _load_07c()
    import h5py
    from scipy.sparse import csr_matrix

    print(f"=== E-step convergence probe ===\nrun={run_dir}\nbudgets={budgets}  "
          f"n_cells={n_cells}  patience={patience}", flush=True)

    m = h.load_model(run_dir)
    psi = np.asarray(m["psi"], dtype=np.float32)
    K = psi.shape[0]
    psi_t = torch.tensor(psi, dtype=torch.float32, device=h.DEVICE)
    dir_conc = m.get("dir_conc"); junc_prior = bool(m.get("junc_specific_prior", True))
    pi = np.asarray(m["pi"], float) if m.get("pi") is not None else None
    alpha_pi = np.asarray(m["alpha_pi"], float) if m.get("alpha_pi") is not None else None
    bb_conc = m.get("bb_conc")
    print(f"K={K}  bb_conc={bb_conc}", flush=True)

    # one chunk of held-out cells, full-coverage counts (same input 07c uses)
    with h5py.File(test_h5mu, "r") as f:
        s = f["mod/splicing"]
        N = min(n_cells, int(s["obs/_index"].shape[0]))
        jr = np.array(s["layers/junc_ratio_original"][:N], dtype=np.float32)
        atse = h.read_sparse_layer(s["layers/cell_by_cluster_matrix_original"]).tocsr()[:N].astype(np.float32)
    atse_d = atse.toarray()
    junc = np.rint(jr * atse_d).astype(np.float32)
    del jr
    chunk = h.build_chunk_adata(csr_matrix(junc), csr_matrix(atse_d))
    del junc, atse_d
    print(f"chunk: {N} cells x {psi.shape[1]} junctions", flush=True)

    # SAME phi init for every budget, so differences are budget alone, not init noise
    phi0 = h.sample_pi_phi_init(pi, K, N, np.random.default_rng(0))

    phis = {}
    for b in budgets:
        h._E_STEP_CONV.clear()
        print(f"\n--- budget {b} epochs ---", flush=True)
        phis[b] = h.infer_phi_chunk(chunk, psi_t, K, dir_conc, junc_prior, pi, alpha_pi,
                                    bb_conc, b, patience, phi0)
        ran, early, cap = h._E_STEP_CONV[-1]
        print(f"  ran {ran}/{cap} epochs, early_stopped={early}", flush=True)

    ref = phis[budgets[-1]]
    rows = []
    for i, b in enumerate(budgets):
        p = phis[b]
        d = np.abs(p - ref).sum(1)
        prev = phis[budgets[i - 1]] if i > 0 else None
        dprev = np.abs(p - prev).sum(1) if prev is not None else None
        pn = p / (p.sum(1, keepdims=True) + 1e-12)
        perp = float(np.median(np.exp(-(pn * np.log(pn + 1e-12)).sum(1))))
        ran, early, cap = h._E_STEP_CONV[-1] if b == budgets[-1] else (np.nan, np.nan, b)
        rows.append({
            "epochs": b,
            "mean_L1_vs_max": float(d.mean()),
            "max_L1_vs_max": float(d.max()),
            "mean_L1_vs_prev": float(dprev.mean()) if dprev is not None else np.nan,
            "corr_vs_max": float(np.corrcoef(p.ravel(), ref.ravel())[0, 1]),
            "argmax_agree_vs_max": float((p.argmax(1) == ref.argmax(1)).mean()),
            "phi_perplexity": perp,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("\n=== CONVERGENCE ===", flush=True)
    print(df.to_string(index=False), flush=True)

    # verdict: phi is converged at the largest budget only if the last step barely moved it
    last = df.iloc[-1]["mean_L1_vs_prev"]
    print("\n  Interpretation: mean_L1_vs_prev is how much phi still moved on the LAST doubling.")
    print(f"  last step ({budgets[-2]} -> {budgets[-1]} epochs): mean_L1 = {last:.4f}")
    if last > 0.05:
        print(f"  !! NOT CONVERGED even at {budgets[-1]} epochs -- phi is still moving. The epoch")
        print(f"     budget is a live knob on every 04e/07c number. Raise it, or fix early stopping")
        print(f"     (min_delta is absolute and inert; use min_delta_rel).")
    else:
        print(f"  phi is stable by {budgets[-1]} epochs.")
        ok = df[df.mean_L1_vs_max < 0.05]["epochs"]
        if len(ok):
            print(f"  cheapest budget within 0.05 mean-L1 of the reference: {int(ok.iloc[0])} epochs")
    print(f"\nwrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
