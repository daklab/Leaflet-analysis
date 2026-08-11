"""Permuted negative-control ATSEs: a null with REAL dphi and no true signal.

The gap this fills
------------------
Setting `delta` in `P(TV > delta)` needs to know the TV that appears when there is NO real
difference -- the model's resolution floor. Every null tried so far measures it at the wrong scale:

  permuting GROUP LABELS      drives E[dphi] -> 0, so the test reverts to the point null and calls
                              24,422 of 24,475 events.
  the donor-split null (A53)  keeps dphi honest but tiny: ||dphi||_1 ~ 0.12 against ~1.0 for a real
                              cell-type contrast, because both arms are the same cell type. TV
                              scales with ||dphi||, so the floor it reports is far too small.
  a rotation null             replaces dphi with a random zero-sum vector at matched ||dphi||_1.
                              Measured ~10x anti-conservative: the real dphi is structurally
                              aligned with the direction psi actually varies in, and a random
                              direction is not, so the null comes out systematically too small.

This construction avoids all three. Permute a RANDOM SUBSET of ATSEs and leave the rest intact:

  * the permuted ATSEs are null BY CONSTRUCTION -- no truth labels needed, so it works on the atlas
  * the untouched 90% still pin phi, so dphi keeps its true magnitude and direction
  * coverage is preserved exactly: a cell's whole count vector for that event moves together with
    its own N_e, so the marginal depth distribution over cells is unchanged and only the
    association between CELL IDENTITY (hence program loading) and USAGE is destroyed

delta can then be read off a high quantile of the permuted ATSEs' TV distribution, which makes it a
measured property of the fit rather than a number chosen by looking at an answer.

Validation gate, before this is trusted anywhere
-----------------------------------------------
On simulations the truly-null ATSEs are known, so the construction can be checked directly:
compare the permuted ATSEs' TV distribution against the TRUE nulls'. They should MATCH. Tighter
means dphi has collapsed and any delta derived here would be too small (anti-conservative); looser
means the permutation destroyed more than the cell-to-usage association and delta would be too
large. `A39` looked sound on paper and was anti-conservative for a reason only visible after it was
built, so the same mistake is available here and the gate is not optional.
"""

import numpy as np


def permute_atse_counts(junc, clus, atse_idx, n_atse, frac, seed=0, exclude=None):
    """Permute cells' count vectors within a random subset of ATSEs.

    junc, clus : [N, J] float arrays, modified on a COPY and returned
    atse_idx   : [J] int, which ATSE each junction belongs to
    frac       : fraction of ATSEs to permute
    exclude    : optional array of ATSE indices never to permute

    Returns (junc_p, clus_p, permuted_atse_indices).
    """
    rng = np.random.default_rng(seed)
    N = junc.shape[0]
    cand = np.arange(n_atse)
    if exclude is not None and len(exclude):
        cand = np.setdiff1d(cand, np.asarray(exclude))
    n_pick = int(round(frac * n_atse))
    picked = np.sort(rng.choice(cand, min(n_pick, len(cand)), replace=False))

    junc_p, clus_p = junc.copy(), clus.copy()
    # Precompute the column groups once; atse_idx is flat so this is a single pass.
    order = np.argsort(atse_idx, kind="stable")
    bounds = np.searchsorted(atse_idx[order], np.arange(n_atse + 1))
    for e in picked:
        cols = order[bounds[e]:bounds[e + 1]]
        if len(cols) < 2:
            continue
        # ONE permutation of cells for the whole event. Permuting junctions independently would
        # break the sum_j y_j == N_e identity and destroy the compositional structure the model is
        # built on -- the point is to destroy WHICH CELL has WHICH usage, not the usage itself.
        p = rng.permutation(N)
        junc_p[np.ix_(np.arange(N), cols)] = junc[np.ix_(p, cols)]
        clus_p[np.ix_(np.arange(N), cols)] = clus[np.ix_(p, cols)]
    return junc_p, clus_p, picked


def summarise(tv_permuted, tv_other, label=""):
    """Print the floor and what it implies for delta."""
    q = np.quantile(tv_permuted, [0.5, 0.90, 0.95, 0.99])
    print(f"  {label}permuted (null by construction), n={len(tv_permuted):,}: "
          f"median {q[0]:.4f}  p90 {q[1]:.4f}  p95 {q[2]:.4f}  p99 {q[3]:.4f}")
    if tv_other is not None and len(tv_other):
        print(f"  {label}untouched ATSEs, n={len(tv_other):,}: "
              f"median {np.median(tv_other):.4f}  p95 {np.quantile(tv_other, 0.95):.4f}")
    print(f"  -> delta at the 95th percentile of the floor = {q[2]:.3f}")
    return float(q[2])
