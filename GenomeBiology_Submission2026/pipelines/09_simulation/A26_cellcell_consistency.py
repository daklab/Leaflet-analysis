"""Do independently seeded fits agree on WHICH CELLS RESEMBLE WHICH?

Every per-program comparison runs into the same obstacle: programs are exchangeable, and matching them
between fits is near-arbitrary when they resemble one another (on the atlas at K=30, 62% of programs
match a second partner within 0.02 of the one Hungarian picks). Any statistic built on that matching
inherits the ambiguity.

The cell-cell similarity matrix avoids it completely. S[i, j] = corr(phi_i, phi_j) asks only whether
two cells use the programs alike, which is unchanged if the programs are relabelled or the same
variation is divided among them differently. It is also the structure that matters downstream: whether
cells of the same type are grouped together does not depend on what the programs are called.

Simulation makes it a real test rather than a description, because the true phi is known:
  - agreement BETWEEN seeds says the structure is a property of the data, not of one optimisation
  - agreement with TRUTH says that structure is the right one -- an initialisation or a K that made
    every run agree on the same wrong grouping would score well on the first and badly on the second

Run across dispersion levels, since overdispersion is what sets the reproducibility ceiling.

Writes an npz for plotting: the subsampled similarity matrices, the true-program labels for ordering,
and the pairwise agreements.

Usage: A26_cellcell_consistency.py <out_npz> [n_seeds] [n_cells]
"""
from __future__ import annotations
import sys, itertools
import numpy as np, pandas as pd, anndata as ad
sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from leafletfa.minibatch import LeafletFAMinibatch

SIM = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION"
LEVELS = [("binom", "no overdispersion"), ("bb20", "mild (c=20)"), ("bb2", "atlas-like (c=2)")]


def cellcell(phi):
    """Cell x cell correlation of the loading vectors. Invariant to any relabelling of programs."""
    Z = np.asarray(phi, np.float32)
    Z = Z - Z.mean(1, keepdims=True)
    n = np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    Z = Z / n
    return Z @ Z.T


def offdiag(S):
    n = S.shape[0]
    iu = np.triu_indices(n, 1)
    return S[iu]


def main():
    out_npz = sys.argv[1]
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    n_cells = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
    store, rows = {}, []
    for tag, lab in LEVELS:
        a = ad.read_h5ad(f"{SIM}/sim_heart_K5_{tag}_eff0.25_frac0.3_seed0.h5ad")
        g = lambda k: np.asarray(a.layers[k].todense() if hasattr(a.layers[k], "todense")
                                 else a.layers[k], dtype=np.float32)
        junc, clus = g("cell_by_junction_matrix"), g("cell_by_cluster_matrix")
        tp = np.asarray(a.obsm["true_phi"], float)
        K = tp.shape[1]
        # a fixed subsample, stratified by true program so every program is represented
        rng = np.random.default_rng(0)
        lab_true = tp.argmax(1)
        per = max(1, n_cells // K)
        idx = np.sort(np.concatenate([rng.choice(np.where(lab_true == k)[0],
                                                 size=min(per, (lab_true == k).sum()), replace=False)
                                      for k in range(K)]))
        print(f"\n=== {lab} ===  {len(idx)} cells subsampled, K={K}", flush=True)
        S = {}
        for s in range(n_seeds):
            m = LeafletFAMinibatch(junc, clus, K=K, junc_specific_prior=True,
                                   likelihood="betabinomial")
            m.fit(num_steps=5000, subsample_size=2048, lr=0.05, gamma=0.1, seed=s,
                  log_every=5000, verbose=False)
            S[s] = cellcell(m.phi_mean().cpu().numpy()[idx])
        S_true = cellcell(tp[idx])
        between = [float(np.corrcoef(offdiag(S[x]), offdiag(S[y]))[0, 1])
                   for x, y in itertools.combinations(range(n_seeds), 2)]
        vs_truth = [float(np.corrcoef(offdiag(S[s]), offdiag(S_true))[0, 1]) for s in S]
        rows.append(dict(dispersion=tag, label=lab, n_cells=len(idx), K=K,
                         between_seeds=float(np.mean(between)),
                         between_seeds_min=float(np.min(between)),
                         vs_truth=float(np.mean(vs_truth))))
        print(f"  cell-cell structure: between seeds r = {np.mean(between):.3f}"
              f"   vs TRUTH r = {np.mean(vs_truth):.3f}", flush=True)
        store[f"S_{tag}_true"] = S_true.astype(np.float32)
        for s in S:
            store[f"S_{tag}_seed{s}"] = S[s].astype(np.float32)
        store[f"order_{tag}"] = lab_true[idx]

    d = pd.DataFrame(rows)
    np.savez_compressed(out_npz, **store, summary=d.to_records(index=False))
    d.to_csv(out_npz.replace(".npz", ".csv"), index=False)
    print("\n=== summary ===", flush=True)
    print(d[["label", "between_seeds", "vs_truth"]].round(3).to_string(index=False), flush=True)
    print("\n  between_seeds  do independent fits group cells the same way?", flush=True)
    print("  vs_truth       is that grouping the RIGHT one? Read them together: agreeing on a wrong"
          "\n                 grouping would score high on the first and low on the second.", flush=True)
    print(f"\nwrote {out_npz}", flush=True)


if __name__ == "__main__":
    main()
