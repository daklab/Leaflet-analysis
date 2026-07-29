"""Do independent seeds agree on WHICH JUNCTIONS ARE UP IN WHICH CELL TYPES?

This is the question the atlas exists to answer, and it is not the same question as "do the latent
variables reproduce". Latent-variable agreement is hard to interpret and, for psi, actively
misleading: 88% of psi's variance is a per-junction baseline shared by every program, and at K=30
62% of programs match a second partner within 0.02 of the one Hungarian chooses, so a matched
correlation of ~0.96 is close to guaranteed and is weak evidence that the same programs were found.

The biological output needs none of that machinery. Cell-type labels are FIXED, so the quantity

    rate[c, j] = (mean phi over cells of type c) @ psi[:, j]

is directly comparable between fits -- no permutation, no matching, no ambiguity. Its cell-type
deviation, rate[c,j] - mean_c' rate[c',j], is "junction j is elevated in cell type c", which is the
statement the paper makes.

Reported:
  RATE r        correlation of the predicted (cell type x junction) splicing rates
  EFFECT SLOPE  regression of one fit's deviations on the other's. NOT the correlation: correlation
                is scale-invariant and would read ~1.0 even under systematic attenuation.
  TOP-N         overlap of the N most cell-type-elevated pairs -- the strongest signals, which are
                what gets followed up.
  JACCARD       overlap of the called sets at a fixed effect threshold. Lower than the rate
                correlation by construction: thresholding a continuous quantity flips whatever sits
                near the boundary, so read it alongside TOP-N rather than on its own.

Usage: A25_celltype_junction_reproducibility.py <model_dir> <tasks csv> <labels h5ad> <out_csv>
"""
from __future__ import annotations
import sys, os, json, itertools
import numpy as np, pandas as pd, h5py


def celltypes(path, key="broad_cell_type"):
    with h5py.File(path, "r") as f:
        s = f["mod/splicing"] if "mod/splicing" in f else f
        g = s[f"obs/{key}"]
        if isinstance(g, h5py.Group):
            cats = [c.decode() if isinstance(c, bytes) else str(c) for c in np.array(g["categories"])]
            return np.array(cats, dtype=object)[np.array(g["codes"])]
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in np.array(g)])


def main():
    md, tasks_s, labels_h5, out_csv = sys.argv[1:5]
    tasks = [int(x) for x in tasks_s.split(",")]
    delta = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1
    labs = celltypes(labels_h5)
    keep = [t for t, n in pd.Series(labs).value_counts().items() if n >= 50]

    dev, K = {}, None
    for t in tasks:
        f = f"{md}/mb_run_{t}/fit.npz"
        if not os.path.exists(f):
            print(f"task {t}: missing"); continue
        z = np.load(f)
        phi = np.asarray(z["phi"], np.float32)
        if phi.shape[0] != len(labs):
            print(f"task {t}: phi has {phi.shape[0]:,} rows, labels {len(labs):,} -- skipped"); continue
        K = phi.shape[1]
        M = np.vstack([phi[labs == c].mean(0) for c in keep])          # (n_ct, K)
        R = M @ np.asarray(z["psi"], np.float32)                       # (n_ct, J)
        dev[t] = dict(rate=R, dev=R - R.mean(0, keepdims=True),
                      seed=json.load(open(f"{md}/mb_run_{t}/result.json"))["seed"])
    if len(dev) < 2:
        sys.exit("need >=2 fits")
    print(f"K={K}, {len(dev)} seeds, {len(keep)} cell types, "
          f"{dev[tasks[0]]['rate'].shape[1]:,} junctions\n", flush=True)

    rows = []
    for a, b in itertools.combinations(sorted(dev), 2):
        A, B = dev[a], dev[b]
        rec = dict(pair=f"seed{A['seed']}_vs_seed{B['seed']}", K=K)
        rec["rate_r"] = float(np.corrcoef(A["rate"].ravel(), B["rate"].ravel())[0, 1])
        da, db = A["dev"].ravel(), B["dev"].ravel()
        rec["effect_slope"] = float(np.polyfit(da, db, 1)[0])
        rec["effect_r"] = float(np.corrcoef(da, db)[0, 1])
        for n in (100, 1000, 10000, 50000):
            ia = set(np.argsort(-da)[:n].tolist()); ib = set(np.argsort(-db)[:n].tolist())
            rec[f"top{n}_overlap"] = len(ia & ib) / n
        SA, SB = A["dev"] > delta, B["dev"] > delta
        rec["n_calls_a"] = int(SA.sum()); rec["n_calls_b"] = int(SB.sum())
        rec["jaccard"] = float((SA & SB).sum() / max((SA | SB).sum(), 1))
        rows.append(rec)
        print(f"  {rec['pair']}: rate r {rec['rate_r']:.4f}  slope {rec['effect_slope']:.3f}  "
              f"top-1000 {rec['top1000_overlap']:.1%}  Jaccard {rec['jaccard']:.3f}", flush=True)

    d = pd.DataFrame(rows); d.to_csv(out_csv, index=False)
    print(f"\n=== K={K} summary — the biological output ===", flush=True)
    print(f"  predicted (cell type x junction) rate      r = {d.rate_r.mean():.4f}", flush=True)
    print(f"  effect-size slope (1.0 = no attenuation)     {d.effect_slope.mean():.3f}", flush=True)
    for n in (100, 1000, 10000, 50000):
        print(f"  top-{n:<6} strongest signals agree        {d[f'top{n}_overlap'].mean():.1%}", flush=True)
    print(f"  called sets at effect > {delta}, Jaccard        {d.jaccard.mean():.3f}"
          f"   ({d.n_calls_a.mean():,.0f} calls/fit)", flush=True)
    print(f"\nwrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
