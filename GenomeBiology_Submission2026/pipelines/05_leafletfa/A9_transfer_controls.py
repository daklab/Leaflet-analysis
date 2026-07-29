"""Is the train-vs-held-out agreement real, and do independent fits find the same structure?

Two objections to the A7 map, both fair, both answered here with controls rather than assertion.

OBJECTION 1 -- "train and held-out look suspiciously similar; is anything actually being tested?"
    A correlation of 0.998 between the two maps is only impressive against a scale. Three references:

      SAMPLING FLOOR  the same model, the same training cells, but a random 12,000-cell subsample --
                      i.e. how much two maps differ for no reason but sample size. The held-out map is
                      built from 12,000 cells and the training map from 92,894, so part of any gap is
                      sampling, not transfer. THIS is the number train-vs-test should be judged
                      against: if transfer sits at the sampling floor, transfer costs nothing.
      SHUFFLED NULL   held-out cells with their cell-type labels permuted. Destroys the correspondence
                      while leaving every marginal intact, so it isolates what the labels contribute.
      CROSS-MODEL     the training map of one seed against the held-out map of a DIFFERENT seed, after
                      matching programs. If this scored as well as within-model, the map would not be
                      model-specific and the whole comparison would be vacuous.

OBJECTION 1a -- "0.998 feels too high; is the correlation itself flattering?"
    It is. The map is mostly near-zero entries with a handful of large ones, so Pearson r is carried by
    whether the big blocks agree and cannot say much about the rest. Reported alongside it, therefore:
    Spearman (rank-based, so magnitude cannot carry it), mean absolute error, the same statistics
    restricted to entries above 0.05 (where r cannot coast on the zeros), and the largest single
    disagreement. The relative error on substantive entries is the number to quote in text; the
    Pearson r is only meaningful against the sampling floor.

OBJECTION 1b -- "is the held-out phi actually learned, or does it just fall back on the prior?"
    If the E-step had no signal to work with, every held-out cell would land on the prior mean pi: the
    per-cell perplexity would rise to K, no cell would commit to a program, and the map could still
    look fine because every cell type would be averaging the same thing. Reported here as perplexity
    and max-loading for held-out vs training cells against what the prior alone would give.

OBJECTION 2 -- "across seeds the programs are in a different order, so how can they be compared?"
    They cannot be compared directly, and Hungarian matching only papers over it with a choice of our
    own. The way out is to compare something PERMUTATION-INVARIANT.

    For each model, correlate every pair of cell types' program-loading profiles -> a cell-type x
    cell-type matrix. Reordering, or even relabelling, the programs leaves that matrix exactly
    unchanged, because it only ever asks "do these two cell types use programs the same way". So two
    fits with completely different program orderings can be compared with no matching at all. What it
    measures -- which cell types share splicing structure -- is also the biologically meaningful part.

Usage: A9_transfer_controls.py <model_dir> <train_h5ad> <test_h5mu> <out_prefix> [K] [n_cells]
"""
from __future__ import annotations
import sys, os, json
import numpy as np, pandas as pd, h5py
from scipy.optimize import linear_sum_assignment

MIN_CELLS = 50
TASKS_BY_K = {10: (0, 1, 2), 20: (3, 4, 5), 30: (6, 7, 8)}


def celltypes(path, key="broad_cell_type"):
    with h5py.File(path, "r") as f:
        s = f["mod/splicing"] if "mod/splicing" in f else f
        g = s[f"obs/{key}"]
        if isinstance(g, h5py.Group):
            cats = [c.decode() if isinstance(c, bytes) else str(c)
                    for c in np.array(g["categories"])]
            return np.array(cats, dtype=object)[np.array(g["codes"])], s["obs/_index"].shape[0]
        return (np.array([x.decode() if isinstance(x, bytes) else str(x) for x in np.array(g)]),
                s["obs/_index"].shape[0])


def cmap_of(phi, ct, types):
    """Mean loading per cell type -> (n_types, K)."""
    return np.vstack([phi[ct == t].mean(0) for t in types])


def gram(M):
    """Cell-type x cell-type profile correlation. INVARIANT to any permutation of the programs."""
    Z = (M - M.mean(1, keepdims=True)) / np.maximum(M.std(1, keepdims=True), 1e-12)
    return (Z @ Z.T) / M.shape[1]


def offdiag(G):
    return G[~np.eye(G.shape[0], dtype=bool)]


def r(a, b):
    return float(np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1])


def main():
    model_dir, train_h5ad, test_h5mu, out_prefix = sys.argv[1:5]
    K = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    n_cells = int(sys.argv[6]) if len(sys.argv) > 6 else 12000
    tasks = TASKS_BY_K[K]

    ct_tr, n_tr = celltypes(train_h5ad)
    ct_te_all, n_te = celltypes(test_h5mu)
    idx = np.sort(np.random.default_rng(0).choice(n_te, size=min(n_cells, n_te), replace=False))
    ct_te = ct_te_all[idx]

    def phi_te_of(task):
        """The held-out phi a7_all_models.sh cached for this task (K=10 shares A5's cache)."""
        for p in (f"{model_dir}/A7_phi_task{task}.npz", f"{model_dir}/A5_repro_K10_phi_cache.npz"):
            if os.path.exists(p):
                z = np.load(p)
                if f"phi_{task}" in z.files:
                    return z[f"phi_{task}"]
        return None

    maps, rows, prior_rows, err_rows = {}, [], [], []
    for t in tasks:
        f = f"{model_dir}/A7_prog_ct_task{t}.npz"
        if not os.path.exists(f):
            print(f"task {t}: {f} missing -- run a7_all_models.sh first", flush=True)
            continue
        z = np.load(f, allow_pickle=True)
        types = [str(x) for x in z["types"]]
        phi_tr = np.asarray(np.load(f"{model_dir}/mb_run_{t}/fit.npz")["phi"])
        seed = json.load(open(f"{model_dir}/mb_run_{t}/result.json"))["seed"]
        maps[t] = dict(types=types, train=z["train"], test=z["test"], seed=seed, phi_tr=phi_tr)

    if not maps:
        sys.exit("no A7 outputs found")
    types = maps[tasks[0]]["types"]
    print(f"K={K}: {len(maps)} models, {len(types)} cell types, "
          f"{n_tr:,} train / {len(ct_te):,} held-out cells\n", flush=True)

    # ---- objection 1: is train-vs-test better than the sampling floor? --------------------------
    print("=== 1. train vs held-out, against three reference points ===", flush=True)
    print(f"{'':6} {'transfer':>9} {'sampling':>9} {'shuffled':>9} {'x-model':>9}", flush=True)
    rng = np.random.default_rng(1)
    for t, m in maps.items():
        M_tr, M_te = m["train"], m["test"]
        # sampling floor: the SAME cells the training map came from, subsampled to the test-set size,
        # averaged over draws -- so the comparison is like-for-like on n
        floor = []
        for _ in range(20):
            sub = rng.choice(n_tr, size=len(ct_te), replace=False)
            ct_s, phi_s = ct_tr[sub], m["phi_tr"][sub]
            ok = [x for x in types if (ct_s == x).sum() >= 5]
            if len(ok) < len(types) // 2:
                continue
            keep = [types.index(x) for x in ok]
            floor.append(r(M_tr[keep], cmap_of(phi_s, ct_s, ok)))
        # shuffled null: the held-out map is intact but its cell types are permuted, so every
        # marginal survives and only the type-to-type correspondence is destroyed
        sh = [r(M_tr, M_te[rng.permutation(len(types))]) for _ in range(20)]
        # cross-model: this model's training map vs another seed's held-out map, programs matched
        xm = []
        for t2, m2 in maps.items():
            if t2 == t:
                continue
            C = np.corrcoef(M_tr.T, m2["test"].T)[:M_tr.shape[1], M_tr.shape[1]:]
            _, col = linear_sum_assignment(-np.nan_to_num(C))
            xm.append(r(M_tr, m2["test"][:, col]))
        row = dict(K=K, task=t, seed=m["seed"], transfer=r(M_tr, M_te),
                   sampling_floor=float(np.mean(floor)), shuffled=float(np.mean(sh)),
                   cross_model=float(np.mean(xm)) if xm else np.nan)
        rows.append(row)
        print(f"seed{m['seed']} {row['transfer']:9.4f} {row['sampling_floor']:9.4f} "
              f"{row['shuffled']:9.4f} {row['cross_model']:9.4f}", flush=True)

    d = pd.DataFrame(rows)
    print(f"\n  transfer {d.transfer.mean():.4f} vs sampling floor {d.sampling_floor.mean():.4f}"
          f"  -> transfer costs {d.sampling_floor.mean() - d.transfer.mean():+.4f}", flush=True)
    print(f"  shuffled labels collapse to {d.shuffled.mean():.4f}, so the agreement is carried by the"
          f" labels, not by the marginals", flush=True)
    print(f"  a DIFFERENT seed's held-out map scores {d.cross_model.mean():.4f} -- the map is"
          f" {'model-specific' if d.cross_model.mean() < d.transfer.mean() - 0.005 else 'NOT model-specific'}",
          flush=True)

    # ---- objection 1a: is r flattered by the many near-zero entries? ---------------------------
    print("\n=== 1a. the same agreement on metrics r cannot flatter ===", flush=True)
    from scipy.stats import spearmanr
    print(f"{'':7} {'pearson':>8} {'spearman':>9} {'MAE':>8} {'r|>0.05':>9} {'rel.err':>8} "
          f"{'max|d|':>8}", flush=True)
    for t, m in maps.items():
        a, b = m["train"], m["test"]
        big = a > 0.05
        row = dict(K=K, task=t, seed=m["seed"], pearson=r(a, b),
                   spearman=float(spearmanr(a.ravel(), b.ravel())[0]),
                   mae=float(np.abs(a - b).mean()),
                   pearson_big=float(np.corrcoef(a[big], b[big])[0, 1]),
                   rel_err_big=float(np.abs(a[big] - b[big]).mean() / a[big].mean()),
                   max_abs_diff=float(np.abs(a - b).max()), n_big=int(big.sum()))
        err_rows.append(row)
        print(f"seed{m['seed']} {row['pearson']:8.4f} {row['spearman']:9.4f} {row['mae']:8.4f} "
              f"{row['pearson_big']:9.4f} {row['rel_err_big']:8.1%} {row['max_abs_diff']:8.4f}",
              flush=True)
    er = pd.DataFrame(err_rows)
    print(f"\n  QUOTE THIS: cell-type program usage reproduces on held-out cells to within "
          f"{er.rel_err_big.mean():.1%} relative error on the {int(er.n_big.mean())} substantive "
          f"entries (loading > 0.05).", flush=True)
    print(f"  The Pearson r of {er.pearson.mean():.4f} is only informative against the sampling "
          f"floor -- on its own it is inflated by the many near-zero entries "
          f"(Spearman {er.spearman.mean():.4f}).", flush=True)

    # ---- objection 1b: is held-out phi sharp, or prior-dominated? ------------------------------
    print("\n=== 1b. is the held-out phi learned, or the prior? ===", flush=True)
    import gzip, pickle

    def perp(p):
        p = np.clip(np.asarray(p, float), 1e-12, None)
        p = p / p.sum(1, keepdims=True)
        return np.exp(-(p * np.log(p)).sum(1))

    print(f"{'':7} {'train perp':>11} {'test perp':>10} {'prior perp':>11} {'test max phi':>13}",
          flush=True)
    for t, m in maps.items():
        pkl = f"{model_dir}/eval_grouped/task_{t}/model/leafletfa_model.pkl.gz"
        if not os.path.exists(pkl):
            continue
        with gzip.open(pkl, "rb") as fh:
            pi = np.asarray(pickle.load(fh)["pi"], float).ravel()
        pi = pi / pi.sum()
        te = phi_te_of(t)
        if te is None:
            continue
        pp = float(np.exp(-(pi * np.log(np.clip(pi, 1e-12, None))).sum()))
        row = dict(K=K, task=t, seed=m["seed"], train_perplexity=float(np.median(perp(m["phi_tr"]))),
                   test_perplexity=float(np.median(perp(te))), prior_perplexity=pp,
                   test_max_phi=float(np.median(te.max(1))),
                   train_max_phi=float(np.median(m["phi_tr"].max(1))))
        prior_rows.append(row)
        print(f"seed{m['seed']} {row['train_perplexity']:11.2f} {row['test_perplexity']:10.2f} "
              f"{pp:11.2f} {row['test_max_phi']:13.3f}", flush=True)
    if prior_rows:
        pr = pd.DataFrame(prior_rows)
        print(f"  held-out cells commit as sharply as training cells "
              f"({pr.test_perplexity.mean():.2f} vs {pr.train_perplexity.mean():.2f} of "
              f"K={K}), far from the {pr.prior_perplexity.mean():.2f} the prior alone would give "
              f"-- the E-step is using the held-out reads, not falling back on pi.", flush=True)

    # ---- objection 2: permutation-invariant agreement between seeds -----------------------------
    print("\n=== 2. cell-type similarity structure, which needs no program matching ===", flush=True)
    G = {t: dict(train=gram(m["train"]), test=gram(m["test"])) for t, m in maps.items()}
    ts = sorted(G)
    grows = []
    for i, a in enumerate(ts):
        for b in ts[i + 1:]:
            v = r(offdiag(G[a]["train"]), offdiag(G[b]["train"]))
            print(f"  seed{maps[a]['seed']} vs seed{maps[b]['seed']}: "
                  f"cell-type structure agrees at r = {v:.4f}", flush=True)
            grows.append(dict(K=K, comparison=f"seed{maps[a]['seed']}_vs_seed{maps[b]['seed']}",
                              gram_r=v))
    for t in ts:
        v = r(offdiag(G[t]["train"]), offdiag(G[t]["test"]))
        grows.append(dict(K=K, comparison=f"seed{maps[t]['seed']}_train_vs_test", gram_r=v))
        print(f"  seed{maps[t]['seed']} train vs held-out: r = {v:.4f}", flush=True)

    d.to_csv(f"{out_prefix}_controls.csv", index=False)
    if prior_rows:
        pd.DataFrame(prior_rows).to_csv(f"{out_prefix}_prior_check.csv", index=False)
    if err_rows:
        pd.DataFrame(err_rows).to_csv(f"{out_prefix}_error_metrics.csv", index=False)
    pd.DataFrame(grows).to_csv(f"{out_prefix}_gram.csv", index=False)
    np.savez_compressed(f"{out_prefix}_gram.npz", types=np.array(types, dtype=object),
                        **{f"gram_train_{t}": G[t]["train"] for t in ts},
                        **{f"gram_test_{t}": G[t]["test"] for t in ts},
                        seeds=np.array([maps[t]["seed"] for t in ts]), K=K)
    print(f"\nwrote {out_prefix}_controls.csv, {out_prefix}_gram.csv, {out_prefix}_gram.npz",
          flush=True)


if __name__ == "__main__":
    main()
