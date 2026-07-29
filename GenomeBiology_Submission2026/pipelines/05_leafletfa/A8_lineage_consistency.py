"""Are the lineage relationships in the program x cell-type map a property of the data, or of one fit?

A7 shows, for a single model, that the programs organise the atlas along recognisable lineages. Three
of those observations are specific enough to be worth stating in the paper:

  NEURON SPLIT   the model does not learn one generic "neuron" program -- excitatory and hippocampal
                 neurons resemble each other far more than either resembles inhibitory neurons.
  MICROGLIA      microglia group with the myeloid types rather than with the macroglia (astrocyte,
                 oligodendrocyte), consistent with their myeloid rather than neuroectodermal origin.
  LYMPHOID SPLIT the immune compartment is resolved rather than lumped -- B/T/NK resemble each other
                 more than they resemble monocytes, macrophages and granulocytes.

A claim read off one heatmap is an anecdote. This script re-tests all three on every model, across
random seeds and across K.

Each claim is posed as an ORDERING BETWEEN TWO PROFILE SIMILARITIES rather than as a statement about
which program a cell type loads. That matters twice over. Program identity is arbitrary and not
comparable across fits, so an index-based test would need cross-model matching; a relational test
needs none. And an index-based test is not comparable across K either: asking whether two cell types
share their top two programs gets strictly harder as K grows, so a model that resolves astrocytes and
oligodendrocytes into separate programs -- the finer, better answer -- would be scored as a failure.
An earlier version of this script made exactly that mistake and appeared to show the claims did not
replicate; they do.

Only the training-side map is used, which comes straight out of `fit.npz` -- no E-step, no GPU.

Usage: A8_lineage_consistency.py <model_dir> <train_h5ad> <out_csv>
"""
from __future__ import annotations
import sys, json, os
import numpy as np, pandas as pd, h5py

MIN_CELLS = 50          # a cell type must have this many cells for its mean loading to be stable

# the cell types each claim is about, lower-cased; labels in the atlas are upper-case
NEURON_EXC = ["excitatory neuron", "hippocampal neuron"]
NEURON_INH = ["inhibitory neuron"]
MACROGLIA = ["astrocyte", "oligodendrocyte"]
MICROGLIA = ["microglia"]
LYMPHOID = ["b cell", "t cell", "nk cell"]
MYELOID = ["monocyte", "macrophage", "granulocyte", "myeloid immune cell"]


def celltypes(path, key="broad_cell_type"):
    with h5py.File(path, "r") as f:
        s = f["mod/splicing"] if "mod/splicing" in f else f
        g = s[f"obs/{key}"]
        if isinstance(g, h5py.Group):
            cats = [c.decode() if isinstance(c, bytes) else str(c)
                    for c in np.array(g["categories"])]
            return np.array(cats, dtype=object)[np.array(g["codes"])]
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in np.array(g)])


def prof_sim(a, b):
    """Similarity between two cell types' program-loading profiles."""
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def mean_sim(M, group_a, group_b):
    """Mean profile similarity between two groups of cell types (excluding self-pairs)."""
    A = [t for t in group_a if t in M.index]
    B = [t for t in group_b if t in M.index]
    if not A or not B:
        return np.nan
    v = [prof_sim(M.loc[x].values, M.loc[y].values) for x in A for y in B if x != y]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else np.nan


def claims(M: pd.DataFrame) -> dict:
    """M: cell type (lower-cased, index) x program.

    Each claim is an ORDERING between two profile similarities. Stating them relationally makes them
    invariant to K and to the arbitrary labelling of programs: "excitatory neurons resemble each other
    more than they resemble inhibitory neurons" means the same thing at K=10 and K=30, whereas "they
    share their top two programs" becomes strictly harder to satisfy as K grows and would score finer
    resolution as a failure. NaN where a required cell type is absent -- never silently a pass.
    """
    out = {}

    # 1. the excitatory types resemble each other more than either resembles inhibitory neurons
    within = mean_sim(M, NEURON_EXC, NEURON_EXC)
    between = mean_sim(M, NEURON_EXC, NEURON_INH)
    out["exc_within"], out["exc_vs_inh"] = within, between
    out["neuron_exc_inh_resolved"] = (None if not np.isfinite(within * between)
                                      else bool(within > between))

    # 2. microglia resemble the myeloid types more than they resemble the macroglia
    mic_mye = mean_sim(M, MICROGLIA, MYELOID)
    mic_mac = mean_sim(M, MICROGLIA, MACROGLIA)
    out["microglia_vs_myeloid"], out["microglia_vs_macroglia"] = mic_mye, mic_mac
    out["microglia_groups_myeloid"] = (None if not np.isfinite(mic_mye * mic_mac)
                                       else bool(mic_mye > mic_mac))

    # 3. lymphoid types resemble each other more than they resemble myeloid types
    lym_in = mean_sim(M, LYMPHOID, LYMPHOID)
    lym_mye = mean_sim(M, LYMPHOID, MYELOID)
    out["lymphoid_within"], out["lymphoid_vs_myeloid"] = lym_in, lym_mye
    out["lymphoid_myeloid_resolved"] = (None if not np.isfinite(lym_in * lym_mye)
                                        else bool(lym_in > lym_mye))
    return out


def main():
    model_dir, train_h5ad, out_csv = sys.argv[1:4]
    ct = np.array([str(t).lower() for t in celltypes(train_h5ad)])
    keep = [t for t, n in pd.Series(ct).value_counts().items() if n >= MIN_CELLS]
    print(f"{len(ct):,} training cells, {len(keep)} cell types with >={MIN_CELLS} cells\n", flush=True)

    rows = []
    for task in range(9):
        fit = f"{model_dir}/mb_run_{task}/fit.npz"
        if not os.path.exists(fit):
            print(f"task {task}: no fit.npz", flush=True)
            continue
        r = json.load(open(f"{model_dir}/mb_run_{task}/result.json"))
        phi = np.asarray(np.load(fit)["phi"])
        if phi.shape[0] != len(ct):
            print(f"task {task}: phi has {phi.shape[0]:,} rows, labels have {len(ct):,} -- skipped",
                  flush=True)
            continue
        M = pd.DataFrame(np.vstack([phi[ct == t].mean(0) for t in keep]), index=keep)
        c = claims(M)
        rows.append(dict(task=task, K=int(r["K"]), seed=int(r["seed"]), **c))
        flag = {True: "yes", False: "NO ", None: " - "}
        print(f"  K={r['K']:>2} seed={r['seed']}  "
              f"neuron {c['exc_within']:+.2f} vs {c['exc_vs_inh']:+.2f} {flag[c['neuron_exc_inh_resolved']]}   "
              f"microglia {c['microglia_vs_myeloid']:+.2f} vs {c['microglia_vs_macroglia']:+.2f} "
              f"{flag[c['microglia_groups_myeloid']]}   "
              f"lymphoid {c['lymphoid_within']:+.2f} vs {c['lymphoid_vs_myeloid']:+.2f} "
              f"{flag[c['lymphoid_myeloid_resolved']]}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print("\n=== how often does each ordering hold, and by how much? ===", flush=True)
    from scipy.stats import wilcoxon
    for name, a, b in (("neuron_exc_inh_resolved", "exc_within", "exc_vs_inh"),
                       ("microglia_groups_myeloid", "microglia_vs_myeloid", "microglia_vs_macroglia"),
                       ("lymphoid_myeloid_resolved", "lymphoid_within", "lymphoid_vs_myeloid")):
        v = df[name].dropna()
        d = (df[a] - df[b]).dropna()
        # paired across the nine models: is the gap consistently positive, not merely positive on average?
        p = wilcoxon(d)[1] if len(d) >= 6 and d.abs().sum() > 0 else np.nan
        print(f"  {name:<28} {int(v.sum())}/{len(v)} models   "
              f"gap {d.mean():+.3f} (min {d.min():+.3f})   Wilcoxon P={p:.4f}", flush=True)
    print(f"\nwrote {out_csv}", flush=True)
    print("A claim that holds on some models but not others is a property of that fit, not of the "
          "data, and should not be stated in the paper without the caveat.", flush=True)


if __name__ == "__main__":
    main()
