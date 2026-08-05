#!/usr/bin/env python
"""FIG2-style recovery panels for the Dirichlet-multinomial work.

Run: python panels_fig2_style.py --dir <DM_TEST> --sim-dir <SIMULATION> --out <panels>

Matches the layout of `figures/FIGURE2_simulation_imputation/plots/FIG2_simulation.png`: recovery
metrics on the y-axis, simulated effect size on the x-axis, one line per model with error bars
across seeds. Here the four "methods" are the four cells of the likelihood x psi-parameterisation
factorial, and the extra dimension is the GENERATOR -- whether the counts were drawn as
independent Binomials (the current model's own assumption) or as one multinomial per ATSE (what
the real atlas looks like).

  a  Effect-size recovery -- Spearman between true per-ATSE delta-psi and recovered. The headline
     recovery metric, same as FIG2 panel a.
  b  Psi recovery -- correlation of recovered psi to truth after Hungarian program matching.
     Programs are arbitrary labels, so matching first is required or disagreement is manufactured.
  c  Program separation -- silhouette of phi against the true program label. Says whether the
     latent space still resolves the programs, which is what makes any of the rest meaningful.
  d  Composition validity -- fraction of program-ATSE pairs whose psi sums into [0.9, 1.1]. The
     property the DM and the simplex exist to guarantee.
  e  Calibration and power at nominal 0.05, as bars, so the recovery story and the DS story sit
     in one figure rather than two.

Labels 1-indexed; colours from the project palette.
"""

import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/"
                   "figures/_common")
import leaflet_theme as th  # noqa: E402

CELLS = ["betabinomial_beta", "betabinomial_simplex",
         "dirichletmultinomial_beta", "dirichletmultinomial_simplex"]
LAB = {"betabinomial_beta": "BB + beta (current)", "betabinomial_simplex": "BB + simplex",
       "dirichletmultinomial_beta": "DM + beta", "dirichletmultinomial_simplex": "DM + simplex"}


def colours():
    p = th.PALETTE
    return {"betabinomial_beta": p["grey"], "betabinomial_simplex": p["ochre"],
            "dirichletmultinomial_beta": p["indigo"],
            "dirichletmultinomial_simplex": p["teal"]}


def load_sim(path, cache={}):
    if path in cache:
        return cache[path]
    f = h5py.File(path, "r")

    def cat(node):
        if isinstance(node, h5py.Group):
            c = np.array([x.decode() if isinstance(x, bytes) else x
                          for x in node["categories"][:]])
            return c[node["codes"][:]]
        return np.asarray(node[:])

    ev = cat(f["var"]["event_id"]).astype(str)
    role = cat(f["var"]["junction_role"]).astype(str)
    uq, inv = np.unique(ev, return_inverse=True)
    ids = np.array([x.decode() if isinstance(x, bytes) else x for x in f["uns"]["atse_ids"][:]])
    order = {a: i for i, a in enumerate(ids.astype(str))}
    tidx = np.array([order[a] for a in uq])
    psi_skip = np.asarray(f["uns"]["true_psi_skip"])[:, tidx]
    K = psi_skip.shape[0]
    tpsi = np.zeros((K, len(ev)))
    sk = role == "J3"
    tpsi[:, sk] = psi_skip[:, inv[sk]]
    tpsi[:, ~sk] = (1 - psi_skip[:, inv[~sk]]) / 2
    d = dict(inv=inv, n_atse=len(uq), tpsi=tpsi, psi_skip=psi_skip,
             delta=np.asarray(f["uns"]["atse_delta_psi"])[tidx],
             prog=cat(f["obs"]["true_program"]).astype(str), role=role)
    f.close()
    cache[path] = d
    return d


def silhouette(X, labels, max_n=3000, seed=0):
    """Mean silhouette on a subsample, Euclidean on phi."""
    from sklearn.metrics import silhouette_score
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(labels), size=min(max_n, len(labels)), replace=False)
    if len(np.unique(labels[idx])) < 2:
        return np.nan
    return float(silhouette_score(X[idx], labels[idx]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--sim-dir", default="/gpfs/commons/groups/knowles_lab/Karin/"
                                         "Leaflet-analysis-WD/SIMULATION")
    ap.add_argument("--out", required=True)
    ap.add_argument("--experiment", choices=["sweep", "factorial"], default="sweep",
                    help="'sweep' = fresh simulation per seed, 4 effect sizes (the recovery "
                         "curves). 'factorial' = one simulation refit under different model "
                         "seeds, 2 effect sizes. Pooling them conflates two different questions.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    th.apply_theme()
    C = colours()

    rows = []
    for p in sorted(glob.glob(os.path.join(args.dir, "fit_*.npz"))):
        m = re.match(r"fit_(.+?)_(betabinomial|dirichletmultinomial)_(beta|simplex)_seed(\d)\.npz",
                     os.path.basename(p))
        if not m:
            continue
        sim, cell, seed = m.group(1), f"{m.group(2)}_{m.group(3)}", int(m.group(4))
        # TWO experiments share this directory and must not be pooled:
        #   old factorial : sim seed is always 0, only the MODEL seed varies -> "does the fit
        #                   reproduce on identical data" (2 effect sizes)
        #   new sweep     : sim seed varies WITH the model seed -> "does it reproduce on fresh
        #                   data" (4 effect sizes)
        # They answer different questions. Select on the SIM seed, which is in the sim name.
        sim_seed = int(re.search(r"_seed(\d)$", sim).group(1))
        if args.experiment == "sweep" and sim_seed != seed:
            continue
        if args.experiment == "factorial" and sim_seed != 0:
            continue
        h5 = os.path.join(args.sim_dir, f"{sim}.h5ad")
        if not os.path.exists(h5):
            continue
        s = load_sim(h5)
        z = np.load(p, allow_pickle=True)
        psi, phi = z["psi_mean"], z["phi"]
        K = psi.shape[0]

        # match programs to truth before comparing anything
        Cm = np.corrcoef(psi, s["tpsi"])[:K, K:]
        r_, c_ = linear_sum_assignment(-np.nan_to_num(Cm))
        psi_r = float(np.mean([Cm[i, j] for i, j in zip(r_, c_)]))

        # recovered per-ATSE spread = max-min over programs on the skip junction
        skip = np.array([np.flatnonzero(s["inv"] == e)[
            np.flatnonzero(s["role"][s["inv"] == e] == "J3")[0]] for e in range(s["n_atse"])])
        rec = psi[:, skip].max(0) - psi[:, skip].min(0)
        sp = float(spearmanr(s["delta"], rec).statistic)

        sums = np.zeros((K, s["n_atse"]))
        np.add.at(sums.T, s["inv"], psi.T)
        rows.append(dict(sim=sim, cell=cell, seed=seed,
                         gen="compositional" if "dm50" in sim else "independent",
                         eff=float(re.search(r"eff([\d.]+)_frac", sim).group(1)),
                         psi_r=psi_r, delta_spearman=sp,
                         sil=silhouette(phi, s["prog"]),
                         valid=float(((sums >= 0.9) & (sums <= 1.1)).mean())))
        print(f"  {sim[:34]:<34} {cell:<30} s{seed}  psi_r {psi_r:.3f}  "
              f"dSpearman {sp:.3f}  sil {rows[-1]['sil']:.3f}  valid {rows[-1]['valid']:.3f}",
              flush=True)
    R = pd.DataFrame(rows)
    if R.empty:
        print("no fits found")
        return
    R.to_csv(os.path.join(args.dir, "fig2style_metrics.tsv"), sep="\t", index=False)

    # DS metrics for panel e
    fs = [f for f in glob.glob(os.path.join(args.dir, "a36_*.tsv")) if "_summary" not in f]
    D = pd.concat([pd.read_csv(f, sep="\t") for f in fs], ignore_index=True)
    D["gen"] = np.where(D["sim"].str.contains("dm50"), "compositional", "independent")
    D["eff"] = D["sim"].str.extract(r"eff([\d.]+)_frac")[0].astype(float)
    A = D[np.isclose(D.alpha, 0.05)]

    gens = ["independent", "compositional"]
    metrics = [("delta_spearman", "Effect-size recovery\n(Spearman $\\Delta\\psi$)", "a"),
               ("psi_r", "$\\psi$ recovery\n(corr to truth)", "b"),
               ("sil", "Program separation\n(silhouette, true program)", "c"),
               ("valid", "Composition validity\n(fraction summing to 1)", "d")]

    fig, axes = plt.subplots(2, 4, figsize=th.figsize("double", h=5.0), sharex="col")
    for gi, gen in enumerate(gens):
        for mi, (mkey, mlab, _) in enumerate(metrics):
            ax = axes[gi, mi]
            g = R[R.gen == gen]
            for cell in CELLS:
                gg = g[g.cell == cell]
                if not len(gg):
                    continue
                agg = gg.groupby("eff")[mkey].agg(["mean", "std"])
                ax.errorbar(agg.index, agg["mean"], yerr=agg["std"], marker="o", ms=4,
                            lw=1.6, capsize=2, color=C[cell],
                            label=LAB[cell] if (gi == 0 and mi == 0) else None)
            ax.set_xticks(sorted(R.eff.unique()))
            ax.set_xscale("log")   # FIG2 sweeps effect geometrically
            if gi == 1:
                ax.set_xlabel("simulated effect size")
            if mi == 0:
                ax.set_ylabel(f"{gen} generator", fontsize=th.PT["label"])
            if gi == 0:
                ax.set_title(mlab, fontsize=th.PT["annot"])
    axes[0, 0].legend(frameon=False, fontsize=th.PT["legend"], loc="lower right")
    for mi, (_, _, letter) in enumerate(metrics):
        th.panel_label(axes[0, mi], letter)
    fig.tight_layout()
    th.save(fig, os.path.join(args.out, f"fig2style_recovery_{args.experiment}"))
    plt.close(fig)

    # panel e: calibration and power bars
    effs = sorted(A.eff.unique())
    combos = [(g, e) for g in gens for e in effs]
    fig2, ax2 = plt.subplots(1, len(combos), figsize=th.figsize("double", h=2.6), sharey="row")
    ax2 = np.atleast_1d(ax2)
    for i, (gen, eff) in enumerate(combos):
        ax = ax2[i]
        sub = A[(A.gen == gen) & np.isclose(A.eff, eff)]
        x = np.arange(len(CELLS))
        mu = [sub[sub.cell == c].realised_fdp.mean() for c in CELLS]
        sd = [sub[sub.cell == c].realised_fdp.std() for c in CELLS]
        ax.bar(x, mu, 0.7, yerr=sd, capsize=2, color=[C[c] for c in CELLS], linewidth=0)
        ax.axhline(0.05, color=th.INK, lw=0.9, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([LAB[c].split(" (")[0].replace(" + ", "\n+") for c in CELLS],
                           fontsize=th.PT["tick"])
        ax.set_title(f"{gen}, effect {eff}", fontsize=th.PT["annot"])
    ax2[0].set_ylabel("realised FDP at $\\alpha$=0.05")
    ax2[0].annotate("nominal", (-0.4, 0.055), fontsize=th.PT["annot"], color=th.INK)
    th.panel_label(ax2[0], "e")
    fig2.tight_layout()
    th.save(fig2, os.path.join(args.out, f"fig2style_calibration_{args.experiment}"))
    plt.close(fig2)

    print(f"\n=== summary, mean over seeds ===")
    print(R.groupby(["gen", "eff", "cell"])[
        ["delta_spearman", "psi_r", "sil", "valid"]].mean().round(4).to_string())
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
