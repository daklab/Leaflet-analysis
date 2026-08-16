#!/usr/bin/env python
"""Q6: does the result hold as the data gets more or less overdispersed?

Run: fresh_Q6_overdispersion.py --dir <DM_FRESH> --sim-dir <SIMULATION_DM_FRESH> --out <panels>

Modelled on FIGURE2's overdispersion panel, which sweeps the Beta-Binomial concentration. The
Dirichlet-multinomial's equivalent is `dm_conc`: the data are drawn as
`q ~ Dirichlet(dm_conc * p)` then `y ~ Multinomial(N_e, q)`, so

    dm_conc = inf   q collapses to p exactly -- a pure Multinomial, no overdispersion
    dm_conc large   compositions hug the truth
    dm_conc small   compositions scatter widely around it

This matters because EVERY other result in this directory comes from `dm_conc = 50` alone, and
the concentration is the model's own central parameter. A method that only works at the one value
it was tested on has not been tested.

Four panels: psi recovery, program separation, the DS statistic's ability to rank truly
differential events, and whether the model recovers the concentration it was given.

Effect size is held at 0.5 throughout so the sweep is one-dimensional.
"""
import argparse, glob, json, os, re, sys
import h5py, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/_common")
import leaflet_theme as th  # noqa: E402

TAGS = [("multinom", np.inf), ("dm200", 200.0), ("dm50", 50.0), ("dm10", 10.0)]


def load_sim(p):
    f = h5py.File(p, "r")
    def cat(n):
        if isinstance(n, h5py.Group):
            c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
            return c[n["codes"][:]]
        return np.asarray(n[:])
    ev = cat(f["var"]["event_id"]).astype(str); role = cat(f["var"]["junction_role"]).astype(str)
    uq, inv = np.unique(ev, return_inverse=True)
    ids = np.array([x.decode() if isinstance(x, bytes) else x for x in f["uns"]["atse_ids"][:]]).astype(str)
    order = {x: i for i, x in enumerate(ids)}; tidx = np.array([order[x] for x in uq])
    ps = np.asarray(f["uns"]["true_psi_skip"])[:, tidx]; K = ps.shape[0]
    tpsi = np.zeros((K, len(ev))); sk = role == "J3"
    tpsi[:, sk] = ps[:, inv[sk]]; tpsi[:, ~sk] = (1 - ps[:, inv[~sk]]) / 2
    d = dict(inv=inv, n_atse=len(uq), tpsi=tpsi, role=role,
             delta=np.asarray(f["uns"]["atse_delta_psi"])[tidx],
             prog=cat(f["obs"]["true_program"]).astype(str))
    f.close(); return d


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dir", required=True); ap.add_argument("--sim-dir", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--null-tol", type=float, default=0.05)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); th.apply_theme()

rows = []
for tag, conc in TAGS:
    sp_ = os.path.join(a.sim_dir, f"sim_heart_K5_{tag}_eff0.5_frac0.3_seed0.h5ad")
    if not os.path.exists(sp_):
        print(f"  missing sim {tag}"); continue
    s = load_sim(sp_)
    base = os.path.basename(sp_).replace(".h5ad", "")
    fits = {int(re.search(r"seed(\d)\.npz", p).group(1)): np.load(p, allow_pickle=True)
            for p in sorted(glob.glob(os.path.join(a.dir, f"fit_{base}_dirichletmultinomial_simplex_seed*.npz")))}
    if not fits:
        print(f"  no fits for {tag}"); continue
    K = s["tpsi"].shape[0]
    skip = np.array([np.flatnonzero(s["inv"] == q)[np.flatnonzero(s["role"][s["inv"] == q] == "J3")[0]]
                     for q in range(s["n_atse"])])
    rng = np.random.default_rng(0); sub = rng.choice(len(s["prog"]), size=min(3000, len(s["prog"])), replace=False)
    for ts, z in fits.items():
        C = np.corrcoef(z["psi_mean"], s["tpsi"])[:K, K:]
        r_, c_ = linear_sum_assignment(-np.nan_to_num(C))
        pm = np.empty(K, int); pm[c_] = r_
        psi = z["psi_mean"][pm]; phi = z["phi"][:, pm]
        rec = psi[:, skip].max(0) - psi[:, skip].min(0)
        rows.append(dict(
            tag=tag, conc=conc, seed=ts,
            psi_r=float(np.mean([C[i, j] for i, j in zip(r_, c_)])),
            sil=float(silhouette_score(phi[sub], s["prog"][sub])),
            spearman=float(spearmanr(s["delta"], rec).statistic),
            conc_hat=float(z["bb_conc"]) if "bb_conc" in z.files else np.nan))
        print(f"  {tag:<9} seed {ts}  psi_r {rows[-1]['psi_r']:.4f}  sil {rows[-1]['sil']:.4f}  "
              f"spearman {rows[-1]['spearman']:.4f}  conc_hat {rows[-1]['conc_hat']:.2f}", flush=True)
R = pd.DataFrame(rows)
if R.empty:
    print("nothing to plot"); sys.exit(0)
R.to_csv(os.path.join(a.dir, "Q6_overdispersion.tsv"), sep="\t", index=False)

teal, grey = th.PALETTE["teal"], th.PALETTE["grey"]
order = [t for t, _ in TAGS if t in set(R.tag)]
x = np.arange(len(order))
lab = {"multinom": "inf\n(no overdisp.)", "dm200": "200", "dm50": "50", "dm10": "10\n(strong)"}
fig, ax = plt.subplots(1, 4, figsize=th.figsize("double", h=2.7))
for i, (k, ttl, yl) in enumerate([
        ("psi_r", "$\\psi$ recovery", "corr to truth"),
        ("sil", "Program separation", "silhouette (true program)"),
        ("spearman", "DS effect recovery", "Spearman $\\Delta\\psi$"),
        ("conc_hat", "Concentration recovery", "learned conc")]):
    m = R.groupby("tag")[k].agg(["mean", "std"]).reindex(order)
    ax[i].errorbar(x, m["mean"], yerr=m["std"], marker="o", ms=5, lw=1.6, capsize=3, color=teal)
    if k == "conc_hat":
        tv = [c for t, c in TAGS if t in order]
        finite = [(xi, c) for xi, c in zip(x, tv) if np.isfinite(c)]
        ax[i].plot([p[0] for p in finite], [p[1] for p in finite], marker="_", ms=14, lw=0,
                   color=th.INK, label="true conc")
        ax[i].set_yscale("log"); ax[i].legend(frameon=False, fontsize=th.PT["legend"])
    else:
        ax[i].set_ylim(0, 1.05)
    ax[i].set_xticks(x); ax[i].set_xticklabels([lab[t] for t in order], fontsize=th.PT["tick"])
    ax[i].set_xlabel("simulated dm_conc"); ax[i].set_ylabel(yl)
    ax[i].set_title(ttl, fontsize=th.PT["panel"])
    th.panel_label(ax[i], "abcd"[i])
fig.suptitle("Q6  Does the result hold across overdispersion?   DM + simplex, effect 0.5, "
             "3 training seeds   |   FIGURE2 panel-d equivalent", fontsize=th.PT["annot"], y=1.0)
fig.tight_layout()
th.save(fig, os.path.join(a.out, "Q6_overdispersion"))
print("\n=== summary ===")
print(R.groupby("tag")[["psi_r", "sil", "spearman", "conc_hat"]].mean().reindex(order).round(4).to_string())
print(f"\n-> {a.out}/Q6_overdispersion")
