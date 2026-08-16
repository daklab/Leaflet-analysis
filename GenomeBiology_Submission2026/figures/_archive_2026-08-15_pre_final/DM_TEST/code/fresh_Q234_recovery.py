#!/usr/bin/env python
"""Q2, Q3, Q4 for the fresh DM+simplex runs.

Run: fresh_Q234_recovery.py --dir <DM_FRESH> --sim-dir <SIMULATION_DM_FRESH> --out <panels>

One simulation per effect size, three TRAINING seeds each, so truth is identical across seeds and
the comparisons below are all well posed.

Q2  DO TRAINING SEEDS AGREE?
    Per effect size, across seed pairs: correlation of learned psi, of the per-ATSE DS statistic
    TV, and the overlap of the top-500 called ATSEs. Programs are Hungarian-matched on psi and the
    SAME permutation is applied to phi -- permuting one without the other pairs every term with
    the wrong program, which is a mistake that made an earlier version of this measurement report
    0.47 where the true value was 0.94.

Q3  DOES LEARNED PSI MATCH SIMULATED PSI?
    Scatter of learned against true psi over all programs and junctions, plus the correlation as
    a function of effect size. Attenuation appears as points falling below the diagonal.

Q4  DO WE CALL THE CORRECT DS ATSEs?
    Realised false discovery proportion and power against the known truth, over the effect axis,
    at each nominal alpha. Plus the discrimination view: recovered per-ATSE spread for truly
    differential versus truly null events, which is what any threshold has to separate.

Labels 1-indexed; colours from the project palette.
"""
import argparse, glob, os, re, sys
import h5py, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/_common")
import leaflet_theme as th  # noqa: E402


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


def tv_stat(psi, phi, c1, c2, inv, n_atse):
    d = (phi[c1].mean(0) - phi[c2].mean(0)) @ psi
    tv = np.zeros(n_atse); np.add.at(tv, inv, np.abs(d)); return 0.5 * tv


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dir", required=True)
ap.add_argument("--sim-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--null-tol", type=float, default=0.05)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); th.apply_theme()

effs, data = [], {}
for sp_ in sorted(glob.glob(os.path.join(a.sim_dir, "sim_heart_K5_dm50_eff*_seed0.h5ad"))):
    base = os.path.basename(sp_).replace(".h5ad", "")
    eff = float(re.search(r"eff([\d.]+)_", base).group(1))
    fits = {}
    for p in sorted(glob.glob(os.path.join(a.dir, f"fit_{base}_dirichletmultinomial_simplex_seed*.npz"))):
        fits[int(re.search(r"seed(\d)\.npz", p).group(1))] = np.load(p, allow_pickle=True)
    if len(fits) < 1: continue
    effs.append(eff); data[eff] = (load_sim(sp_), fits, base)
effs = sorted(effs)
print(f"[1] {len(effs)} effect sizes: {effs}; seeds per effect: "
      f"{ {e: sorted(data[e][1]) for e in effs} }")

# ---------- Q2: seed agreement ----------
rows = []
for e in effs:
    s, fits, _ = data[e]; seeds = sorted(fits); K = fits[seeds[0]]["psi_mean"].shape[0]
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            A, B = fits[seeds[i]], fits[seeds[j]]
            C = np.corrcoef(A["psi_mean"], B["psi_mean"])[:K, K:]
            r_, c_ = linear_sum_assignment(-np.nan_to_num(C))
            perm = np.empty(K, int); perm[r_] = c_
            bpsi, bphi = B["psi_mean"][perm], B["phi"][:, perm]
            n1, n2 = np.argsort(-A["phi"].sum(0))[:2]
            c1 = np.flatnonzero(A["phi"].argmax(1) == n1); c2 = np.flatnonzero(A["phi"].argmax(1) == n2)
            if min(len(c1), len(c2)) < 50: continue
            t1 = tv_stat(A["psi_mean"], A["phi"], c1, c2, s["inv"], s["n_atse"])
            t2 = tv_stat(bpsi, bphi, c1, c2, s["inv"], s["n_atse"])
            rows.append(dict(eff=e, pair=f"{seeds[i]}-{seeds[j]}",
                             psi_r=float(np.mean([C[x, y] for x, y in zip(r_, c_)])),
                             tv_r=float(np.corrcoef(t1, t2)[0, 1]),
                             top500=len(np.intersect1d(np.argsort(-t1)[:500], np.argsort(-t2)[:500]))/500))
Q2 = pd.DataFrame(rows)

# ---------- Q3: psi recovery ----------
rows = []
for e in effs:
    s, fits, _ = data[e]; K = s["tpsi"].shape[0]
    for ts, z in fits.items():
        C = np.corrcoef(z["psi_mean"], s["tpsi"])[:K, K:]
        r_, c_ = linear_sum_assignment(-np.nan_to_num(C))
        rows.append(dict(eff=e, seed=ts,
                         psi_r=float(np.mean([C[i, j] for i, j in zip(r_, c_)])),
                         perm=" ".join(str(int(x)) for x in c_)))   # string, not list
Q3 = pd.DataFrame(rows)

# ---------- Q4: DS calling ----------
fs = [f for f in glob.glob(os.path.join(a.dir, "a36_*.tsv")) if "_summary" not in f]
Q4 = pd.concat([pd.read_csv(f, sep="\t") for f in fs], ignore_index=True) if fs else pd.DataFrame()
if len(Q4):
    Q4["eff"] = Q4["sim"].str.extract(r"eff([\d.]+)_")[0].astype(float)

teal, grey, ochre = th.PALETTE["teal"], th.PALETTE["grey"], th.PALETTE["ochre"]
fig, ax = plt.subplots(1, 4, figsize=th.figsize("double", h=2.8))

m = Q2.groupby("eff")[["psi_r", "tv_r", "top500"]].agg(["mean", "std"])
for k, col, lab in (("psi_r", teal, "learned $\\psi$"), ("tv_r", ochre, "DS statistic (TV)"),
                    ("top500", grey, "top-500 overlap")):
    ax[0].errorbar(m.index, m[(k, "mean")], yerr=m[(k, "std")], marker="o", ms=4, lw=1.6,
                   capsize=2, color=col, label=lab)
ax[0].set_xscale("log"); ax[0].set_xticks(effs); ax[0].set_xticklabels([str(e) for e in effs])
ax[0].set_ylim(0, 1.05); ax[0].set_xlabel("simulated effect size")
ax[0].set_ylabel("agreement between training seeds")
ax[0].set_title("Q2  do seeds agree?", fontsize=th.PT["panel"])
ax[0].legend(frameon=False, fontsize=th.PT["legend"], loc="lower right"); th.panel_label(ax[0], "a")

g = Q3.groupby("eff").psi_r.agg(["mean", "std"])
ax[1].errorbar(g.index, g["mean"], yerr=g["std"], marker="o", ms=4, lw=1.6, capsize=2, color=teal)
ax[1].set_xscale("log"); ax[1].set_xticks(effs); ax[1].set_xticklabels([str(e) for e in effs])
ax[1].set_ylim(0, 1.05); ax[1].set_xlabel("simulated effect size")
ax[1].set_ylabel("corr(learned $\\psi$, true $\\psi$)")
ax[1].set_title("Q3  does $\\psi$ match truth?", fontsize=th.PT["panel"]); th.panel_label(ax[1], "b")

e0 = effs[-1]; s, fits, _ = data[e0]; z = fits[sorted(fits)[0]]; K = s["tpsi"].shape[0]
C = np.corrcoef(z["psi_mean"], s["tpsi"])[:K, K:]
r_, c_ = linear_sum_assignment(-np.nan_to_num(C))
perm = np.empty(K, int); perm[c_] = r_
rng = np.random.default_rng(0); sub = rng.choice(s["tpsi"].shape[1], size=min(6000, s["tpsi"].shape[1]), replace=False)
ax[2].scatter(s["tpsi"][:, sub].ravel(), z["psi_mean"][perm][:, sub].ravel(), s=1.2, c=teal,
              linewidths=0, rasterized=True)
ax[2].plot([0, 1], [0, 1], color=th.INK, lw=0.8, ls="--", zorder=0)
ax[2].set_xlim(0, 1); ax[2].set_ylim(0, 1)
ax[2].set_xlabel("true $\\psi$"); ax[2].set_ylabel("learned $\\psi$")
ax[2].set_title(f"Q3  effect {e0}, seed {sorted(fits)[0]}", fontsize=th.PT["panel"])
th.panel_label(ax[2], "c")

if len(Q4):
    A = Q4[np.isclose(Q4.alpha, 0.05)]
    f_ = A.groupby("eff").realised_fdp.agg(["mean", "std"])
    p_ = A.groupby("eff").power.agg(["mean", "std"])
    ax[3].errorbar(f_.index, f_["mean"], yerr=f_["std"], marker="o", ms=4, lw=1.6, capsize=2,
                   color=teal, label="realised FDP")
    ax[3].errorbar(p_.index, p_["mean"], yerr=p_["std"], marker="s", ms=4, lw=1.6, capsize=2,
                   color=grey, label="power")
    ax[3].axhline(0.05, color=th.INK, lw=0.9, ls="--")
    ax[3].annotate("nominal 0.05", (effs[0], 0.07), fontsize=th.PT["annot"], color=th.INK)
    ax[3].set_xscale("log"); ax[3].set_xticks(effs); ax[3].set_xticklabels([str(e) for e in effs])
    ax[3].set_xlabel("simulated effect size"); ax[3].set_ylabel("at nominal $\\alpha$ = 0.05")
    ax[3].set_title("Q4  correct DS calls?", fontsize=th.PT["panel"])
    ax[3].legend(frameon=False, fontsize=th.PT["legend"], loc="best")
else:
    ax[3].axis("off")
th.panel_label(ax[3], "d")
fig.suptitle("DM + simplex on fresh compositional simulations   |   one simulation per effect, "
             "three training seeds", fontsize=th.PT["annot"], y=1.0)
fig.tight_layout()
th.save(fig, os.path.join(a.out, "Q2Q3Q4_recovery"))
Q2.to_csv(os.path.join(a.dir, "Q2_seed_agreement.tsv"), sep="\t", index=False)
Q3.to_csv(os.path.join(a.dir, "Q3_psi_recovery.tsv"), sep="\t", index=False)
print("\n=== Q2 seed agreement ===");  print(Q2.groupby("eff")[["psi_r","tv_r","top500"]].mean().round(4).to_string())
print("\n=== Q3 psi recovery ===");    print(Q3.groupby("eff").psi_r.agg(["mean","std"]).round(4).to_string())
if len(Q4):
    print("\n=== Q4 DS calls at nominal 0.05 ===")
    print(Q4[np.isclose(Q4.alpha,0.05)].groupby("eff")[["realised_fdp","power","frac_valid"]].mean().round(4).to_string())
print(f"\n-> {a.out}/Q2Q3Q4_recovery")
