#!/usr/bin/env python
"""Fig 2 supp: what happens on the simulations when the model rank is misspecified (K = 3, 5, 10; true K = 5).
Per model K, effect in {0.25, 0.5, 1.0}, 3 model seeds: DS calibration and power for the DM LRT (A45, BH,
null tol 0.05) and the posterior test at delta 0.15 (A62, per contrast), plus program recovery (silhouette
of learned phi vs true state, and AUROC of differential-ATSE detection from learned psi) via sim_benchmark.
mean +- SD over seeds (x contrasts for the posterior test)."""
import os, sys, glob, re
import numpy as np, pandas as pd, anndata as ad
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
HERE = os.path.dirname(os.path.abspath(__file__)); FIGROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, FIGROOT); sys.path.insert(0, os.path.join(FIGROOT, "_common"))
import sim_benchmark as SB
from leaflet_theme import apply_theme, PALETTE, METHOD_COLORS, figsize, panel_label, despine, save
W = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD"; M = f"{W}/MOUSE_SPLICING_FOUNDATION"
DIRS = {3: f"{M}/DM_SHIPPED_SIM_K3", 5: f"{M}/DM_SHIPPED_SIM", 10: f"{M}/DM_SHIPPED_SIM_K10"}
EFFS = [0.25, 0.5, 1.0]; LFA = METHOD_COLORS["LeafletFA"]; GREY = PALETTE["grey"]
OUT = os.path.join(FIGROOT, "FIGURE2_simulation_imputation", "data", "sim_K_sweep.tsv")


def collect():
    rows = []
    for K, d in DIRS.items():
        for eff in EFFS:
            for f in glob.glob(f"{d}/ds/a45_lrt_dm50_eff{eff}_frac0.3_seed0_seed*.tsv"):
                ms = int(re.search(r"_seed(\d)\.tsv$", f).group(1)); t = pd.read_csv(f, sep="\t")
                q = multipletests(t.p, method="fdr_bh")[1]; called = q <= 0.05; null = t.true_delta <= 0.05
                rows.append(dict(K=K, eff=eff, ms=ms, test="LRT", ratio=np.mean(null[called]) / 0.05 if called.any() else np.nan, power=np.mean(called[~null])))
            if K == 5:   # the K=5 a62 files were run at delta 0.05; the delta sweep holds delta 0.15
                t = pd.read_csv(f"{d}/ds/delta_sweep_full.tsv", sep="\t"); t = t[(t.delta == 0.15) & (t.eff == eff)]
                for _, r in t.iterrows(): rows.append(dict(K=K, eff=eff, ms=r.ms, test="posterior", ratio=r.ratio, power=r.power))
            else:
                for f in glob.glob(f"{d}/ds/a62_heart_K5_dm50_eff{eff}_frac0.3_seed0_fitseed*.tsv"):
                    ms = int(re.search(r"fitseed(\d)\.tsv$", f).group(1)); t = pd.read_csv(f, sep="\t"); t = t[t.alpha == 0.05]
                    for _, r in t.iterrows(): rows.append(dict(K=K, eff=eff, ms=ms, test="posterior", ratio=r.fdp_ratio, power=r.power, contrast=r.contrast))
            sim = ad.read_h5ad(f"{W}/SIMULATION/sim_heart_K5_dm50_eff{eff}_frac0.3_seed0.h5ad"); T = SB.truth_of(sim)
            for f in glob.glob(f"{d}/fit_sim_heart_K5_dm50_eff{eff}_frac0.3_seed0_dirichletmultinomial_simplex_seed*.npz"):
                ms = int(re.search(r"seed(\d)\.npz$", f).group(1)); z = np.load(f, allow_pickle=True)
                m = SB.metrics(T, np.asarray(z["psi_mean"]), np.asarray(z["phi"]))
                rows.append(dict(K=K, eff=eff, ms=ms, test="recovery", sil=m["sil"], auroc=m["auroc"], ap=m["ap"], psi_r=m["psi_r"]))
    df = pd.DataFrame(rows); df.to_csv(OUT, sep="\t", index=False); return df


def main():
    apply_theme()
    df = pd.read_csv(OUT, sep="\t") if os.path.exists(OUT) and "--reuse" in sys.argv else collect()
    fig, axes = plt.subplots(1, 4, figsize=figsize("double", h=2.4))
    cols = {0.25: PALETTE["ochre"], 0.5: PALETTE["indigo"], 1.0: LFA}
    for ax, (test, col, ylab, logy) in zip(axes[:2], [("LRT", "ratio", "realised FDP / nominal (LRT)", True), ("posterior", "ratio", "realised FDP / nominal\n(posterior $P(TV>0.15)$)", True)]):
        for eff in EFFS:
            g = df[(df.test == test) & (df.eff == eff)].groupby("K")[col].agg(["mean", "std"])
            ax.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), fmt="o-", color=cols[eff], ms=3.2, lw=1.3, capsize=2, label=f"effect {eff}")
        ax.axhline(1, color=GREY, ls=":", lw=0.8); ax.set_yscale("log"); ax.set_xticks([3, 5, 10]); ax.set_xlabel("model K (true K = 5)"); ax.set_ylabel(ylab)
    ax = axes[2]
    for eff in EFFS:
        for test, ls in [("LRT", "-"), ("posterior", "--")]:
            g = df[(df.test == test) & (df.eff == eff)].groupby("K")["power"].agg(["mean", "std"])
            ax.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), fmt="o" + ls, color=cols[eff], ms=3, lw=1.2, capsize=2, label=f"{test}, effect {eff}")
    ax.set_ylim(0, 1.02); ax.set_xticks([3, 5, 10]); ax.set_xlabel("model K (true K = 5)"); ax.set_ylabel("power at $\\alpha$ = 0.05")
    ax = axes[3]
    for eff in EFFS:
        g = df[(df.test == "recovery") & (df.eff == eff)].groupby("K")[["sil", "auroc"]].agg(["mean", "std"])
        ax.errorbar(g.index, g[("auroc", "mean")], yerr=g[("auroc", "std")].fillna(0), fmt="o-", color=cols[eff], ms=3.2, lw=1.3, capsize=2, label=f"AUROC, effect {eff}")
        ax.errorbar(g.index, g[("sil", "mean")], yerr=g[("sil", "std")].fillna(0), fmt="s--", color=cols[eff], ms=3, lw=1.1, capsize=2, label=f"silhouette, effect {eff}")
    ax.set_ylim(0, 1.02); ax.set_xticks([3, 5, 10]); ax.set_xlabel("model K (true K = 5)"); ax.set_ylabel("recovery: DS-ATSE AUROC (solid),\n$\\phi$ silhouette vs true state (dashed)")
    axes[0].legend(frameon=False, fontsize=5); axes[2].legend(frameon=False, fontsize=4.5, ncol=2); axes[3].legend(frameon=False, fontsize=4.5, ncol=1)
    for ax, L in zip(axes, "abcd"): despine(ax); panel_label(ax, L)
    fig.suptitle("Simulations: model rank misspecified (mean $\\pm$ SD over 3 seeds; posterior test also over contrasts)", fontsize=8, y=1.02)
    fig.tight_layout(); save(fig, os.path.join(FIGROOT, "FIGURE2_simulation_imputation", "plots", "FIG2_supp_sim_K_sweep"))
    pd.set_option("display.width", 200)
    print(df[df.test != "recovery"].groupby(["test", "K", "eff"])[["ratio", "power"]].mean().round(2).unstack("K").to_string())
    print(df[df.test == "recovery"].groupby(["K", "eff"])[["sil", "auroc", "psi_r"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
