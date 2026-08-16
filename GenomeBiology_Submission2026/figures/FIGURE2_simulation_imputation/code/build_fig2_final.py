#!/usr/bin/env python
"""Figure 2, rebuilt: what the model recovers, and where it stops.

Why a rebuild rather than an edit
---------------------------------
Every panel of the previous FIG2_simulation.png validated the BetaBinomial and Binomial models.
Neither is shipped. A figure that validates a method the paper no longer proposes is worse than no
figure, so the panels are replaced rather than adjusted.

The panels, and what each is allowed to claim
---------------------------------------------
Row 1 -- the model recovers the generative parameters.
  a  effect-size recovery against NMF at matched rank. NMF is scored at its BEST gauge of the two
     (0-impute and mean-impute), so the baseline is not handicapped by a choice we made for it.
  b  concentration recovery. conc is the effective sample size, so this is the panel that says the
     overdispersion parameter means something rather than absorbing slack.
  c  psi recovery against read depth. Recovery RISES with coverage; a method scoring 1.0 at 1-2
     reads would be fitting noise, so the direction is the result.

Row 2 -- differential splicing, which is what the model is for.
  d  DS calibration: realised FDP divided by nominal, against simulated effect, LeafletFA's LRT
     against LeafCutter on the SAME simulations with the SAME truth and the same null tolerance.
     1.0 is honest; above 1.0 is over-calling.
  e  DS power at nominal 0.05. Shown beside d and never alone: a miscalibrated test calls more by
     construction, so power without the calibration panel next to it rewards over-calling.
  f  phi against ground truth by regime -- the limitation panel. This is the only place phi's
     accuracy is measured against truth, and it is what justifies the shipped recipe rather than
     leaving it an unexplained tuning choice.

What the figure deliberately does NOT include
---------------------------------------------
The posterior TV test. It is 4-7x anti-conservative on these same simulations once phi is fixed,
and is dominated by the LRT on both axes, so putting it here would either mislead or need a
paragraph of apology. It goes in the supplement as a diagnostic (build_fig2_supp_posterior.py)
where the null-floor mechanism can be shown properly.

Caveat that belongs in the caption: these simulations sit at median 33 reads per ATSE against the
atlas's 16, so every power number here is an OPTIMISTIC bound on real data.
"""

import os, sys, glob, re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, FIGROOT)
sys.path.insert(0, os.path.join(FIGROOT, "_common"))
from leaflet_theme import (apply_theme, PALETTE, METHOD_COLORS, figsize,   # noqa: E402
                           panel_label, despine, save)

W = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD"
M = f"{W}/MOUSE_SPLICING_FOUNDATION"
DATA = os.path.join(FIGROOT, "FIGURE2_simulation_imputation", "data")
PLOTS = os.path.join(FIGROOT, "FIGURE2_simulation_imputation", "plots")
LFA, LC = METHOD_COLORS["LeafletFA"], METHOD_COLORS["NMF"]
GREY = PALETTE["grey"]


# ---------------------------------------------------------------------------------------------
def panel_a(ax):
    """Effect-size recovery vs NMF. NMF gets its best gauge at every effect."""
    d = pd.read_csv(f"{DATA}/dm_vs_nmf.tsv", sep="\t")
    d["fam"] = np.where(d.model.str.startswith("LeafletFA"), "LeafletFA", "NMF")
    g = (d.groupby(["fam", "model", "effect"])["spearman_delta"].mean().reset_index())
    lfa = g[g.fam == "LeafletFA"].groupby("effect")["spearman_delta"].mean()
    # best of the two NMF gauges per effect -- deliberately generous to the baseline
    nmf = g[g.fam == "NMF"].groupby("effect")["spearman_delta"].max()
    ax.plot(lfa.index, lfa.values, "o-", color=LFA, label="LeafletFA", ms=3.2, lw=1.4)
    ax.plot(nmf.index, nmf.values, "s--", color=LC, label="NMF (best gauge)", ms=3.2, lw=1.4)
    ax.axhline(0, color=GREY, lw=0.6, ls=":")
    ax.set_xlabel("simulated effect size")
    ax.set_ylabel("recovery of $\\Delta\\psi$\n(Spearman vs truth)")
    ax.set_xscale("log"); ax.set_xticks([0.1, 0.25, 0.5, 1.0])
    ax.set_xticklabels(["0.1", "0.25", "0.5", "1.0"])
    ax.legend(frameon=False, loc="upper left")


def panel_b(ax):
    """Concentration recovery. conc IS the effective sample size, so this must be near the line."""
    d = pd.read_csv(f"{M}/DM_FRESH/Q6_overdispersion.tsv", sep="\t")
    d = d[np.isfinite(pd.to_numeric(d["conc"], errors="coerce"))]
    d["conc"] = d["conc"].astype(float)
    g = d.groupby("conc")["conc_hat"].agg(["mean", "std"]).reset_index()
    lim = [5, 400]
    ax.plot(lim, lim, ls=":", color=GREY, lw=0.8, zorder=0)
    ax.errorbar(g["conc"], g["mean"], yerr=g["std"].fillna(0), fmt="o",
                color=LFA, ms=4, lw=1.2, capsize=2)
    for _, r in g.iterrows():
        ax.annotate(f"{r['mean']:.0f}", (r["conc"], r["mean"]), textcoords="offset points",
                    xytext=(5, -7), fontsize=6, color=LFA)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("simulated concentration")
    ax.set_ylabel("recovered concentration")
    ax.annotate("identity", (330, 250), fontsize=6, color=GREY, ha="right")


def panel_c(ax):
    """psi recovery vs depth, by effect. Rising with coverage is the correct direction."""
    d = pd.read_csv(f"{M}/DM_FRESH/Q5_recovery_by_coverage.tsv", sep="\t")
    effs = sorted(d.eff.unique())
    cols = [PALETTE[k] for k in ("olive", "teal", "indigo", "ochre")]
    for e, c in zip(effs, cols):
        s = d[d.eff == e].groupby("bin").agg(depth=("depth", "mean"), r=("r", "mean"))
        ax.plot(s.depth, s.r, "o-", color=c, ms=2.6, lw=1.2, label=f"effect {e:g}")
    ax.set_xscale("log")
    ax.set_xlabel("reads per ATSE")
    ax.set_ylabel("$\\psi$ recovery (r vs truth)")
    ax.legend(frameon=False, loc="lower right", ncol=1)


# ---------------------------------------------------------------------------------------------
def _ds_tables():
    lrt = pd.read_csv(f"{M}/DM_SHIPPED_SIM/ds/summary_lrt.tsv", sep="\t")
    lc = pd.read_csv(f"{M}/LC_SIM_DM/summary_leafcutter_dm.tsv", sep="\t")
    return lrt[lrt.alpha == 0.05], lc[lc.alpha == 0.05]


def panel_d(ax):
    """Calibration: realised FDP / nominal. 1.0 is honest, above is over-calling."""
    lrt, lc = _ds_tables()
    a = lrt.groupby("eff")["ratio"].agg(["mean", "std"])
    b = lc.groupby("eff")["ratio"].agg(["mean", "std"])
    ax.axhline(1.0, color=GREY, lw=0.8, ls=":", zorder=0)
    ax.errorbar(a.index, a["mean"], yerr=a["std"].fillna(0), fmt="o-", color=LFA,
                ms=3.2, lw=1.4, capsize=2, label="LeafletFA (DM LRT)")
    ax.errorbar(b.index, b["mean"], yerr=b["std"].fillna(0), fmt="s--", color=LC,
                ms=3.2, lw=1.4, capsize=2, label="LeafCutter")
    ax.set_yscale("log")
    ax.set_xscale("log"); ax.set_xticks([0.1, 0.2, 0.35, 0.5, 1.0])
    ax.set_xticklabels(["0.1", "0.2", "0.35", "0.5", "1.0"])
    ax.set_xlabel("simulated effect size")
    ax.set_ylabel("realised FDP / nominal\n(at $\\alpha$ = 0.05)")
    ax.annotate("honest", (1.02, 1.0), xycoords=("axes fraction", "data"),
                fontsize=6, color=GREY, va="center")
    ax.legend(frameon=False, loc="upper right")


def panel_e(ax):
    """Power at nominal 0.05. Only meaningful read beside panel d."""
    lrt, lc = _ds_tables()
    a = lrt.groupby("eff")["power"].agg(["mean", "std"])
    b = lc.groupby("eff")["power"].agg(["mean", "std"])
    ax.errorbar(a.index, a["mean"], yerr=a["std"].fillna(0), fmt="o-", color=LFA,
                ms=3.2, lw=1.4, capsize=2, label="LeafletFA (DM LRT)")
    ax.errorbar(b.index, b["mean"], yerr=b["std"].fillna(0), fmt="s--", color=LC,
                ms=3.2, lw=1.4, capsize=2, label="LeafCutter")
    ax.set_xscale("log"); ax.set_xticks([0.1, 0.2, 0.35, 0.5, 1.0])
    ax.set_xticklabels(["0.1", "0.2", "0.35", "0.5", "1.0"])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("simulated effect size")
    ax.set_ylabel("power (at $\\alpha$ = 0.05)")
    ax.legend(frameon=False, loc="lower right")


# The regime arms, in the order they read as an argument rather than alphabetically. `dc5` is
# delta = K on these K=5 simulations, i.e. the SHIPPED recipe, so it is the bar that has to be
# picked out; `dc25` is delta = 5K and is in the figure precisely because it is WORSE than doing
# nothing, which is what makes delta = K a floor to sit on rather than a knob to turn up.
REGIME_ARMS = [
    ("ctrl", "control\n(conc 50, full depth)"),
    ("base", "atlas regime,\n$\\delta$ learned"),
    ("dc25", "$\\delta$ = 5K"),
    # mathtext, not plain text: the theme's sentence-casing capitalises a leading bare "m" and
    # silently renders this as "M/(m-1)", which reads as a different quantity.
    ("wt",   "$m/(m\\!-\\!1)$ weight"),
    ("both", "$\\delta$ = K + weight"),
    ("dc5",  "$\\delta$ = K  (shipped)"),
]


def panel_f(ax):
    """phi against truth, by regime. The limitation panel, and the one that justifies the recipe."""
    d = pd.read_csv(f"{M}/ATLAS_DM/regime/a56_scores.tsv", sep="\t")
    g = d.groupby("arm")["phi_r_true"].agg(["mean", "std"])
    arms = [(k, lab) for k, lab in REGIME_ARMS if k in g.index]
    y = np.arange(len(arms))
    vals = [g.loc[k, "mean"] for k, _ in arms]
    errs = [0.0 if not np.isfinite(g.loc[k, "std"]) else g.loc[k, "std"] for k, _ in arms]
    cols = [LFA if k == "dc5" else GREY for k, _ in arms]
    ax.barh(y, vals, xerr=errs, color=cols, height=0.62,
            error_kw=dict(lw=0.8, capsize=2, ecolor=PALETTE["olive"]))
    for yi, v in zip(y, vals):
        ax.annotate(f"{v:.2f}", (v, yi), textcoords="offset points", xytext=(4, 0),
                    va="center", fontsize=6, color=PALETTE["olive"])
    ax.set_yticks(y); ax.set_yticklabels([lab for _, lab in arms], fontsize=6)
    ax.set_xlabel("$\\phi$ recovery vs truth (r)")
    ax.set_xlim(0, 1.14); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])


def panel_g(ax):
    """Choosing delta: the calibration-power trade-off, measured rather than asserted.

    delta is the effect size the test is asked to resolve. Too small and it sits under psi's own
    posterior noise, so the test over-calls; too large and real signal is discarded. Plotting
    realised-FDP-over-nominal against power, with delta as the point label, shows the whole trade in
    one panel and makes the operating point a reading rather than a claim."""
    d = pd.read_csv(f"{M}/DM_SHIPPED_SIM/ds/delta_sweep_full.tsv", sep="\t")
    # An FDP over a handful of calls is noise, not a measurement -- drop those cells rather than
    # letting a ratio computed from 0.2 called events set the shape of the curve.
    g = (d.groupby(["eff", "delta"])
           .agg(ratio=("ratio", "mean"), power=("power", "mean"), n=("n_called", "mean"))
           .reset_index())
    g = g[g.n >= 10]
    effs = sorted(g.eff.unique())
    cols = {e: c for e, c in zip(effs, [PALETTE[k] for k in
                                        ("rose", "ochre", "indigo", "teal", "olive")] * 3)}
    ax.axvline(1.0, color=GREY, lw=0.8, ls=":", zorder=0)
    for e in effs:
        s = g[g.eff == e].sort_values("delta")
        if len(s) < 2:
            continue
        ax.plot(s.ratio, s.power, "-o", color=cols[e], ms=3, lw=1.2, label=f"effect {e:g}")
        for _, r in s.iterrows():
            ax.annotate(f"{r['delta']:.2f}", (r["ratio"], r["power"]), fontsize=5,
                        color=cols[e], textcoords="offset points", xytext=(3, 3))
    ax.set_xscale("log")
    ax.set_xlabel("realised FDP / nominal\n(1.0 = honest)")
    ax.set_ylabel("power")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right", fontsize=5)


def panel_h(ax):
    """How much of the atlas any delta can reach.

    Effect sizes measured by LEAFCUTTER on the atlas, independent of our model. The test cannot
    speak to events below delta, so this is the panel that scopes every posterior-test claim, and it
    belongs in the main text rather than being left for a reader to derive."""
    tv = np.load(f"{M}/LEAFCUTTER_CMP_DELTAONLY/leafcutter_atlas_tv.npy")
    grid = np.linspace(0, 0.4, 200)
    frac = [(tv > t).mean() for t in grid]
    ax.plot(grid, frac, color=LFA, lw=1.6)
    for dl, lab in [(0.10, None), (0.15, "shipped")]:
        f = (tv > dl).mean()
        ax.axvline(dl, color=PALETTE["olive"] if dl == 0.15 else GREY,
                   lw=0.9, ls="--" if dl == 0.15 else ":")
        ax.annotate(f"$\\delta$={dl:.2f}\n{100*f:.0f}%", (dl, f), fontsize=6,
                    color=PALETTE["olive"] if dl == 0.15 else GREY,
                    textcoords="offset points", xytext=(5, 12))
    ax.set_xlabel("effect size measured by LeafCutter (TV)")
    ax.set_ylabel("fraction of atlas\n(contrast, ATSE) pairs above")
    ax.set_ylim(0, 1.0)


def panel_j(ax):
    """DS at the data's REAL overdispersion (concentration 2, the measured effective sample size)
    against the clean simulations (concentration 50). Answers "your simulations are cleaner than
    your data" with the number rather than a caveat."""
    d = pd.read_csv(f"{M}/DM_SHIPPED_SIM_CONC2/summary_conc2_vs_conc50.tsv", sep="\t")
    g = d.groupby(["method", "conc", "eff"]).agg(ratio=("ratio", "mean"), power=("power", "mean")).reset_index()
    ax.axhline(1.0, color=GREY, lw=0.8, ls=":", zorder=0)
    for meth, col, mk in [("LeafletFA LRT", LFA, "o"), ("LeafCutter", LC, "s")]:
        for conc, ls, alpha in [(50, "-", 0.45), (2, "-", 1.0)]:
            s_ = g[(g.method == meth) & (g.conc == conc)].sort_values("eff")
            ax.plot(s_.eff, s_.ratio, marker=mk, ls=ls, color=col, alpha=alpha, ms=3.5, lw=1.4,
                    label=f"{meth.replace(' LRT','')}, conc {conc}" + (" (atlas-like)" if conc == 2 else ""))
    ax.set_yscale("log"); ax.set_xscale("log"); ax.set_xticks([0.25, 0.5, 1.0]); ax.set_xticklabels(["0.25", "0.5", "1.0"])
    ax.set_xlabel("simulated effect size")
    ax.set_ylabel("realised FDP / nominal\n(at $\\alpha$ = 0.05)")
    ax.legend(frameon=False, fontsize=5, loc="upper right")
    ax.annotate("honest", (1.02, 1.0), xycoords=("axes fraction", "data"), fontsize=6, color=GREY, va="center")


# ---------------------------------------------------------------------------------------------
def main():
    apply_theme()
    fig, axes = plt.subplots(3, 3, figsize=figsize("double", h=7.6))
    for ax, fn, letter in zip(axes.ravel(),
                              [panel_a, panel_b, panel_c, panel_d, panel_e, panel_f,
                               panel_g, panel_h, panel_j],
                              "abcdefghi"):
        try:
            fn(ax)
        except Exception as e:                       # a broken panel must not kill the figure
            ax.text(0.5, 0.5, f"{letter}: {type(e).__name__}\n{e}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=5, color=PALETTE["grey"], wrap=True)
            print(f"  panel {letter} FAILED: {type(e).__name__}: {e}", flush=True)
        despine(ax)
        panel_label(ax, letter)
    fig.tight_layout(w_pad=1.6, h_pad=2.0)
    os.makedirs(PLOTS, exist_ok=True)
    save(fig, os.path.join(PLOTS, "FIG2_final"))
    print(f"-> {PLOTS}/FIG2_final.[pdf|png]")


if __name__ == "__main__":
    main()
