#!/usr/bin/env python
"""The results, as figures. Run once the factorial and cross-seed analyses have landed.

Run: python panels_results.py --dir <DM_TEST> --out <panels dir>

  a  CALIBRATION CURVES -- realised FDP against nominal alpha, one line per model cell, faceted
     by generator (binom vs dm) and effect size. The diagonal is perfect calibration; above it
     is anti-conservative. This is the headline: how far each cell sits from the diagonal, and
     whether that changes when the data is generated compositionally.
  b  POWER against nominal alpha, same facets. Read with (a): a cell that hugs the diagonal
     but sits at the bottom of (b) is trustworthy and blind.
  c  THE TRADE -- realised FDP against power at nominal 0.05, one point per cell per facet.
     Down and to the right is better. Makes the calibration/power trade a single glance.
  d  CROSS-SEED agreement -- for each cell, the correlation of the ATSE-level TV statistic
     between seed pairs, plus the within-ATSE sum correlation (the unidentified direction the
     simplex is supposed to pin) and the composition correlation (the biology).
  e  JOINT STRUCTURE -- observed within-event junction correlation against what each likelihood
     implies. The BetaBinomial implies exactly zero and cannot represent the competition for a
     fixed total; the DM implies a negative value.

Colours from the project palette; labels 1-indexed.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/"
                   "figures/_common")
import leaflet_theme as th  # noqa: E402

CELL_ORDER = ["betabinomial_beta", "betabinomial_simplex",
              "dirichletmultinomial_beta", "dirichletmultinomial_simplex"]
CELL_LABEL = {"betabinomial_beta": "BB + beta (current)",
              "betabinomial_simplex": "BB + simplex",
              "dirichletmultinomial_beta": "DM + beta",
              "dirichletmultinomial_simplex": "DM + simplex"}


def cell_colours():
    p = th.PALETTE
    return {"betabinomial_beta": p["grey"], "betabinomial_simplex": p["ochre"],
            "dirichletmultinomial_beta": p["indigo"],
            "dirichletmultinomial_simplex": p["teal"]}


def facet_of(sim):
    gen = "compositional (DM)" if "dm50" in sim else "independent (Binom)"
    eff = "0.25" if "eff0.25" in sim else "0.50"
    return f"{gen}\neffect {eff}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    th.apply_theme()
    C = cell_colours()

    files = sorted(f for f in glob.glob(os.path.join(args.dir, "a36_*.tsv"))
                   if "_summary" not in os.path.basename(f))
    if not files:
        print(f"no a36_*.tsv under {args.dir}")
        return
    D = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)
    D["facet"] = D["sim"].map(facet_of)
    facets = sorted(D.facet.unique())
    print(f"loaded {len(files)} cells across {len(facets)} facets")
    for fc in facets:
        have = D[D.facet == fc].groupby("cell").seed.nunique().to_dict()
        print(f"  {fc.replace(chr(10),' '):<32} {have}")

    # ── a + b: calibration and power curves ─────────────────────────────────────────────
    for metric, lab, fname in (("realised_fdp", "realised FDP", "calibration"),
                               ("power", "power", "power")):
        fig, axes = plt.subplots(1, len(facets), figsize=th.figsize("double", h=2.7),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        for ax, fc in zip(axes, facets):
            g = D[D.facet == fc]
            for cell in CELL_ORDER:
                gg = g[g.cell == cell]
                if not len(gg):
                    continue
                m = gg.groupby("alpha")[metric].mean()
                ax.plot(m.index, m.values, marker="o", ms=3, lw=1.6, color=C[cell],
                        label=CELL_LABEL[cell])
            if metric == "realised_fdp":
                lim = [1e-3, 0.25]
                ax.plot(lim, lim, color=th.INK, lw=0.8, ls="--", zorder=0)
                ax.set_xscale("log"); ax.set_yscale("log")
                ax.set_xlim(*lim); ax.set_ylim(*lim)
            ax.set_xlabel("nominal $\\alpha$")
            ax.set_title(fc, fontsize=th.PT["annot"])
        axes[0].set_ylabel(lab)
        axes[0].legend(frameon=False, fontsize=th.PT["legend"], loc="best")
        if metric == "realised_fdp":
            axes[0].annotate("above the dashed line\n= anti-conservative", (0.0015, 0.09),
                             fontsize=th.PT["annot"], color=th.INK)
        th.panel_label(axes[0], "a" if metric == "realised_fdp" else "b")
        fig.tight_layout()
        th.save(fig, os.path.join(args.out, f"results_{fname}"))
        plt.close(fig)

    # ── c: the trade ────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(facets), figsize=th.figsize("double", h=2.7), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fc in zip(axes, facets):
        A = D[(D.facet == fc) & np.isclose(D.alpha, args.alpha)]
        for cell in CELL_ORDER:
            gg = A[A.cell == cell]
            if not len(gg):
                continue
            ax.scatter(gg.power.mean(), gg.realised_fdp.mean(), s=42, color=C[cell],
                       label=CELL_LABEL[cell], zorder=3, edgecolors="white", linewidths=0.8)
        ax.axhline(args.alpha, color=th.INK, lw=0.9, ls="--")
        ax.set_xlabel("power"); ax.set_title(fc, fontsize=th.PT["annot"])
    axes[0].set_ylabel(f"realised FDP at $\\alpha$={args.alpha}")
    axes[0].legend(frameon=False, fontsize=th.PT["legend"], loc="best")
    axes[0].annotate("nominal", (axes[0].get_xlim()[0], args.alpha), fontsize=th.PT["annot"],
                     color=th.INK, xytext=(2, 3), textcoords="offset points")
    th.panel_label(axes[0], "c")
    fig.tight_layout()
    th.save(fig, os.path.join(args.out, "results_trade"))
    plt.close(fig)

    # ── d: cross-seed ───────────────────────────────────────────────────────────────────
    sp = os.path.join(args.dir, "a41_cross_seed.tsv")
    if os.path.exists(sp):
        S = pd.read_csv(sp, sep="\t")
        S["facet"] = S["sim"].map(facet_of)
        mets = [("sum_r", "within-ATSE sum\n(unidentified)"), ("comp_r", "composition\n(biology)"),
                ("tv_r", "TV statistic\n(the DS call)")]
        fig, axes = plt.subplots(1, len(mets), figsize=th.figsize("double", h=2.7), sharey=True)
        for ax, (m, lab) in zip(np.atleast_1d(axes), mets):
            piv = S.pivot_table(index="cell", columns="facet", values=m)
            piv = piv.reindex([c for c in CELL_ORDER if c in piv.index])
            x = np.arange(len(piv)); w = 0.8 / max(piv.shape[1], 1)
            for i, fc in enumerate(piv.columns):
                ax.bar(x + i * w - 0.4 + w / 2, piv[fc].values, w,
                       color=[th.PALETTE["teal"], th.PALETTE["ochre"],
                              th.PALETTE["indigo"], th.PALETTE["rose"]][i % 4],
                       linewidth=0, label=fc.replace("\n", ", ") if m == mets[0][0] else None)
            ax.set_xticks(x)
            ax.set_xticklabels([CELL_LABEL[c].split(" (")[0] for c in piv.index],
                               rotation=30, ha="right", fontsize=th.PT["tick"])
            ax.set_ylim(0, 1.05)
            ax.set_title(lab, fontsize=th.PT["annot"])
        np.atleast_1d(axes)[0].set_ylabel("across-seed correlation")
        np.atleast_1d(axes)[0].legend(frameon=False, fontsize=th.PT["legend"], loc="lower left")
        th.panel_label(np.atleast_1d(axes)[0], "d")
        fig.tight_layout()
        th.save(fig, os.path.join(args.out, "results_cross_seed"))
        plt.close(fig)
        print("  cross-seed panel written")
    else:
        print("  (no a41_cross_seed.tsv yet -- panel d skipped)")

    # ── e: joint structure ──────────────────────────────────────────────────────────────
    jp = os.path.join(args.dir, "a41_joint_structure.tsv")
    if os.path.exists(jp):
        J = pd.read_csv(jp, sep="\t")
        J["facet"] = J["sim"].map(facet_of)
        fig, ax = plt.subplots(figsize=th.figsize("single", h=2.7))
        cells = [c for c in CELL_ORDER if c in set(J.cell)]
        x = np.arange(len(cells))
        obs = J.groupby("cell").observed_mean_corr.mean().reindex(cells)
        imp = J.groupby("cell").implied_mean_corr.mean().reindex(cells)
        ax.bar(x - 0.2, obs.values, 0.4, color=th.PALETTE["grey"], linewidth=0, label="observed")
        ax.bar(x + 0.2, imp.values, 0.4, color=th.PALETTE["teal"], linewidth=0,
               label="implied by model")
        ax.axhline(0, color=th.INK, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([CELL_LABEL[c].split(" (")[0] for c in cells], rotation=30,
                           ha="right", fontsize=th.PT["tick"])
        ax.set_ylabel("within-event junction correlation")
        ax.set_title("Joint structure", fontsize=th.PT["panel"])
        ax.legend(frameon=False, fontsize=th.PT["legend"])
        th.panel_label(ax, "e")
        fig.tight_layout()
        th.save(fig, os.path.join(args.out, "results_joint_structure"))
        plt.close(fig)
        print("  joint-structure panel written")
    else:
        print("  (no a41_joint_structure.tsv yet -- panel e skipped)")

    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
