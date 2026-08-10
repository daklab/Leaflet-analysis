#!/usr/bin/env python
"""FIGURE 2 panels: what the atlas's data regime does to the per-cell encoding.

The result these panels carry
-----------------------------
Every simulation validating LeafletFA ran far from the data it is applied to. Measured, the atlas
has a median of 6.0 reads per observed cell-junction entry with 25.2% at 1-2 reads, while the
simulations sat at 33.0 and 9.3% -- roughly 5.5x deeper -- and used concentrations of 10-200 against
the atlas's fitted ~2. Moving a simulation INTO the atlas's regime (concentration 2, depth thinned
to a median of 6.0 reads, matched to the atlas exactly) reproduces a failure that had only ever been
seen on real data: psi survives, phi collapses, and the group contrast stops being reproducible
between identical runs.

Because this is simulation, phi can be scored against obsm['true_phi'] -- which is impossible on the
atlas, where there is nothing to compare against and only run-to-run agreement can be measured.
That is the whole reason the diagnosis had to move here.

Panels
------
a  the joint criterion. Effect size recovered (x) against run-to-run spread (y). The target is the
   bottom-right corner: right size AND reproducible. Neither axis alone is sufficient -- an arm that
   shrinks every contrast to zero is perfectly reproducible and useless, which is not hypothetical
   (delta=5K lands bottom-left-ish on size while being the WORST on spread).
b  phi against the truth, per arm, with the control as the reference line. This is the quantity the
   atlas cannot measure.
c  what breaks and what does not: psi cross-seed agreement barely moves while phi's collapses. The
   same signature appears on the atlas, which is what ties the simulation to the real failure.

Usage:
    python build_regime_phi_panels.py [--scores <a56_scores.tsv>]
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGROOT = "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures"
sys.path.insert(0, FIGROOT)
sys.path.insert(0, os.path.join(FIGROOT, "_common"))
from leaflet_theme import (apply_theme, PALETTE, figsize, panel_label,  # noqa: E402
                           despine, save)

WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
DEFAULT_SCORES = f"{WD}/ATLAS_DM/regime/a56_scores.tsv"

# Display order and labels. Position encodes the arm -- NOT colour: there are six arms and the
# palette is validated for five series, so using colour here would break greyscale/dichromat
# separation. Colour is reserved for the one distinction that matters (control vs atlas regime vs
# the shipped recipe).
ORDER = ["ctrl", "base", "dc5", "dc25", "wt", "both"]
# Pre-cased deliberately. leaflet_theme.save() sentence-cases tick labels whenever ANY label on
# the axis starts with a lowercase letter -- which turned "m/(m-1)" into "M/(m-1)", i.e. changed a
# symbol. No label here starts with a lowercase letter, so the hook leaves them alone and the
# mathtext m stays lowercase.
LABEL = {
    "ctrl": "Validated\nregime",
    "base": "Atlas\nregime",
    "dc5":  "$\\delta$ = K",
    "dc25": "$\\delta$ = 5K",
    "wt":   "$m/(m\\!-\\!1)$\nweight",
    "both": "Both",
}
# teal = the shipped recipe, olive = the control we are trying to get back to, grey = context
COLOR = {
    "ctrl": PALETTE["olive"],
    "base": PALETTE["grey"],
    "dc5":  PALETTE["grey"],
    "dc25": PALETTE["grey"],
    "wt":   PALETTE["grey"],
    "both": PALETTE["teal"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=DEFAULT_SCORES)
    ap.add_argument("--out", default=os.path.join(
        FIGROOT, "FIGURE2_simulation_imputation", "plots", "FIG2_regime_phi_collapse"))
    args = ap.parse_args()

    d = pd.read_csv(args.scores, sep="\t").set_index("arm")
    d = d.loc[[a for a in ORDER if a in d.index]]
    arms = list(d.index)

    apply_theme()
    fig, axes = plt.subplots(1, 3, figsize=figsize("double", 2.9))

    # ── a. the joint criterion ─────────────────────────────────────────────────────────────
    ax = axes[0]
    for a in arms:
        ax.scatter(d.loc[a, "contrast_mag_vs_true"], d.loc[a, "contrast_cv"],
                   s=46, color=COLOR[a], zorder=3,
                   edgecolor="white", linewidth=0.6)
    # label points without overlapping: control and 'both' are the two that matter
    off = {"ctrl": (-0.02, 0.012), "base": (0.015, 0.006), "dc5": (0.015, 0.004),
           "dc25": (0.015, 0.004), "wt": (0.015, 0.006), "both": (0.012, -0.016)}
    for a in arms:
        dx, dy = off.get(a, (0.01, 0.01))
        ax.annotate(LABEL[a].replace("\n", " "),
                    (d.loc[a, "contrast_mag_vs_true"] + dx, d.loc[a, "contrast_cv"] + dy),
                    fontsize=5.6, color=COLOR[a], va="center")
    ax.axvline(1.0, color=PALETTE["grey"], lw=0.6, ls=":", zorder=1)
    ax.set_xlabel("effect size recovered (1.0 = true)")
    ax.set_ylabel("run-to-run spread (CV)")
    ax.set_xlim(0.22, 1.16)
    ax.set_ylim(-0.02, 0.245)
    ax.invert_yaxis()          # so "better" is UP as well as right
    # An inverted axis is easy to misread, so say which way is good rather than relying on it.
    ax.annotate("", xy=(1.10, 0.012), xytext=(1.10, 0.105),
                arrowprops=dict(arrowstyle="->", lw=0.7, color=PALETTE["grey"]))
    ax.text(1.115, 0.058, "better", rotation=90, va="center", ha="left",
            fontsize=5.6, color=PALETTE["grey"])
    despine(ax)
    panel_label(ax, "a")

    # ── b. phi against the truth ───────────────────────────────────────────────────────────
    ax = axes[1]
    x = np.arange(len(arms))
    ax.bar(x, d["phi_r_true"].values, width=0.68,
           color=[COLOR[a] for a in arms], zorder=3)
    ax.axhline(d.loc["ctrl", "phi_r_true"], color=PALETTE["olive"], lw=0.7, ls="--", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[a] for a in arms], fontsize=5.4)
    ax.set_ylabel("corr($\\phi$, true $\\phi$)")
    ax.set_ylim(0, 1.05)
    despine(ax)
    panel_label(ax, "b")

    # ── c. what breaks, and what does not ──────────────────────────────────────────────────
    ax = axes[2]
    w = 0.36
    ax.bar(x - w / 2, d["psi_r_seed"].values, width=w, color=PALETTE["indigo"],
           label="$\\psi$ (programs)", zorder=3)
    ax.bar(x + w / 2, d["phi_r_seed"].values, width=w, color=PALETTE["rose"],
           label="$\\phi$ (per-cell)", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[a] for a in arms], fontsize=5.4)
    ax.set_ylabel("cross-seed agreement (r)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=5.6, loc="upper right", handlelength=1.1)
    despine(ax)
    panel_label(ax, "c")

    fig.tight_layout(w_pad=1.6)
    save(fig, args.out)
    print(f"saved -> {args.out}.pdf / .png")

    # the numbers the caption has to quote, so they are never retyped by hand
    c, b, t = d.loc["ctrl"], d.loc["base"], d.loc["both"]
    print("\ncaption numbers")
    print(f"  control        phi_r_true {c.phi_r_true:.3f}  mag {c.contrast_mag_vs_true:.3f}  "
          f"cv {c.contrast_cv:.3f}  ARI {c.phi_ari:.3f}")
    print(f"  atlas regime   phi_r_true {b.phi_r_true:.3f}  mag {b.contrast_mag_vs_true:.3f}  "
          f"cv {b.contrast_cv:.3f}  ARI {b.phi_ari:.3f}")
    print(f"  both           phi_r_true {t.phi_r_true:.3f}  mag {t.contrast_mag_vs_true:.3f}  "
          f"cv {t.contrast_cv:.3f}  ARI {t.phi_ari:.3f}")
    gap = lambda x0, x1, x2: (x2 - x0) / (x1 - x0)       # noqa: E731
    print(f"  gap recovered  phi {100*gap(b.phi_r_true, c.phi_r_true, t.phi_r_true):.0f}%  "
          f"mag {100*gap(b.contrast_mag_vs_true, c.contrast_mag_vs_true, t.contrast_mag_vs_true):.0f}%  "
          f"cv {100*gap(b.contrast_cv, c.contrast_cv, t.contrast_cv):.0f}%")


if __name__ == "__main__":
    main()
