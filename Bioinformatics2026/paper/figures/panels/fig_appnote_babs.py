"""Alternative Figure AN panel b (Karin 2026-11-13; the current panel b is kept): check 10.1 as absolute errors.

Per bin (TRAIN cells per type; TRAIN reads at the event), both atlases: mean TV error against the held-out TEST pseudobulk
of raw TRAIN pseudobulk (gray open marker) and LeafletFA (teal filled), joined by a line, with the 10.1 bootstrap CI over
cell types (1,000, seed 0). Reference levels (amendment 9 note 1, not tests): "no cell-type information" = the across-type
pooled TRAIN psi used for every type (gray dashed); "noise floor" = TEST split halves scaled to full depth (gray band from 0).
Above a bin: the change in error, LeafletFA vs raw, where the 10.1 difference CI excludes 0. Under a bin (gray): share of
catalog rows (junction x type) in that bin, TMS / Allen. Atlas by marker shape (TMS square, Allen circle).
Input: $V/c14_shrinkage/<ds>_abs_panel.tsv (c14_reference.py). No new analysis."""
import os
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from style import axes_in, color, container_mm, sub_slot

V = "/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation"
FACETS = [("size", ["10-49", "50-199", "200-999", ">=1000"], "TRAIN cells per type"),
          ("reads", ["0", "1-19", "20-99", ">=100"], "TRAIN reads at the event")]


def _table():
    return pd.concat([pd.read_csv(os.path.join(V, "c14_shrinkage", f"{ds}_abs_panel.tsv"), sep="\t") for ds in ("TMS", "Allen")],
                     ignore_index=True)


def panel_b_abs(cfg, fig):
    w, h = container_mm(fig)
    A = _table()
    sm = cfg["style"]["font"]["small_pt"]
    ink, gray, light = cfg["palette"]["ink"], cfg["palette"]["gray"], cfg["palette"]["light_gray"]
    teal = color(cfg, "leafletfa")
    ref_lw = cfg["style"]["line"]["reference_pt"]
    top = max(A.raw_hi.max(), A.upper_hi.max(), A.leafletfa_hi.max()) * 1.38
    mk = {"TMS": cfg["data"]["tms_data"]["marker"], "Allen": cfg["data"]["allen_data"]["marker"]}
    leg_h = 11.0
    for f, (by, bins, xlabel) in enumerate(FACETS):
        ax = axes_in(sub_slot(fig, f * w / 2, 0, w / 2, h - leg_h), (13.0 if f == 0 else 6.5, 19.0, 1.5, 1.5))
        for i, b in enumerate(bins):
            for o, ds in ((-0.21, "TMS"), (0.21, "Allen")):
                r = A[(A.dataset == ds) & (A.by == by) & (A.bin == b)].iloc[0]
                x = i + o
                ax.fill_between([x - 0.18, x + 0.18], 0, r.floor, color=light, alpha=0.6, lw=0, zorder=0)
                ax.plot([x - 0.18, x + 0.18], [r.upper] * 2, color=gray, lw=ref_lw, ls=(0, (3, 2)), zorder=1)
                xr, xl = x - 0.07, x + 0.07
                ax.plot([xr, xl], [r.raw, r.leafletfa], color=gray, lw=0.6, zorder=2)
                ax.vlines(xr, r.raw_lo, r.raw_hi, color=gray, lw=0.8, zorder=2)
                ax.vlines(xl, r.leafletfa_lo, r.leafletfa_hi, color=teal, lw=0.8, zorder=3)
                ax.plot(xr, r.raw, ls="none", marker=mk[ds], ms=3.0, mfc="white", mec=gray, mew=0.7, zorder=3)
                ax.plot(xl, r.leafletfa, ls="none", marker=mk[ds], ms=3.0, mfc=teal, mec=teal, mew=0.7, zorder=4)
                if r.diff_ci_excludes_0:             # fixed rows at the top: TMS above Allen, over each atlas's markers
                    ch = -r.pct_reduction
                    ax.text(x, 0.985 if ds == "TMS" else 0.905, f"{ch:+.0f}%".replace("-", "−"), transform=ax.get_xaxis_transform(),
                            ha="center", va="top", fontsize=sm, color=teal if ch < 0 else ink)
            s = [A[(A.dataset == ds) & (A.by == by) & (A.bin == b)].iloc[0].catalog_rows_share for ds in ("TMS", "Allen")]
            ax.text(i, -0.25, f"{100 * s[0]:.0f}%\n{100 * s[1]:.0f}%", transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=sm, color=gray, linespacing=1.0)
        ax.set_xticks(np.arange(len(bins)))
        ax.set_xticklabels([b.replace(">=", "≥").replace("-", "–") for b in bins], rotation=35, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.66, len(bins) - 0.5); ax.set_ylim(0, top)
        ax.set_xlabel(xlabel); ax.xaxis.set_label_coords(0.5, -0.40); ax.xaxis.label.set_va("top")
        if f == 0:
            ax.set_ylabel("Error against held-out cells\n(0 = identical)")
        else:
            ax.set_yticklabels([])
    lx = axes_in(sub_slot(fig, 0, h - leg_h, w, leg_h), (13.0, 0.3, 1.5, 0.3)); lx.axis("off")
    hs = [Line2D([], [], ls="none", marker="o", ms=3.0, mfc=teal, mec=teal, label="LeafletFA"),
          Line2D([], [], ls="none", marker="o", ms=3.0, mfc="white", mec=gray, mew=0.7, label="Raw pseudobulk"),
          Line2D([], [], ls="none", marker=mk["TMS"], ms=3.0, mfc=ink, mec=ink, label="TMS"),
          Line2D([], [], color=gray, lw=ref_lw, ls=(0, (3, 2)), label="No cell-type information"),
          Patch(facecolor=light, alpha=0.6, lw=0, label="Noise floor"),
          Line2D([], [], ls="none", marker=mk["Allen"], ms=3.0, mfc=ink, mec=ink, label="Allen Brain")]
    lx.text(0.0, 0.0, "Gray under bins: % of catalog rows (TMS, Allen)", transform=lx.transAxes, ha="left", va="bottom",
            fontsize=sm, color=gray)
    lx.legend(handles=hs, loc="upper left", ncol=3, frameon=False, fontsize=sm, handlelength=1.2, columnspacing=0.9,
              labelspacing=0.25, borderpad=0.1, borderaxespad=0)


PANELS = {"ANbabs_b": panel_b_abs}
SOURCE_DATA = {"ANbabs_b": lambda cfg: {"figAN_b_abs_numbers": _table()}}
