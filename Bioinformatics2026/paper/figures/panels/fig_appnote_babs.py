"""Figure AN panel b as absolute errors (Karin 2026-11-13): check 10.1 errors with check 14's shrinkage comparator and the
amendment 9 note 1 reference levels. Two layouts from one drawing function:
  main (ANbabsR_b, 85 x 50 mm, the current panel b slot): only the "TRAIN reads at the event" facet; used as figure AN's
       panel b (fig_appnote.panel_b; FIGAN_PANEL_B=diff restores the difference panel).
  supplementary (ANbabs_b): both facets (TRAIN cells per type; TRAIN reads at the event).

Per bin and atlas (TMS square, Allen circle): mean TV error against the held-out TEST pseudobulk of raw TRAIN pseudobulk
(gray open), LeafletFA (teal filled), joined by a line, and check 14 comparator B, shrinkage to the across-type pooled TRAIN
psi (indigo, "Shrinkage to pooled ψ"), each with the 10.1 bootstrap CI over cell types (1,000, seed 0). Reference levels
(not tests): "no cell-type information" = pooled TRAIN psi for every type (gray dashed); "noise floor" = TEST split halves
scaled to full depth (gray band from 0). Above a bin: change in error, LeafletFA vs raw, where the 10.1 difference CI
excludes 0 (top row TMS, second row Allen). Under a bin (gray): % of catalog rows in the bin, TMS over Allen.
Input: $V/c14_shrinkage/<ds>_abs_panel.tsv (c14_reference.py, c14_abs_B.py). No new analysis."""
import os
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from style import axes_in, color, container_mm, sub_slot

V = "/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation"
FACETS = {"size": (["10-49", "50-199", "200-999", ">=1000"], "TRAIN cells per type"),
          "reads": (["0", "1-19", "20-99", ">=100"], "TRAIN reads at the event")}
B_LABEL = "Shrinkage to pooled ψ"


def _table():
    return pd.concat([pd.read_csv(os.path.join(V, "c14_shrinkage", f"{ds}_abs_panel.tsv"), sep="\t", dtype={"bin": str})
                      for ds in ("TMS", "Allen")], ignore_index=True)


def _style(cfg):
    p = cfg["palette"]
    return dict(sm=cfg["style"]["font"]["small_pt"], ink=p["ink"], gray=p["gray"], light=p["light_gray"], indigo=p["indigo"],
                teal=color(cfg, "leafletfa"), ref_lw=cfg["style"]["line"]["reference_pt"],
                mk={"TMS": cfg["data"]["tms_data"]["marker"], "Allen": cfg["data"]["allen_data"]["marker"]})


def _facet(ax, A, by, S, top, rotate):
    bins, xlabel = FACETS[by]
    for i, b in enumerate(bins):
        for o, ds in ((-0.22, "TMS"), (0.22, "Allen")):
            r = A[(A.dataset == ds) & (A.by == by) & (A.bin == b)].iloc[0]
            x = i + o
            ax.fill_between([x - 0.2, x + 0.2], 0, r.floor, color=S["light"], alpha=0.6, lw=0, zorder=0)
            ax.plot([x - 0.2, x + 0.2], [r.upper] * 2, color=S["gray"], lw=S["ref_lw"], ls=(0, (3, 2)), zorder=1)
            xr, xl, xb = x - 0.12, x, x + 0.12
            ax.plot([xr, xl], [r.raw, r.leafletfa], color=S["gray"], lw=0.6, zorder=2)
            for xx, lo, hi, c in ((xr, r.raw_lo, r.raw_hi, S["gray"]), (xl, r.leafletfa_lo, r.leafletfa_hi, S["teal"]),
                                  (xb, r.B_lo, r.B_hi, S["indigo"])):
                ax.vlines(xx, lo, hi, color=c, lw=0.8, zorder=3)
            ax.plot(xr, r.raw, ls="none", marker=S["mk"][ds], ms=2.8, mfc="white", mec=S["gray"], mew=0.7, zorder=4)
            ax.plot(xl, r.leafletfa, ls="none", marker=S["mk"][ds], ms=2.8, mfc=S["teal"], mec=S["teal"], mew=0.7, zorder=5)
            ax.plot(xb, r.B, ls="none", marker=S["mk"][ds], ms=2.8, mfc=S["indigo"], mec=S["indigo"], mew=0.7, zorder=4)
            if r.diff_ci_excludes_0:
                ch = -r.pct_reduction
                ax.text(x, 0.985 if ds == "TMS" else 0.90, f"{ch:+.0f}%".replace("-", "−"), transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=S["sm"], color=S["teal"] if ch < 0 else S["ink"])
        s = [A[(A.dataset == ds) & (A.by == by) & (A.bin == b)].iloc[0].catalog_rows_share for ds in ("TMS", "Allen")]
        ax.text(i, -0.25 if rotate else -0.13, f"{100 * s[0]:.0f}%\n{100 * s[1]:.0f}%", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=S["sm"], color=S["gray"], linespacing=1.0)
    ax.set_xticks(np.arange(len(bins)))
    lab = [b.replace(">=", "≥").replace("-", "–") for b in bins]
    if rotate:
        ax.set_xticklabels(lab, rotation=35, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticklabels(lab)
    ax.set_xlim(-0.66, len(bins) - 0.5); ax.set_ylim(0, top)
    ax.set_xlabel(xlabel); ax.xaxis.set_label_coords(0.5, -0.40 if rotate else -0.37); ax.xaxis.label.set_va("top")


def _handles(S):
    return [Line2D([], [], ls="none", marker="o", ms=2.8, mfc=S["teal"], mec=S["teal"], label="LeafletFA"),
            Line2D([], [], ls="none", marker="o", ms=2.8, mfc=S["indigo"], mec=S["indigo"], label=B_LABEL),
            Line2D([], [], ls="none", marker="o", ms=2.8, mfc="white", mec=S["gray"], mew=0.7, label="Raw pseudobulk"),
            Line2D([], [], color=S["gray"], lw=S["ref_lw"], ls=(0, (3, 2)), label="All-cells average"),
            Patch(facecolor=S["light"], alpha=0.6, lw=0, label="Best achievable"),
            Line2D([], [], ls="none", marker=S["mk"]["TMS"], ms=2.8, mfc=S["ink"], mec=S["ink"], label="TMS"),
            Line2D([], [], ls="none", marker=S["mk"]["Allen"], ms=2.8, mfc=S["ink"], mec=S["ink"], label="Allen Brain")]


def _top(A, bys):
    a = A[A.by.isin(bys)]
    return max(a.raw_hi.max(), a.upper_hi.max(), a.leafletfa_hi.max(), a.B_hi.max()) * 1.38


def panel_b_main(cfg, fig):
    """85 x 50 mm (Karin 2026-11-13, simplified): one facet per atlas (TMS, Allen), bins 1-19, 20-99, >=100 TRAIN reads
    (the 0-read bin has no catalog row and raw equals the all-cells average there); one marker shape; one percent label per
    bin (LeafletFA vs raw, where the 10.1 CI excludes 0); catalog shares go to the caption facts."""
    w, h = container_mm(fig); A = _table(); S = _style(cfg)
    bins = ["1-19", "20-99", ">=100"]; leg_h = 7.0
    a = A[(A.by == "reads") & A.bin.isin(bins)]
    top = max(a.raw_hi.max(), a.upper_hi.max(), a.leafletfa_hi.max(), a.B_hi.max()) * 1.18
    wl = 13.0; wf = (w - wl - 3.0) / 2
    for f, (ds, title) in enumerate((("TMS", "TMS"), ("Allen", "Allen Brain"))):
        x0 = 0 if f == 0 else wl + wf + 1.5
        ax = axes_in(sub_slot(fig, x0, 0, (wl + wf) if f == 0 else wf + 1.5, h - leg_h), (wl if f == 0 else 1.5, 9.0, 0.5, 3.2))
        for i, b in enumerate(bins):
            r = a[(a.dataset == ds) & (a.bin == b)].iloc[0]
            ax.fill_between([i - 0.36, i + 0.36], 0, r.floor, color=S["light"], alpha=0.6, lw=0, zorder=0)
            ax.plot([i - 0.36, i + 0.36], [r.upper] * 2, color=S["gray"], lw=S["ref_lw"], ls=(0, (3, 2)), zorder=1)
            xr, xl, xb = i - 0.2, i, i + 0.2
            ax.plot([xr, xl], [r.raw, r.leafletfa], color=S["gray"], lw=0.6, zorder=2)
            for xx, lo, hi, c in ((xr, r.raw_lo, r.raw_hi, S["gray"]), (xl, r.leafletfa_lo, r.leafletfa_hi, S["teal"]),
                                  (xb, r.B_lo, r.B_hi, S["indigo"])):
                ax.vlines(xx, lo, hi, color=c, lw=0.8, zorder=3)
            ax.plot(xr, r.raw, ls="none", marker="o", ms=2.8, mfc="white", mec=S["gray"], mew=0.7, zorder=4)
            ax.plot(xl, r.leafletfa, ls="none", marker="o", ms=2.8, mfc=S["teal"], mec=S["teal"], zorder=5)
            ax.plot(xb, r.B, ls="none", marker="o", ms=2.8, mfc=S["indigo"], mec=S["indigo"], zorder=4)
            if r.diff_ci_excludes_0:
                ch = -r.pct_reduction
                ax.text(i, 0.99, f"{ch:+.0f}%".replace("-", "−"), transform=ax.get_xaxis_transform(), ha="center", va="top",
                        fontsize=S["sm"], color=S["teal"] if ch < 0 else S["ink"])
        ax.set_xticks(np.arange(len(bins))); ax.set_xticklabels([b.replace(">=", "≥").replace("-", "–") for b in bins])
        ax.set_xlim(-0.5, len(bins) - 0.5); ax.set_ylim(0, top)
        ax.set_title(title, fontsize=S["sm"], loc="left", pad=1.0, color=S["ink"])
        if f == 0:
            ax.set_ylabel("Error against held-out cells\n(0 = identical)")
        else:
            ax.set_yticklabels([])
    xl_ = axes_in(sub_slot(fig, wl, 0, w - wl, 4.0), (0, 0, 0, 0)); xl_.axis("off")
    xl_.text(0.5, 0.15, "TRAIN reads at the event", ha="center", va="bottom", fontsize=cfg["style"]["font"]["size_pt"],
             color=S["ink"], transform=xl_.transAxes)
    lx = axes_in(sub_slot(fig, 0, h - leg_h, w, leg_h), (wl, 0.2, 0.5, 0.2)); lx.axis("off")
    hs = [hh for hh in _handles(S) if hh.get_label() not in ("TMS", "Allen Brain")]
    lx.legend(handles=hs, loc="center left", ncol=3, frameon=False, fontsize=S["sm"], handlelength=1.2, columnspacing=0.8,
              labelspacing=0.2, borderpad=0.0, borderaxespad=0)


def panel_b_supp(cfg, fig):
    """both facets; legend strip on top"""
    w, h = container_mm(fig); A = _table(); S = _style(cfg)
    leg_h = 12.0; top = _top(A, ["size", "reads"])
    for f, by in enumerate(("size", "reads")):
        ax = axes_in(sub_slot(fig, f * w / 2, 0, w / 2, h - leg_h), (13.0 if f == 0 else 6.5, 19.0, 1.5, 1.5))
        _facet(ax, A, by, S, top, rotate=True)
        if f == 0:
            ax.set_ylabel("Error against held-out cells\n(0 = identical)")
        else:
            ax.set_yticklabels([])
    lx = axes_in(sub_slot(fig, 0, h - leg_h, w, leg_h), (13.0, 0.3, 1.5, 0.3)); lx.axis("off")
    lx.legend(handles=_handles(S), loc="upper left", ncol=3, frameon=False, fontsize=S["sm"], handlelength=1.2, columnspacing=0.9,
              labelspacing=0.25, borderpad=0.1, borderaxespad=0)
    lx.text(1.0, 0.0, "Gray under bins: % of catalog rows (TMS over Allen)", transform=lx.transAxes, ha="right", va="bottom",
            fontsize=S["sm"], color=S["gray"])


PANELS = {"ANbabsR_b": panel_b_main, "ANbabs_b": panel_b_supp}
SOURCE_DATA = {"ANbabsR_b": lambda cfg: {"figAN_b_abs_numbers": _table()},
               "ANbabs_b": lambda cfg: {"figAN_b_abs_numbers": _table()}}
