#!/usr/bin/env python
"""Figure 2 supplement: the panels that left the main figure when it was rebuilt as v2.
K = 5 on every simulation; SD bars are over 3 model seeds where the underlying table has them."""
import os, sys
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import build_fig2_final as F
from leaflet_theme import apply_theme, figsize, panel_label, despine, save, PALETTE

def main():
    apply_theme()
    fig, axes = plt.subplots(2, 3, figsize=figsize("double", h=5.4))
    for ax, fn, L, title in zip(axes.ravel(), [F.panel_b, F.panel_c, F.panel_f, F.panel_g, F.panel_h, F.panel_j], "abcdef",
                                ["concentration recovery", "$\\psi$ recovery vs read depth", "$\\phi$ vs truth by training regime",
                                 "choosing $\\delta$: calibration vs power", "how much of the atlas any $\\delta$ reaches", "DS at concentration 2 (atlas-like)"]):
        try: fn(ax)
        except Exception as e:
            ax.text(.5, .5, f"{L}: {type(e).__name__}\n{e}", ha="center", va="center", transform=ax.transAxes, fontsize=5, color=PALETTE["grey"], wrap=True)
            print(f"  panel {L} FAILED: {e}")
        ax.set_title(title, fontsize=7, loc="left"); despine(ax); panel_label(ax, L)
    fig.tight_layout(w_pad=1.4, h_pad=1.8)
    save(fig, os.path.join(F.PLOTS, "FIG2_supp_from_v1")); print(f"-> {F.PLOTS}/FIG2_supp_from_v1")

if __name__ == "__main__":
    main()
