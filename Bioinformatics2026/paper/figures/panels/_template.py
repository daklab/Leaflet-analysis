"""Figure N: <one-sentence claim of the figure>.

Copy this file to figN.py, rename the functions, register the module in make_figures.FIGURE_MODULES and add the panel
sizes / layout / inputs to paper_config.yaml. Panel functions draw only: no saving, sizing, fonts or literal colors."""
import os

import pandas as pd

from style import axes_in, color, method_style, reference_line  # noqa: F401

FIG = "N"


def _inp(cfg, key):
    """Path of an input file: cfg["inputs"]["<analysis>"] holds a folder and file names (see paper_config.yaml)."""
    block = cfg["inputs"]["<analysis>"]
    return os.path.join(block["dir"], block[key])


def panel_Nb(cfg, fig):
    """<What the panel shows, in one line.> <Details: data, threshold, what the marks mean.>
    Input: inputs.<analysis>.<key>."""
    d = pd.read_csv(_inp(cfg, "<key>"), sep="\t")
    ax = axes_in(fig, (9.5, 8, 2, 2.5))     # margins in mm: left, bottom, right, top
    ax.plot(d["x"], d["y"], **method_style(cfg, "splicevi"))   # color, marker, line style, label from `methods:`
    ax.set_xlabel("<x label (unit)>")
    ax.set_ylabel("<y label (unit)>")
    ax.legend(loc="best")


PANELS = {"Nb": panel_Nb}
