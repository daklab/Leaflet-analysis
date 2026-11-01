"""Figure 1: ATSEmapper defines splicing events, and the mouse catalog they form is sparse per cell.

Panels with code: 1b catalog per dataset, 1c event types and junctions per ATSE, 1d sparsity per cell.
1a (ATSEmapper schematic) is drawn by Karin in Affinity (slot size in paper_config.yaml). Inputs: inputs.fig1 (written by
scripts/fig1_catalog_summaries.py from the June 2026 build). Panel functions draw only."""
import os

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from style import axes_in, color, sub_slot  # noqa: F401

FIG = "1"
DATASETS = [("TMS", "tms_data"), ("Allen", "allen_data")]
COMMA = FuncFormatter(lambda x, _: f"{x:,.0f}")


def _inp(cfg, key):
    block = cfg["inputs"]["fig1"]
    return os.path.join(block["dir"], block[key])


def _fill(cfg, key):
    """filled for TMS, open for Allen, as in the `data` block"""
    e = cfg["data"][key]; c = color(cfg, key)
    return dict(facecolor=c if e["filled"] else "white", edgecolor=c, linewidth=0.6)


def _log_ticks(ax, lo, hi, axis="y"):
    k = np.arange(int(np.floor(lo)), int(np.ceil(hi)) + 1)
    (ax.set_yticks if axis == "y" else ax.set_xticks)(k)
    (ax.set_yticklabels if axis == "y" else ax.set_xticklabels)([f"$10^{{{x}}}$" for x in k])


def panel_1b(cfg, fig):
    """Catalog per dataset: cells, and junctions, ATSEs and genes with reads in that dataset (log scale, values printed).
    Input: inputs.fig1.catalog."""
    d = pd.read_csv(_inp(cfg, "catalog"), sep="\t").set_index("dataset")
    q = [("cells", "Cells"), ("junctions", "Junctions"), ("atses", "ATSEs"), ("genes", "Genes")]
    ax = axes_in(fig, (17, 8, 12, 2.5))
    h = 0.36
    for i, (ds, key) in enumerate(DATASETS):
        y = np.arange(len(q)) + (i - 0.5) * h
        v = [d.loc[ds, c] for c, _ in q]
        ax.barh(y, v, height=h, label=cfg["data"][key]["label"], **_fill(cfg, key))
        for yy, vv in zip(y, v):
            ax.text(vv * 1.08, yy, f"{vv:,}", va="center", ha="left", fontsize=cfg["style"]["font"]["small_pt"])
    ax.set_xscale("log"); ax.set_xlim(5e3, 2.5e5)
    ax.set_xticks([1e4, 1e5]); ax.set_xticklabels(["$10^{4}$", "$10^{5}$"])
    ax.set_yticks(np.arange(len(q))); ax.set_yticklabels([l for _, l in q]); ax.invert_yaxis()
    ax.set_xlabel("Count (log scale)")
    ax.legend(loc="lower right", frameon=False)


def panel_1c(cfg, fig):
    """Event types of the 24,475 ATSEs (geometric rules, DEFINITIONS.md 1.6) and junctions per ATSE.
    Inputs: inputs.fig1.event_types, inputs.fig1.junctions_per_atse."""
    from style import container_mm
    w, h = container_mm(fig)
    E = pd.read_csv(_inp(cfg, "event_types"), sep="\t")
    ax = axes_in(sub_slot(fig, 0, 0, w * 0.55, h), (19, 8, 6, 2.5))
    ax.barh(np.arange(len(E)), E.atses, color=cfg["palette"]["dark_gray"], height=0.6)
    for i, r in E.iterrows():
        ax.text(r.atses * 1.02, i, f"{100 * r.share:.0f}%", va="center", fontsize=cfg["style"]["font"]["small_pt"])
    ax.set_yticks(np.arange(len(E))); ax.set_yticklabels([t.replace("alt", "Alt.").replace("skipped", "Skipped").replace("complex", "Complex")
                                                          for t in E.event_type]); ax.invert_yaxis()
    ax.set_xlabel("ATSEs"); ax.set_xlim(0, E.atses.max() * 1.3); ax.xaxis.set_major_formatter(COMMA)
    ax.set_xticks([0, 4000, 8000])
    M = pd.read_csv(_inp(cfg, "junctions_per_atse"), sep="\t")
    m7 = M[M.junctions_per_atse >= 7].atses.sum()
    M = pd.concat([M[M.junctions_per_atse < 7], pd.DataFrame([dict(junctions_per_atse=7, atses=m7)])])
    ax2 = axes_in(sub_slot(fig, w * 0.55, 0, w * 0.45, h), (8, 8, 1.5, 2.5))
    ax2.bar(M.junctions_per_atse, M.atses, color=cfg["palette"]["dark_gray"], width=0.7)
    ax2.set_xticks(M.junctions_per_atse); ax2.set_xticklabels([str(x) if x < 7 else "≥7" for x in M.junctions_per_atse])
    ax2.set_xlabel("Junctions per ATSE"); ax2.set_ylabel("ATSEs"); ax2.yaxis.set_major_formatter(COMMA)


def panel_1d(cfg, fig):
    """Sparsity per cell, TMS and Allen: junctions and ATSEs with reads per cell (violin + box, log10), and reads per
    observed (cell, ATSE) (fraction of observed events per log2 bin). Inputs: inputs.fig1.per_cell, .reads_per_event."""
    from style import container_mm
    w, h = container_mm(fig)
    C = pd.read_csv(_inp(cfg, "per_cell"), sep="\t")
    fs = cfg["style"]["font"]["small_pt"]
    for j, (col, lab) in enumerate((("junctions_with_reads", "Junctions with reads\nper cell"), ("atses_with_reads", "ATSEs with reads\nper cell"))):
        ax = axes_in(sub_slot(fig, j * w * 0.3, 0, w * 0.3, h), (10, 8, 1.5, 2.5))
        vals = [np.log10(np.maximum(C[C.dataset == ds][col].values, 1)) for ds, _ in DATASETS]
        vp = ax.violinplot(vals, positions=[0, 1], showextrema=False, widths=0.8)
        for b, (ds, key) in zip(vp["bodies"], DATASETS):
            b.set_facecolor(color(cfg, key) if cfg["data"][key]["filled"] else "white"); b.set_edgecolor(color(cfg, key))
            b.set_alpha(0.35 if cfg["data"][key]["filled"] else 1); b.set_linewidth(0.6)
        ax.boxplot(vals, positions=[0, 1], widths=0.18, showfliers=False, medianprops=dict(color=cfg["palette"]["ink"], linewidth=1),
                   boxprops=dict(linewidth=0.6), whiskerprops=dict(linewidth=0.6), capprops=dict(linewidth=0))
        for i, v in enumerate(vals):
            ax.text(i, np.quantile(v, 0.999) + 0.05, f"{10 ** np.median(v):,.0f}", ha="center", va="bottom", fontsize=fs)
        ax.set_xticks([0, 1]); ax.set_xticklabels([cfg["data"][k]["label"] for _, k in DATASETS])
        lo, hi = min(map(np.min, vals)), max(map(np.max, vals))
        _log_ticks(ax, lo, hi); ax.set_ylabel(lab)
    R = pd.read_csv(_inp(cfg, "reads_per_event"), sep="\t")
    ax = axes_in(sub_slot(fig, 0.6 * w, 0, 0.4 * w, h), (11.5, 8, 2.5, 2.5))
    for ds, key in DATASETS:
        r = R[R.dataset == ds]; f = r.observed_events / r.observed_events.sum()
        ax.plot(np.log2(r.reads_from), f, color=color(cfg, key), linestyle=cfg["data"][key]["linestyle"],
                marker=cfg["data"][key]["marker"], ms=2.5, mfc=color(cfg, key) if cfg["data"][key]["filled"] else "white",
                label=cfg["data"][key]["label"])
    ax.set_xticks([0, 2, 4, 6, 8]); ax.set_xticklabels(["1", "4", "16", "64", "256"])
    ax.set_xlabel("Reads per observed\n(cell, ATSE)"); ax.set_ylabel("Fraction of observed\n(cell, ATSE)")
    ax.set_xlim(-0.5, 9); ax.set_ylim(0, None)
    ax.legend(loc="upper right", frameon=False)


PANELS = {"1b": panel_1b, "1c": panel_1c, "1d": panel_1d}
SOURCE_DATA = {
    "1b": lambda cfg: {"fig1b_catalog": pd.read_csv(_inp(cfg, "catalog"), sep="\t")},
    "1c": lambda cfg: {"fig1c_event_types": pd.read_csv(_inp(cfg, "event_types"), sep="\t"),
                       "fig1c_junctions_per_atse": pd.read_csv(_inp(cfg, "junctions_per_atse"), sep="\t")},
    "1d": lambda cfg: {"fig1d_quantiles": pd.read_csv(_inp(cfg, "quantiles"), sep="\t"),
                       "fig1d_reads_per_event": pd.read_csv(_inp(cfg, "reads_per_event"), sep="\t")},
}
