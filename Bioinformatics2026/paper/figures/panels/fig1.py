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
    """the dataset's color (data block: slate = TMS, sage = Allen Brain)"""
    c = color(cfg, key)
    return dict(facecolor=c, edgecolor=c, linewidth=0.6)


def _log_ticks(ax, lo, hi, axis="y"):
    k = np.arange(int(np.floor(lo)), int(np.ceil(hi)) + 1)
    (ax.set_yticks if axis == "y" else ax.set_xticks)(k)
    (ax.set_yticklabels if axis == "y" else ax.set_xticklabels)([f"$10^{{{x}}}$" for x in k])


Q1B = [("cells", "Cells"), ("junctions", "Junctions with reads"), ("atses", "ATSEs with reads"), ("genes", "Genes with reads")]


def panel_1b(cfg, fig):
    """Catalog per dataset as small multiples: one row per quantity, each on its own linear scale starting at 0 (values
    printed; no shared axis). Input: inputs.fig1.catalog."""
    from style import container_mm
    w, h = container_mm(fig)
    d = pd.read_csv(_inp(cfg, "catalog"), sep="\t").set_index("dataset")
    fs = cfg["style"]["font"]["small_pt"]
    top, rowh = 6.0, (h - 6.0 - 2.0) / len(Q1B)
    for i, (col, lab) in enumerate(Q1B):
        ax = axes_in(sub_slot(fig, 0, h - top - (i + 1) * rowh, w, rowh), (30, 1.0, 18, 1.0))
        v = [d.loc[ds, col] for ds, _ in DATASETS]
        ax.barh([0, 1], v, height=0.75, color=[color(cfg, k) for _, k in DATASETS])
        for yy, vv in zip([0, 1], v):
            ax.text(vv + max(v) * 0.02, yy, f"{vv:,}", va="center", ha="left", fontsize=fs)
        ax.set_xlim(0, max(v) * 1.18); ax.set_ylim(1.6, -0.6)
        ax.set_yticks([0.5]); ax.set_yticklabels([lab]); ax.tick_params(axis="y", length=0)
        ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
    import matplotlib.patches as mpatches
    hs = [mpatches.Patch(color=color(cfg, k), label=cfg["data"][k]["label"]) for _, k in DATASETS]
    lax = axes_in(sub_slot(fig, 0, h - top, w, top), (30, 0, 2, 0)); lax.axis("off")
    lax.legend(handles=hs, loc="center left", ncol=2, frameon=False)


def panel_1b_table(cfg, fig):
    """Alternative to 1b: the catalog per dataset as a compact table (no bars). Input: inputs.fig1.catalog."""
    d = pd.read_csv(_inp(cfg, "catalog"), sep="\t").set_index("dataset")
    ax = axes_in(fig, (1, 1, 1, 1)); ax.axis("off")
    fs = cfg["style"]["font"]["size_pt"]
    cols = [cfg["data"][k]["label"] for _, k in DATASETS]
    ys = np.linspace(0.78, 0.08, len(Q1B))
    for j, (ds, k) in enumerate(DATASETS):
        x = 0.62 + 0.22 * j
        ax.text(x, 0.95, cols[j], ha="right", va="center", fontsize=fs, fontweight="bold", color=color(cfg, k))
        for y, (col, _) in zip(ys, Q1B):
            ax.text(x, y, f"{d.loc[ds, col]:,}", ha="right", va="center", fontsize=fs)
    for y, (_, lab) in zip(ys, Q1B):
        ax.text(0.02, y, lab, ha="left", va="center", fontsize=fs)
    ax.plot([0.02, 0.86], [0.87, 0.87], color=cfg["palette"]["ink"], linewidth=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def panel_1c(cfg, fig):
    """Event types of the 24,475 ATSEs (geometric rules, DEFINITIONS.md 1.6) and junctions per ATSE.
    Inputs: inputs.fig1.event_types, inputs.fig1.junctions_per_atse."""
    from style import container_mm
    w, h = container_mm(fig)
    E = pd.read_csv(_inp(cfg, "event_types"), sep="\t")
    ax = axes_in(sub_slot(fig, 0, 0, w * 0.55, h), (19, 8, 6, 2.5))
    ax.barh(np.arange(len(E)), E.atses, color=cfg["palette"]["dark_gray"], height=0.6)
    for i, r in E.iterrows():
        ax.text(r.atses * 1.03, i, f"{r.atses:,}\n({100 * r.share:.0f}%)", va="center", fontsize=cfg["style"]["font"]["small_pt"], linespacing=0.95)
    ax.set_yticks(np.arange(len(E))); ax.set_yticklabels([t.replace("alt", "Alt.").replace("skipped", "Skipped").replace("complex", "Complex")
                                                          for t in E.event_type]); ax.invert_yaxis()
    ax.set_xlim(0, E.atses.max() * 1.6); ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
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
        ax = axes_in(sub_slot(fig, j * w * 0.31, 0, w * 0.31, h), (10, 8, 1.0, 2.5))
        vals = [np.log10(np.maximum(C[C.dataset == ds][col].values, 1)) for ds, _ in DATASETS]
        vp = ax.violinplot(vals, positions=[0, 1], showextrema=False, widths=0.8)
        for b, (ds, key) in zip(vp["bodies"], DATASETS):
            b.set_facecolor(color(cfg, key)); b.set_edgecolor(color(cfg, key)); b.set_alpha(0.55); b.set_linewidth(0.6)
        ax.boxplot(vals, positions=[0, 1], widths=0.18, showfliers=False, medianprops=dict(color=cfg["palette"]["ink"], linewidth=1),
                   boxprops=dict(linewidth=0.6), whiskerprops=dict(linewidth=0.6), capprops=dict(linewidth=0))
        for i, v in enumerate(vals):
            ax.text(i, np.quantile(v, 0.999) + 0.05, f"{10 ** np.median(v):,.0f}", ha="center", va="bottom", fontsize=fs)
        ax.set_xticks([0, 1]); ax.set_xticklabels([cfg["data"][k]["label"].replace(" ", "\n") for _, k in DATASETS])
        lo, hi = min(map(np.min, vals)), max(map(np.max, vals))
        _log_ticks(ax, lo, hi); ax.set_ylabel(lab)
    R = pd.read_csv(_inp(cfg, "reads_per_event"), sep="\t")
    ax = axes_in(sub_slot(fig, 0.62 * w, 0, 0.38 * w, h), (10, 11, 1.5, 2.5))
    for ds, key in DATASETS:
        r = R[R.dataset == ds]; f = r.observed_events / r.observed_events.sum()
        ax.plot(np.log2(r.reads_from), f, color=color(cfg, key), linestyle=cfg["data"][key]["linestyle"],
                marker=cfg["data"][key]["marker"], ms=2.2, mfc=color(cfg, key),
                label=cfg["data"][key]["label"])
    ax.set_xticks([0, 2, 4, 6, 8]); ax.set_xticklabels(["1", "4", "16", "64", "256"])
    ax.set_xlabel("Reads per observed\n(cell, ATSE)"); ax.set_ylabel("Fraction of observed\n(cell, ATSE)")
    ax.set_xlim(-0.5, 9); ax.set_ylim(0, 0.4); ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.legend(loc="upper right", frameon=False, handlelength=1.6, borderaxespad=0.1)


PANELS = {"1b": panel_1b, "1b_table": panel_1b_table, "1c": panel_1c, "1d": panel_1d}
SOURCE_DATA = {
    "1b": lambda cfg: {"fig1b_catalog": pd.read_csv(_inp(cfg, "catalog"), sep="\t")},
    "1b_table": lambda cfg: {},
    "1c": lambda cfg: {"fig1c_event_types": pd.read_csv(_inp(cfg, "event_types"), sep="\t"),
                       "fig1c_junctions_per_atse": pd.read_csv(_inp(cfg, "junctions_per_atse"), sep="\t")},
    "1d": lambda cfg: {"fig1d_quantiles": pd.read_csv(_inp(cfg, "quantiles"), sep="\t"),
                       "fig1d_reads_per_event": pd.read_csv(_inp(cfg, "reads_per_event"), sep="\t")},
}
