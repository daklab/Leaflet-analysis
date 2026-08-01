"""
Universal figure theme for the LeafletFA Genome Biology submission.

Executable version of FIGURE_THEME.md (same directory). Every figure script and notebook
under GenomeBiology_Submission2026/figures/ starts with:

    import sys; sys.path.insert(0, "<...>/GenomeBiology_Submission2026/figures")
    from leaflet_theme import apply_theme, PALETTE, METHOD_COLORS, figsize, panel_label, save
    apply_theme()

Nothing here sets data-dependent defaults — it only fixes type, geometry, color and export
so panels from different scripts composite into one figure without retouching.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont, FontProperties

__all__ = [
    "apply_theme", "PALETTE", "METHOD_COLORS", "CMAP", "WIDTHS_IN",
    "figsize", "panel_label", "despine", "save", "symmetric_norm", "check_fonts",
]

# ── color ───────────────────────────────────────────────────────────────────────────────
# LeafletFA palette. Cool-dominant and deliberately distinct from stock matplotlib/seaborn
# and from Okabe-Ito, but constructed — not eyeballed — on a CIELAB *lightness ladder*:
# each series color sits ~13 L* from its neighbours. That ladder is what makes the set
# survive greyscale printing AND dichromat vision, because dichromats effectively lose the
# red-green axis and read lightness instead. Hue alone never achieves this.
#
# Verified by validate_palette.py: all 10 pairs clear ΔE≥20 (normal), ≥15 (deuteranopia
# and protanopia), and ΔL*≥8 (greyscale). Re-run it after ANY color change.
PALETTE = {
    "teal":   "#006865",   # L* 39 — the method
    "indigo": "#657BB5",   # L* 52
    "ochre":  "#B79B51",   # L* 65
    "rose":   "#ECB0CA",   # L* 78
    "olive":  "#3A4621",   # L* 28
    "grey":   "#6E7478",   # non-series only (see below)
}

# Fixed across every figure — LeafletFA is teal in every panel it appears in.
METHOD_COLORS = {
    "LeafletFA": PALETTE["teal"],
    "NMF":       PALETTE["indigo"],
    "scVI":      PALETTE["ochre"],
    "null":      PALETTE["rose"],
    "shuffled":  PALETTE["rose"],
    "ablation":  PALETTE["olive"],
    "PCA":       PALETTE["olive"],
    "other":     PALETTE["grey"],
}

# HARD LIMIT: five series colors per panel. A sixth categorical color cannot be added
# without breaking greyscale or dichromat separation — this is a property of color space,
# not a lack of effort. Need more than five categories? Facet, or collapse to top-4 + grey.
MAX_SERIES = 5

# `grey` is exempt from the separation checks because it never encodes a compared series —
# it marks non-focal context ("all other cell types", background points). If you find
# yourself needing grey to be *distinguishable* from a series color, it has become a series
# and needs a real color, or the panel needs faceting.

CMAP = {
    "sequential": "mako",     # cool, teal-anchored — matches the palette identity
    "diverging":  "BrBG",     # brown↔teal; CB-safe and on-brand (avoids stock red/blue)
    "psi":        "mako",     # always with vmin=0, vmax=1
    "viridis":    "viridis",  # only when matching a previously published panel
}

_CYCLE = [PALETTE[k] for k in ("teal", "indigo", "ochre", "rose", "olive")]

# ── geometry ────────────────────────────────────────────────────────────────────────────
WIDTHS_IN = {"single": 89 / 25.4, "onehalf": 120 / 25.4, "double": 183 / 25.4}
MAX_HEIGHT_IN = 220 / 25.4

INK = "#222222"       # near-black; pure black is harsh at print size
GRID = "#E6E6E6"

# ── type ────────────────────────────────────────────────────────────────────────────────
# Arimo and Liberation Sans are metric-compatible Arial clones — same advance widths, so a
# panel laid out in either matches Arial exactly. Journal-acceptable substitutes on Linux,
# where Arial itself is not licensed to ship.
FONT_STACK = ["Arial", "Helvetica", "Arimo", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]
PT = {"panel": 8, "label": 7, "tick": 6, "legend": 6, "title": 7, "annot": 6}


def _available_fonts(stack=FONT_STACK):
    """Keep only faces matplotlib can actually resolve, so it never silently falls back
    mid-render (which yields panels in two different faces across a figure)."""
    import logging
    installed = {f.name for f in mpl.font_manager.fontManager.ttflist}
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    return [n for n in stack if n in installed] + ["DejaVu Sans"]


def check_fonts(verbose: bool = True) -> str:
    """Report which face the theme will actually use.

    Genome Biology / Nature expect Arial or Helvetica. Neither ships with Linux, and
    conda-forge carries no Arial-metric font (only DejaVu). If only DejaVu Sans is present
    the figures are still legible and fully embedded, but the typeface will not match the
    journal's house style.

    Fix — install Arimo (Apache-2.0, metric-compatible with Arial), then rebuild
    matplotlib's font cache. NOTE the explicit submodule import: `matplotlib.font_manager`
    is not auto-imported, so `import matplotlib; matplotlib.font_manager...` raises
    AttributeError.

        mkdir -p ~/.local/share/fonts
        for f in Regular Bold Italic BoldItalic; do
          curl -sL -o ~/.local/share/fonts/Arimo-$f.ttf \\
            "https://github.com/googlefonts/Arimo/raw/main/fonts/ttf/Arimo-$f.ttf"
        done
        fc-cache -f ~/.local/share/fonts
        python -c "from matplotlib import font_manager; font_manager._load_fontmanager(try_read_cache=False)"
    """
    resolved = _available_fonts()[0]
    if verbose and resolved not in ("Arial", "Helvetica", "Liberation Sans", "Arimo"):
        print(f"[leaflet_theme] using {resolved!r} — Arial/Helvetica not installed; "
              f"see check_fonts.__doc__ to install a metric-compatible face.")
    return resolved


def apply_theme(scale: float = 1.0) -> None:
    """Install the manuscript theme globally. Call once, at the top of every script.

    scale > 1 enlarges all type proportionally — use only for slides/posters, never for
    the submission (it breaks size consistency between panels).
    """
    try:  # registers 'mako'/'rocket' as matplotlib colormaps
        import seaborn  # noqa: F401
    except ImportError:
        CMAP["sequential"] = CMAP["psi"] = "viridis"

    mpl.rcParams.update({
        # type
        "font.family": "sans-serif",
        "font.sans-serif": _available_fonts(),
        "font.size": PT["tick"] * scale,
        "axes.labelsize": PT["label"] * scale,
        "axes.titlesize": PT["title"] * scale,
        "axes.titleweight": "regular",
        "axes.titlelocation": "left",
        "xtick.labelsize": PT["tick"] * scale,
        "ytick.labelsize": PT["tick"] * scale,
        "legend.fontsize": PT["legend"] * scale,
        "legend.title_fontsize": PT["title"] * scale,
        "figure.titlesize": PT["panel"] * scale,
        "mathtext.default": "regular",

        # ink
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.prop_cycle": mpl.cycler(color=_CYCLE),

        # axes furniture
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,

        # marks
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "scatter.marker": "o",
        "patch.linewidth": 0.5,
        "boxplot.flierprops.markersize": 2,

        # legend
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.labelspacing": 0.35,
        "legend.borderpad": 0.2,
        "legend.columnspacing": 1.0,

        # figure + export
        "figure.dpi": 140,               # on-screen only
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.transparent": False,
        "pdf.fonttype": 42,             # embed as editable text, not outlines
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def figsize(width: str = "double", h: float = 3.0) -> tuple[float, float]:
    """Inches tuple for a Nature column width. width ∈ {single, onehalf, double}."""
    if width not in WIDTHS_IN:
        raise ValueError(f"width must be one of {sorted(WIDTHS_IN)}, got {width!r}")
    if h > MAX_HEIGHT_IN:
        raise ValueError(f"height {h:.2f} in exceeds the {MAX_HEIGHT_IN:.2f} in page limit")
    return (WIDTHS_IN[width], h)


def panel_label(ax, letter: str, dx: float = -0.13, dy: float = 1.06):
    """Bold lowercase panel label, top-left outside the axes."""
    return ax.text(dx, dy, letter.lower(), transform=ax.transAxes,
                   fontsize=PT["panel"], fontweight="bold", va="top", ha="left", color=INK)


def despine(ax, left: bool = False, bottom: bool = False):
    """Top/right are already off via rcParams; pass left/bottom=True to drop those too."""
    if left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    if bottom:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)
    return ax


def sentence_case(s: str) -> str:
    """Upper-case the first letter, leaving the rest of the string alone.

    Skips anything that starts with mathtext ('$\\Psi$ recovery'), a digit, or an already
    upper-case letter — so 'ARI', 'K=10' and '$\\phi$ usage' are never mangled.
    """
    if not s:
        return s
    first = s[0]
    if first == "$" or not first.isalpha() or first.isupper():
        return s
    return first.upper() + s[1:]


def capitalize_labels(obj) -> None:
    """Sentence-case every axis label, title and legend title on a Figure or Axes.

    Called automatically by `save()`, so panels get consistent capitalisation without each
    script having to remember. Safe to call twice.
    """
    axes = obj.get_axes() if hasattr(obj, "get_axes") else [obj]
    for ax in axes:
        ax.set_xlabel(sentence_case(ax.get_xlabel()))
        ax.set_ylabel(sentence_case(ax.get_ylabel()))
        # The theme sets axes.titlelocation='left', so set_title() writes the LEFT title while
        # get_title() defaults to reading the CENTER one — check all three explicitly.
        for loc in ("center", "left", "right"):
            t = ax.get_title(loc=loc)
            if t:
                ax.set_title(sentence_case(t), loc=loc)
        leg = ax.get_legend()
        if leg is not None and leg.get_title() is not None:
            leg.get_title().set_text(sentence_case(leg.get_title().get_text()))
        # Explicit *text* tick labels (categories like 'global prior') get the same treatment.
        # Numeric ticks are left alone — only axes with a fixed list of labels are touched, so
        # '10', 'K=10' and 'P1' are never rewritten.
        # Key off the LOCATOR, not the formatter: set_xticks() installs a FixedLocator, but the
        # formatter may be a FuncFormatter, so an isinstance check on the formatter silently misses
        # every categorical axis.
        import matplotlib.ticker as mticker
        for axis in (ax.xaxis, ax.yaxis):
            if not isinstance(axis.get_major_locator(), mticker.FixedLocator):
                continue
            texts = [t.get_text() for t in axis.get_ticklabels()]
            if any(t and t[0].isalpha() and t[0].islower() for t in texts):
                axis.set_ticklabels([sentence_case(t) for t in texts])


def sig_bracket(ax, x1: float, x2: float, y: float, p: float, dh: float = 0.02,
                fmt: str = "stars"):
    """Draw a significance bracket between two x positions at height `y`.

    fmt='stars' -> ns / * / ** / ***      fmt='p' -> 'P = 0.013' (or 'P < 0.001')
    Returns the top of the annotation so callers can stack brackets.
    """
    if fmt == "stars":
        label = "ns" if p >= 0.05 else ("***" if p < 0.001 else ("**" if p < 0.01 else "*"))
    else:
        label = "P < 0.001" if p < 0.001 else f"P = {p:.3g}"
    h = dh * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.6, c=INK, clip_on=False)
    ax.text((x1 + x2) / 2, y + h, label, ha="center", va="bottom",
            fontsize=PT["annot"], color=INK, clip_on=False)
    return y + h


def symmetric_norm(values, cmap: str = CMAP["diverging"]):
    """(vmin, vmax, cmap) for a diverging map centered on 0 — never let it drift."""
    import numpy as np
    v = float(np.nanmax(np.abs(np.asarray(values, dtype=float))))
    return -v, v, cmap


def save(fig, stem, formats=("pdf", "png"), close: bool = False):
    """Write the panel as vector PDF (submission) + 600 dpi PNG (review). `stem` has no
    extension; parent directories are created.

    Axis labels and titles are sentence-cased on the way out (see `capitalize_labels`), so
    capitalisation is consistent across panels without every script having to remember."""
    from pathlib import Path
    capitalize_labels(fig)
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in formats:
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p, dpi=600 if ext == "png" else None)
        out.append(p)
    if close:
        plt.close(fig)
    return out
