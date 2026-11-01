"""Shared style for the LeafletFA paper figures (copied from the SpliceVI paper, SpliceVI-utils manuscript/paper/figures/style.py,
so both companion papers look the same): reads paper_config.yaml, applies one matplotlib style, sizes figures in
millimeters and saves every panel the same way. Panel code takes colors and labels from here, never hard-coded."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn  # noqa: E402,F401  (registers the mako color map named in `cmaps`)
import yaml  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "..", "paper_config.yaml")
MM = 1 / 25.4


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_dir"] = os.path.dirname(os.path.abspath(path))   # relative paths in the config (e.g. font files) resolve against it
    check_styles(cfg)
    return cfg


STYLE_BLOCKS = ("methods", "data", "misc")


def check_styles(cfg):
    """Each key in exactly one of methods / data / misc, and every color a palette token."""
    seen = {}
    for block in STYLE_BLOCKS:
        for key, v in (cfg.get(block) or {}).items():
            if key in seen:
                raise ValueError(f"style key '{key}' is defined in both '{seen[key]}' and '{block}' of paper_config.yaml")
            seen[key] = block
            tok = v if block == "misc" else v["color"]
            if tok not in cfg["palette"]:
                raise ValueError(f"{block}.{key}: color '{tok}' is not a palette token")


def apply_style(cfg):
    """matplotlib rcParams from the config: fonts (with glyph fallback), sizes in pt, thin axes, no top/right spines,
    editable text in PDF (TrueType, fonttype 42)."""
    st = cfg["style"]
    for f in st["font"].get("files", []):
        f = f if os.path.isabs(f) else os.path.join(cfg.get("_dir", os.path.dirname(DEFAULT_CONFIG)), f)
        if os.path.exists(f):
            fm.fontManager.addfont(f)
    fs, sm, lw = st["font"]["size_pt"], st["font"]["small_pt"], st["line"]
    installed = {f.name for f in fm.fontManager.ttflist}
    family = [f for f in st["font"]["family"] if f in installed] or ["DejaVu Sans"]
    plt.rcParams.update({
        "font.family": family, "font.size": fs,
        "axes.labelsize": fs, "axes.titlesize": fs, "axes.titleweight": "normal", "legend.fontsize": sm,
        "xtick.labelsize": sm, "ytick.labelsize": sm, "legend.title_fontsize": sm,
        "axes.linewidth": lw["axes_pt"], "xtick.major.width": lw["axes_pt"], "ytick.major.width": lw["axes_pt"],
        "xtick.minor.width": lw["axes_pt"] * 0.8, "ytick.minor.width": lw["axes_pt"] * 0.8,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2.5,
        "lines.linewidth": lw["data_pt"], "lines.markersize": lw["marker_pt"], "patch.linewidth": lw["axes_pt"],
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "legend.frameon": False, "legend.handlelength": 1.4, "legend.handletextpad": 0.4, "legend.borderaxespad": 0.2,
        "legend.labelspacing": 0.3, "legend.columnspacing": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": st["dpi"]["color"], "figure.dpi": 100, "savefig.transparent": False,
        "mathtext.fontset": "custom", "mathtext.rm": family[0], "mathtext.it": family[0] + ":italic",   # math in the text font
        "mathtext.bf": family[0] + ":bold",
        "axes.prop_cycle": matplotlib.cycler(color=[cfg["palette"][k] for k in ("teal", "indigo", "ochre", "olive", "gray")]),   # theme order; panels should still name keys
    })


def _entry(cfg, key):
    for block in STYLE_BLOCKS:
        if key in (cfg.get(block) or {}):
            return block, cfg[block][key]
    raise KeyError(f"'{key}' is not in methods / data / misc of paper_config.yaml (add new methods to `methods:`)")


def color(cfg, key):
    """Hex color of a key from methods / data / misc."""
    block, e = _entry(cfg, key)
    return cfg["palette"][e if block == "misc" else e["color"]]


def resolve_color(cfg, name):
    """Hex color for a palette token or a methods / data / misc key (for config blocks that may name either)."""
    return cfg["palette"][name] if name in cfg["palette"] else color(cfg, name)


def method_style(cfg, key, ms=None):
    """Plot kwargs for a method or dataset from `methods:` / `data:` (color token, marker, line style, filled or open
    marker), e.g. ax.plot(x, y, **method_style(cfg, "splicevi")). Paper model = solid + filled; variants = dashed + open."""
    block, m = _entry(cfg, key)
    if block == "misc":
        raise KeyError(f"'{key}' is a misc color, use color(cfg, '{key}')")
    c = cfg["palette"][m["color"]]
    return dict(color=c, marker=m["marker"], linestyle=m["linestyle"], label=m["label"],
                markersize=ms or cfg["style"]["line"]["marker_pt"], markeredgecolor=c, markeredgewidth=0.7,
                markerfacecolor=c if m["filled"] else "white")


class Slot:
    """A panel's rectangle on an assembled page: the page Figure plus the rectangle's position and size in mm."""

    def __init__(self, page, x_mm, y_mm, w_mm, h_mm):
        self.page, self.x, self.y, self.w, self.h = page, x_mm, y_mm, w_mm, h_mm


def axes_in(container, pad_mm=(9.5, 8.0, 2.0, 2.5)):
    """One Axes inside a panel container -- a standalone Figure at the panel's size, or a Slot of the assembled page --
    with fixed margins in mm (left, bottom, right, top), so the axes sit identically in both outputs."""
    l, b, r, t = pad_mm
    if isinstance(container, Slot):
        page = container.page
        PW, PH = page.get_size_inches() / MM
        x, y, w, h = container.x, container.y, container.w, container.h
        return page.add_axes([(x + l) / PW, (y + b) / PH, (w - l - r) / PW, (h - b - t) / PH])
    W, H = container.get_size_inches() / MM
    return container.add_axes([l / W, b / H, 1 - (l + r) / W, 1 - (b + t) / H])


def container_mm(container):
    """(w, h) in mm of a panel container (a standalone Figure or a Slot of the assembled page)."""
    if isinstance(container, Slot):
        return container.w, container.h
    w, h = container.get_size_inches() / MM
    return w, h


def sub_slot(container, x_mm, y_mm, w_mm, h_mm):
    """A rectangle inside a panel container (x, y from its bottom-left, in mm), usable with axes_in(), so one panel can
    hold facets or a legend strip and still draw identically standalone and on the assembled page."""
    base = container if isinstance(container, Slot) else Slot(container, 0, 0, *container_mm(container))
    return Slot(base.page, base.x + x_mm, base.y + y_mm, w_mm, h_mm)


def reference_line(cfg, ax, y=None, x=None, **kw):
    style = dict(color=color(cfg, "full_depth"), lw=cfg["style"]["line"]["reference_pt"], ls=(0, (3, 2)), zorder=0)
    style.update(kw)
    return ax.axhline(y, **style) if y is not None else ax.axvline(x, **style)


def save(cfg, fig, path_base, formats=("pdf", "tiff", "png")):
    """PDF (vector, editable text), TIFF at the color dpi, PNG at the review dpi; "svg" on request (text kept as text)."""
    os.makedirs(os.path.dirname(path_base), exist_ok=True)
    if "pdf" in formats:     # clean metadata: no creator / producer / date (and so no paths or versions)
        fig.savefig(path_base + ".pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None})
    if "svg" in formats:
        fig.savefig(path_base + ".svg", metadata={"Creator": None, "Date": None})
    if "tiff" in formats:
        fig.savefig(path_base + ".tiff", dpi=cfg["style"]["dpi"]["color"], pil_kwargs={"compression": "tiff_lzw"})
    if "png" in formats:
        fig.savefig(path_base + "_review.png", dpi=cfg["style"]["dpi"]["review"], metadata={"Software": None})
    plt.close(fig)
    return path_base + ".pdf"
