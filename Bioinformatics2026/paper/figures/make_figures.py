#!/usr/bin/env python3
"""
make_figures.py -- build LeafletFA paper figure panels (copied from the SpliceVI paper's make_figures.py) from evaluation outputs, using paper_config.yaml.

Outputs per figure, under <output_dir>/fig<N>/:
  panels/<id>.pdf (+ .tiff at 350 dpi, _review.png)  each panel alone at its final printed size, for arranging in
                                                      Affinity Designer
  source_data/<name>.tsv                              the numbers behind each data panel (journal source data), from the
                                                      figure module's SOURCE_DATA functions
  fig<N>.pdf (+ .tiff, _review.png)                   the whole figure assembled at journal width with panel letters,
                                                      drawn by the same panel code (vector), to check the panels together

Usage:  python make_figures.py --list
        python make_figures.py --figure 3            # every panel of Figure 3 + preview
        python make_figures.py --panels 3d 3e        # only these panels (+ preview of their figure)
        python make_figures.py --all
        [--config <paper_config.yaml>]
"""
import argparse
import importlib
import os
import sys

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "panels"))
from style import DEFAULT_CONFIG, MM, Slot, apply_style, load_config, save  # noqa: E402

FIGURE_MODULES = {"1": "fig1"}   # supplementary figures by content; numbers live in the captions   # add figures here as they are written


def panels_of(fig_id):
    return importlib.import_module(FIGURE_MODULES[fig_id]).PANELS


def panel_size(cfg, fig_id, pid, row=()):
    """(w, h) in mm from the config; panels without code (schematics) take the row's height and 56 mm."""
    pc = cfg["figures"][fig_id]["panels"]
    if pid in pc:
        return pc[pid]["w"], pc[pid]["h"]
    return 56, max((pc[p]["h"] for p in row if p in pc), default=50)


def letter(cfg, pid):
    l = pid.rsplit("_", 1)[1] if "_" in pid else pid.lstrip("0123456789")   # "3b" -> "b", "S_umaps_a" -> "a"
    if len(l) != 1:          # named panels (e.g. 1logo) get no letter
        return ""
    return l.upper() if cfg["style"]["panel_label"]["case"] == "upper" else l.lower()


def build_panel(cfg, fig_id, pid):
    """One panel alone at its final size -> <output_dir>/fig<N>/panels/<pid>.*"""
    w, h = panel_size(cfg, fig_id, pid)
    fig = plt.figure(figsize=(w * MM, h * MM))
    panels_of(fig_id)[pid](cfg, fig)
    formats = cfg["figures"][fig_id].get("formats", ("pdf", "tiff", "png"))
    path = save(cfg, fig, os.path.join(cfg["output_dir"], f"fig{fig_id}", "panels", pid), formats=formats)
    write_source_data(cfg, fig_id, pid)
    return path


def write_source_data(cfg, fig_id, pid):
    """Tables behind a panel -> <output_dir>/fig<N>/source_data/<name>.tsv, if the module defines SOURCE_DATA[pid]
    (a function cfg -> {name: DataFrame}). Panel functions stay draw-only."""
    fn = getattr(importlib.import_module(FIGURE_MODULES[fig_id]), "SOURCE_DATA", {}).get(pid)
    if fn is None:
        return
    d = os.path.join(cfg["output_dir"], f"fig{fig_id}", "source_data")
    os.makedirs(d, exist_ok=True)
    for name, df in fn(cfg).items():
        df.to_csv(os.path.join(d, name + ".tsv"), sep="\t", index=False)
        print(f"[FIG]   source data {name}.tsv")


def _cell_size(cfg, fig_id, cell, row, gap_mm):
    """(w, h) of one layout cell: a panel id, or a list of panel ids stacked top to bottom (width = widest, height = sum + gaps)."""
    if isinstance(cell, list):
        sizes = [panel_size(cfg, fig_id, p, row) for p in cell]
        return max(w for w, _ in sizes), sum(h for _, h in sizes) + gap_mm * (len(sizes) - 1)
    return panel_size(cfg, fig_id, cell, row)


def build_assembled(cfg, fig_id, gap_mm=2.0):
    """The whole figure on one page of journal width: every panel drawn by the same function into its slot of the
    layout, plus panel letters; slots without code are gray placeholders (schematics drawn in Affinity). A layout row is a
    list of cells; a cell is a panel id or a list of panel ids stacked top to bottom (e.g. [1a, [1c, 1b]])."""
    fc, st = cfg["figures"][fig_id], cfg["style"]
    page_w = st["width_mm"][fc["width"]]
    flat = lambda row: [p for c in row for p in (c if isinstance(c, list) else [c])]   # noqa: E731
    rows = [(row, max(_cell_size(cfg, fig_id, c, flat(row), gap_mm)[1] for c in row)) for row in fc["layout"]]
    page_h = sum(h for _, h in rows) + gap_mm * (len(rows) - 1)
    page = plt.figure(figsize=(page_w * MM, page_h * MM))
    fns, y = panels_of(fig_id), page_h
    for row, rh in rows:
        x = 0.0
        for cell in row:
            cw, _ = _cell_size(cfg, fig_id, cell, flat(row), gap_mm)
            first = cell[0] if isinstance(cell, list) else cell
            x = fc["panels"].get(first, {}).get("x", x)   # optional fixed left edge (mm), e.g. to center a small panel
            if x + cw > page_w + 0.01:
                print(f"[FIG] WARNING: figure {fig_id} row {row} is wider than {page_w} mm")
            yc = y
            for pid in (cell if isinstance(cell, list) else [cell]):
                w, h = panel_size(cfg, fig_id, pid, flat(row))
                dy = fc["panels"].get(pid, {}).get("dy", 0)   # optional dy (mm) nudges a panel down
                slot = Slot(page, x, yc - h - dy, w, h)
                if pid in fns:
                    fns[pid](cfg, slot)
                else:
                    page.patches.append(plt.Rectangle((x / page_w, (yc - h - dy) / page_h), w / page_w, h / page_h,
                                                      transform=page.transFigure, color="#EEEEEE", zorder=-1))
                    page.text((x + w / 2) / page_w, (yc - h / 2 - dy) / page_h, f"{pid}: " + fc["panels"].get(pid, {}).get("note", "schematic\n(Affinity)") + f"\n{w:g} x {h:g} mm", ha="center", va="center", color="#777777")
                page.text(x / page_w, yc / page_h, letter(cfg, pid), ha="left", va="top", fontsize=st["font"]["panel_label_pt"],
                          fontweight="bold" if cfg["style"]["panel_label"]["bold"] else "normal")
                yc -= h + gap_mm
            x += cw + gap_mm
        y -= rh + gap_mm
    path = save(cfg, page, os.path.join(cfg["output_dir"], f"fig{fig_id}", f"fig{fig_id}"))
    print(f"[FIG] assembled {path} ({page_w} x {page_h:.0f} mm)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--figure", nargs="*", default=[])
    ap.add_argument("--panels", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    apply_style(cfg)
    if a.list:
        for fid in FIGURE_MODULES:
            for pid, fn in panels_of(fid).items():
                print(f"{pid}: {(fn.__doc__ or '').strip().splitlines()[0]}")
        return
    figs = list(FIGURE_MODULES) if a.all else a.figure
    todo = [(f, p) for f in figs for p in panels_of(f)] + [(p.rsplit("_", 1)[0] if "_" in p else p.rstrip("abcdefghijklmnopqrstuvwxyz"), p) for p in a.panels]
    if not todo:
        ap.error("nothing to do: give --figure, --panels, --all or --list")
    for fid, pid in todo:
        print(f"[FIG] {pid}: {build_panel(cfg, fid, pid)}")
    for fid in sorted({f for f, _ in todo}):
        build_assembled(cfg, fid)


if __name__ == "__main__":
    main()
