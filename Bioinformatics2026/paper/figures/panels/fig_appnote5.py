"""Application Note figure, VARIANT B (five panels, Karin's decision pending): the same panels as figure AN without Nrxn1.
a catalog and model + legend for c-e   b check 10.1   c Gria2 (Allen)   d Fgfr2 (TMS)   e Ptprc (TMS; was f in figure AN)
Nrxn1 SS4 is not drawn here; it is row 3 of the supplementary gallery (figure SG, panel SG_c), unchanged.
Every panel is drawn by fig_appnote.py with the same rules; no event was substituted for Nrxn1."""
from fig_appnote import EVENTS, event_data, panel_a as _a, panel_b as _b, event_panel, _a_tables, _accuracy

FIG = "AN5"
EV5 = {"c": EVENTS["c"], "d": EVENTS["d"], "e": EVENTS["f"]}   # Ptprc keeps its spec; only its letter changes


def panel_a(cfg, fig):
    """Catalog, model placeholder and the legend for c-e (as AN_a with the legend range c-e)."""
    _a(cfg, fig, last="e")


def panel_b(cfg, fig):
    """Check 10.1 (as AN_b)."""
    _b(cfg, fig)


def panel_c(cfg, fig):
    """Gria2 flip/flop (Allen), as AN_c."""
    event_panel(cfg, fig, EV5["c"])


def panel_d(cfg, fig):
    """Fgfr2 IIIb/IIIc (TMS), as AN_d."""
    event_panel(cfg, fig, EV5["d"])


def panel_e(cfg, fig):
    """Ptprc (TMS), as AN_f."""
    event_panel(cfg, fig, EV5["e"])


PANELS = {"AN5_a": panel_a, "AN5_b": panel_b, "AN5_c": panel_c, "AN5_d": panel_d, "AN5_e": panel_e}
SOURCE_DATA = {
    "AN5_a": lambda cfg: dict(zip(("figAN5_a_catalog", "figAN5_a_event_types"), (lambda C, E: (C.reset_index(), E))(*_a_tables(cfg)))),
    "AN5_b": lambda cfg: {"figAN5_b_accuracy_10_1": _accuracy()},
    **{f"AN5_{p}": (lambda p: lambda cfg: {f"figAN5_{p}_{EV5[p]['gene']}": event_data(cfg, EV5[p])[0]})(p) for p in "cde"},
}
