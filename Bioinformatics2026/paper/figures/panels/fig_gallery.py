"""Supplementary gallery: the known-biology events of GALLERY_KNOWN_BIOLOGY_DRAFT.md (list final, Karin 2026-11-10, 2eae6cc),
every row shown whatever it gives, in the event-panel style of the Application Note figure (fig_appnote.event_panel; rules in its
docstring). Junctions are chosen by annotation (paper/scripts/event_annotation.py, GENCODE vM19). Each title carries the row's
evidence type and expected direction as written in the list. Rows 7 (Ctnnd1) and 10 (Pkm) have no ATSE in the catalog."""
from fig_appnote import ANNOTATION, color, event_data, event_panel, axes_in, sub_slot, container_mm  # noqa: F401

FIG = "SG"
ROWS = [  # (panel, spec, evidence; expected) -- text from the final gallery list
    ("a", dict(key="fgfr2", gene="Fgfr2", title="IIIb vs IIIc", scheme="tms_epithelial"),
     "Mouse in vivo (Esrp knockout epidermis) + cell lines; expected: IIIb higher in epithelial types"),
    ("b", dict(key="ptprc", gene="Ptprc", title="variable exons", scheme="tms_immune", pick="ptprc_10_4"),
     "Not verified; descriptive (differs between immune types)"),
    ("c", dict(key="nrxn1_ss4", gene="Nrxn1", title="SS4", scheme="allen_inhibitory_split"),   # SpliceVI §111 SS4 (Karin 2026-11-11)
     "Mouse cell types (Nguyen 2016); expected: SS4 included in PV and SST, skipped in excitatory"),
    ("d", dict(key="gria2", gene="Gria2", title="flip/flop", scheme="allen"),
     "Rat brain regions (abstract only); descriptive"),
    ("e", dict(key="cd44", gene="Cd44", title="variable exons v8–v10", scheme="tms_epithelial"),
     "Cell lines (v8–v10) + mouse in vivo; expected: v8–v10 inclusion higher in epithelial types"),
    ("f", dict(key="enah", gene="Enah", title="exon 11a (by length)", scheme="tms_epithelial"),
     "Cell lines + mouse in vivo; expected: included in epithelial, skipped in mesenchymal"),
    ("g", dict(key="ctnnd1", gene="Ctnnd1", title="exons 2–3", scheme="tms_epithelial"),
     "Cell lines; expected: exons 2–3 higher in mesenchymal types"),
    ("h", dict(key="ptbp2", gene="Ptbp2", title="exon 10", scheme="tms_neural"),
     "Development; expected: included in neurons, skipped in non-neuronal cells"),
    ("i", dict(key="dlg4", gene="Dlg4", title="exon 18", scheme="tms_neural"),
     "Development; expected: included late in neuronal maturation"),
    ("j", dict(key="pkm", gene="Pkm", title="M1/M2 exons", scheme="tms_epithelial"),
     "Not verified (rat); descriptive"),
]


def _legend(cfg, fig):
    """legend for all gallery panels, drawn in the empty Pkm panel"""
    w, h = container_mm(fig)
    sm = cfg["style"]["font"]["small_pt"]; ink, gray = cfg["palette"]["ink"], cfg["palette"]["gray"]
    teal = color(cfg, "leafletfa")
    gx = axes_in(sub_slot(fig, 0, 0, w, h * 0.55), (6, 1, 2, 1)); gx.axis("off"); gx.set_xlim(0, 1); gx.set_ylim(0, 1)
    gx.text(0.0, 0.97, "All panels", ha="left", va="top", fontsize=sm, color=ink, fontweight="bold")
    gx.plot([0.02, 0.09], [0.68, 0.68], color=teal, lw=0.4, alpha=0.55); gx.scatter([0.055], [0.68], s=6, color=teal, lw=0)
    gx.text(0.12, 0.68, "LeafletFA ψ, 95% predictive interval (where held-out cells have\n≥ 100 reads and ≥ 10 cells with reads)", ha="left", va="center", fontsize=sm, color=ink, linespacing=1.0)
    gx.scatter([0.055], [0.36], s=10, facecolors="none", edgecolors=gray, linewidths=0.5)
    gx.text(0.12, 0.36, "Raw TRAIN pseudobulk ψ (area ~ log junction reads)", ha="left", va="center", fontsize=sm, color=ink)
    gx.text(0.0, 0.05, f"Exons: {ANNOTATION}", ha="left", va="bottom", fontsize=sm, color=gray)


def _make(spec, sub):
    def f(cfg, fig):
        event_panel(cfg, fig, spec, subtitle=sub)
        if spec["key"] == "pkm":
            _legend(cfg, fig)
    f.__doc__ = f"{spec['gene']} {spec['title']} ({sub})"
    return f


PANELS = {f"SG_{p}": _make(spec, sub) for p, spec, sub in ROWS}
SOURCE_DATA = {f"SG_{p}": (lambda spec: lambda cfg: ({f"figSG_{spec['key']}": event_data(cfg, spec)[0]} if event_data(cfg, spec) else {}))(spec)
               for p, spec, _ in ROWS}
