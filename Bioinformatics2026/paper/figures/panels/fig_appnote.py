"""Application Note figure (one figure, six panels; APPLICATION_NOTE_FIGURE_PLAN.md, layout approved 2026-11-06) and the shared
event-panel code used by the supplementary gallery (fig_gallery.py).

a  catalog per atlas and event types (ledger inputs) + model schematic placeholder (Karin finishes it) + legend for c-f
b  check 10.1: held-out error, LeafletFA minus raw TRAIN pseudobulk, by cell-type size and by TRAIN reads, both atlases
c  Gria2 flip/flop (Allen)   d  Fgfr2 IIIb/IIIc (TMS)   e  Nrxn1 SS4 skipping (Allen)   f  Ptprc (TMS)

Event panels, fixed rules (no hand-picking):
- Cell types shown: every type (TMS tissue-cell type; Allen cell group) with >= 20 TRAIN reads and >= 3 TRAIN cells with reads
  at the ATSE (the browser's "range" support rule), grouped by lineage, sorted within a lineage by LeafletFA psi of the first
  junction shown (highest at top).
- Lineages: GenomeBiology_Submission2026/cell_type_maps/mouse_lineage_map.json (one definition for the browser and the figures):
  TMS by broad cell type, Allen by class with inhibitory groups split by cluster-name prefix; each panel uses one of its schemes.
  The TMS Epithelial / Mesenchymal lineages are asserted equal to the check 10 keyword lists (c10_catalog.EPI / MES).
- Junctions shown: chosen by annotation in paper/scripts/event_annotation.py (GENCODE vM19; rules in its docstring), read from
  <output_dir>/data/figAN_event_junctions.tsv. Ptprc (panel f): among the lead junctions of the check 10.4 Ptprc calls
  (c10_TMS_s0_ptprc_direction.tsv), the one with the most TRAIN reads over all types; labeled by its exons from that table.
- Marks: LeafletFA point estimate (check 10 point, seed 0) in teal, with the check 12 (a)+(b) interval (thin, semi-transparent)
  where held-out TEST cells have >= 100 reads and >= 10 cells with reads at the ATSE (point only elsewhere); raw TRAIN pseudobulk
  psi as a gray ring, area ~ log junction reads. Variant (FIGAN_GREY_NOIV=1): types without a validated interval drawn in light gray.
Panel functions draw only; numbers come from _event() / _accuracy() (cached) and are written as source data."""
import functools
import json
import os

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.path import Path

from style import axes_in, color, container_mm, sub_slot

FIG = "AN"
V = "/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation"
PRE = os.path.join(V, "portal_browser", "precomputed")
B = "/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_splicevi_baseline"
FIT = {ds: os.path.join(B, ds, "full_K20_s0_f64", "fit", "fit.npz") for ds in ("TMS", "Allen")}   # seed-0 fits (RELEASE_MANIFEST.md)
NUMBERS = "/gpfs/commons/home/kisaev/Leaflet-analysis-bioinformatics/Bioinformatics2026/manuscript/numbers.tex"
LINEAGE_MAP = "/gpfs/commons/home/kisaev/Leaflet-analysis-bioinformatics/GenomeBiology_Submission2026/cell_type_maps/mouse_lineage_map.json"
ANNOTATION = "GENCODE vM19 (Ensembl 94, GRCm38)"
MIN_READS, MIN_CELLS = 20, 3
EPI = ("EPITHELIAL", "KERATINOCYTE", "BASAL", "HEPATOCYTE", "DUCTAL", "INTESTINAL", "ISLET", "KIDNEY CELL", "LUMINAL")   # c10_catalog
MES = ("FIBROBLAST", "MESENCHYMAL", "STROMAL", "SMOOTH MUSCLE", "PERICYTE", "SKELETAL MUSCLE", "CARDIAC MUSCLE", "VLMC")
GREY_NOIV = os.environ.get("FIGAN_GREY_NOIV") == "1"
# key -> event panel spec; junctions from the annotation table (key), lineage scheme from the lineage map
EVENTS = {
    "c": dict(key="gria2", gene="Gria2", title="flip/flop", scheme="allen"),
    "d": dict(key="fgfr2", gene="Fgfr2", title="IIIb vs IIIc", scheme="tms_epithelial"),
    "e": dict(key="nrxn1_ss4", gene="Nrxn1", title="SS4", scheme="allen_inhibitory_split"),   # Karin 2026-11-11: variant C is panel e
    "f": dict(key="ptprc", gene="Ptprc", title="variable exons", scheme="tms_immune", pick="ptprc_10_4"),
}
# Panel e = Nrxn1 SS4 by the SpliceVI Fig 2 definition (atse_7, 90-nt exon; event_annotation.py), Karin 2026-11-11 (variant C).
# The old panel e (atse_2, the note-4 30-nt exon, an early Nrxn-alpha site, NOT SS4) is kept only as variant A:
if os.environ.get("FIGAN_PANEL_E") == "atse2":
    EVENTS["e"] = dict(key="nrxn1", gene="Nrxn1", title="early site (atse_2; not SS4)", scheme="allen_inhibitory_split")


# ---------------------------------------------------------------- data
@functools.lru_cache(None)
def _catalog():
    v = pd.read_parquet(os.path.join(PRE, "catalog_junctions.parquet"))
    cat = json.load(open(os.path.join(PRE, "catalog_events.json")))
    return v, cat


@functools.lru_cache(None)
def _lineage_map():
    return json.load(open(LINEAGE_MAP))


def junction_table(cfg):
    return pd.read_csv(os.path.join(cfg["output_dir"], "data", "figAN_event_junctions.tsv"), sep="\t")


@functools.lru_cache(None)
def _ds(ds):
    E = np.load(os.path.join(V, "c10", f"{ds}_s0_estimates.npz"), allow_pickle=True)
    IV = np.load(os.path.join(V, "c12", "export", f"{ds}_intervals.npz"))
    T = np.load(os.path.join(PRE, f"{ds}_train_tables.npz"))
    meta = json.load(open(os.path.join(PRE, f"{ds}_record.json")))["types_meta"]
    types = E["types"].astype(str)
    assert np.array_equal(types, T["types"].astype(str)) and np.array_equal(types, IV["types"].astype(str))
    assert [m["type"] for m in meta] == list(types)
    z = np.load(FIT[ds], allow_pickle=True)   # junction / ATSE order of the fit (= estimates, train tables, intervals)
    jid, inv, ev = z["junction_id"].astype(str), np.asarray(z["inv_ev"], np.int64), z["event_id"].astype(str)
    assert E["point"].shape == (len(types), len(jid))
    return dict(types=types, point=E["point"], lo=IV["lo"], hi=IV["hi"], jreads=T["jreads"], ereads=T["ereads"], cov=T["cov"],
                treads=IV["test_reads"], tcells=IV["test_cells"], meta=meta, jid=jid, inv=inv, ev=ev, tau2=str(IV["tau2"]))


def lineage_of(ds, m):
    """fine lineage of a type from the lineage map (TMS: broad cell type; Allen: class, inhibitory split by cluster-name prefix)"""
    L = _lineage_map()
    if ds == "TMS":
        lin = L["tms_broad_to_lineage"][m["cell_type"]]
        ct = m["cell_type"].upper()   # the map reproduces the check 10 keyword lists
        kw = "Epithelial" if any(k in ct for k in EPI) else "Mesenchymal" if any(k in ct for k in MES) else None
        assert (lin in ("Epithelial", "Mesenchymal")) == (kw is not None) and (kw is None or kw == lin), (m["cell_type"], lin, kw)
        return lin
    lin = L["allen_class_to_lineage"].get(m.get("allen_class"), L["allen_class_to_lineage"]["_other"])
    if lin == "Inhibitory":
        lin = next((lab for pre, lab in L["allen_inhibitory_prefix"] if m["type"].split(" ")[0] == pre), L["allen_inhibitory_other"])
    return lin


def scheme(name):
    return _lineage_map()["schemes"][name]


def collapse(sch, lin):
    m = scheme(sch)["map"]
    return m.get(lin, m.get("_other"))


@functools.lru_cache(None)
def _event_cached(cfg_dir, spec_items):
    import yaml
    cfg = yaml.safe_load(open(os.path.join(cfg_dir, "paper_config.yaml")))
    spec = dict(spec_items)
    JT = junction_table(cfg); JT = JT[JT.key == spec["key"]]
    if JT.junction_id.isna().all():
        return None
    ds = JT.dataset.iloc[0]; event = JT.event_id.iloc[0]; d = _ds(ds)
    e = int(np.flatnonzero(d["ev"] == event)[0])
    jx = {x: i for i, x in enumerate(d["jid"])}
    if spec.get("pick") == "ptprc_10_4":
        P = pd.read_csv(os.path.join(V, "tsv", "c10_TMS_s0_ptprc_direction.tsv"), sep="\t")
        cand = [jx[x] for x in P.lead_junction.unique()]
        js = [cand[int(np.argmax(d["jreads"][:, cand].sum(0)))]]
        labels = {r.junction_id: f"{r.label} ({r.transcripts.split()[0]})" for r in JT.itertuples()}
    else:
        js = [jx[x] for x in JT.junction_id]
        labels = dict(zip(JT.junction_id, JT.label))
    assert all(d["inv"][j] == e for j in js)
    keep = (d["ereads"][:, e] >= MIN_READS) & (d["cov"][:, e] >= MIN_CELLS)
    rows = []
    for k in np.flatnonzero(keep):
        m = d["meta"][k]; fine = lineage_of(ds, m)
        for n, j in enumerate(js):
            er = float(d["ereads"][k, e])
            rows.append(dict(key=spec["key"], dataset=ds, event_id=event, gene=spec["gene"], junction=d["jid"][j], junction_label=labels[d["jid"][j]],
                             junction_index=n, cell_type=d["types"][k], lineage_fine=fine, lineage=collapse(spec["scheme"], fine),
                             psi=float(d["point"][k, j]), interval_lo=float(d["lo"][k, j]), interval_hi=float(d["hi"][k, j]),
                             has_interval=bool(np.isfinite(d["lo"][k, j])), raw_psi=float(d["jreads"][k, j]) / er,
                             junction_reads=int(d["jreads"][k, j]), atse_reads=int(er), cells_with_reads=int(d["cov"][k, e]),
                             test_reads=int(d["treads"][k, e]), test_cells=int(d["tcells"][k, e])))
    D = pd.DataFrame(rows)
    order = scheme(spec["scheme"])["order"]
    first = D[D.junction_index == 0].copy()
    first["lin_rank"] = first.lineage.map({l: i for i, l in enumerate(order)})
    first = first.sort_values(["lin_rank", "psi"], ascending=[True, False])
    D["row"] = D.cell_type.map({t: i for i, t in enumerate(first.cell_type)})
    v, cat = _catalog()
    exons = cat["structure"][event]
    jall = d["jid"][d["inv"] == e]
    return D.sort_values(["junction_index", "row"]).reset_index(drop=True), list(d["jid"][js]), exons, list(jall), order, ds


def event_data(cfg, spec):
    return _event_cached(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), tuple(sorted(spec.items())))


def _event(pid, cfg=None):
    import style
    cfg = cfg or style.load_config()
    return event_data(cfg, EVENTS[pid])


@functools.lru_cache(None)
def _accuracy():
    out = []
    for ds in ("TMS", "Allen"):
        A = pd.read_csv(os.path.join(V, "tsv", f"c10_{ds}_s0_accuracy.tsv"), sep="\t")
        out.append(A)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- panel a
def panel_a(cfg, fig, last="f"):
    """Catalog per atlas (cells, cell types, junctions, ATSEs, genes; ledger inputs) and event types of the 24,475 ATSEs;
    model schematic placeholder. Inputs: inputs.fig1.catalog, inputs.fig1.event_types, c10 estimates (types per atlas)."""
    w, h = container_mm(fig)
    fs, sm = cfg["style"]["font"]["size_pt"], cfg["style"]["font"]["small_pt"]
    ink = cfg["palette"]["ink"]
    C, Ev = _a_tables(cfg)
    ax = axes_in(sub_slot(fig, 0, h * 0.50, w * 0.52, h * 0.50), (1, 0.5, 1, 3.5)); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    cols = [("TMS", "tms_data"), ("Allen", "allen_data"), ("Combined", None)]
    xs = (0.51, 0.72, 1.0)
    ax.text(0.0, 0.97, "Catalog", ha="left", va="top", fontsize=fs, color=ink)
    for x, (ds, key) in zip(xs, cols):
        ax.text(x, 0.75, (cfg["data"][key]["label"] if key else ds).replace(" ", "\n"), ha="right", va="bottom", fontsize=sm,
                color=color(cfg, key) if key else ink, fontweight="bold", linespacing=1.0)
    rows = [("cells", "Cells"), ("types", "Cell types / groups"), ("junctions", "Junctions"), ("atses", "ATSEs"), ("genes", "Genes")]
    for i, (col, lab) in enumerate(rows):
        y = 0.67 - i * 0.12
        ax.text(0.0, y, lab, ha="left", va="center", fontsize=sm, color=ink)
        for x, (ds, _) in zip(xs, cols):
            ax.text(x, y, f"{int(C.loc[ds, col]):,}", ha="right", va="center", fontsize=sm, color=ink)
    ax.text(0.0, 0.0, "One catalog; atlas columns: ≥ 1 read in that atlas", ha="left", va="bottom", fontsize=sm,
            color=cfg["palette"]["gray"])
    bx = axes_in(sub_slot(fig, 0, 0, w * 0.52, h * 0.47), (19, 1.0, 12, 4.0))
    bx.barh(np.arange(len(Ev)), Ev.atses, color=cfg["palette"]["dark_gray"], height=0.62)
    for i, r in Ev.iterrows():
        bx.text(r.atses + Ev.atses.max() * 0.03, i, f"{r.atses:,} ({100 * r.share:.0f}%)", va="center", fontsize=sm, color=ink)
    bx.set_yticks(np.arange(len(Ev))); bx.set_yticklabels([(t[0].upper() + t[1:]).replace("mutually exclusive", "mutually\nexclusive") for t in Ev.event_type]); bx.invert_yaxis()
    bx.set_xlim(0, Ev.atses.max() * 1.9); bx.set_xticks([]); bx.spines["bottom"].set_visible(False); bx.tick_params(axis="y", length=0)
    bx.set_title("Event types (24,475 ATSEs)", fontsize=sm, loc="left", pad=1.5, color=ink)
    px = axes_in(sub_slot(fig, w * 0.54, h * 0.44, w * 0.46, h * 0.56), (1, 0.5, 1, 3.5)); px.axis("off"); px.set_xlim(0, 1); px.set_ylim(0, 1)
    px.add_patch(mpatches.FancyBboxPatch((0.02, 0.04), 0.96, 0.9, boxstyle="round,pad=0,rounding_size=0.03", fc="#F2F3F4",
                                         ec=cfg["palette"]["light_gray"], lw=0.5))
    px.text(0.5, 0.70, "Model schematic (placeholder)", ha="center", va="center", fontsize=fs, color=cfg["palette"]["gray"])
    px.text(0.5, 0.36, "cells × junctions ≈ φ × ψ\nDirichlet-multinomial per ATSE", ha="center", va="center",
            fontsize=sm, color=cfg["palette"]["gray"], linespacing=1.3)
    # one legend for panels c-f (shared encoding)
    gx = axes_in(sub_slot(fig, w * 0.54, 0, w * 0.46, h * 0.44), (1, 0.5, 0.5, 0.5)); gx.axis("off"); gx.set_xlim(0, 1); gx.set_ylim(0, 1)
    teal, gray = color(cfg, "leafletfa"), cfg["palette"]["gray"]
    gx.text(0.0, 1.0, f"Panels c–{last}", ha="left", va="top", fontsize=sm, color=ink, fontweight="bold")
    gx.plot([0.03, 0.13], [0.72, 0.72], color=teal, lw=0.4, alpha=0.55); gx.scatter([0.08], [0.72], s=6, color=teal, lw=0)
    gx.text(0.17, 0.72, "LeafletFA ψ, 95% predictive interval\n(where held-out cells have ≥ 100 reads\nand ≥ 10 cells with reads)", ha="left", va="center", fontsize=sm, color=ink, linespacing=1.0)
    gx.scatter([0.08], [0.40], s=10, facecolors="none", edgecolors=gray, linewidths=0.5)
    gx.text(0.17, 0.40, "Raw TRAIN pseudobulk ψ\n(area ~ log junction reads)", ha="left", va="center", fontsize=sm, color=ink, linespacing=1.0)
    gx.text(0.0, 0.0, "Exons: GENCODE vM19 (GRCm38)" + ("\nLight gray: no validated interval" if GREY_NOIV else ""), ha="left", va="bottom",
            fontsize=sm, color=gray)


def _a_tables(cfg):
    blk = cfg["inputs"]["fig1"]
    C = pd.read_csv(os.path.join(blk["dir"], blk["catalog"]), sep="\t").set_index("dataset")
    C["types"] = [len(np.load(os.path.join(V, "c10", f"{ds}_s0_estimates.npz"), allow_pickle=True)["types"]) for ds in C.index]
    Ev = pd.read_csv(os.path.join(blk["dir"], blk["event_types"]), sep="\t")
    import re   # the ledger macros (numbers.tex) must print the same numbers
    mac = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open(NUMBERS).read()))
    for ds in C.index:
        for col in ("cells", "junctions", "atses", "genes"):
            assert mac[f"Catalog{col.capitalize() if col != 'atses' else 'ATSEs'}{ds}"].replace(",", "") == str(int(C.loc[ds, col])), (ds, col)
    for t, m in (("alt 3' site", "EventShareAltThree"), ("alt 5' site", "EventShareAltFive"), ("skipped exon", "EventShareSkippedExon"), ("complex", "EventShareComplex")):
        assert mac[m] == f"{100 * Ev.set_index('event_type').loc[t, 'share']:.0f}\\%", (t, mac[m])
    # one catalog for both atlases (Karin 2026-11-13): both fits and the catalog summary use the same var (24,475 ATSEs,
    # 79,768 junctions); the per-atlas columns count those with >= 1 read in that atlas, so each atlas is a subset.
    num = lambda k: int(mac[k].replace(",", ""))
    C.loc["Combined"] = dict(cells=num("CatalogCellsMouse"), junctions=num("CatalogJunctionsMouse"), atses=num("CatalogATSEsMouse"),
                             genes=num("CatalogGenesMouse"), types=int(C.loc[["TMS", "Allen"], "types"].sum()))
    assert C.loc["Combined", "cells"] == C.loc[["TMS", "Allen"], "cells"].sum()
    for col in ("junctions", "atses", "genes"):
        assert (C.loc[["TMS", "Allen"], col] <= C.loc["Combined", col]).all(), col
    # split "complex" into linked MXE and other complex with the browser's 5 classes (precompute.py); the other classes
    # must agree with the geometric rule's counts, and linked + other must equal the ledger's complex
    cl = pd.Series(json.load(open(os.path.join(V, "portal_browser", "precomputed", "catalog_events.json")))["classes"]).value_counts()
    assert cl.sum() == C.loc["Combined", "atses"]
    E = Ev.set_index("event_type").atses
    assert (cl["alt3"], cl["alt5"], cl["skipped_exon"]) == (E["alt 3' site"], E["alt 5' site"], E["skipped exon"])
    assert cl["mxe_linked"] + cl["complex"] == E["complex"]
    Ev = pd.concat([Ev[Ev.event_type != "complex"],
                    pd.DataFrame(dict(event_type=["linked mutually exclusive exons", "other complex"],
                                      atses=[cl["mxe_linked"], cl["complex"]]))], ignore_index=True)
    Ev["share"] = Ev.atses / Ev.atses.sum()
    return C, Ev


# ---------------------------------------------------------------- panel b
B_FACETS = [("size", ["10-49", "50-199", "200-999", ">=1000"], "TRAIN cells per type"),
            ("reads", ["0", "1-19", "20-99", ">=100"], "TRAIN reads at the event")]


def panel_b(cfg, fig):
    """Panel b (Karin 2026-11-13): absolute errors, reads facet, with shrinkage comparator and reference levels
    (fig_appnote_babs.panel_b_main). FIGAN_PANEL_B=diff gives the earlier difference panel (panel_b_diff)."""
    if os.environ.get("FIGAN_PANEL_B", "abs") == "diff":
        return panel_b_diff(cfg, fig)
    import fig_appnote_babs
    return fig_appnote_babs.panel_b_main(cfg, fig)


def panel_b_diff(cfg, fig):
    """Check 10.1: error vs held-out TEST pseudobulk (TV), LeafletFA minus raw TRAIN pseudobulk, with 95% CI, by cell-type
    size and by TRAIN reads at the event, TMS and Allen. Input: tsv/c10_{TMS,Allen}_s0_accuracy.tsv (saved check 10)."""
    w, h = container_mm(fig)
    A = _accuracy()
    sm = cfg["style"]["font"]["small_pt"]
    lo_y = min(A.diff_lo.min(), -0.07); hi_y = max(A.diff_hi.max(), 0.01)
    for f, (by, bins, xlabel) in enumerate(B_FACETS):
        ax = axes_in(sub_slot(fig, f * w / 2, 0, w / 2, h), (12.5 if f == 0 else 6.5, 12.0, 1.5, 5.5))
        ax.axhline(0, color=cfg["palette"]["gray"], lw=cfg["style"]["line"]["reference_pt"], ls=(0, (3, 2)), zorder=0)
        # atlases in neutral black / gray (Karin 2026-11-10: not the lineage or dataset colors)
        for o, (ds, key, c) in zip((-0.13, 0.13), (("TMS", "tms_data", cfg["palette"]["ink"]), ("Allen", "allen_data", cfg["palette"]["gray"]))):
            S = A[(A.dataset == ds) & (A.by == by)].set_index("bin").loc[bins]
            x = np.arange(len(bins)) + o
            ax.vlines(x, S.diff_lo, S.diff_hi, color=c, lw=0.8)
            ax.plot(x, S["diff"], ls="none", marker=cfg["data"][key]["marker"], ms=3.2, mfc=c if ds == "TMS" else "white", mec=c, mew=0.8,
                    label=cfg["data"][key]["label"])
        ax.set_xticks(np.arange(len(bins))); ax.set_xticklabels([b.replace(">=", "≥").replace("-", "–") for b in bins], rotation=35, ha="right", rotation_mode="anchor")
        ax.set_xlabel(xlabel); ax.set_ylim(lo_y * 1.08, hi_y + 0.012); ax.set_xlim(-0.5, len(bins) - 0.5)
        if f == 0:
            ax.set_ylabel("Held-out error,\nLeafletFA − raw")
            ax.legend(loc="lower right", frameon=False, handletextpad=0.2, borderaxespad=0.1)
            # direction arrows beside the y-axis (Karin 2026-11-13; replace the gray sentence above the axes)
            y0, y1 = ax.get_ylim(); z = (0 - y0) / (y1 - y0)              # axes fraction of the zero line
            for (a, b), lab in (((z - 0.06, 0.20), "LeafletFA better"), ((z + 0.04, 0.97), "raw better")):
                ax.annotate("", xy=(0.035, b), xytext=(0.035, a), xycoords="axes fraction",
                            arrowprops=dict(arrowstyle="-|>,head_length=0.25,head_width=0.12", color=cfg["palette"]["gray"],
                                            lw=cfg["style"]["line"]["reference_pt"], shrinkA=0, shrinkB=0))
                ax.text(0.07, (a + b) / 2, lab, transform=ax.transAxes, ha="left", va="center", fontsize=sm,
                        color=cfg["palette"]["gray"])
        else:
            ax.set_yticklabels([])


# ---------------------------------------------------------------- panels c-f
def _xmap(exons, J):
    """the browser's compressed coordinate (app.js structureSVG): exon segments 10 + sqrt(len), introns 8 + 7 log10(len + 1)"""
    jl = [(int(j.rsplit("_", 3)[1]) + 1, int(j.rsplit("_", 3)[2]) + 1) for j in J]          # half-open, as the browser
    ex = [(a, b + 1, kind) for a, b, _, kind in exons]
    pts = sorted({p for l, r in jl for p in (l, r)} | {p for a, b, _ in ex for p in (a, b)})
    inex = lambda u, v: any(a <= u and b >= v for a, b, _ in ex)   # noqa: E731
    wts = [(10 + np.sqrt(pts[i + 1] - pts[i])) if inex(pts[i], pts[i + 1]) else (8 + 7 * np.log10(pts[i + 1] - pts[i] + 1)) for i in range(len(pts) - 1)]
    cum = np.concatenate([[0], np.cumsum(wts)]) / sum(wts)
    pos = dict(zip(pts, cum))
    return pos, jl, ex


def _structure(cfg, ax, exons, Jall, shown):
    pos, jl, ex = _xmap(exons, Jall)
    ink, gray, light = cfg["palette"]["ink"], cfg["palette"]["gray"], cfg["palette"]["light_gray"]
    ax.axis("off"); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.32, 1.32)
    ax.plot([-0.02, 1.02], [0, 0], color=light, lw=0.5, zorder=0)
    for a, b, kind in ex:
        x0, x1 = pos[a], pos[b]
        ax.add_patch(mpatches.Rectangle((x0, -0.14), max(x1 - x0, 0.004), 0.28, fc=gray if kind == 0 else "none", ec=gray, lw=0.5,
                                        ls="-" if kind == 0 else (0, (2, 1.5)), zorder=2))
    spans = sorted(range(len(jl)), key=lambda i: pos[jl[i][1]] - pos[jl[i][0]])
    hmax = len(jl)
    labels = {j: f"J{n + 1}" for n, j in enumerate(shown)}
    for rank, i in enumerate(spans):
        l, r = jl[i]; xl, xr = pos[l], pos[r]; ht = 0.32 + 0.62 * (rank + 1) / hmax
        is_shown = Jall[i] in labels
        verts = [(xl, 0.14), (xl, 0.14 + ht * 1.33), (xr, 0.14 + ht * 1.33), (xr, 0.14)]
        ax.add_patch(mpatches.PathPatch(Path(verts, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]), fc="none",
                                        ec=color(cfg, "leafletfa") if is_shown else light, lw=1.0 if is_shown else 0.6, zorder=3 if is_shown else 1))
        if is_shown:
            ax.text((xl + xr) / 2, 0.14 + ht * 1.0 + 0.04, labels[Jall[i]], ha="center", va="bottom", fontsize=cfg["style"]["font"]["small_pt"],
                    color=ink, fontweight="bold")


def event_panel(cfg, fig, spec, subtitle=None):
    """One known event: structure on top, psi per cell type below (rules in the module docstring). spec: EVENTS entry or a
    gallery entry; subtitle: extra text after the title (the gallery's evidence type)."""
    w, h = container_mm(fig)
    fs, sm = cfg["style"]["font"]["size_pt"], cfg["style"]["font"]["small_pt"]
    ink, gray, light = cfg["palette"]["ink"], cfg["palette"]["gray"], cfg["palette"]["light_gray"]
    teal = color(cfg, "leafletfa")
    title_h = 4.0 if subtitle is None else 8.0
    out = event_data(cfg, spec)
    tax = axes_in(sub_slot(fig, 0, h - title_h, w, title_h), (4, 0, 1, 0)); tax.axis("off")
    if out is None:
        tax.text(0, 0.5, f"$\\it{{{spec['gene']}}}$ {spec['title']}", ha="left", va="center", fontsize=fs, color=ink)
        if subtitle:
            tax.text(0, -0.2, subtitle, ha="left", va="center", fontsize=sm, color=gray)
        nx = axes_in(sub_slot(fig, 0, 0, w, h - title_h), (4, 2, 2, 2)); nx.axis("off")
        nx.text(0.5, 0.85, "No ATSE of this gene in the catalog\n(not shown; nothing to estimate)", ha="center", va="center", fontsize=sm, color=gray)
        return
    D, shown, exons, Jall, order, ds = out
    yt = 0.78 if subtitle else 0.5
    tax.text(0, yt, f"$\\it{{{spec['gene']}}}$ {spec['title']} ({ds})", ha="left", va="center", fontsize=fs, color=ink)
    tax.text(1, yt, f"{D.cell_type.nunique()} {'cell groups' if ds == 'Allen' else 'cell types'}", ha="right", va="center", fontsize=sm, color=gray)
    if subtitle:
        tax.text(0, 0.28, subtitle, ha="left", va="center", fontsize=sm, color=gray)
    top_h = min(13.0, 0.26 * h)
    sax = axes_in(sub_slot(fig, 0, h - title_h - top_h, w, top_h), (24, 0.5, 2, 0.5))
    _structure(cfg, sax, exons, Jall, shown)
    lab_w = 24.0
    ncol = len(shown)
    body_h = h - title_h - top_h - 0.5
    col_w = (w - lab_w - 2) / ncol
    nrow = int(D.row.max()) + 1
    lin = D[D.junction_index == 0].groupby("lineage", sort=False).row.agg(["min", "max", "count"])
    present = [l for l in order if l in lin.index]
    gap = 1.2   # rows between lineages
    row_lin = D[D.junction_index == 0].set_index("row").lineage
    # every lineage block is at least one 6-pt label tall (2.4 mm), so small groups (e.g. PV, SST) get their own label space;
    # a small group's rows are centered in its block
    ax_h_mm = body_h - 7.2 - 3.6
    cnt = [int(lin.loc[l, "count"]) for l in present]
    k = 1
    for _ in range(20):
        tot = sum(max(c, k) for c in cnt) + gap * (len(present) - 1)
        k_new = int(np.ceil(2.4 / (ax_h_mm / tot)))
        if k_new == k:
            break
        k = k_new
    base, y = {}, 0.0
    for l, c in zip(present, cnt):
        size = max(c, k)
        base[l] = y + (size - c) / 2 - lin.loc[l, "min"]
        y += size + gap
    ytot = y - gap
    ypos = lambda r: np.array([x + base[row_lin[x]] for x in np.atleast_1d(r)], dtype=float)   # noqa: E731
    rmax = max(D.junction_reads.max(), 1)
    for n, j in enumerate(shown):
        ax = axes_in(sub_slot(fig, lab_w + n * col_w, 0, col_w, body_h), (2.0 if n else 0.5, 7.2, 1.0, 3.6))
        S = D[D.junction_index == n]
        y = ypos(S.row.values)
        for g in (0, 0.5, 1):
            ax.axvline(g, color=light, lw=0.4, zorder=0)
        iv = S.has_interval.values
        ax.hlines(y[iv], S.interval_lo.values[iv], S.interval_hi.values[iv], color=teal, lw=0.4, alpha=0.55, zorder=2)
        rs = 1.0 + 9.0 * np.log10(S.junction_reads.values + 1) / np.log10(rmax + 1)
        pc = np.where(iv | (not GREY_NOIV), teal, light)          # variant: no validated interval -> light gray
        rc = np.where(iv | (not GREY_NOIV), gray, light)
        ax.scatter(S.raw_psi, y, s=rs, facecolors="none", edgecolors=list(rc), linewidths=0.45, zorder=3)
        ax.scatter(S.psi, y, s=4.5, c=list(pc), linewidths=0, zorder=4)
        ax.set_ylim(ytot - 0.5, -0.5); ax.set_xlim(-0.03, 1.03); ax.set_yticks([])
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("ψ", labelpad=1)
        ax.set_title(f"J{n + 1}: {S.junction_label.iloc[0]}", fontsize=sm, color=ink, pad=1.5)
    lax = axes_in(sub_slot(fig, 0, 0, lab_w, body_h), (0.5, 7.2, 0.3, 3.6)); lax.axis("off")
    lax.set_ylim(ytot - 0.5, -0.5); lax.set_xlim(0, 1)
    cmap = scheme(spec["scheme"])["colors"]
    sep = 2.2 / ax_h_mm * ytot                      # minimum label spacing (one 6-pt line) in row units
    spans = [(l, ypos(lin.loc[l, "min"])[0], ypos(lin.loc[l, "max"])[0]) for l in present]
    lab_y = [(y0 + y1) / 2 for _, y0, y1 in spans]
    for i in range(1, len(lab_y)):                  # push labels apart, top to bottom, then shift the block up if it overflows
        lab_y[i] = max(lab_y[i], lab_y[i - 1] + sep)
    if lab_y and lab_y[-1] > ytot - 0.5 - sep / 2:   # overflow at the bottom: pull up only the labels that collide
        lab_y[-1] = ytot - 0.5 - sep / 2
        for i in range(len(lab_y) - 2, -1, -1):
            lab_y[i] = min(lab_y[i], lab_y[i + 1] - sep)
    for (l, y0, y1), ly in zip(spans, lab_y):
        lax.add_patch(mpatches.Rectangle((0.90, y0 - 0.45), 0.07, y1 - y0 + 0.9, fc=cfg["palette"][cmap[l]], ec="none"))
        lax.text(0.80, ly, f"{l} ({int(lin.loc[l, 'count'])})", ha="right", va="center", fontsize=sm, color=ink, linespacing=1.0)
        lax.plot([0.81, 0.89], [ly, (y0 + y1) / 2], color=light, lw=0.4)   # leader from label to its strip


def panel_c(cfg, fig):
    """Gria2 flip/flop (Allen); rules in the module docstring."""
    event_panel(cfg, fig, EVENTS["c"])


def panel_d(cfg, fig):
    """Fgfr2 IIIb/IIIc (TMS); rules in the module docstring."""
    event_panel(cfg, fig, EVENTS["d"])


def panel_e(cfg, fig):
    """Nrxn1 SS4 (Allen); rules in the module docstring."""
    event_panel(cfg, fig, EVENTS["e"])


def panel_f(cfg, fig):
    """Ptprc (TMS); rules in the module docstring."""
    event_panel(cfg, fig, EVENTS["f"])


PANELS = {"AN_a": panel_a, "AN_b": panel_b, "AN_c": panel_c, "AN_d": panel_d, "AN_e": panel_e, "AN_f": panel_f}
SOURCE_DATA = {
    "AN_a": lambda cfg: dict(zip(("figAN_a_catalog", "figAN_a_event_types"), (lambda C, E: (C.reset_index(), E))(*_a_tables(cfg)))),
    "AN_b": lambda cfg: ({"figAN_b_accuracy_10_1": _accuracy()} if os.environ.get("FIGAN_PANEL_B", "abs") == "diff" else
                         {"figAN_b_accuracy_10_1": _accuracy(), "figAN_b_abs_numbers": __import__("fig_appnote_babs")._table()}),
    **{f"AN_{p}": (lambda p: lambda cfg: {f"figAN_{p}_{EVENTS[p]['gene']}": event_data(cfg, EVENTS[p])[0]})(p) for p in "cdef"},
}
