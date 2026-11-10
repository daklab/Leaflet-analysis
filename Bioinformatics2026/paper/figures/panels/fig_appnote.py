"""Application Note figure (one figure, six panels; APPLICATION_NOTE_FIGURE_PLAN.md, layout approved 2026-11-06).

a  catalog per atlas and event types (ledger inputs) + model schematic placeholder (Karin finishes it)
b  check 10.1: held-out error, LeafletFA minus raw TRAIN pseudobulk, by cell-type size and by TRAIN reads, both atlases
c  Gria2 flip/flop (Allen)   d  Fgfr2 IIIb/IIIc (TMS)   e  Nrxn1 SS4 (Allen)   f  Ptprc (TMS)

Panels c-f, fixed rules (no hand-picking):
- Cell types shown: every type (TMS tissue-cell type; Allen cell group) with >= 20 TRAIN reads and >= 3 TRAIN cells with reads
  at the ATSE (the browser's "range" support rule, build_portal.RANGE_READS / RANGE_CELLS), grouped by lineage, sorted within
  a lineage by LeafletFA psi of the first junction shown (highest at top).
- Lineages: Allen = class of the group's TRAIN cells (Glutamatergic -> excitatory, GABAergic -> inhibitory, else non-neuronal;
  the browser's rule). TMS = the broad cell type of the label (cell-type map: the part after the tissue), Fgfr2 by the check 10
  epithelial / mesenchymal keyword lists (c10_catalog.EPI / MES), Ptprc by B cell / T cell / other immune / non-immune.
- Junctions shown: c Gria2 = the two junctions from the shared left site of the MXE pair (L, b1), (a1, R), (L, b2), (a2, R)
  (DEFINITIONS.md 1.6) into each of the two ~115-nt exons; d Fgfr2 = the IIIb junction (c10_catalog.FGFR2_IIIB);
  e Nrxn1 = among junctions with an end at the annotated 30-nt SS4 exon (c10_known_biology_fix.SS4), the one with the most TRAIN
  reads over all types; f Ptprc = among the lead junctions of the check 10.4 Ptprc calls (c10_TMS_s0_ptprc_direction.tsv), the
  one with the most TRAIN reads over all types.
- Marks: LeafletFA point estimate (check 10 point, seed 0) in teal, with the check 12 (a)+(b) interval where held-out TEST cells
  have >= 100 reads and >= 10 cells with reads at the ATSE (point only elsewhere); raw TRAIN pseudobulk psi as a gray ring,
  area by junction reads.
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
MIN_READS, MIN_CELLS = 20, 3
EPI = ("EPITHELIAL", "KERATINOCYTE", "BASAL", "HEPATOCYTE", "DUCTAL", "INTESTINAL", "ISLET", "KIDNEY CELL", "LUMINAL")   # c10_catalog
MES = ("FIBROBLAST", "MESENCHYMAL", "STROMAL", "SMOOTH MUSCLE", "PERICYTE", "SKELETAL MUSCLE", "CARDIAC MUSCLE", "VLMC")
FGFR2_IIIB = "chr7_130199905_130200982_-"
SS4_NRXN1 = ("chr17", 90992115, 90992144)
OTHER_IMMUNE = ("NK CELL", "MACROPHAGE", "MONOCYTE", "GRANULOCYTE", "DENDRITIC CELL", "MYELOID IMMUNE CELL", "MICROGLIA",
                "GENERAL IMMUNE CELL", "HEMATOPOIETIC PROGENITOR")
EVENTS = {
    "c": dict(ds="Allen", event="ENSMUSG00000033981.14_atse_3", gene="Gria2", title="flip/flop exons", rule="gria2"),
    "d": dict(ds="TMS", event="ENSMUSG00000030849.18_atse_3", gene="Fgfr2", title="IIIb vs IIIc", rule="fgfr2"),
    "e": dict(ds="Allen", event="ENSMUSG00000024109.18_atse_2", gene="Nrxn1", title="SS4", rule="nrxn1"),
    "f": dict(ds="TMS", event="ENSMUSG00000026395.16_atse_2", gene="Ptprc", title="variable exons", rule="ptprc"),
}
# lineage -> palette token (strip and label only; data marks stay LeafletFA teal and raw gray). <= 3 colors + light gray.
LINEAGES = {
    "allen": [("Excitatory", "indigo"), ("Inhibitory", "ochre"), ("Non-neuronal", "light_gray")],
    "fgfr2": [("Epithelial", "indigo"), ("Mesenchymal", "ochre"), ("Other", "light_gray")],
    "ptprc": [("B cell", "indigo"), ("T cell", "ochre"), ("Other immune", "olive"), ("Non-immune", "light_gray")],
}


# ---------------------------------------------------------------- data
@functools.lru_cache(None)
def _catalog():
    v = pd.read_parquet(os.path.join(PRE, "catalog_junctions.parquet"))
    cat = json.load(open(os.path.join(PRE, "catalog_events.json")))
    return v, cat


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


def _lineage(rule, m):
    if rule in ("gria2", "nrxn1"):
        c = m.get("allen_class")
        return "Excitatory" if c == "Glutamatergic" else "Inhibitory" if c == "GABAergic" else "Non-neuronal"
    ct = m["cell_type"].upper()
    if rule == "fgfr2":
        return "Epithelial" if any(k in ct for k in EPI) else "Mesenchymal" if any(k in ct for k in MES) else "Other"
    return "B cell" if ct == "B CELL" else "T cell" if ct == "T CELL" else "Other immune" if ct in OTHER_IMMUNE else "Non-immune"


def _lineage_key(rule):
    return "allen" if rule in ("gria2", "nrxn1") else rule


def _junctions(rule, d, e):
    """indices (fit order) of the junctions shown, by the rules in the module docstring"""
    J = np.flatnonzero(d["inv"] == e)
    ids = d["jid"][J]
    a = np.array([int(x.rsplit("_", 3)[1]) for x in ids]); b = np.array([int(x.rsplit("_", 3)[2]) for x in ids])
    if rule == "fgfr2":
        return [J[list(ids).index(FGFR2_IIIB)]]
    if rule == "gria2":
        # MXE pair (L, b1), (a1, R), (L, b2), (a2, R): exon 1 = [b1 + 1, a1] before exon 2 = [b2 + 1, a2], both > 100 nt
        pairs = set()
        for i in range(len(J)):
            for k in range(len(J)):
                if a[i] != a[k] or not b[i] < b[k]:
                    continue
                for p in range(len(J)):
                    for q in range(len(J)):
                        if (b[i] < a[p] <= b[k] and a[q] > b[k] and b[p] == b[q] and b[p] > a[q]
                                and a[p] - b[i] > 100 and a[q] - b[k] > 100):
                            pairs.add((i, k))
        assert len(pairs) == 1, f"Gria2 MXE rule: {pairs}"
        i, k = pairs.pop()
        return [J[i], J[k]]
    if rule == "nrxn1":
        c, s, t = SS4_NRXN1
        cand = [J[m] for m in range(len(J)) if b[m] == s - 1 or a[m] == t]
    else:
        P = pd.read_csv(os.path.join(V, "tsv", "c10_TMS_s0_ptprc_direction.tsv"), sep="\t")
        cand = [J[list(ids).index(x)] for x in P.lead_junction.unique()]
    tot = d["jreads"][:, cand].sum(0)
    return [cand[int(np.argmax(tot))]]


@functools.lru_cache(None)
def _event(pid):
    spec = EVENTS[pid]; d = _ds(spec["ds"])
    e = int(np.flatnonzero(d["ev"] == spec["event"])[0])
    js = _junctions(spec["rule"], d, e)
    keep = (d["ereads"][:, e] >= MIN_READS) & (d["cov"][:, e] >= MIN_CELLS)
    rows = []
    for k in np.flatnonzero(keep):
        m = d["meta"][k]
        for n, j in enumerate(js):
            er = float(d["ereads"][k, e])
            rows.append(dict(panel=pid, dataset=spec["ds"], event_id=spec["event"], gene=spec["gene"], junction=d["jid"][j], junction_index=n,
                             cell_type=d["types"][k], lineage=_lineage(spec["rule"], m), psi=float(d["point"][k, j]),
                             interval_lo=float(d["lo"][k, j]), interval_hi=float(d["hi"][k, j]),
                             has_interval=bool(np.isfinite(d["lo"][k, j])), raw_psi=float(d["jreads"][k, j]) / er,
                             junction_reads=int(d["jreads"][k, j]), atse_reads=int(er), cells_with_reads=int(d["cov"][k, e]),
                             test_reads=int(d["treads"][k, e]), test_cells=int(d["tcells"][k, e])))
    D = pd.DataFrame(rows)
    order = [l for l, _ in LINEAGES[_lineage_key(spec["rule"])]]
    first = D[D.junction_index == 0].copy()
    first["lin_rank"] = first.lineage.map({l: i for i, l in enumerate(order)})
    first = first.sort_values(["lin_rank", "psi"], ascending=[True, False])
    D["row"] = D.cell_type.map({t: i for i, t in enumerate(first.cell_type)})
    v, cat = _catalog()
    exons = cat["structure"][spec["event"]]
    jall = d["jid"][d["inv"] == e]
    return D.sort_values(["junction_index", "row"]).reset_index(drop=True), list(d["jid"][js]), exons, list(jall), order


@functools.lru_cache(None)
def _accuracy():
    out = []
    for ds in ("TMS", "Allen"):
        A = pd.read_csv(os.path.join(V, "tsv", f"c10_{ds}_s0_accuracy.tsv"), sep="\t")
        out.append(A)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- panel a
def panel_a(cfg, fig):
    """Catalog per atlas (cells, cell types, junctions, ATSEs, genes; ledger inputs) and event types of the 24,475 ATSEs;
    model schematic placeholder. Inputs: inputs.fig1.catalog, inputs.fig1.event_types, c10 estimates (types per atlas)."""
    w, h = container_mm(fig)
    fs, sm = cfg["style"]["font"]["size_pt"], cfg["style"]["font"]["small_pt"]
    ink = cfg["palette"]["ink"]
    C, Ev = _a_tables(cfg)
    ax = axes_in(sub_slot(fig, 0, h * 0.50, w * 0.52, h * 0.50), (1, 0.5, 1, 3.5)); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    cols = [("TMS", "tms_data"), ("Allen", "allen_data")]
    ax.text(0.0, 0.97, "Catalog", ha="left", va="top", fontsize=fs, color=ink)
    for j, (ds, key) in enumerate(cols):
        ax.text(0.68 + 0.30 * j, 0.80, cfg["data"][key]["label"], ha="right", va="center", fontsize=sm, color=color(cfg, key), fontweight="bold")
    rows = [("cells", "Cells"), ("types", "Cell types / groups"), ("junctions", "Junctions"), ("atses", "ATSEs"), ("genes", "Genes")]
    for i, (col, lab) in enumerate(rows):
        y = 0.66 - i * 0.145
        ax.text(0.0, y, lab, ha="left", va="center", fontsize=sm, color=ink)
        for j, (ds, _) in enumerate(cols):
            ax.text(0.68 + 0.30 * j, y, f"{int(C.loc[ds, col]):,}", ha="right", va="center", fontsize=sm, color=ink)
    bx = axes_in(sub_slot(fig, 0, 0, w * 0.52, h * 0.47), (19, 1.0, 12, 4.0))
    bx.barh(np.arange(len(Ev)), Ev.atses, color=cfg["palette"]["dark_gray"], height=0.62)
    for i, r in Ev.iterrows():
        bx.text(r.atses + Ev.atses.max() * 0.03, i, f"{r.atses:,} ({100 * r.share:.0f}%)", va="center", fontsize=sm, color=ink)
    bx.set_yticks(np.arange(len(Ev))); bx.set_yticklabels([t[0].upper() + t[1:] for t in Ev.event_type]); bx.invert_yaxis()
    bx.set_xlim(0, Ev.atses.max() * 1.9); bx.set_xticks([]); bx.spines["bottom"].set_visible(False); bx.tick_params(axis="y", length=0)
    bx.set_title("Event types (24,475 ATSEs)", fontsize=sm, loc="left", pad=1.5, color=ink)
    px = axes_in(sub_slot(fig, w * 0.54, h * 0.36, w * 0.46, h * 0.64), (1, 0.5, 1, 3.5)); px.axis("off"); px.set_xlim(0, 1); px.set_ylim(0, 1)
    px.add_patch(mpatches.FancyBboxPatch((0.02, 0.04), 0.96, 0.9, boxstyle="round,pad=0,rounding_size=0.03", fc="#F2F3F4",
                                         ec=cfg["palette"]["light_gray"], lw=0.5))
    px.text(0.5, 0.70, "Model schematic (placeholder)", ha="center", va="center", fontsize=fs, color=cfg["palette"]["gray"])
    px.text(0.5, 0.36, "cells × junctions ≈ φ × ψ\nDirichlet-multinomial per ATSE\nK = 20 programs", ha="center", va="center",
            fontsize=sm, color=cfg["palette"]["gray"], linespacing=1.3)
    # one legend for panels c-f (shared encoding)
    gx = axes_in(sub_slot(fig, w * 0.54, 0, w * 0.46, h * 0.36), (1, 0.5, 0.5, 0.5)); gx.axis("off"); gx.set_xlim(0, 1); gx.set_ylim(0, 1)
    teal, gray = color(cfg, "leafletfa"), cfg["palette"]["gray"]
    gx.text(0.0, 0.95, "Panels c–f", ha="left", va="top", fontsize=sm, color=ink, fontweight="bold")
    gx.plot([0.03, 0.13], [0.68, 0.68], color=teal, lw=0.6); gx.scatter([0.08], [0.68], s=6, color=teal, lw=0)
    gx.text(0.17, 0.68, "LeafletFA ψ, 95% predictive interval\n(where held-out cells have ≥ 100 reads\nand ≥ 10 cells with reads)", ha="left", va="center", fontsize=sm, color=ink, linespacing=1.0)
    gx.scatter([0.08], [0.16], s=10, facecolors="none", edgecolors=gray, linewidths=0.5)
    gx.text(0.17, 0.16, "Raw TRAIN pseudobulk ψ\n(area ~ log junction reads)", ha="left", va="center", fontsize=sm, color=ink, linespacing=1.0)


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
    return C, Ev


# ---------------------------------------------------------------- panel b
B_FACETS = [("size", ["10-49", "50-199", "200-999", ">=1000"], "TRAIN cells per type"),
            ("reads", ["0", "1-19", "20-99", ">=100"], "TRAIN reads at the event")]


def panel_b(cfg, fig):
    """Check 10.1: error vs held-out TEST pseudobulk (TV), LeafletFA minus raw TRAIN pseudobulk, with 95% CI, by cell-type
    size and by TRAIN reads at the event, TMS and Allen. Input: tsv/c10_{TMS,Allen}_s0_accuracy.tsv (saved check 10)."""
    w, h = container_mm(fig)
    A = _accuracy()
    sm = cfg["style"]["font"]["small_pt"]
    lo_y = min(A.diff_lo.min(), -0.07); hi_y = max(A.diff_hi.max(), 0.01)
    for f, (by, bins, xlabel) in enumerate(B_FACETS):
        ax = axes_in(sub_slot(fig, f * w / 2, 0, w / 2, h), (12.5 if f == 0 else 6.5, 12.0, 1.5, 5.5))
        ax.axhline(0, color=cfg["palette"]["gray"], lw=cfg["style"]["line"]["reference_pt"], ls=(0, (3, 2)), zorder=0)
        for o, (ds, key) in zip((-0.13, 0.13), (("TMS", "tms_data"), ("Allen", "allen_data"))):
            S = A[(A.dataset == ds) & (A.by == by)].set_index("bin").loc[bins]
            x = np.arange(len(bins)) + o
            c = color(cfg, key)
            ax.vlines(x, S.diff_lo, S.diff_hi, color=c, lw=0.8)
            ax.plot(x, S["diff"], ls="none", marker=cfg["data"][key]["marker"], ms=3.2, mfc=c, mec=c, label=cfg["data"][key]["label"])
        ax.set_xticks(np.arange(len(bins))); ax.set_xticklabels([b.replace(">=", "≥").replace("-", "–") for b in bins], rotation=35, ha="right", rotation_mode="anchor")
        ax.set_xlabel(xlabel); ax.set_ylim(lo_y * 1.08, hi_y + 0.012); ax.set_xlim(-0.5, len(bins) - 0.5)
        if f == 0:
            ax.set_ylabel("Held-out error,\nLeafletFA − raw (TV)")
            ax.legend(loc="lower right", frameon=False, handletextpad=0.2, borderaxespad=0.1)
        else:
            ax.set_yticklabels([])
        ax.text(0.02, 1.0, "LeafletFA closer to held-out cells below 0", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=sm, color=cfg["palette"]["gray"]) if f == 0 else None


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
    ax.axis("off"); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.32, 1.12)
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


def _jdesc(j, exons, rule, n):
    s, e = int(j.rsplit("_", 3)[1]), int(j.rsplit("_", 3)[2])
    if rule == "gria2":
        ex = [x for x in exons if x[0] == e + 1]
        return f"into exon {ex[0][0]:,}–{str(ex[0][1])[-3:]}" if ex else j
    return {"fgfr2": "IIIb junction", "nrxn1": "SS4 junction", "ptprc": "lead junction (10.4)"}[rule]


def event_panel(cfg, fig, pid):
    spec = EVENTS[pid]
    D, shown, exons, Jall, order = _event(pid)
    w, h = container_mm(fig)
    fs, sm = cfg["style"]["font"]["size_pt"], cfg["style"]["font"]["small_pt"]
    ink, gray = cfg["palette"]["ink"], cfg["palette"]["gray"]
    teal = color(cfg, "leafletfa")
    top_h, title_h = 13.0, 4.0
    tax = axes_in(sub_slot(fig, 0, h - title_h, w, title_h), (4, 0, 1, 0)); tax.axis("off")
    tax.text(0, 0.5, f"$\\it{{{spec['gene']}}}$ {spec['title']} ({spec['ds']})", ha="left", va="center", fontsize=fs, color=ink)
    tax.text(1, 0.5, f"{D.cell_type.nunique()} {'cell groups' if spec['ds'] == 'Allen' else 'cell types'}", ha="right", va="center", fontsize=sm, color=gray)
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
    ypos = lambda r: np.atleast_1d(r) + gap * np.array([present.index(row_lin[x]) for x in np.atleast_1d(r)])   # noqa: E731
    ytot = nrow + gap * (len(lin) - 1)
    rmax = max(D.junction_reads.max(), 1)
    lkey = _lineage_key(spec["rule"])
    for n, j in enumerate(shown):
        ax = axes_in(sub_slot(fig, lab_w + n * col_w, 0, col_w, body_h), (2.0 if n else 0.5, 7.2, 1.0, 3.6))
        S = D[D.junction_index == n]
        y = ypos(S.row.values)
        for g in (0, 0.5, 1):
            ax.axvline(g, color=cfg["palette"]["light_gray"], lw=0.4, zorder=0)
        iv = S.has_interval.values
        ax.hlines(y[iv], S.interval_lo.values[iv], S.interval_hi.values[iv], color=teal, lw=0.6, zorder=2)
        rs = 1.0 + 9.0 * np.log10(S.junction_reads.values + 1) / np.log10(rmax + 1)
        ax.scatter(S.raw_psi, y, s=rs, facecolors="none", edgecolors=gray, linewidths=0.45, zorder=3)
        ax.scatter(S.psi, y, s=4.5, color=teal, linewidths=0, zorder=4)
        ax.set_ylim(ytot - 0.5, -0.5); ax.set_xlim(-0.03, 1.03); ax.set_yticks([])
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("ψ", labelpad=1)
        ax.set_title(f"J{n + 1}: {_jdesc(j, exons, spec['rule'], n)}", fontsize=sm, color=ink, pad=1.5)
    lax = axes_in(sub_slot(fig, 0, 0, lab_w, body_h), (0.5, 7.2, 0.3, 3.6)); lax.axis("off")
    lax.set_ylim(ytot - 0.5, -0.5); lax.set_xlim(0, 1)
    cmap = dict(LINEAGES[lkey])
    for l, r in lin.iterrows():
        y0, y1 = ypos(r["min"])[0], ypos(r["max"])[0]
        lax.add_patch(mpatches.Rectangle((0.90, y0 - 0.45), 0.07, y1 - y0 + 0.9, fc=cfg["palette"][cmap[l]], ec="none"))
        lax.text(0.86, (y0 + y1) / 2, f"{l} ({int(r['count'])})", ha="right", va="center", fontsize=sm, color=ink, linespacing=1.0)


def panel_c(cfg, fig):
    """Gria2 flip/flop (Allen); rules in the module docstring."""
    event_panel(cfg, fig, "c")


def panel_d(cfg, fig):
    """Fgfr2 IIIb/IIIc (TMS); rules in the module docstring."""
    event_panel(cfg, fig, "d")


def panel_e(cfg, fig):
    """Nrxn1 SS4 (Allen); rules in the module docstring."""
    event_panel(cfg, fig, "e")


def panel_f(cfg, fig):
    """Ptprc (TMS); rules in the module docstring."""
    event_panel(cfg, fig, "f")


PANELS = {"AN_a": panel_a, "AN_b": panel_b, "AN_c": panel_c, "AN_d": panel_d, "AN_e": panel_e, "AN_f": panel_f}
SOURCE_DATA = {
    "AN_a": lambda cfg: dict(zip(("figAN_a_catalog", "figAN_a_event_types"), (lambda C, E: (C.reset_index(), E))(*_a_tables(cfg)))),
    "AN_b": lambda cfg: {"figAN_b_accuracy_10_1": _accuracy()},
    **{f"AN_{p}": (lambda p: lambda cfg: {f"figAN_{p}_{EVENTS[p]['gene']}": _event(p)[0]})(p) for p in "cdef"},
}
