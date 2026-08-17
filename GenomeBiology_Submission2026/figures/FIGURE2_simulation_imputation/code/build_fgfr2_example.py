#!/usr/bin/env python
"""Fgfr2 exon IIIb / IIIc, raw vs imputed, per cell type, with genomic context and program usage.

ATSE 3 of Fgfr2 (chr7, minus strand): shared upstream exon (ends 130,200,982) spliced to exon IIIb
(starts 130,199,905; epithelial isoform) or exon IIIc (starts 130,198,563; mesenchymal isoform).
Imputed = the DM posterior mean (gamma_e rho_cj + y_cj)/(gamma_e + N_ce), rho = phi . psi from the
shipped K=20 fit; for cells with no reads at the event it equals the prior phi . psi.

a  gene model with the two junctions as arcs
b  epithelial and mesenchymal cell types (>= 5 covered cells): raw IIIb usage of covered cells (jitter + violin, left) and
   imputed IIIb usage of ALL cells of that type (violin, right); types ordered by lineage then raw
c  mean phi of the same cell types, programs as SP1..SP20 (SP1 = largest pi)
"""
import os, sys
import numpy as np, pandas as pd, h5py, torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
FIGROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, FIGROOT); sys.path.insert(0, os.path.join(FIGROOT, "_common"))
sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa")
sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from leaflet_theme import apply_theme, PALETTE, METHOD_COLORS, figsize, panel_label, despine, save
from programs import sp_order
from _phi_rows import phi_row_of
from leafletfa.dirichlet_multinomial import segment_softmax
_spec = importlib.util.spec_from_file_location("viz", "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/08_leafcutter_ds/04_visualize_ds_atse.py")
viz = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(viz)
import gffutils

M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
H5 = f"{M}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad"
FITS = [f"{M}/ATLAS_DM/runs_final_deltaonly/dm_simplex_K20_seed{s}_b8192/fit.npz" for s in (0, 1, 2)]
FIT = FITS[0]
EV = "ENSMUSG00000030849.18_atse_3"
J_IIIB, J_IIIC = "chr7_130199905_130200982_-", "chr7_130198563_130200982_-"
EPI = ("EPITHELIAL", "KERATINOCYTE", "BASAL", "HEPATOCYTE", "DUCTAL", "INTESTINAL", "ISLET", "KIDNEY CELL", "LUMINAL")
MES = ("FIBROBLAST", "MESENCHYMAL", "STROMAL", "SMOOTH MUSCLE", "PERICYTE", "SKELETAL MUSCLE", "CARDIAC MUSCLE", "VLMC")
NEU = ("NEURON", "OLIGODENDROCYTE", "ASTROCYTE", "MICROGLIA", "STEM CELL")
CACHE = os.path.join(FIGROOT, "FIGURE2_simulation_imputation", "data", "fgfr2_cells.npz")


def cat(g):
    c = np.array([x.decode() if isinstance(x, bytes) else x for x in g["categories"][:]]); return c[g["codes"][:]]


def arc(ax, x0, x1, h, color, lw, label=None):
    verts = [(x0, 0), ((x0 + x1) / 2, h), (x1, 0)]
    ax.add_patch(mpatches.PathPatch(Path(verts, [Path.MOVETO, Path.CURVE3, Path.CURVE3]), fc="none", ec=color, lw=lw, zorder=4))
    if label: ax.text((x0 + x1) / 2, h * 0.7, label, ha="center", va="bottom", fontsize=6.5, color=color)


def lineage(t):
    u = t.upper()
    if any(k in u for k in EPI): return "epithelial"
    if any(k in u for k in MES): return "mesenchymal"
    if any(k in u for k in NEU): return "neural / glial"
    return "other"


def load():
    if os.path.exists(CACHE):
        c = np.load(CACHE, allow_pickle=True); Y = c["Y"]; lab = c["lab"]; jn = c["jn"]
    else:
        z0 = np.load(FIT, allow_pickle=True)
        fev = np.asarray(z0["event_id"]).astype(str); inv = np.asarray(z0["inv_ev"]).astype(np.int64)
        e = int(np.flatnonzero(fev == EV)[0]); cols = np.flatnonzero(inv == e)
        with h5py.File(H5, "r") as f:
            jid = cat(f["var/junction_id"]) if isinstance(f["var/junction_id"], h5py.Group) else f["var/junction_id"][:].astype(str)
            lab = cat(f["obs/tissue_cell_type"]); N = len(lab)
            g = f["layers/cell_by_junction_matrix"]; ip = g["indptr"][:]
            Y = np.zeros((N, len(cols)), np.float32); CH = 20000
            for s0 in range(0, N, CH):
                s1 = min(N, s0 + CH); a, b = int(ip[s0]), int(ip[s1]); idx = g["indices"][a:b]; dat = g["data"][a:b]
                rowid = np.repeat(np.arange(s0, s1), np.diff(ip[s0:s1 + 1]))
                for jj, cc in enumerate(cols):
                    m = idx == cc; Y[rowid[m], jj] += dat[m]
        jn = jid[cols]; np.savez_compressed(CACHE, Y=Y, lab=lab, jn=jn)
    ib = int(np.flatnonzero(jn == J_IIIB)[0]); Nt = Y.sum(1)
    raw = np.where(Nt > 0, Y[:, ib] / np.maximum(Nt, 1), np.nan)
    df = pd.DataFrame(dict(lab=lab, raw=raw, n=Nt)); df["lin"] = [lineage(t) for t in df.lab]
    seeds = []
    for fp in FITS:
        z = np.load(fp, allow_pickle=True)
        fev = np.asarray(z["event_id"]).astype(str); inv = np.asarray(z["inv_ev"]).astype(np.int64)
        e = int(np.flatnonzero(fev == EV)[0]); cols = np.flatnonzero(inv == e)
        phi = np.asarray(z["phi"]); conc = float(np.asarray(z["bb_conc_vec"])[e])
        psi = segment_softmax(torch.from_numpy(np.asarray(z["psi_loc"], np.float32)), torch.from_numpy(inv), len(fev)).numpy()[:, cols]
        prow = phi_row_of(z, len(lab)); tr = prow >= 0
        rho = np.full((len(lab), len(cols)), np.nan, np.float32); rho[tr] = phi[prow[tr]] @ psi
        post = (conc * rho + Y) / (conc + Nt[:, None])
        PH = np.full((len(lab), phi.shape[1]), np.nan, np.float32); PH[tr] = phi[prow[tr]]
        seeds.append(dict(post=post[:, ib], prior=rho[:, ib], PH=PH, order=sp_order(z), psi_b=psi[:, ib], tr=tr))
    df["tr"] = seeds[0]["tr"]
    for i, sd in enumerate(seeds): df[f"post_{i}"] = sd["post"]; df[f"prior_{i}"] = sd["prior"]
    df["post"] = df[[f"post_{i}" for i in range(len(seeds))]].mean(1)
    df["prior"] = df[[f"prior_{i}" for i in range(len(seeds))]].mean(1)
    return df, seeds, jn


def main():
    apply_theme()
    df, seeds, jn = load()
    K = len(seeds[0]["order"])
    ct = df[df.tr].groupby("lab").agg(ncell=("tr", "size"), ncov=("n", lambda x: int((x > 0).sum())), lin=("lin", "first"),
                                     raw_mean=("raw", "mean"), post_mean=("post", "mean"))
    keep = ct[(ct.ncov >= 5) & ct.lin.isin(["epithelial", "mesenchymal"])].copy()
    keep["lin_rank"] = keep.lin.map({"epithelial": 0, "mesenchymal": 1, "neural / glial": 2, "other": 3})
    keep = keep.sort_values(["lin_rank", "raw_mean"], ascending=[True, False])
    types = list(keep.index)
    col = {"epithelial": PALETTE["ochre"], "mesenchymal": PALETTE["teal"], "neural / glial": PALETTE["indigo"], "other": PALETTE["grey"]}

    fig = plt.figure(figsize=figsize("double", h=8.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.9, 1.5, 1.6], hspace=0.65, wspace=0.55)
    # a: gene model
    ax = fig.add_subplot(gs[0, :])
    parts = [j.split("_") for j in jn]; chrom = parts[0][0]; strand = parts[0][3]
    starts = np.array([int(p[1]) for p in parts]); ends = np.array([int(p[2]) for p in parts])
    # window: the shared upstream exon (ends 130,200,982), IIIb, IIIc, and the shared downstream exon
    # (starts 130,196,374), with margin, so every exon of the switch is drawn with its introns
    lo, hi = 130195800, 130201700
    tx = viz.get_transcripts(gffutils.FeatureDB(viz.DEFAULT_DB), chrom, lo, hi, strand, max_tx=6)
    ybot = viz.draw_exons(ax, tx, lo, hi) if tx else -0.7
    for s, e_, j in zip(starts, ends, jn):
        isb = j == J_IIIB
        arc(ax, s, e_, 1.4 if isb else 2.0, col["epithelial"] if isb else col["mesenchymal"], 1.8, "exon 7 $\\rightarrow$ IIIb (epithelial)" if isb else "exon 7 $\\rightarrow$ IIIc (mesenchymal)")
    for xx, nm in [(130200982 - 60, "exon 7"), ((130199905 + 130199757) / 2, "IIIb"), ((130198563 + 130198418) / 2, "IIIc"), (130196374 - 80, "exon 10")]:
        ax.text(xx, 0.45, nm, ha="center", va="bottom", fontsize=5.5, color="#333333")
    ax.set_xlim(lo, hi); ax.set_ylim(ybot - 0.4, 2.8); ax.set_yticks([]); ax.spines[["left", "top", "right"]].set_visible(False)
    ax.set_xlabel(f"{chrom} ({strand})", fontsize=6); ax.ticklabel_format(axis="x", style="plain", useOffset=False); ax.tick_params(axis="x", labelsize=5.5)
    ax.set_title("$Fgfr2$ ATSE 3: mutually exclusive exons IIIb / IIIc", fontsize=8, loc="left"); panel_label(ax, "a", dx=-0.02)

    # b: per cell type raw vs imputed (imputed = mean over the 3 seeds per cell; seed means as ticks)
    ax = fig.add_subplot(gs[1, :])
    for i, t in enumerate(types):
        s = df[(df.lab == t) & df.tr]; c = col[keep.loc[t, "lin"]]
        v = s.raw.dropna().values; rng = np.random.default_rng(i)
        vp = ax.violinplot([v], positions=[i - 0.28], widths=0.27, showextrema=False, points=40)
        for b in vp["bodies"]: b.set_facecolor(c); b.set_alpha(0.3); b.set_edgecolor("none")
        ax.scatter(i - 0.28 + rng.normal(0, 0.04, len(v)), v, s=2.5, color=c, alpha=0.6, lw=0)
        ax.hlines(v.mean(), i - 0.41, i - 0.15, color="black", lw=1.1, zorder=5)
        u = s.prior.dropna().values
        vp = ax.violinplot([u], positions=[i], widths=0.27, showextrema=False, points=40)
        for b in vp["bodies"]: b.set_facecolor("white"); b.set_alpha(1); b.set_edgecolor(c); b.set_linewidth(0.7); b.set_hatch("////")
        ax.hlines(u.mean(), i - 0.13, i + 0.13, color="black", lw=1.1, zorder=5)
        w = s.post.dropna().values
        vp = ax.violinplot([w], positions=[i + 0.28], widths=0.27, showextrema=False, points=40)
        for b in vp["bodies"]: b.set_facecolor(c); b.set_alpha(0.85); b.set_edgecolor("black"); b.set_linewidth(0.4)
        ax.hlines(w.mean(), i + 0.15, i + 0.41, color="black", lw=1.1, zorder=5)
        for k_ in range(3):
            ax.plot(i + 0.45, s[f"post_{k_}"].mean(), marker="_", ms=4, color="black", mew=0.8, zorder=6)
        ax.text(i, -0.12, f"{len(v)}", ha="center", fontsize=4.5, color=c)
        ax.text(i, -0.2, f"{len(w):,}", ha="center", fontsize=4.5, color="black")
    ax.text(-0.9, -0.12, "covered", ha="right", fontsize=4.5); ax.text(-0.9, -0.2, "all cells", ha="right", fontsize=4.5)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([t.replace("_", " ").replace("Non-Myeloid ", "").replace("Isocortex ", "").lower() for t in types], rotation=60, ha="right", fontsize=5)
    for lbl, t in zip(ax.get_xticklabels(), types): lbl.set_color(col[keep.loc[t, "lin"]])
    ax.set_ylim(-0.25, 1.08); ax.set_xlim(-1.2, len(types) - 0.4); ax.set_ylabel("IIIb usage per cell")
    from matplotlib.lines import Line2D
    h = [mpatches.Patch(facecolor=PALETTE["grey"], alpha=0.3, label="raw, covered cells (jitter + violin)"),
         mpatches.Patch(facecolor="white", edgecolor=PALETTE["grey"], hatch="////", label="LeafletFA prior $\\phi\\cdot\\psi$ (no reads used), all cells"),
         mpatches.Patch(facecolor=PALETTE["grey"], alpha=0.85, edgecolor="black", label="LeafletFA posterior mean (prior + own reads), all cells; ticks = each seed")]
    h += [mpatches.Patch(facecolor=col[k], label=k) for k in ["epithelial", "mesenchymal"]]
    ax.legend(handles=h, frameon=False, fontsize=4.8, ncol=3, loc="upper right", bbox_to_anchor=(1, 1.14))
    despine(ax); panel_label(ax, "b", dx=-0.02)

    # c: phi heatmap of the same cell types, one per seed, each clustered on its own phi, SP order per seed
    from scipy.cluster.hierarchy import linkage, leaves_list
    for k_, sd in enumerate(seeds):
        ax = fig.add_subplot(gs[2, k_])
        order = sd["order"]
        Mphi = np.vstack([np.nanmean(sd["PH"][(df.lab == t).values], 0) for t in types])[:, order]
        co = leaves_list(linkage(Mphi, "average", metric="cityblock")); ctypes = [types[i] for i in co]
        im = ax.imshow(Mphi[co].T, aspect="auto", cmap="Greens", vmin=0, vmax=0.8)
        ax.set_xticks(range(len(ctypes))); ax.set_xticklabels([t.replace("_", " ").replace("Non-Myeloid ", "").lower() for t in ctypes], rotation=70, ha="right", fontsize=4)
        for lbl, t in zip(ax.get_xticklabels(), ctypes): lbl.set_color(col[keep.loc[t, "lin"]])
        ax.set_yticks(range(K)); ax.set_yticklabels([f"SP{r+1} ($\\psi_{{IIIb}}$ {sd['psi_b'][order[r]]:.2f})" for r in range(K)], fontsize=4)
        if k_ == 0: ax.set_ylabel("program (SP1 = largest $\\pi$), with its IIIb usage")
        ax.set_title(f"seed {k_}: mean $\\phi$, columns clustered", fontsize=6.5, loc="left")
        if k_ == 0: panel_label(ax, "c", dx=-0.02)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("mean $\\phi$", fontsize=6); cb.ax.tick_params(labelsize=5)
    out = os.path.join(FIGROOT, "FIGURE2_simulation_imputation", "plots", "FIG2_supp_Fgfr2_IIIb_IIIc")
    save(fig, out); print("->", out)


if __name__ == "__main__":
    main()
