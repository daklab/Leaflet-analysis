#!/usr/bin/env python
"""Figure 3: differential splicing on the atlas -- pseudobulk vs single cells, and what phi adds.

The argument the figure makes, in the order the panels make it
--------------------------------------------------------------
LeafCutter on pseudobulked cells is the realistic best practice for Smart-seq2 DS today: aggregate
cells within a cell type into replicates, test with a Dirichlet-multinomial GLM. Rows 1-2 ask what
happens when the SAME test is instead run on every individual cell. Row 3 asks what the model adds
on top -- the denoised psi entering through the posterior effect-size test.

    a  calibration on each method's own null            pseudobulk vs single cells, same test
    b  calibration by event size m                      where pseudobulk breaks (11x at m=9)
    c  calls at MATCHED realised FDP                    a data-design result, said as such
    d  where the two disagree                           agreement rises with effect; split along m
    e  posterior DS: consensus across seeds             phi enters here, and only here
    f  posterior test vs LeafCutter on the SAME question  precise, not sensitive
    g  worked example: Fbln2                            program decomposition, endothelial vs fibroblast

The `--design phi` LRT is deliberately absent: it uses phi only as a covariate and refits every
composition from raw counts, so it hollows out the model's contribution. phi is shown through the
posterior test, which is the only arm that uses the denoised psi.
"""
import os, sys, glob, re
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
FIGROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, FIGROOT); sys.path.insert(0, os.path.join(FIGROOT, "_common"))
from leaflet_theme import apply_theme, PALETTE, METHOD_COLORS, figsize, panel_label, despine, save

M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
O = f"{M}/LEAFCUTTER_CMP_DELTAONLY"
LCD = f"{M}/LEAFCUTTER_DS_2026-08"
PLOTS = os.path.join(FIGROOT, "FIGURE3_leafcutter_ds", "plots")
LFA, LC, GREY, OLIVE = METHOD_COLORS["LeafletFA"], METHOD_COLORS["NMF"], PALETTE["grey"], PALETTE["olive"]
LAB_LFA = "LeafletFA (single cells)"
LAB_LC = "LeafCutter (pseudobulk)"


def _lc_null():
    fs = glob.glob(f"{LCD}/permutation_null/*/ds_results_cluster_significance.txt")
    t = pd.concat([pd.read_csv(f, sep="\t") for f in fs])
    t = t[t.status == "Success"].dropna(subset=["p"])
    return t


def panel_a(ax):
    """QQ of null p-values on the -log10 scale: over-calling sits ABOVE the diagonal."""
    lc = _lc_null(); lfa = pd.read_csv(f"{O}/d52/a52_matched_null.tsv", sep="\t")
    for p, c, lab in [(lc.p.values, LC, LAB_LC), (lfa.p.values, LFA, LAB_LFA)]:
        p = np.sort(p[np.isfinite(p) & (p > 0)]); n = len(p)
        exp = -np.log10((np.arange(1, n + 1) - 0.5) / n); obs = -np.log10(p)
        idx = np.unique(np.geomspace(1, n, 800).astype(int) - 1)
        ax.plot(exp[idx], obs[idx], "-", color=c, lw=1.5, label=lab)
    lim = [0, 5]
    ax.plot(lim, lim, ls=":", color=GREY, lw=0.8)
    for a in (0.05, 0.01):
        ax.axvline(-np.log10(a), color=GREY, lw=0.5, ls="--")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("expected $-\\log_{10} p$ under the null"); ax.set_ylabel("observed $-\\log_{10} p$")
    ax.legend(frameon=False, loc="upper left", fontsize=5.5)
    ax.annotate("above diagonal = over-calling", (0.97, 0.05), xycoords="axes fraction",
                fontsize=5.5, color=GREY, ha="right")


def panel_b(ax):
    """calibration ratio at alpha 0.05 by m."""
    lc = _lc_null(); lfa = pd.read_csv(f"{O}/d52/a52_matched_null.tsv", sep="\t")
    rows = []
    for m in range(2, 10):
        k1 = (lc.df.values == m - 1); k2 = (lfa.m.values == m)
        if k1.sum() < 200 or k2.sum() < 200: continue
        rows.append((m, (lc.p.values[k1] <= .05).mean() / .05, (lfa.p.values[k2] <= .05).mean() / .05))
    r = pd.DataFrame(rows, columns=["m", "lc", "lfa"])
    ax.plot(r.m, r.lc, "s--", color=LC, ms=3.5, lw=1.4, label=LAB_LC)
    ax.plot(r.m, r.lfa, "o-", color=LFA, ms=3.5, lw=1.4, label=LAB_LFA)
    ax.axhline(1, ls=":", color=GREY, lw=0.8)
    ax.set_yscale("log"); ax.set_xlabel("junctions per ATSE (m)")
    ax.set_ylabel("realised FDP / nominal\n(at $\\alpha$ = 0.05, own null)")
    ax.annotate("honest", (9.1, 1.0), fontsize=5.5, color=GREY, va="center")
    ax.legend(frameon=False, loc="upper left", fontsize=5.5)


def panel_c(ax):
    """calls at matched realised FDP, and the nominal counts greyed."""
    d = pd.read_csv(f"{O}/analysis_group/a50_contrast_comparison.tsv", sep="\t")
    x = np.arange(2); w = 0.36
    nom = [d.lc_nom.median(), d.lfa_nom.median()]
    cal = [d.lc_cal.median(), d.lfa_cal.median()]
    ax.bar(x - w/2, nom, w, color=GREY, alpha=0.45, label="nominal q$\\leq$0.05")
    ax.bar(x + w/2, cal, w, color=[LC, LFA], label="matched realised FDP 0.05")
    for i, (a, b) in enumerate(zip(nom, cal)):
        ax.annotate(f"{int(a):,}", (i - w/2, a), ha="center", va="bottom", fontsize=5.5, color=GREY)
        ax.annotate(f"{int(b):,}", (i + w/2, b), ha="center", va="bottom", fontsize=6,
                    color=[LC, LFA][i], fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["LeafCutter\n(7 v 8 pseudobulk)", "LeafletFA\n(all single cells)"], fontsize=6)
    ax.set_ylabel("ATSEs called per contrast\n(median over 70)")
    ax.legend(frameon=False, fontsize=5.5, loc="upper left")


def panel_d(ax):
    """Jaccard decomposition by LeafCutter's own measured effect."""
    d = pd.read_csv(f"{O}/analysis_group/jaccard_decomposition.tsv.gz", sep="\t")
    d = d[(d.a | d.b)].copy()
    d["cat"] = np.select([d.a & d.b, d.a & ~d.b], ["both", "LeafletFA only"], "LeafCutter only")
    d["bin"] = pd.cut(d.tv_lc, [0, .05, .10, .15, .20, .30, 1.0],
                      labels=["<0.05", "0.05-0.1", "0.1-0.15", "0.15-0.2", "0.2-0.3", ">0.3"])
    t = pd.crosstab(d["bin"], d["cat"], normalize="index")[["LeafCutter only", "both", "LeafletFA only"]]
    bottom = np.zeros(len(t)); x = np.arange(len(t))
    for col, c in [("LeafCutter only", LC), ("both", OLIVE), ("LeafletFA only", LFA)]:
        ax.bar(x, t[col].values, bottom=bottom, color=c, width=0.75, label=col)
        bottom += t[col].values
    ax.set_xticks(x); ax.set_xticklabels(t.index, fontsize=5.5, rotation=0)
    ax.set_xlabel("effect size measured by LeafCutter (TV)")
    ax.set_ylabel("share of called ATSEs")
    ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=5, loc="upper right", ncol=1)


def panel_e(ax):
    """posterior DS at delta 0.15: single-seed vs consensus, and rank consistency."""
    paths = {0: f"{O}/a51_delta0.15/a51_group_ds_posterior.tsv",
             1: f"{O}/a51_seed1_delta0.15/a51_group_ds_posterior.tsv",
             2: f"{O}/a51_seed2_delta0.15/a51_group_ds_posterior.tsv"}
    D = {s: pd.read_csv(p, sep="\t", usecols=["contrast", "cluster", "tv_post_mean", "q"]) for s, p in paths.items()}
    m = D[0].merge(D[1], on=["contrast", "cluster"], suffixes=("_0", "_1")).merge(
        D[2].rename(columns={"tv_post_mean": "tv_post_mean_2", "q": "q_2"}), on=["contrast", "cluster"])
    votes = (m.q_0 <= .05).astype(int) + (m.q_1 <= .05).astype(int) + (m.q_2 <= .05).astype(int)
    per = [int((m[f"q_{s}"] <= .05).sum()) for s in range(3)]
    cons2 = int((votes >= 2).sum()); cons3 = int((votes == 3).sum())
    x = np.arange(5)
    vals = per + [cons2, cons3]
    cols = [GREY, GREY, GREY, LFA, OLIVE]
    ax.bar(x, vals, color=cols, width=0.7)
    for i, v in enumerate(vals):
        ax.annotate(f"{v/1000:.1f}k", (i, v), ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks(x); ax.set_xticklabels(["seed 0", "seed 1", "seed 2", "$\\geq$2 of 3\n(shipped)", "all 3"], fontsize=5.5)
    ax.set_ylabel(f"ATSEs called, $P(TV>0.15)$, q$\\leq$0.05\n(sum over 70 contrasts)")
    r = np.median([m.tv_post_mean_0.corr(m.tv_post_mean_1, method="spearman"),
                   m.tv_post_mean_0.corr(m.tv_post_mean_2, method="spearman"),
                   m.tv_post_mean_1.corr(m.tv_post_mean_2, method="spearman")])
    ax.annotate(f"TV rank consistency\nacross seeds: $\\rho$ = {r:.2f}", (0.98, 0.95),
                xycoords="axes fraction", ha="right", va="top", fontsize=6, color=OLIVE)


def panel_f(ax):
    """posterior test vs LeafCutter on LeafCutter's own large-effect set."""
    d = pd.read_csv(f"{O}/analysis_matched/a63_matched_delta0.15.tsv", sep="\t")
    x = np.arange(2); w = 0.36
    ax.bar(x - w/2, [d.recall_lc.median(), d.prec_lc.median()], w, color=LC, label=LAB_LC)
    ax.bar(x + w/2, [d.recall_post.median(), d.prec_post.median()], w, color=LFA, label="LeafletFA posterior $P(TV>0.15)$")
    for i, (a, b) in enumerate(zip([d.recall_lc.median(), d.prec_lc.median()],
                                   [d.recall_post.median(), d.prec_post.median()])):
        ax.annotate(f"{a:.2f}", (i - w/2, a), ha="center", va="bottom", fontsize=5.5, color=LC)
        ax.annotate(f"{b:.2f}", (i + w/2, b), ha="center", va="bottom", fontsize=5.5, color=LFA)
    ax.set_xticks(x); ax.set_xticklabels(["recall of\nLeafCutter TV > 0.15", "precision vs\nLeafCutter TV > 0.15"], fontsize=5.5)
    ax.set_ylim(0, 1); ax.set_ylabel("on LeafCutter's own\nlarge-effect set")
    ax.legend(frameon=False, fontsize=5, loc="upper left")


def panel_g(ax):
    """Fbln2 worked example -- reuse the built panel as an image."""
    p = os.path.join(FIGROOT, "FIGURE3_leafcutter_ds",
                     "FIG3_supp_Fbln2_program_decomposition_K20_seed0_SHIPPED.png")
    ax.imshow(mpimg.imread(p)); ax.axis("off")


def main():
    apply_theme()
    fig = plt.figure(figsize=figsize("double", h=7.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.15])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]),
            fig.add_subplot(gs[2, :])]
    for ax, fn, L in zip(axes, [panel_a, panel_b, panel_c, panel_d, panel_e, panel_f, panel_g], "abcdefg"):
        try:
            fn(ax)
        except Exception as e:
            ax.text(.5, .5, f"{L}: {type(e).__name__}\n{e}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=5, color=GREY, wrap=True)
            print(f"  panel {L} FAILED: {type(e).__name__}: {e}", flush=True)
        if L != "g": despine(ax)
        panel_label(ax, L)
    fig.tight_layout(w_pad=1.6, h_pad=2.0)
    os.makedirs(PLOTS, exist_ok=True)
    save(fig, os.path.join(PLOTS, "FIG3_draft"))
    print(f"-> {PLOTS}/FIG3_draft.[pdf|png]")


if __name__ == "__main__":
    main()
