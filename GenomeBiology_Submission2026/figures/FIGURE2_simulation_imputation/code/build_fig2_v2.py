#!/usr/bin/env python
"""Figure 2 v2 -- the story in three rows, one idea per panel.

Row 1  SETUP: what was simulated.
  a  cells (true phi) in UMAP, coloured by true program -- 6,986 heart cells, K=5, real coverage
  b  one cassette exon simulated DIFFERENTIAL: true skip-PSI per program, and the observed per-cell
     skip PSI of the cells grouped by their program (violins) -- what the model has to find
  c  the same for one NON-differential exon -- what it must not call
Row 2  RECOVERY: does the model get the programs back, without imputation?
  d  learned phi (LeafletFA) in UMAP, coloured by TRUE program, silhouette on the latent space
  e  NMF at the same rank (mean-impute, its best gauge), same colouring
  f  recovery of the between-program effect (Delta psi) vs truth, LeafletFA against NMF, by effect
Row 3  TEST + IMPUTATION: does it find the right ATSEs, calibrated, and impute better than NMF?
  g  DS calibration (realised FDP / nominal) by simulated effect: LeafletFA LRT, LeafletFA posterior
     P(TV > 0.15), LeafCutter on the same simulations and truth
  h  DS power beside it (never alone)
  i  held-out imputation on the atlas: Pearson vs % reads dropped, DM posterior mean vs NMF -- the
     one panel not from simulations (A3; PLACEHOLDER until the sweep lands)

Everything else from FIG2_final (concentration recovery, psi vs depth, delta trade-off, atlas
reach, conc-2 regime, phi by regime) moves to the supplement: build_fig2_supp.py.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, FIGROOT); sys.path.insert(0, os.path.join(FIGROOT, "_common"))
from leaflet_theme import apply_theme, PALETTE, METHOD_COLORS, figsize, panel_label, despine, save

W = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD"
M = f"{W}/MOUSE_SPLICING_FOUNDATION"
DATA = os.path.join(FIGROOT, "FIGURE2_simulation_imputation", "data")
PLOTS = os.path.join(FIGROOT, "FIGURE2_simulation_imputation", "plots")
LFA, LC, GREY = METHOD_COLORS["LeafletFA"], METHOD_COLORS["NMF"], PALETTE["grey"]
PROG_COLS = [PALETTE["teal"], PALETTE["indigo"], PALETTE["ochre"], PALETTE["rose"], PALETTE["olive"]]
P = np.load(os.path.join(DATA, "fig2_v2_prep.npz"), allow_pickle=True)


def _umap(ax, emb, prog, title, sil=None):
    rng = np.random.default_rng(0); o = rng.permutation(len(prog))
    for k in range(5):
        m = prog[o] == k
        ax.scatter(emb[o][m, 0], emb[o][m, 1], s=1.2, color=PROG_COLS[k], lw=0, alpha=0.7, label=f"program {k+1}")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(title, fontsize=7, loc="left")
    if sil is not None:
        ax.annotate(f"silhouette {sil:.2f}", (0.98, 0.02), xycoords="axes fraction", ha="right", fontsize=6, color=GREY)


def panel_a(ax):
    _umap(ax, P["emb_true"], P["prog"], f"simulated cells: true $\\phi$\n({int(P['n_cells']):,} heart cells, K = 5 programs, {int(P['n_atse']):,} cassette-exon ATSEs)")
    ax.legend(frameon=False, fontsize=5, markerscale=6, loc="upper right", handletextpad=0.2)


def _example(ax, tag, title):
    tp = P[f"{tag}_true"]; obs = P[f"{tag}_obs"]; n = P[f"{tag}_n"]; prog = P["prog"]
    for k in range(5):
        m = (prog == k) & np.isfinite(obs)
        v = obs[m]
        if len(v) >= 5:
            vp = ax.violinplot([v], positions=[k], widths=0.8, showextrema=False, points=50)
            for b in vp["bodies"]: b.set_facecolor(PROG_COLS[k]); b.set_alpha(0.35); b.set_edgecolor("none")
            rng = np.random.default_rng(k); sel = v if len(v) <= 150 else rng.choice(v, 150, replace=False)
            ax.scatter(k + rng.normal(0, 0.07, len(sel)), sel, s=1.5, color=PROG_COLS[k], alpha=0.5, lw=0)
            ax.text(k, -0.13, f"n={m.sum()}", ha="center", fontsize=5, color=PROG_COLS[k])
        ax.hlines(tp[k], k - 0.4, k + 0.4, color="black", lw=1.4, zorder=5)
    ax.plot([], [], color="black", lw=1.4, label="true skip-PSI of program")
    ax.set_xticks(range(5)); ax.set_xticklabels([f"P{k+1}" for k in range(5)], fontsize=6)
    ax.set_ylim(-0.2, 1.05); ax.set_ylabel("skip-junction usage\n(observed per cell)")
    ax.set_title(title, fontsize=7, loc="left")
    ax.legend(frameon=False, fontsize=5, loc="upper right")


def panel_b(ax):
    _example(ax, "ds", f"a DIFFERENTIAL ATSE (30% of ATSEs)\nmax between-program $\\Delta\\psi$ = {float(P['ds_delta']):.2f}")


def panel_c(ax):
    _example(ax, "null", "a NON-differential ATSE (70%)\nsame usage in every program")


def panel_d(ax):
    _umap(ax, P["emb_lfa"], P["prog"], "learned $\\phi$, LeafletFA fit at K = 5\n(coloured by TRUE program)", float(P["sil_lfa"]))


def panel_e(ax):
    _umap(ax, P["emb_nmf"], P["prog"], "NMF at the same rank\n(mean-imputed input)", float(P["sil_nmf"]))


def panel_f(ax):
    """Detecting WHICH ATSEs are differential across programs, label-free: score each ATSE by
    max_k psi_hat - min_k psi_hat over the learned programs, truth = simulated is_differential.
    AUROC (solid) and AUPRC (dashed); NMF at its best gauge per effect."""
    d = pd.read_csv(f"{DATA}/dm_vs_nmf.tsv", sep="\t")
    d["fam"] = np.where(d.model.str.startswith("LeafletFA"), "LeafletFA", "NMF")
    for metric, ls, lab in [("auroc", "-", "AUROC"), ("ap", "--", "AUPRC")]:
        g = d.groupby(["fam", "model", "effect"])[metric].mean().reset_index()
        lfa = d[d.fam == "LeafletFA"].groupby("effect")[metric].agg(["mean", "std"])   # 3 model seeds
        nmf = g[g.fam == "NMF"].groupby("effect")[metric].max()                       # deterministic
        ax.errorbar(lfa.index, lfa["mean"], yerr=lfa["std"].fillna(0), fmt="o" + ls, color=LFA, ms=3.2, lw=1.4, capsize=2, label=f"LeafletFA {lab}")
        ax.plot(nmf.index, nmf.values, "s" + ls, color=LC, ms=3.2, lw=1.4, label=f"NMF {lab}")
    ax.axhline(0.5, color=GREY, lw=0.6, ls=":"); ax.annotate("AUROC chance", (0.1, 0.5), fontsize=5, color=GREY, va="bottom")
    ax.axhline(0.3, color=GREY, lw=0.6, ls=":"); ax.annotate("AUPRC chance (30% differential)", (0.1, 0.3), fontsize=5, color=GREY, va="bottom")
    ax.set_xscale("log"); ax.set_xticks([0.1, 0.25, 0.5, 1.0]); ax.set_xticklabels(["0.1", "0.25", "0.5", "1.0"])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("simulated effect size"); ax.set_ylabel("differential-ATSE detection\nfrom learned $\\psi$ (no labels)")
    ax.set_title("which ATSEs differ between programs (K = 5; mean $\\pm$ SD, 3 seeds)", fontsize=7, loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=5, ncol=2)


def _ds():
    lrt = pd.read_csv(f"{M}/DM_SHIPPED_SIM/ds/summary_lrt.tsv", sep="\t"); lrt = lrt[lrt.alpha == 0.05]
    lc = pd.read_csv(f"{M}/LC_SIM_DM/summary_leafcutter_dm.tsv", sep="\t"); lc = lc[lc.alpha == 0.05]
    po = pd.read_csv(f"{M}/DM_SHIPPED_SIM/ds/delta_sweep_full.tsv", sep="\t")
    # delta = 0.15 is only achievable where the simulated effect exceeds it (at effect 0.15 the largest
    # true delta anywhere is 0.147), so the posterior series starts at 0.25
    po = po[(po.delta == 0.15) & (po.eff >= 0.25)]
    return lrt, lc, po


def _ds_panel(ax, col, ylabel, logy):
    lrt, lc, po = _ds()
    for d, c, mk, ls, lab in [(lrt, LFA, "o", "-", "LeafletFA, DM LRT (single cells)"),
                              (po, PALETTE["olive"], "^", "-", "LeafletFA, posterior $P(TV>0.15)$"),
                              (lc, LC, "s", "--", "LeafCutter (pseudobulk)")]:
        g = d.groupby("eff")[col].agg(["mean", "std"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), fmt=mk + ls, color=c, ms=3.2, lw=1.3, capsize=2, label=lab, alpha=0.9)
    ax.set_xscale("log"); ax.set_xticks([0.1, 0.2, 0.35, 0.5, 1.0]); ax.set_xticklabels(["0.1", "0.2", "0.35", "0.5", "1.0"])
    ax.set_xlabel("simulated effect size"); ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log"); ax.axhline(1.0, color=GREY, lw=0.8, ls=":", zorder=0)
        ax.annotate("calibrated", (1.02, 1.0), xycoords=("axes fraction", "data"), fontsize=6, color=GREY, va="center")
    else:
        ax.set_ylim(0, 1.02)


def panel_g(ax):
    _ds_panel(ax, "ratio", "realised FDP / nominal\n(at $\\alpha$ = 0.05)", True)
    ax.set_title("does it over- or under-call?  (mean $\\pm$ SD, 3 seeds $\\times$ contrasts)", fontsize=7, loc="left")
    ax.legend(frameon=False, fontsize=5, loc="upper right")


def panel_h(ax):
    _ds_panel(ax, "power", "power (at $\\alpha$ = 0.05)", False)
    ax.set_title("does it find the differential ATSEs?", fontsize=7, loc="left")


def panel_i(ax):
    """held-out imputation on the atlas (A3, 70/30 split, 3 seeds where available).
    Metric = median within-junction Pearson across held-out cells (does the method rank the cells of a
    junction?), on entries whose reads were ALL removed -- the imputation task proper. Inset: the same
    on all covered entries, where methods that keep the cell's own reads (LeafletFA posterior mean,
    MeanBayes) are far above NMF and near each other. Cell-type-mean baseline dropped (uses labels)."""
    import glob
    fs = sorted(glob.glob(f"{M}/IMPUTATION_DM/results/a3_heldout70_seed*.csv"))
    d = pd.concat([pd.read_csv(f).assign(seed=i) for i, f in enumerate(fs)]); d = d[d.dataset == "all"]
    meth = [("lf", "LeafletFA (prior $\\phi\\cdot\\psi$)", LFA, "o", "-"), ("nmfm", "NMF, mean-imputed", LC, "s", "--"),
            ("nmf0", "NMF, zero-imputed", PALETTE["indigo"], "s", ":"), ("mb", "junction mean", GREY, "^", "-")]
    L = d[d.entry_set == "lost_to_downsampling"]
    for k, lab, c, mk, ls in meth:
        g = L.groupby("downsampling_pct")[f"median_pearson_{k}"].agg(["mean", "std"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), fmt=mk + ls, color=c, ms=3, lw=1.3, capsize=2, label=lab)
    ax.set_xlabel("% of reads removed from held-out cells"); ax.set_ylabel("within-junction Pearson\n(entries with NO reads left)")
    ax.set_ylim(-0.005, 0.12); ax.set_xticks([10, 25, 50, 75, 90, 99, 100])
    ax.set_title(f"held-out imputation, atlas 70/30 (K = 20, {len(fs)} seed{'s' if len(fs) > 1 else ''})", fontsize=7, loc="left")
    ax.legend(frameon=False, fontsize=5, loc="upper right")
    ins = ax.inset_axes([0.42, 0.42, 0.55, 0.42])
    A = d[d.entry_set == "all_covered"]
    for k, lab, c, mk, ls in [("lf_bb", "LeafletFA posterior mean", LFA, "o", "-"), ("mb", "junction mean + own reads", GREY, "^", "-"), ("nmfm", "NMF mean-imputed", LC, "s", "--")]:
        g = A.groupby("downsampling_pct")[f"median_pearson_{k}"].agg(["mean", "std"])
        ins.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), fmt=mk + ls, color=c, ms=2, lw=1, capsize=1.5, label=lab)
    ins.set_ylim(0, 1); ins.set_xticks([10, 50, 90, 100]); ins.tick_params(labelsize=4.5)
    ins.set_title("all covered entries\n(cell's remaining reads used)", fontsize=4.8); ins.legend(frameon=False, fontsize=4, loc="lower left")
    despine(ins)


def main():
    apply_theme()
    fig, axes = plt.subplots(3, 3, figsize=figsize("double", h=8.2))
    for ax, fn, letter in zip(axes.ravel(), [panel_a, panel_b, panel_c, panel_d, panel_e, panel_f, panel_g, panel_h, panel_i], "abcdefghi"):
        try:
            fn(ax)
        except Exception as e:
            ax.text(0.5, 0.5, f"{letter}: {type(e).__name__}\n{e}", ha="center", va="center", transform=ax.transAxes, fontsize=5, color=GREY, wrap=True)
            print(f"  panel {letter} FAILED: {type(e).__name__}: {e}", flush=True)
        despine(ax); panel_label(ax, letter)
    fig.tight_layout(w_pad=1.4, h_pad=1.8)
    save(fig, os.path.join(PLOTS, "FIG2_v2")); print(f"-> {PLOTS}/FIG2_v2.[pdf|png]")


if __name__ == "__main__":
    main()
