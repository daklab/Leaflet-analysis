"""Candidate figures from the validation checks (Karin 2026-11-16 overnight list, item 3). Candidates only: the current
figure AN and its panel b are unchanged. Built by figures/build_validation_figures.py (no change to make_figures.py or
paper_config.yaml). Every number is read from the validation outputs under $V; nothing is computed here beyond selecting rows.

  ANb17_b   85 x 50 mm, figure AN panel b candidate: check 17 (pairwise TMS), excess confirmation by read bin.
  ANb18_b   85 x 50 mm, the same for check 18a (refit on half the mice, confirmed only in unseen mice).
  Sds_a     183 x 120 mm, supplementary: checks 16, 16b (designs 1 and 2), 17, 18a and 18b, LeafletFA - comparator excess by bin.
  Ssim_a    183 x 112 mm, supplementary: simulations, checks 15 (amendment 10) and 15b (amendment 11).
  Sscale_a  183 x 58 mm, supplementary: scaling runs (time and memory against cells).

Colors (palette tokens; Karin's reserved table): LeafletFA teal; raw pseudobulk indigo; LeafCutter ochre; shrinkage
comparator B olive; scQuint olive in the scaling figure (no B there); reference lines gray dashed."""
import os

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from style import axes_in, container_mm, sub_slot

V = "/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation"
BINS = ["1-19", "20-99", "100-499", ">=500"]
BIN_LAB = ["1–19", "20–99", "100–499", "≥ 500"]
DEPTHS = [("full", "Full"), ("f30", "30 %"), ("f10", "10 %")]
MS = 2.8


def _S(cfg):
    p = cfg["palette"]
    return dict(sm=cfg["style"]["font"]["small_pt"], fs=cfg["style"]["font"]["size_pt"], ink=p["ink"], gray=p["gray"],
                light=p["light_gray"], lfa=p["teal"], raw=p["indigo"], lc=p["ochre"], B=p["olive"], scq=p["olive"],
                ref=cfg["style"]["line"]["reference_pt"])


def _letter(ax, s, S, dx_mm=1.0):
    """bold lowercase panel letter at the axes' top-left, dx_mm to the left of the axes edge"""
    W = ax.figure.get_size_inches()[0] * 25.4; aw = ax.get_position().width * W
    ax.text(-dx_mm / aw, 1.0, s, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, fontweight="bold", color=S["ink"])


def _ci(ax, x, y, lo, hi, c, marker, filled=True):
    ax.vlines(x, lo, hi, color=c, lw=0.8, zorder=3)
    ax.plot(x, y, ls="none", marker=marker, ms=MS, mfc=c if filled else "white", mec=c, mew=0.7, zorder=4)


# ---------------------------------------------------------------- data
def c17_table(check="17"):
    """check 17 ($V/c17/summary.tsv, column pairs) or check 18a ($V/c18/18a_summary.tsv, same layout, column units)"""
    path = os.path.join(V, "c17", "summary.tsv") if check == "17" else os.path.join(V, "c18", "18a_summary.tsv")
    s = pd.read_csv(path, sep="\t").rename(columns={"units": "pairs"})
    s = s[s.scope == "all"]
    rows = []
    for b in BINS:
        r = s[(s.bin == b) & (s.comparison == "LeafletFA vs raw_pseudobulk")]
        for m in ("LeafletFA", "raw_pseudobulk"):
            x = r[r.method == m].iloc[0]
            rows.append(dict(bin=b, method=m, pairs=int(x.pairs), calls=int(x.calls), excess=x.excess, lo=x.excess_lo, hi=x.excess_hi,
                             source=f"{os.path.basename(path)}, LeafletFA vs raw_pseudobulk"))
        r = s[(s.bin == b) & (s.comparison == "LeafletFA vs LeafCutter")]
        for m, lab in (("LeafletFA", "LeafletFA_subset"), ("LeafCutter", "LeafCutter")):
            x = r[r.method == m]
            if len(x):
                x = x.iloc[0]
                rows.append(dict(bin=b, method=lab, pairs=int(x.pairs), calls=int(x.calls), excess=x.excess, lo=x.excess_lo,
                                 hi=x.excess_hi, source=f"{os.path.basename(path)}, LeafletFA vs LeafCutter (50-pair subset)"))
    return pd.DataFrame(rows)


def ds_tables():
    """LeafletFA - comparator excess per design, bin and comparator (compared rows only)."""
    out = []
    d = pd.read_csv(os.path.join(V, "c16", "differences.tsv"), sep="\t")
    d = d[(d.design == "primary") & d.comparator.isin(["raw_pseudobulk", "LeafCutter"]) & d.both_compared]
    out.append(d.assign(check="16", units=np.nan)[["check", "bin", "comparator", "units", "diff", "diff_lo", "diff_hi"]])
    for k in ("1", "2"):
        d = pd.read_csv(os.path.join(V, "c16b", f"design{k}_differences.tsv"), sep="\t")
        d = d[d.types > 0]
        out.append(d.assign(check=f"16b_d{k}", units=d.types)[["check", "bin", "comparator", "units", "diff", "diff_lo", "diff_hi"]])
    d = pd.read_csv(os.path.join(V, "c17", "differences.tsv"), sep="\t")
    d = d[(d.scope == "all") & d.compared]
    out.append(d.assign(check="17", units=d.pairs)[["check", "bin", "comparator", "units", "diff", "diff_lo", "diff_hi"]])
    for k in ("18a", "18b"):
        d = pd.read_csv(os.path.join(V, "c18", f"{k}_differences.tsv"), sep="\t")
        d = d[(d.scope == "all") & d.compared]
        out.append(d.assign(check=k)[["check", "bin", "comparator", "units", "diff", "diff_lo", "diff_hi"]])
    return pd.concat(out, ignore_index=True)


def sim_tables():
    m1, m2, lc = [], [], []
    for chk, base in (("15", os.path.join(V, "c15", "score")), ("15b", None)):
        for dp, _ in DEPTHS:
            p1 = os.path.join(base, f"{dp}_m1.tsv") if base else os.path.join(V, "c15b", "acc", "score", f"{dp}_m1.tsv")
            t = pd.read_csv(p1, sep="\t", dtype={"bin": str}); m1.append(t.assign(check=chk))
            p2 = os.path.join(base, f"{dp}_m2.tsv") if base else os.path.join(V, "c15b", "diff", "score", f"{dp}_m2.tsv")
            m2.append(pd.read_csv(p2, sep="\t", dtype={"bin": str}).assign(check=chk))
            p3 = os.path.join(base, f"{dp}_m2_lc.tsv") if base else os.path.join(V, "c15b", "diff", "score", f"{dp}_m2_lc.tsv")
            lc.append(pd.read_csv(p3, sep="\t", dtype={"bin": str}).assign(check=chk))
    return pd.concat(m1, ignore_index=True), pd.concat(m2, ignore_index=True), pd.concat(lc, ignore_index=True)


def scaling_table():
    return pd.read_csv(os.path.join(V, "scaling", "scaling.tsv"), sep="\t")


# ---------------------------------------------------------------- ANb17_b
def panel_b17(cfg, fig, check="17"):
    """Check 17 (amendment 13), TMS cell-type pairs: excess confirmation (confirmed in half-B TEST cells minus the
    within-mouse shuffled rate) of each method's top 5 calls per pair and read bin, 95 % pair-bootstrap intervals.
    LeafletFA and raw on all sampled pairs (n under the bins); LeafCutter on the 50-pair subset (41 and 48 pairs), none
    testable below 100 reads. Input: $V/c17/summary.tsv."""
    w, h = container_mm(fig); S = _S(cfg); T = c17_table(check); leg_h = 10.0
    ax = axes_in(sub_slot(fig, 0, 0, w, h - leg_h), (13.0, 11.0, 0.8, 1.5))
    off = {"LeafletFA": -0.27, "raw_pseudobulk": -0.09, "LeafletFA_subset": 0.09, "LeafCutter": 0.27}
    sty = {"LeafletFA": (S["lfa"], "o"), "raw_pseudobulk": (S["raw"], "s"), "LeafletFA_subset": (S["lfa"], "o", False),
           "LeafCutter": (S["lc"], "D")}
    for i, b in enumerate(BINS):
        for m in off:
            r = T[(T.bin == b) & (T.method == m)]
            if len(r):
                r = r.iloc[0]; _ci(ax, i + off[m], r.excess, r.lo, r.hi, *sty[m])
            elif m == "LeafCutter":
                ax.text(i + 0.18, 0.012, "n.t.", ha="center", va="bottom", fontsize=S["sm"] - 1, color=S["gray"])
    ax.axhline(0, color=S["gray"], lw=S["ref"], ls=(0, (3, 2)), zorder=0)
    def npairs(b):
        a = T[(T.bin == b) & (T.method == "LeafletFA")].pairs.iloc[0]; c = T[(T.bin == b) & (T.method == "LeafCutter")]
        return f"({a:,} / {c.pairs.iloc[0]})" if len(c) else f"({a:,})"
    ax.set_xticks(range(len(BINS))); ax.set_xticklabels([f"{l}\n{npairs(b)}" for l, b in zip(BIN_LAB, BINS)])
    ax.set_xlim(-0.5, len(BINS) - 0.5)
    ax.set_ylim(0, max(T.hi.max() * 1.08, 0.3))
    ax.set_ylabel("Confirmed in held-out\nmice, above shuffled")
    ax.set_xlabel("Reads at the event, less-covered type (pairs: all / subset)", labelpad=1.5)
    lx = axes_in(sub_slot(fig, 0, h - leg_h, w, leg_h), (13.0, 0.2, 0.5, 0.2)); lx.axis("off")
    hs = [Line2D([], [], ls="none", marker="o", ms=MS, mfc=S["lfa"], mec=S["lfa"], label="LeafletFA"),
          Line2D([], [], ls="none", marker="s", ms=MS, mfc=S["raw"], mec=S["raw"], label="Raw pseudobulk"),
          Line2D([], [], ls="none", marker="o", ms=MS, mfc="white", mec=S["lfa"], mew=0.7, label="LeafletFA, same 50-pair subset"),
          Line2D([], [], ls="none", marker="D", ms=MS, mfc=S["lc"], mec=S["lc"], label="LeafCutter, 50-pair subset")]
    lx.legend(handles=hs, loc="upper left", ncol=2, frameon=False, fontsize=S["sm"], handlelength=1.0, columnspacing=0.7,
              borderpad=0.0, borderaxespad=0)
    lx.text(1.0, 0.0, "n.t. = no pair with 5 events LeafCutter could test", transform=lx.transAxes, ha="right", va="bottom",
            fontsize=S["sm"] - 1, color=S["gray"])


# ---------------------------------------------------------------- Sds_a
DS_FACETS = [("16", "Check 16: type vs rest,\ntop 500 per bin", "FAIL"),
             ("16b_d1", "Check 16b design 1: 5 calls\nper type, same types", "BETTER"),
             ("16b_d2", "Check 16b design 2: types\nwhere all methods call", "FAIL"),
             ("17", "Check 17: cell-type pairs,\n5 calls per pair", "BETTER"),
             ("18a", "Check 18a: pairs; refit on half the mice,\nconfirmed only in unseen mice", "BETTER"),
             ("18b", "Check 18b: type vs rest; refit on half the\nmice, confirmed only in unseen mice", "BETTER")]


def panel_ds(cfg, fig):
    """Checks 16, 16b, 17 and 18 (TMS): LeafletFA minus comparator excess confirmation by read bin, 95 % bootstrap
    intervals (over types in 16/16b/18b, over pairs in 17/18a), with the pre-registered verdict of each design. 16-17:
    calls on half-A TRAIN cells of the production fit, confirmation in half-B TEST cells. 18: LeafletFA refit on half-A
    mice only, confirmation in all cells of half-B mice. Dashed line 0; dotted line the -0.05 non-inferiority margin.
    LeafCutter appears only where it was compared. Inputs: $V/c16/differences.tsv, $V/c16b/design{1,2}_differences.tsv,
    $V/c17/differences.tsv, $V/c18/18{a,b}_differences.tsv."""
    w, h = container_mm(fig); S = _S(cfg); T = ds_tables(); leg_h = 6.0; xlab_h = 4.0
    lo_, hi_ = min(T.diff_lo.min(), -0.1) - 0.02, T.diff_hi.max() + 0.08
    wl = 12.5; nc = 3; wf = (w - wl - 1.0) / nc; rowh = (h - leg_h - xlab_h) / 2
    for f, (key, title, verdict) in enumerate(DS_FACETS):
        r_, c_ = divmod(f, nc)
        x0 = 0 if c_ == 0 else wl + c_ * wf; y0 = xlab_h + (1 - r_) * rowh
        ax = axes_in(sub_slot(fig, x0, y0, (wl + wf) if c_ == 0 else wf, rowh), ((wl if c_ == 0 else 2.5), 7.5, 0.5, 9.0))
        t = T[T.check == key]
        for i, b in enumerate(BINS):
            for comp, dx, c, mk in (("raw_pseudobulk", -0.13, S["raw"], "s"), ("LeafCutter", 0.13, S["lc"], "D")):
                r = t[(t.bin == b) & (t.comparator == comp)]
                if len(r):
                    r = r.iloc[0]; _ci(ax, i + dx, r["diff"], r.diff_lo, r.diff_hi, c, mk)
        ax.axhline(0, color=S["gray"], lw=S["ref"], ls=(0, (3, 2)), zorder=0)
        ax.axhline(-0.05, color=S["gray"], lw=S["ref"], ls=(0, (1, 1.5)), zorder=0)
        ax.set_xticks(range(len(BINS))); ax.set_xticklabels(BIN_LAB)
        ax.set_xlim(-0.5, len(BINS) - 0.5); ax.set_ylim(lo_, hi_)
        ax.set_title(title, fontsize=S["sm"], loc="left", pad=8.0, color=S["ink"], linespacing=1.0)
        ax.text(0.02, 0.98, f"Pre-registered: {verdict}", transform=ax.transAxes, ha="left", va="top", fontsize=S["sm"],
                color=S["ink"], fontweight="bold" if verdict == "FAIL" else "normal")
        _letter(ax, "abcdef"[f], S, dx_mm=1.0 if c_ else 10.5)
        if c_ == 0:
            ax.set_ylabel("LeafletFA minus comparator\n(excess confirmation)")
        else:
            ax.set_yticklabels([])
    xl = axes_in(sub_slot(fig, wl, 0, w - wl, xlab_h), (0, 0, 0, 0)); xl.axis("off")
    xl.text(0.5, 0.0, "Reads at the event in the calling half (pairs: less-covered type)", ha="center", va="bottom",
            fontsize=S["fs"], color=S["ink"], transform=xl.transAxes)
    lx = axes_in(sub_slot(fig, 0, h - leg_h, w, leg_h), (wl, 0.2, 0.5, 0.2)); lx.axis("off")
    hs = [Line2D([], [], ls="none", marker="s", ms=MS, mfc=S["raw"], mec=S["raw"], label="vs raw pseudobulk"),
          Line2D([], [], ls="none", marker="D", ms=MS, mfc=S["lc"], mec=S["lc"], label="vs LeafCutter"),
          Line2D([], [], color=S["gray"], lw=S["ref"], ls=(0, (1, 1.5)), label="−0.05 margin")]
    lx.legend(handles=hs, loc="center left", ncol=3, frameon=False, fontsize=S["sm"], handlelength=1.4, columnspacing=1.0,
              borderpad=0.0, borderaxespad=0)
    lx.text(1.0, 0.5, "16b combined reading: at least partly explained by concentration", transform=lx.transAxes,
            ha="right", va="center", fontsize=S["sm"], color=S["ink"])


# ---------------------------------------------------------------- Ssim_a
def panel_sim(cfg, fig):
    """Simulations. Top: cell-type psi accuracy (mean TV error against the simulated truth, lower is better) for
    LeafletFA, raw pseudobulk and shrinkage comparator B: own-truth pairs by depth in check 15 and 15b, and all pairs by
    TRAIN reads at full depth in 15b. Bottom: planted vs null ranking (pooled AUROC, 95 % type-bootstrap interval for
    LeafletFA and LeafCutter): LeafletFA and raw by depth in 15 and 15b, and LeafletFA vs LeafCutter on the units both
    score in 15b. Labels give the report's readings. Inputs: $V/c15/score/<depth>_m1/_m2/_m2_lc.tsv,
    $V/c15b/{acc,diff}/score/<depth>_m1/_m2/_m2_lc.tsv."""
    w, h = container_mm(fig); S = _S(cfg); M1, M2, LC = sim_tables(); leg_h = 6.0
    rowh = (h - leg_h) / 2; wl = 12.5; wf = (w - wl - 1.0) / 3
    top_lab = [("15", "Check 15: own-truth pairs", "Descriptive, no pass rule;\nshift design confounded"),
               ("15b", "Check 15b: own-truth pairs", "Worse than raw and B\nat every depth"),
               ("15b_reads", "Check 15b: all pairs, full depth", "Better than raw at 1–99 reads;\nB better overall")]
    acc_max = 0
    axes_top = []
    for f, (key, title, note) in enumerate(top_lab):
        x0 = 0 if f == 0 else wl + f * wf
        ax = axes_in(sub_slot(fig, x0, rowh, (wl + wf) if f == 0 else wf, rowh), ((wl if f == 0 else 3.5), 9.5, 0.5, 7.5))
        if key == "15b_reads":
            t = M1[(M1.check == "15b") & (M1.depth == "full") & (M1.pairs == "all") & (M1.by == "reads")].set_index("bin")
            xs = ["0", "1-19", "20-99", ">=100"]; xl = ["0", "1–19", "20–99", "≥ 100"]
        else:
            t = M1[(M1.check == key) & (M1.pairs == "own") & (M1.by == "all")].set_index("depth")
            xs = [d for d, _ in DEPTHS]; xl = [l for _, l in DEPTHS]
        for col, c, mk, lab in (("err_leafletfa", S["lfa"], "o", "LeafletFA"), ("err_raw", S["raw"], "s", "Raw"), ("err_B", S["B"], "^", "B")):
            y = [t.loc[x, col] for x in xs]
            ax.plot(range(len(xs)), y, color=c, lw=0.8, marker=mk, ms=MS, mfc=c, mec=c, zorder=3)
            acc_max = max(acc_max, max(y))
        ax.set_xticks(range(len(xs))); ax.set_xticklabels(xl); ax.set_xlim(-0.4, len(xs) - 0.6)
        ax.set_title(title, fontsize=S["sm"], loc="left", pad=1.5, color=S["ink"])
        ax.text(0.98, 0.04, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=S["sm"] - 1, color=S["ink"], linespacing=1.0)
        ax.set_xlabel("TRAIN reads at the event" if key == "15b_reads" else "Sequencing depth", labelpad=1.5)
        _letter(ax, "abc"[f], S, dx_mm=1.0 if f else 10.5)
        if f == 0:
            ax.set_ylabel("Error against truth\n(mean TV)")
        axes_top.append(ax)
    for ax in axes_top:
        ax.set_ylim(0, acc_max * 1.35)
    for ax in axes_top[1:]:
        ax.set_yticklabels([])
    bot = [("15", "Check 15: planted vs null", "Inflated: one program tracks\nthe shifted half (AUROC 1.00)"),
           ("15b", "Check 15b: planted vs null", "Half check partial (0.66–0.71)"),
           ("15b_lc", "Check 15b: units LeafCutter tests", "LeafCutter ranks better\non shared units")]
    for f, (key, title, note) in enumerate(bot):
        x0 = 0 if f == 0 else wl + f * wf
        ax = axes_in(sub_slot(fig, x0, 0, (wl + wf) if f == 0 else wf, rowh), ((wl if f == 0 else 3.5), 9.5, 0.5, 7.5))
        xs = [d for d, _ in DEPTHS]
        if key == "15b_lc":
            t = LC[(LC.check == "15b") & (LC.by == "all")].set_index("depth")
            series = (("auroc_lfa", S["lfa"], "o"), ("auroc_lc", S["lc"], "D"))
        else:
            t = M2[(M2.check == key) & (M2.by == "all")].set_index("depth")
            series = (("auroc_lfa", S["lfa"], "o"), ("auroc_raw", S["raw"], "s"))
        for k, (col, c, mk) in enumerate(series):
            dx = -0.08 if k == 0 else 0.08
            for i, x in enumerate(xs):
                r = t.loc[x]
                lo, hi = (r[col + "_lo"], r[col + "_hi"]) if col + "_lo" in t.columns else (r[col], r[col])
                _ci(ax, i + dx, r[col], lo, hi, c, mk)
            ax.plot(np.arange(len(xs)) + dx, [t.loc[x, col] for x in xs], color=c, lw=0.6, zorder=2)
        ax.axhline(0.5, color=S["gray"], lw=S["ref"], ls=(0, (3, 2)), zorder=0)
        ax.set_ylim(0.45, 1.05); ax.set_xticks(range(len(xs))); ax.set_xticklabels([l for _, l in DEPTHS]); ax.set_xlim(-0.4, len(xs) - 0.6)
        ax.set_title(title, fontsize=S["sm"], loc="left", pad=1.5, color=S["ink"])
        ax.text(0.98, 0.12, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=S["sm"] - 1, color=S["ink"], linespacing=1.0)
        ax.set_xlabel("Sequencing depth", labelpad=1.5)
        _letter(ax, "def"[f], S, dx_mm=1.0 if f else 10.5)
        if f == 0:
            ax.set_ylabel("Planted vs null\n(AUROC)")
        else:
            ax.set_yticklabels([])
    lx = axes_in(sub_slot(fig, 0, h - leg_h, w, leg_h), (wl, 0.2, 0.5, 0.2)); lx.axis("off")
    hs = [Line2D([], [], color=S["lfa"], lw=0.8, marker="o", ms=MS, label="LeafletFA"),
          Line2D([], [], color=S["raw"], lw=0.8, marker="s", ms=MS, label="Raw pseudobulk"),
          Line2D([], [], color=S["B"], lw=0.8, marker="^", ms=MS, label="Shrinkage to pooled ψ (B)"),
          Line2D([], [], color=S["lc"], lw=0.8, marker="D", ms=MS, label="LeafCutter")]
    lx.legend(handles=hs, loc="center left", ncol=4, frameon=False, fontsize=S["sm"], handlelength=1.6, columnspacing=1.0,
              borderpad=0.0, borderaxespad=0)


# ---------------------------------------------------------------- Sscale_a
def panel_scale(cfg, fig):
    """Scaling runs already done (record 2026-11-14): LeafletFA 2,000 steps on one RTX PRO 6000 Blackwell GPU at
    5,000 to 54,694 TMS TRAIN cells, one 16-CPU run at 5,000 cells, and scQuint (16 CPUs, to its early stop). a training
    time; b peak memory (GPU memory and host memory of the GPU runs; scQuint host memory). Input: $V/scaling/scaling.tsv."""
    import matplotlib.ticker as mt
    w, h = container_mm(fig); S = _S(cfg); T = scaling_table()
    g = T[(T.method == "LeafletFA") & (T.device == "GPU")].sort_values("cells")
    c = T[(T.method == "LeafletFA") & (T.device == "CPU")].iloc[0]; q = T[T.method == "scQuint"].sort_values("cells")
    leg_h = 9.0
    axa = axes_in(sub_slot(fig, 0, 0, w / 2, h), (12.5, 9.0, 2.0, leg_h))
    axb = axes_in(sub_slot(fig, w / 2, 0, w / 2, h), (12.5, 9.0, 2.0, leg_h))
    axa.plot(g.cells, g.fit_min, color=S["lfa"], lw=1.0, marker="o", ms=MS, label="LeafletFA, GPU, 2,000 steps")
    axa.plot([c.cells], [c.fit_min], ls="none", marker="o", ms=MS + 0.6, mfc="white", mec=S["lfa"], mew=0.8, label="LeafletFA, 16 CPUs, 2,000 steps")
    axa.plot(q.cells, q.fit_min, color=S["scq"], lw=1.0, marker="s", ms=MS, label="scQuint, 16 CPUs, to early stop")
    axa.set(xscale="log", yscale="log", ylim=(1, 400), ylabel="Training time (min)")
    axb.plot(g.cells, g.peak_gpu_gb, color=S["lfa"], lw=1.0, marker="o", ms=MS, label="LeafletFA, GPU memory")
    axb.plot(g.cells, g.peak_host_gb, color=S["lfa"], lw=1.0, ls=(0, (3, 2)), marker="o", ms=MS, mfc="white", label="LeafletFA, host memory (GPU run)")
    axb.plot(q.cells, q.peak_host_gb, color=S["scq"], lw=1.0, marker="s", ms=MS, label="scQuint, host memory")
    axb.set(xscale="log", ylim=(0, None), ylabel="Peak memory (GB)")
    for ax, l in ((axa, "a"), (axb, "b")):
        ax.set_xticks([5000, 10000, 25000, 54694]); ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.xaxis.set_minor_locator(mt.NullLocator()); ax.set_xlabel("TMS cells")
        if ax is axa:
            ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda x, _: f"{x:g}"))
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, frameon=False, fontsize=S["sm"], handlelength=1.6,
                  labelspacing=0.2, borderpad=0.0, borderaxespad=0.3)
        _letter(ax, l, S, dx_mm=10.5)
    axa.text(0.98, 0.04, f"CPU / GPU per step at 5,000 cells: {c.s_per_step / g[g.cells == 5000].s_per_step.iloc[0]:.0f}×",
             transform=axa.transAxes, ha="right", va="bottom", fontsize=S["sm"] - 1, color=S["ink"]).set_y(0.16)


def panel_b18(cfg, fig):
    """As ANb17_b for check 18a (refit on half-A mice; confirmation only in half-B mice the model never saw).
    Input: $V/c18/18a_summary.tsv."""
    panel_b17(cfg, fig, check="18a")


PANELS = {"ANb17_b": panel_b17, "ANb18_b": panel_b18, "Sds_a": panel_ds, "Ssim_a": panel_sim, "Sscale_a": panel_scale}
SOURCE_DATA = {"ANb17_b": lambda cfg: {"figAN_b17_check17_excess": c17_table()},
               "ANb18_b": lambda cfg: {"figAN_b18_check18a_excess": c17_table("18a")},
               "Sds_a": lambda cfg: {"figS_ds_validation_differences": ds_tables()},
               "Ssim_a": lambda cfg: dict(zip(("figS_sim_m1", "figS_sim_m2", "figS_sim_m2_lc"), sim_tables())),
               "Sscale_a": lambda cfg: {"figS_scaling": scaling_table()}}
