#!/usr/bin/env python
"""Calibration panels for the DM likelihood-ratio test.

Each panel answers one question that a number alone leaves ambiguous.

  a  QQ of p-values under PERMUTED phi, against Uniform. Permutation destroys the phi<->counts
     pairing, so every event is null by construction and an honest p-value must lie on the
     diagonal. Bowing ABOVE it means p is too small -- anti-conservative. This is the panel the
     whole argument rests on, and it is what showed the chi-square drifting 0.62 -> 3.82 with
     effect size, which a true null cannot do.
  b  Nominal alpha vs realised FDP against known truth, with the y=x diagonal. The incumbent
     fixed-delta rule sits above the line (2.9x at 0.05); the LRT should sit on or below it.
  c  Same QQ, fixed vs free concentration. Holding conc at the model's single global value
     misspecifies the DM for events whose overdispersion differs, and a misspecified DM inflates
     the LRT. This is the before/after for that.
  d  P-value histograms, observed vs permuted. Observed should be a spike near zero on a flat
     bulk; permuted should be flat. The bulk height IS pi0, so the plot makes the null fraction
     readable rather than asserted.
  e  LRT statistic against its chi-square reference, faceted by event size m. Shows WHERE Wilks
     breaks rather than only that it does. m=2 is 44% of the atlas and cannot appear in the
     cassette-exon simulations at all, so this panel is only fully populated on real data.
  f  Null statistic scale against ATSE coverage. Explains why three earlier null-estimation
     attempts all failed in the tail: the pooled null is a mixture over coverage (5.6x scale range
     across deciles), so no single-family fit can match its tail.

Labels are 1-indexed where programs appear. Run:
    panels_lrt_calibration.py --lrt-dir <DM_FRESH_LRT> --free-dir <..._FREECONC> --out <panels>
"""

import argparse, glob, os, re, sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/"
                   "figures/_common")
import leaflet_theme as th  # noqa: E402


def eff_of(path):
    m = re.search(r"eff([0-9.]+)", os.path.basename(path))
    return float(m.group(1)) if m else np.nan


def seed_of(path):
    m = re.search(r"seed(\d+)\.tsv$", os.path.basename(path))
    return int(m.group(1)) if m else 0


def load_dir(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "a45_lrt_*.tsv"))):
        try:
            t = pd.read_csv(f, sep="\t")
        except Exception:
            continue
        if not len(t):
            continue
        t["effect"] = eff_of(f)
        t["seed"] = seed_of(f)
        t["src"] = os.path.basename(f)
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def qq(ax, p, label, color, ls="-"):
    """Empirical quantiles of p against Uniform, on log axes so the TAIL is visible -- the tail is
    where every decision is made and where a linear QQ hides everything."""
    p = np.sort(np.asarray(p, float))
    p = p[np.isfinite(p)]
    if not len(p):
        return
    exp = (np.arange(1, len(p) + 1) - 0.5) / len(p)
    ax.plot(exp, p, ls, color=color, lw=1.3, label=label)


def bh(p, alpha):
    n = len(p); o = np.argsort(p); thr = alpha * np.arange(1, n + 1) / n
    ok = p[o] <= thr
    sig = np.zeros(n, bool)
    if ok.any():
        sig[o[:np.flatnonzero(ok)[-1] + 1]] = True
    return sig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    W = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
    ap.add_argument("--lrt-dir", default=f"{W}/DM_FRESH_LRT")
    ap.add_argument("--free-dir", default=f"{W}/DM_FRESH_LRT_FREECONC")
    ap.add_argument("--atlas-dir", default=f"{W}/ATLAS_LRT")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    th.apply_theme()

    F = load_dir(args.lrt_dir)
    G = load_dir(args.free_dir)
    A = load_dir(args.atlas_dir)
    if not len(F):
        print(f"nothing in {args.lrt_dir}"); return
    effs = sorted(F["effect"].dropna().unique())
    cols = dict(zip(effs, [th.PALETTE[k] for k in ("grey", "ochre", "teal", "indigo")][:len(effs)]))
    print(f"[0] fixed-conc {F['src'].nunique()} runs, free-conc {G['src'].nunique() if len(G) else 0}, "
          f"atlas {A['src'].nunique() if len(A) else 0}")

    fig, axes = plt.subplots(2, 3, figsize=th.figsize("double", h=6.2))

    # ── a. QQ under permuted phi ───────────────────────────────────────────────────────
    ax = axes[0, 0]
    for e in effs:
        sub = F[(F.effect == e) & (F.seed == F.seed.min())]
        if "perm_p" in sub:
            qq(ax, sub["perm_p"], f"effect {e:g}", cols[e])
    ax.plot([1e-4, 1], [1e-4, 1], color=th.INK, lw=0.8, ls=":", zorder=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("expected p (Uniform)"); ax.set_ylabel("observed p")
    ax.set_title("permuted $\\phi$: a true null\nbelow the line = p too small", fontsize=th.PT["annot"])
    ax.legend(frameon=False, fontsize=th.PT["legend"], loc="upper left")
    th.panel_label(ax, "a")

    # ── b. nominal vs realised FDP ─────────────────────────────────────────────────────
    ax = axes[0, 1]
    for e in effs:
        sub = F[F.effect == e]
        xs, ys = [], []
        for a in args.alphas:
            fdps = []
            for s in sub["src"].unique():
                t = sub[sub.src == s]
                sig = bh(t["p"].values, a)
                if sig.sum():
                    fdps.append(float((sig & ~t["is_de"].values).sum() / sig.sum()))
            if fdps:
                xs.append(a); ys.append(np.mean(fdps))
        ax.plot(xs, ys, "o-", ms=3, color=cols[e], lw=1.2, label=f"effect {e:g}")
    ax.plot([1e-3, 0.2], [1e-3, 0.2], color=th.INK, lw=0.8, ls=":", zorder=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("nominal $\\alpha$"); ax.set_ylabel("realised FDP")
    ax.set_title("LRT + BH against known truth\nabove the line = more false calls than promised",
                 fontsize=th.PT["annot"])
    ax.legend(frameon=False, fontsize=th.PT["legend"], loc="upper left")
    th.panel_label(ax, "b")

    # ── c. fixed vs free concentration ─────────────────────────────────────────────────
    ax = axes[0, 2]
    if len(G):
        for e in effs:
            f0 = F[(F.effect == e) & (F.seed == 0)]
            g0 = G[G.effect == e]
            if len(f0) and "perm_p" in f0:
                qq(ax, f0["perm_p"], f"{e:g} fixed", cols[e], ls="--")
            if len(g0) and "perm_p" in g0:
                qq(ax, g0["perm_p"], f"{e:g} free", cols[e], ls="-")
        ax.plot([1e-4, 1], [1e-4, 1], color=th.INK, lw=0.8, ls=":", zorder=0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=th.PT["legend"], loc="upper left", ncol=2)
    else:
        ax.text(0.5, 0.5, "free-conc runs\nnot finished", ha="center", va="center",
                transform=ax.transAxes, fontsize=th.PT["annot"], color=th.PALETTE["grey"])
    ax.set_xlabel("expected p (Uniform)"); ax.set_ylabel("observed p")
    ax.set_title("concentration: global (dashed)\nvs per-ATSE (solid)", fontsize=th.PT["annot"])
    th.panel_label(ax, "c")

    # ── d. p-value histograms ──────────────────────────────────────────────────────────
    ax = axes[1, 0]
    e_show = effs[len(effs) // 2] if effs else None
    sub = F[(F.effect == e_show) & (F.seed == 0)]
    bins = np.linspace(0, 1, 41)
    ax.hist(sub["p"], bins=bins, color=th.PALETTE["teal"], alpha=0.75, label="observed")
    if "perm_p" in sub:
        ax.hist(sub["perm_p"], bins=bins, histtype="step", color=th.PALETTE["rose"], lw=1.3,
                label="permuted $\\phi$ (null)")
    pi0 = float((~sub["is_de"]).mean())
    ax.axhline(len(sub) / (len(bins) - 1) * pi0, color=th.INK, ls=":", lw=0.9)
    ax.set_xlabel("p"); ax.set_ylabel("ATSEs")
    ax.set_title(f"effect {e_show:g}: spike = real signal\ndotted line = $\\pi_0$={pi0:.2f} bulk",
                 fontsize=th.PT["annot"])
    ax.legend(frameon=False, fontsize=th.PT["legend"])
    th.panel_label(ax, "d")

    # ── e. statistic vs chi-square, by m ───────────────────────────────────────────────
    ax = axes[1, 1]
    src = A if len(A) else F
    lab = "atlas" if len(A) else "simulation"
    ms = sorted(src["m"].unique())
    mcols = [th.PALETTE[k] for k in ("teal", "indigo", "ochre", "rose", "olive", "grey")]
    from scipy.stats import chi2 as _chi2
    for i, mm in enumerate(ms[:6]):
        s = src[src.m == mm]
        if len(s) < 50 or "perm_stat" not in s:
            continue
        df = float(s["df"].iloc[0])
        # the statistic rescaled to its own reference: an honest chi2 puts this on Uniform
        u = _chi2.sf(s["perm_stat"].values, df)
        qq(ax, u, f"m={mm} (n={len(s):,})", mcols[i % len(mcols)])
    ax.plot([1e-4, 1], [1e-4, 1], color=th.INK, lw=0.8, ls=":", zorder=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("expected p (Uniform)"); ax.set_ylabel("observed p")
    ax.set_title(f"where Wilks breaks, by event size\n({lab}; m=2 is 44% of the atlas)",
                 fontsize=th.PT["annot"])
    ax.legend(frameon=False, fontsize=th.PT["legend"], loc="upper left")
    th.panel_label(ax, "e")

    # ── f. null scale vs coverage ──────────────────────────────────────────────────────
    ax = axes[1, 2]
    for e in effs:
        sub = F[(F.effect == e) & (F.seed == 0) & (~F.is_de)]
        if len(sub) < 100 or "depth" not in sub:
            continue
        q = np.quantile(sub["depth"], np.linspace(0, 1, 11))
        xs, ys = [], []
        for i in range(10):
            m_ = (sub["depth"] >= q[i]) & (sub["depth"] <= q[i + 1])
            if m_.sum() < 20:
                continue
            xs.append(float(np.median(sub["depth"][m_])))
            ys.append(float(np.median(sub["stat"][m_])))
        ax.plot(xs, ys, "o-", ms=3, color=cols[e], lw=1.2, label=f"effect {e:g}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("ATSE coverage (mean reads/cell)")
    ax.set_ylabel("median LRT statistic, true nulls")
    ax.set_title("why a pooled null cannot be fitted:\nthe null scales with coverage",
                 fontsize=th.PT["annot"])
    ax.legend(frameon=False, fontsize=th.PT["legend"], loc="upper left")
    th.panel_label(ax, "f")

    fig.suptitle("Dirichlet-multinomial likelihood-ratio test: is the p-value honest?",
                 fontsize=th.PT["panel"], y=0.995)
    fig.tight_layout()
    th.save(fig, os.path.join(args.out, "lrt_calibration"))
    plt.close(fig)
    print(f"[1] -> {args.out}/lrt_calibration.*")

    # numeric companion, so the panels are never the only record
    rec = []
    for name, D in (("fixed_conc", F), ("free_conc", G), ("atlas", A)):
        if not len(D):
            continue
        for e in sorted(D["effect"].dropna().unique()) or [np.nan]:
            sub = D[D.effect == e] if np.isfinite(e) else D
            for q in (0.01, 0.05, 0.10, 0.20):
                if "perm_p" in sub:
                    rec.append(dict(source=name, effect=e, nominal=q,
                                    perm_frac=float((sub["perm_p"] <= q).mean()),
                                    ratio=float((sub["perm_p"] <= q).mean() / q), n=len(sub)))
    if rec:
        R = pd.DataFrame(rec)
        R.to_csv(os.path.join(args.out, "lrt_calibration_numbers.tsv"), sep="\t", index=False)
        print("\n=== permuted-null type-I error (ratio 1.0 = honest) ===")
        print(R.pivot_table(index=["source", "effect"], columns="nominal",
                            values="ratio").round(2).to_string())


if __name__ == "__main__":
    main()
