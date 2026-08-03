"""Panel builders for Figure 2: cassette-exon simulation (2A-D) + held-out imputation (2E-F).

    import panels as P
    bench = P.load_benchmark()     # 2B-D: LeafletFA vs NMF on simulated cassette exons
    imp   = P.load_imputation()    # 2E-F: imputation vs coverage on the held-out 30%
    struc = P.load_structure()     # supp:  held-out latent structure

NAMING -- "atlas" means the full dataset (Figure 1) and the released full-data models
(*_full_K30_*). It does NOT mean the work in this file. Everything here is the 70/30 HELD-OUT
analysis, so the constants and loaders say `impute` / `heldout`. Figure 3 is a different figure
entirely (calibration against LeafCutter + differential splicing) and lives in FIGURE3_leafcutter_ds.

Layout: this file is in code/, reads from ../data/, writes panels to ../plots/.

Data provenance
---------------
2A-D   data/benchmark_results.csv, data/benchmark_effect_sweep.csv, data/representative_fit.npz,
       data/rerun_logs/{odbench,prbench}_*.csv
       Cassette-exon simulation from pipelines/09_simulation/00_simulate_splicing.py: each ATSE is
       exactly 3 junctions (J1, J2 inclusion; J3 skip), psi[k] = P(skip) per program and
       P(J1) = P(J2) = (1 - psi)/2, so the event collapses to one interpretable number. `atse_total`
       is taken verbatim from a real cell_by_cluster_matrix, so coverage/depth/zero-inflation are
       real. Sweep axis = fraction of differential ATSEs (10/50/90%), 3 seeds, K_true = 5.
       Baselines: NMF at the same rank with 0-impute and mean-impute (NMF has no missing-data model).

2E-F   LEAFLETFA_MODELS/2026-07-30_atlas9_default/eval_grouped/task_*/{04e_allcov*,07c_out}.csv
       9 models = K in {10,20,30} x 3 seeds, trained on the 70% split (default random init,
       Beta-Binomial, junction-specific prior, minibatch B=8192, early stopping), evaluated on the
       held-out 30%. These are the FRESH 2026-07-30 fits; the earlier 2026-07-26_atlas9 models are
       superseded and must not be mixed in.

       Three entry sets are available per level. `fixed_lost_at_hardest` is the one to lead with --
       the same 93.4M entries at every coverage level, so the curve is a curve. But at mild
       downsampling 93% of those entries are still observed and lf_bb/mb/mb_ct pass the observed
       value through, so plot only the pure-prediction methods (lf, nmf0_tr, nmfm_tr, naive) there;
       passthrough methods belong on `all_covered`.
"""
from __future__ import annotations
import os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
_CODE = os.path.dirname(os.path.abspath(__file__))          # .../FIGURE2_.../code
_FIG  = os.path.dirname(_CODE)                              # .../FIGURE2_...
_HERE = f"{_FIG}/data"                                      # inputs
_OUT  = f"{_FIG}/plots"                                     # rendered panels
sys.path.insert(0, f"{os.path.dirname(_FIG)}/_common")
import leaflet_theme as th

# The 70/30 HELD-OUT models -- 9 fits, K in {10,20,30} x 3 seeds, default (random) init,
# Beta-Binomial, junction-specific prior. Deliberately NOT called "atlas": in this paper "atlas"
# means the full dataset (Figure 1) and the released full-data models (*_full_K30_*). These are the
# held-out imputation/transfer models and nothing else.
IMPUTE_DIR = ("/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/"
              "MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/2026-07-30_atlas9_default")

# Fig 2 series. `likelihood` doubles as the arm label: bb/binom for LeafletFA, imputation for NMF.
SIM_SERIES = [
    ("LeafletFA", "bb",          "LeafletFA (Beta-Binomial)", "LeafletFA"),
    ("LeafletFA", "binom",       "LeafletFA (Binomial)",      "ablation"),
    ("NMF",       "0-impute",    "NMF (0-impute)",            "NMF"),
    ("NMF",       "mean-impute", "NMF (mean-impute)",         "null"),
]
SIM_METRICS = {
    "psi_r":          "$\\Psi$ recovery (corr to truth)",
    "spearman_delta": "effect-size recovery\n(Spearman $\\Delta\\Psi$)",
    "ari":            "ARI (cells vs true program)",
    "sil":            "silhouette (true program)",
    "auroc":          "AUROC (differential ATSEs)",
    "precision":      "precision (differential ATSEs)",
}

# Fig 3 imputation methods. lf_bb is the shipped model; *_tr are the FAIR NMF baselines (dictionary
# fit on train, test cells projected). nmf0/nmfm without _tr are fit ON the test cells and mb_ct uses
# cell-type labels, so both are upper bounds rather than fair comparisons.
METHODS = {
    "lf_bb":   ("LeafletFA", "LeafletFA"),
    "nmf0_tr": ("NMF (transfer, 0-impute)", "NMF"),
    "nmfm_tr": ("NMF (transfer, mean-impute)", "null"),
    "mb":      ("junction mean", "ablation"),
    "mb_ct":   ("cell-type mean*", "PCA"),
}
STRUCT_SOURCES = {
    "heldout_inferred": ("LeafletFA $\\phi$", "LeafletFA"),
    "heldout_nmf0":     ("NMF (0-impute)", "NMF"),
    "heldout_nmfm":     ("NMF (mean-impute)", "null"),
}


def _ms(sub: pd.DataFrame, col: str) -> tuple[float, float]:
    """Mean and standard error over seeds."""
    x = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(float)
    if not len(x):
        return np.nan, 0.0
    return float(x.mean()), (float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0)


def _vals(sub: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(float)


def ttest(a: np.ndarray, b: np.ndarray) -> float:
    """Welch's t-test p-value (unequal variances — the safe default for small n)."""
    from scipy import stats
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def _arm(bench: pd.DataFrame, model: str, arm: str, prior: bool | None = True) -> pd.DataFrame:
    s = bench[(bench.model == model) & (bench.likelihood == arm)]
    if model == "LeafletFA" and prior is not None:
        s = s[s.junc_prior.astype(str) == str(prior)]
    return s


# ── loaders ──────────────────────────────────────────────────────────────────
def load_benchmark() -> pd.DataFrame:
    return pd.read_csv(f"{_HERE}/benchmark_results.csv")


def _odbench_csvs() -> list[str]:
    """Overdispersion-arm results, EXCLUDING the abandoned deterministic-init experiment.

    A bare `odbench_*.csv` glob also matches the six `odbench_detinit_bb{2,20}_seed{0,1,2}.csv`
    written during the initialisation investigation, which was closed and reverted -- those runs
    used a different initialiser and were silently being averaged into the overdispersion panels.
    """
    out, dropped = [], []
    for f in sorted(glob.glob(f"{_HERE}/rerun_logs/odbench_*.csv")):
        (dropped if "detinit" in os.path.basename(f) else out).append(f)
    if dropped:
        print(f"  [odbench] excluding {len(dropped)} deterministic-init file(s) from the "
              f"abandoned init experiment: {', '.join(os.path.basename(f) for f in dropped)}")
    return out


def load_representative() -> dict:
    """True program labels and fitted phi for the UMAP panels."""
    z = np.load(f"{_HERE}/representative_fit.npz", allow_pickle=True)
    return {k: np.asarray(z[k]) for k in z.files}


def _task_map() -> dict[int, tuple[int, int]]:
    p = json.load(open(f"{IMPUTE_DIR}/parameter_combinations.json"))
    return {i: (int(r["K"]), int(r["seed"])) for i, r in enumerate(p)}


def _load_heldout(kind: str) -> pd.DataFrame:
    """Concatenate the per-task eval CSVs from the 70/30 held-out models.

    `kind` is the CSV stem, e.g. "07c" for held-out structure or "04e_allcov"/"04e_allcov_fixedset"
    for imputation. Note the imputation files are named `<kind>.csv`, not `<kind>_out.csv`.
    """
    suffix = "_out.csv" if not kind.startswith("04e") else ".csv"
    tmap, frames = _task_map(), []
    for f in sorted(glob.glob(f"{IMPUTE_DIR}/eval_grouped/task_*/{kind}{suffix}")):
        t = int(os.path.basename(os.path.dirname(f)).split("_")[1])
        d = pd.read_csv(f)
        d["task"], d["K_cfg"], d["seed"] = t, tmap[t][0], tmap[t][1]
        frames.append(d)
    if not frames:
        raise FileNotFoundError(f"no {kind}{suffix} under {IMPUTE_DIR}/eval_grouped/")
    return pd.concat(frames, ignore_index=True)


def load_imputation(kind: str = "04e_allcov", entry_set: str = "all_covered",
                    dataset: str = "all") -> pd.DataFrame:
    """Imputation eval, filtered to ONE entry set and ONE dataset scope.

    THE FILTERING IS NOT OPTIONAL. A3 writes several rows per (task, coverage level): one per
    entry set for `dataset == "all"`, plus a per-dataset breakdown. Without filtering, a panel that
    groups only on K and coverage silently averages

        all / all_covered            0.0709      <- 95.7M entries
        all / lost_to_downsampling   0.0504      <-  6.3M entries, a different question
        allen_brain_exons / all_covered  0.0318  <- a SUBSET of row 1, double counted
        tabula_muris_senis / all_covered 0.0426  <- likewise

    into 0.0489, which is not a quantity. Every imputation panel did exactly that before
    2026-08-02.

    entry_set: "all_covered" | "lost_to_downsampling" | "fixed_lost_at_hardest"
        `fixed_lost_at_hardest` holds the SAME entries at every coverage level and is the one to
        read as a curve; the other two are a moving population. See REPRODUCE_ALL.md B3b.
    kind: "04e_allcov" (original) or "04e_allcov_fixedset" (the run that adds the fixed set).
    """
    d = _load_heldout(kind)
    if "entry_set" in d.columns:
        avail = sorted(d.entry_set.dropna().unique())
        if entry_set not in avail:
            raise ValueError(f"entry_set={entry_set!r} not in {kind}; available: {avail}")
        d = d[d.entry_set == entry_set]
    if "dataset" in d.columns:
        d = d[d.dataset == dataset]
    # exactly one row per (model, coverage level) or the mean below is averaging unlike things
    dup = d.groupby(["task", "downsampling_pct"]).size()
    if (dup > 1).any():
        raise ValueError(f"{kind}/{entry_set}/{dataset}: {int((dup > 1).sum())} (task, level) "
                         f"pairs have >1 row after filtering -- a panel would average them")
    return d


def load_structure() -> pd.DataFrame:
    return _load_heldout("07c")



# ── effect-size sweep (2026-07-26) ───────────────────────────────────────────
# The %-differential sweep varies HOW MANY cassettes differ but always makes them maximally
# different (effect=1.0 -> median true delta-Psi = 0.85, far larger than real differential
# splicing). Every method that works at all saturates there. This sweep instead varies HOW MUCH
# they differ at a fixed 30% differential, which is the axis that actually sets difficulty.
EFFECT_TO_DELTA = {0.1: 0.085, 0.25: 0.21, 0.5: 0.42, 1.0: 0.85}   # median true |delta Psi|


def load_effect_sweep() -> pd.DataFrame:
    f = f"{_HERE}/benchmark_effect_sweep.csv"
    if not os.path.exists(f):
        raise FileNotFoundError(f"{f} missing — run the effect sweep first")
    d = pd.read_csv(f)
    d["median_delta_psi"] = d["effect"].map(EFFECT_TO_DELTA)
    return d


def panel_effect_sweep(ax, eff: pd.DataFrame, metric: str = "spearman_delta",
                       series=SIM_SERIES, prior: bool | None = True, title: str = "",
                       x="median_delta_psi") -> None:
    """A recovery metric against the size of the simulated splicing difference.

    Plotted against median true |delta Psi| rather than the raw `effect` parameter, so the x-axis is
    interpretable: real differential splicing is typically delta-Psi 0.05-0.3, i.e. the left half.
    """
    xs = sorted(eff[x].dropna().unique())
    pos = np.arange(len(xs))                       # evenly spaced: these are 4 discrete levels
    for model, arm, lab, ckey in series:
        s = eff[(eff.model == model) & (eff.likelihood == arm)]
        if model == "LeafletFA" and prior is not None:
            s = s[s.junc_prior.astype(str) == str(prior)]
        if not len(s):
            continue
        m = [_ms(s[s[x] == v], metric)[0] for v in xs]
        e = [_ms(s[s[x] == v], metric)[1] for v in xs]
        ax.errorbar(pos, m, yerr=e, fmt="o-", ms=4, capsize=2, label=lab,
                    color=th.METHOD_COLORS[ckey])
    ax.set_xticks(pos); ax.set_xticklabels([f"{v:.1f}" for v in xs])
    ax.set_xlim(-0.35, len(xs) - 0.65)
    ax.set_xlabel("simulated effect size (median $|\\Delta\\Psi|$)")
    ax.set_ylabel(SIM_METRICS.get(metric, metric))
    if title:
        ax.set_title(title, fontsize=th.PT["annot"])
    th.despine(ax)


def panel_prior_vs_effect(ax, eff: pd.DataFrame, metric: str = "psi_r") -> None:
    """Is the junction-specific prior load-bearing? Only visible where the signal is weak.

    At effect=1.0 the two priors are indistinguishable; the question only has an answer in the
    regime where recovery is actually hard, which the %-differential sweep never reached.
    """
    xs = sorted(eff["median_delta_psi"].dropna().unique())
    base = eff[(eff.model == "LeafletFA") & (eff.likelihood == "bb")]
    for val, lab, ckey in [(True, "junction-specific prior", "LeafletFA"),
                           (False, "global prior", "ablation")]:
        s = base[base.junc_prior.astype(str) == str(val)]
        m = [_ms(s[s.median_delta_psi == v], metric)[0] for v in xs]
        e = [_ms(s[s.median_delta_psi == v], metric)[1] for v in xs]
        ax.errorbar(np.arange(len(xs)), m, yerr=e, fmt="o-", ms=4, capsize=2, label=lab,
                    color=th.METHOD_COLORS[ckey])
    ax.set_xticks(np.arange(len(xs))); ax.set_xticklabels([f"{v:.1f}" for v in xs])
    ax.set_xlim(-0.35, len(xs) - 0.65)
    ax.set_xlabel("simulated effect size (median $|\\Delta\\Psi|$)")
    ax.set_ylabel(SIM_METRICS.get(metric, metric))
    th.despine(ax)


# ── likelihood vs the truth's overdispersion ─────────────────────────────────
# The effect sweep simulates PURE BINOMIAL counts, so the Binomial is correctly specified there and
# the Beta-Binomial can only match it, never beat it. To ask whether modelling overdispersion helps,
# the counts themselves have to be overdispersed: these runs redraw the same simulation with a finite
# Beta-Binomial concentration (2, matching the atlas's learned ~1.96, and 20, mild) at a fixed
# effect size of 0.25, and fit both likelihoods to each.
OVERDISP_LABEL = {float("inf"): "none", 20.0: "mild\n$c$=20", 2.0: "strong\n$c$=2"}


def load_overdispersion() -> pd.DataFrame:
    """Binomial-truth rows (effect=0.25 of the main sweep) plus the overdispersed-truth runs."""
    import glob
    frames = []
    base = f"{_HERE}/benchmark_effect_sweep.csv"
    if os.path.exists(base):
        b = pd.read_csv(base)
        b = b[np.isclose(b.effect, 0.25)].copy()      # matched effect size
        b["sim_bb_conc"] = float("inf")               # that sweep was generated Binomial
        frames.append(b)
    od = sorted(_odbench_csvs())
    if od:
        frames.append(pd.concat([pd.read_csv(x) for x in od], ignore_index=True))
    if not frames:
        raise FileNotFoundError("no overdispersion runs found")
    return pd.concat(frames, ignore_index=True)


def panel_overdispersion(ax, od: pd.DataFrame, metric: str = "spearman_delta") -> None:
    """Does modelling overdispersion matter? Only when the data actually are overdispersed."""
    lf = od[(od.model == "LeafletFA") & (od.junc_prior == True)]
    levels = [c for c in (float("inf"), 20.0, 2.0) if (lf.sim_bb_conc == c).any()]
    pos = np.arange(len(levels))
    w = 0.36
    xs, subs = {}, {}
    for i, (arm, lab, ckey) in enumerate([("bb", "Beta-Binomial model", "LeafletFA"),
                                          ("binom", "Binomial model", "null")]):
        s = lf[lf.likelihood == arm]
        m = [_ms(s[s.sim_bb_conc == c], metric)[0] for c in levels]
        e = [_ms(s[s.sim_bb_conc == c], metric)[1] for c in levels]
        x = pos + (i - 0.5) * w
        ax.bar(x, m, w, yerr=e, capsize=2, label=lab, color=th.METHOD_COLORS[ckey])
        xs[arm], subs[arm] = x, s
    # Beta-Binomial vs Binomial WITHIN each overdispersion level -- the comparison the panel exists
    # to make. `ns` is shown rather than omitted, since "no difference when the data are not
    # overdispersed" is half the claim.
    top = max(_ms(subs[a][subs[a].sim_bb_conc == c], metric)[0] +
              _ms(subs[a][subs[a].sim_bb_conc == c], metric)[1]
              for a in ("bb", "binom") for c in levels)
    for gi, c in enumerate(levels):
        a = _vals(subs["bb"][subs["bb"].sim_bb_conc == c], metric)
        b = _vals(subs["binom"][subs["binom"].sim_bb_conc == c], metric)
        p = ttest(a, b)
        if np.isfinite(p):
            y = max(a.mean(), b.mean()) + 0.10 * top
            th.sig_bracket(ax, xs["bb"][gi], xs["binom"][gi], y, p)
    ax.set_xlim(-0.5, len(levels) - 0.5)
    ax.set_ylim(0, top * 1.30)
    ax.set_xticks(pos)
    ax.set_xticklabels([OVERDISP_LABEL[c] for c in levels])
    ax.set_xlabel("simulated overdispersion")
    ax.set_ylabel(SIM_METRICS.get(metric, metric))
    th.despine(ax)


def load_prior_test() -> pd.DataFrame:
    """All c=2 (atlas-like overdispersion) runs: 3 seeds from the overdispersion arm + 7 more."""
    import glob
    fs = sorted(_odbench_csvs()) + \
         sorted(glob.glob(f"{_HERE}/rerun_logs/prbench_*.csv"))
    if not fs:
        raise FileNotFoundError("no overdispersion / prior-test runs found")
    d = pd.concat([pd.read_csv(x) for x in fs], ignore_index=True)
    return d[np.isclose(d.sim_bb_conc, 2.0)]


def panel_prior_interaction(ax, pt: pd.DataFrame, metric: str = "psi_r") -> None:
    """Does the junction-specific Psi prior help? Only under a likelihood that models the noise.

    Ten seeds at the overdispersion the atlas exhibits (c = 2). Under the Beta-Binomial the
    junction-specific prior both raises recovery and sharply reduces run-to-run variance; under the
    Binomial — which has already discarded the overdispersion the prior would regularise against —
    it does nothing.
    """
    lf = pt[pt.model == "LeafletFA"]
    arms = [("bb", "Beta-Binomial"), ("binom", "Binomial")]
    pos, w = np.arange(len(arms)), 0.36
    cells = {}
    for i, (val, lab, ckey) in enumerate([(True, "junction-specific prior", "LeafletFA"),
                                          (False, "global prior", "ablation")]):
        m, e = [], []
        for arm, _ in arms:
            sub = lf[(lf.likelihood == arm) & (lf.junc_prior == val)]
            mm, ee = _ms(sub, metric)
            m.append(mm); e.append(ee); cells[(arm, val)] = sub
        ax.bar(pos + (i - 0.5) * w, m, w, yerr=e, capsize=2, label=lab,
               color=th.METHOD_COLORS[ckey])
    top = max(_ms(cells[k], metric)[0] + _ms(cells[k], metric)[1] for k in cells)
    for gi, (arm, _) in enumerate(arms):
        p = ttest(_vals(cells[(arm, True)], metric), _vals(cells[(arm, False)], metric))
        if np.isfinite(p):
            th.sig_bracket(ax, pos[gi] - 0.5 * w, pos[gi] + 0.5 * w, top * 1.04, p)
    ax.set_xticks(pos); ax.set_xticklabels([lab for _, lab in arms])
    ax.set_xlabel("likelihood (overdispersed counts, $c$=2)")
    ax.set_ylabel(SIM_METRICS.get(metric, metric))
    ax.set_ylim(0, min(1.0, top * 1.22) if metric == "psi_r" else top * 1.22)
    th.despine(ax)


# ── reproducibility on real data (A6) ────────────────────────────────────────
IMPUTE_A6 = f"{IMPUTE_DIR}/A6_seed_consistency.csv"
# Measured on the atlas, where there is no ground truth: models differing only in random seed are
# compared to each other. Agreement falls at each level of the pipeline for a mechanical reason —
# psi pools every training cell, phi_c uses one cell's reads at ~13.5% coverage, and Leiden then
# discretises that noisy phi. Cluster instability is therefore not evidence of an unstable model.
A6_LEIDEN_ARI = 0.305      # from A5_repro_K10.csv, LeafletFA seed-to-seed Leiden ARI


def load_seed_consistency() -> pd.DataFrame:
    if not os.path.exists(IMPUTE_A6):
        raise FileNotFoundError(f"{IMPUTE_A6} missing — run A6_seed_consistency.py")
    return pd.read_csv(IMPUTE_A6)


def panel_seed_consistency(ax, a6: pd.DataFrame) -> None:
    """How much of the fit is reproducible across random initialisations, on the real atlas."""
    psi = a6[a6.comparison.str.startswith("psi")]
    phi = a6[a6.comparison.str.startswith("phi")]
    levels, vals, errs, cols = [], [], [], []
    for K in sorted(psi.K.unique()):
        v = psi[psi.K == K].median_r
        levels.append(f"programs\n$\\Psi$, K={K}"); vals.append(v.mean())
        errs.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
        cols.append(th.METHOD_COLORS["LeafletFA"])
    if len(phi):
        v = phi.median_r
        levels.append("per-cell\nloadings $\\phi$"); vals.append(v.mean())
        errs.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
        cols.append(th.METHOD_COLORS["ablation"])
    levels.append("Leiden\nclusters"); vals.append(A6_LEIDEN_ARI); errs.append(0)
    cols.append(th.METHOD_COLORS["null"])
    pos = np.arange(len(levels))
    ax.bar(pos, vals, 0.65, yerr=errs, capsize=2, color=cols)
    ax.axhline(0.9, ls=":", lw=0.9, color=th.INK)
    ax.set_xticks(pos); ax.set_xticklabels(levels, fontsize=th.PT["annot"])
    ax.set_ylabel("agreement between seeds\n(r, or ARI for clusters)")
    ax.set_ylim(0, 1.05)
    th.despine(ax)


# ── program x cell-type map, train vs held-out (A7) ──────────────────────────
IMPUTE_A7 = f"{IMPUTE_DIR}/A7_prog_ct_K10.npz"


_ACRONYM = {"nk": "NK", "b": "B", "t": "T"}


def _ct_label(t):
    """Sentence-case an atlas cell-type label, keeping acronyms .capitalize() would mangle."""
    w = str(t).lower().split()
    return " ".join([_ACRONYM.get(w[0], w[0].capitalize())] + w[1:]) if w else str(t)


def load_program_celltype():
    if not os.path.exists(IMPUTE_A7):
        raise FileNotFoundError(f"{IMPUTE_A7} missing — run A7_program_celltype_map.py")
    z = np.load(IMPUTE_A7, allow_pickle=True)
    return dict(types=list(z["types"]), train=z["train"], test=z["test"], K=int(z["K"]))


def panel_program_celltype(axes, pc: dict, cbar: bool = True) -> None:
    """Mean loading of each program within each annotated cell type, for training and held-out cells.

    Reported instead of an agreement score against a discrete clustering: cell-type labels come from
    gene expression, so a single ARI penalises any splicing structure that does not coincide with
    cell identity, whereas this shows how the representation relates to cell identity without
    asserting the two should match. Both panels come from the same model, so the programs are the
    same objects and the comparison is direct: whether held-out cells use them as training cells do.
    """
    types, K = pc["types"], pc["K"]
    vmax = max(pc["train"].max(), pc["test"].max())
    for ax, key, title in zip(axes, ("train", "test"),
                              ("Training cells", "Held-out cells")):
        im = ax.imshow(pc[key], aspect="auto", cmap=th.CMAP["sequential"], vmin=0, vmax=vmax)
        # Numbered rather than "P1..P10": at K=30 the prefixed labels collide, and the axis label
        # already says what the numbers are.
        ax.set_xticks(range(K)); ax.set_xticklabels(range(1, K + 1), fontsize=th.PT["annot"])
        ax.set_title(title)
        ax.set_xlabel("splicing program")
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
    axes[0].set_yticks(range(len(types)))
    # atlas annotations are stored upper-case; sentence-case them to match every other panel,
    # keeping acronyms that .capitalize() would otherwise mangle ("NK CELL" -> "Nk cell").
    axes[0].set_yticklabels([_ct_label(t) for t in types], fontsize=th.PT["annot"] - 1)
    axes[1].set_yticks([])
    if cbar:
        cb = axes[1].figure.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=th.PT["annot"])
        cb.set_label("mean loading", fontsize=th.PT["annot"])
    r = np.corrcoef(pc["train"].ravel(), pc["test"].ravel())[0, 1]
    axes[1].text(1.02, -0.13, f"train vs held-out r = {r:.3f}", transform=axes[1].transAxes,
                 ha="right", fontsize=th.PT["annot"], color=th.INK)

# ── Figure 2 panels (simulation) ─────────────────────────────────────────────
def panel_sim_metric(ax, bench: pd.DataFrame, metric: str = "psi_r",
                     series=SIM_SERIES, prior: bool | None = True, title: str = "") -> None:
    """A simulation metric vs the fraction of differential ATSEs, LeafletFA against NMF.

    `prior` selects the LeafletFA junction-prior arm (True = the reference config); NMF rows carry
    junc_prior=False and are always kept, since the prior does not apply to them.
    """
    fr = sorted(bench.frac.unique())
    for model, arm, lab, ckey in series:
        s = bench[(bench.model == model) & (bench.likelihood == arm)]
        if model == "LeafletFA" and prior is not None:
            s = s[s.junc_prior.astype(str) == str(prior)]
        if not len(s):
            continue
        m = [_ms(s[s.frac == f], metric)[0] for f in fr]
        e = [_ms(s[s.frac == f], metric)[1] for f in fr]
        ax.errorbar([f * 100 for f in fr], m, yerr=e, fmt="o-", ms=4, capsize=2,
                    label=lab, color=th.METHOD_COLORS[ckey])
    ax.set_xticks([f * 100 for f in fr])
    ax.set_xlabel("differential ATSEs (%)")
    ax.set_ylabel(SIM_METRICS.get(metric, metric))
    if title:
        ax.set_title(title, fontsize=th.PT["annot"])
    th.despine(ax)


def panel_recovery_combined(ax, bench: pd.DataFrame,
                            metrics=("psi_r", "spearman_delta"),
                            test_against=("NMF", "0-impute")) -> None:
    """Recovery of the truth, pooled over the %-differential sweep.

    Panels A and B used to plot psi recovery and effect-size recovery against the fraction of
    differential ATSEs, but that axis is not what those metrics respond to -- it just split one
    comparison across two panels. Here the sweep levels and seeds are treated as replicates
    (n = 3 fracs x 3 seeds) and the two recovery metrics sit side by side, so the LeafletFA vs NMF
    comparison is a single readable panel with a test on it.
    """
    groups = np.arange(len(metrics))
    w = 0.8 / len(SIM_SERIES)
    xs, subs = {}, {}
    for i, (model, arm, lab, ckey) in enumerate(SIM_SERIES):
        s = _arm(bench, model, arm)
        m = [_ms(s, mt)[0] for mt in metrics]
        e = [_ms(s, mt)[1] for mt in metrics]
        x = groups + (i - (len(SIM_SERIES) - 1) / 2) * w
        ax.bar(x, m, w, yerr=e, capsize=2, label=lab, color=th.METHOD_COLORS[ckey])
        xs[i], subs[i] = x, s
    ax.set_xticks(groups)
    ax.set_xticklabels([SIM_METRICS.get(mt, mt).replace("\n", " ") for mt in metrics])
    ax.set_ylabel("correlation with simulated truth")
    ax.set_title("Do the fitted programs match the ones that were simulated?\n"
                 "pooled over 10/50/90% differential ATSEs x 3 seeds (n = 9)",
                 fontsize=th.PT["annot"], pad=26)

    # ALL pairwise comparisons within each metric, stacked shortest-span-first so the brackets
    # nest instead of crossing. Non-significant pairs are labelled 'ns' rather than dropped —
    # showing only the significant ones would hide the comparisons that did not separate.
    n = len(SIM_SERIES)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pairs.sort(key=lambda p: p[1] - p[0])
    span = ax.get_ylim()[1]
    for gi, mt in enumerate(metrics):
        tops = [_ms(subs[i], mt)[0] + _ms(subs[i], mt)[1] for i in range(n)]
        y = max(tops) + 0.05 * span
        for (i, j) in pairs:
            p = ttest(_vals(subs[i], mt), _vals(subs[j], mt))
            if not np.isfinite(p):
                continue
            th.sig_bracket(ax, xs[i][gi], xs[j][gi], y, p)
            y += 0.055 * span
    # Psi recovery and Spearman(delta-Psi) are both correlations, so the DATA cannot exceed 1.0;
    # only the annotation stack may. Clamp the tick range to [0, 1] and let the brackets sit in the
    # margin above, so the axis never implies a correlation greater than one.
    ax.set_ylim(0, max(1.0, y) * 1.01)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    th.despine(ax)


def panel_atse_illustration(axes, data: dict) -> None:
    """Two simulated cassette exons: one whose splicing differs across programs, one that does not.

    Each ATSE is 3 junctions -- two inclusion (J1, J2) and one skip (J3) -- whose usages sum to 1
    within the event, so a stacked bar per program IS the full ground truth for that cassette.
    """
    jcol = [th.PALETTE["indigo"], th.PALETTE["teal"], th.PALETTE["ochre"]]
    names = ["J1 (inclusion)", "J2 (inclusion)", "J3 (skip)"]
    for ax, key, title in zip(axes, ("psi_diff", "psi_null"),
                              ("differential ATSE", "non-differential ATSE")):
        P = np.asarray(data[key])                      # (K, 3) usage per program
        K = P.shape[0]
        x = np.arange(K)
        bottom = np.zeros(K)
        for j in range(3):
            ax.bar(x, P[:, j], 0.7, bottom=bottom, color=jcol[j],
                   label=names[j] if key == "psi_diff" else None)
            bottom += P[:, j]
        d = float(data["delta_diff" if key == "psi_diff" else "delta_null"])
        ax.set_title(f"{title}\n$\\Delta\\Psi$ = {d:.2f}")
        ax.set_xticks(x); ax.set_xticklabels([f"P{i+1}" for i in range(K)])
        ax.set_ylim(0, 1); ax.set_xlabel("splicing program")
        th.despine(ax)
    axes[0].set_ylabel("junction usage within ATSE")
    # the labelled artists live on axes[0]; the bars fill both panels and the right-hand side is
    # occupied, so the key goes in the row gap, centred under the pair
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, frameon=False, fontsize=th.PT["legend"], ncol=3,
                   loc="upper center", bbox_to_anchor=(1.10, -0.30), columnspacing=1.0,
                   handlelength=1.0)


def panel_umaps(axes, data: dict, keys=("emb_true", "emb_lfa", "emb_nmf0"),
                titles=("simulated (true $\\phi$)", "LeafletFA", "NMF (0-impute)")) -> None:
    """Cells embedded from the true loadings and from what each method recovered, coloured by the
    true program. If a method recovers the simulated structure, its panel reproduces the grouping."""
    lab = np.asarray(data["true_prog"])
    progs = np.unique(lab)
    cmap = plt.get_cmap("tab10")
    for ai, (ax, k, t) in enumerate(zip(axes, keys, titles)):
        if k not in data:
            ax.set_visible(False)
            continue
        e = np.asarray(data[k])
        for p in progs:
            sel = lab == p
            ax.scatter(e[sel, 0], e[sel, 1], color=cmap(int(p) % 10), s=1.2, linewidths=0,
                       alpha=0.8, label=f"P{int(p)+1} (n={int(sel.sum()):,})" if ai == 0 else None)
        # silhouette computed on the LATENT space (phi / W), not on these 2-D coordinates — UMAP
        # exaggerates separation, so scoring the embedding would measure UMAP, not the model.
        sil_key = "sil_" + k.replace("emb_", "")
        ax.set_title(f"{t}\nsilhouette = {float(data[sil_key]):.2f}" if sil_key in data else t)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2" if ai == 0 else "")
        ax.set_xticks([]); ax.set_yticks([])
        # keep a full box so each embedding reads as its own panel
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(0.6)
    axes[0].legend(frameon=False, fontsize=th.PT["legend"], markerscale=4,
                   loc="upper left", bbox_to_anchor=(0.0, -0.06), ncol=3,
                   handletextpad=0.2, columnspacing=0.8)


def panel_sim_prior(ax, bench: pd.DataFrame, metric: str = "precision") -> None:
    """Is the junction-specific psi prior load-bearing? LeafletFA only, by likelihood."""
    arms = [("bb", "Beta-Binomial", "LeafletFA"), ("binom", "Binomial", "ablation")]
    x = np.arange(2)
    w = 0.36
    cells, xs = {}, {}
    for i, (arm, lab, ckey) in enumerate(arms):
        s = bench[(bench.model == "LeafletFA") & (bench.likelihood == arm)]
        groups = [s[s.junc_prior.astype(str) == p] for p in ("False", "True")]
        m = [_ms(g, metric)[0] for g in groups]
        e = [_ms(g, metric)[1] for g in groups]
        pos = x + (i - 0.5) * w
        ax.bar(pos, m, w, yerr=e, capsize=2, label=lab, color=th.METHOD_COLORS[ckey])
        for j, prior in enumerate(("global", "junction")):
            cells[(arm, prior)] = groups[j]
            xs[(arm, prior)] = pos[j]
    # ALL pairwise comparisons across the four bars, shortest span first so brackets nest rather
    # than cross. Previously only the within-likelihood prior comparison was drawn, which left the
    # two brackets sitting at the same height across the whole panel and reading as if they linked
    # the wrong bars.
    keys = [(a, p) for a, _, _ in arms for p in ("global", "junction")]
    pairs = sorted([(k1, k2) for i, k1 in enumerate(keys) for k2 in keys[i + 1:]],
                   key=lambda kp: abs(xs[kp[0]] - xs[kp[1]]))
    top = max(_ms(cells[k], metric)[0] + _ms(cells[k], metric)[1] for k in keys)
    y = top * 1.06
    for k1, k2 in pairs:
        p = ttest(_vals(cells[k1], metric), _vals(cells[k2], metric))
        if not np.isfinite(p):
            continue
        th.sig_bracket(ax, xs[k1], xs[k2], y, p)
        y += 0.09 * top
    ax.set_xticks(x)
    ax.set_xticklabels(["global\nprior", "junction-specific\nprior"])
    ax.set_ylabel(SIM_METRICS.get(metric, metric))
    ax.set_ylim(0, min(1.0, y * 1.04) if metric in ("precision", "auroc", "ari", "sil") else y * 1.04)
    th.despine(ax)


def panel_umap(ax, emb: np.ndarray, labels: np.ndarray, title: str,
               cmap: str = "tab10", s: float = 2.0) -> None:
    """Scatter a 2-D embedding coloured by true program."""
    ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap=cmap, s=s, linewidths=0, alpha=0.85)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)


def umap_of(mat: np.ndarray, seed: int = 0) -> np.ndarray:
    """2-D UMAP of a cell x factor matrix (falls back to PCA if umap-learn is unavailable)."""
    try:
        import umap
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(mat)
    except Exception:
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(mat)


# ── Figure 3 panels (atlas) ──────────────────────────────────────────────────
def panel_imputation_vs_coverage(ax, imp: pd.DataFrame, K: int = 20,
                                 metric: str = "median_pearson") -> None:
    d = imp[imp.K_cfg == K]
    cov = sorted(pd.to_numeric(d.downsampling_pct, errors="coerce").dropna().unique())
    for suf, (lab, ckey) in METHODS.items():
        col = f"{metric}_{suf}"
        if col not in d:
            continue
        m = [_ms(d[d.downsampling_pct == c], col)[0] for c in cov]
        e = [_ms(d[d.downsampling_pct == c], col)[1] for c in cov]
        ax.errorbar(cov, m, yerr=e, fmt="o", ls="--" if suf.endswith("_tr") else "-",
                    ms=3.5, capsize=2, label=lab, color=th.METHOD_COLORS[ckey])
    ax.set_xlabel("reads retained (%)")
    ax.set_ylabel(metric.replace("_", " ") + " (per junction)")
    th.despine(ax)


def panel_calibration_vs_coverage(ax, imp: pd.DataFrame, K: int = 20) -> None:
    d = imp[imp.K_cfg == K]
    cov = sorted(pd.to_numeric(d.downsampling_pct, errors="coerce").dropna().unique())
    for suf, (lab, ckey) in METHODS.items():
        col = f"calib_slope_{suf}"
        if col not in d:
            continue
        m = [_ms(d[d.downsampling_pct == c], col)[0] for c in cov]
        e = [_ms(d[d.downsampling_pct == c], col)[1] for c in cov]
        ax.errorbar(cov, m, yerr=e, fmt="o-", ms=3.5, capsize=2, label=lab,
                    color=th.METHOD_COLORS[ckey])
    ax.axhline(1.0, ls="--", lw=0.9, color=th.INK)
    ax.set_xlabel("reads retained (%)")
    ax.set_ylabel("calibration slope")
    th.despine(ax)


def panel_structure(ax, struc: pd.DataFrame,
                    metric: str = "ari_leiden_vs_broad_cell_type") -> None:
    d = struc[struc.metric == metric]
    Ks = sorted(d.K_cfg.unique())
    w = 0.8 / max(len(STRUCT_SOURCES), 1)
    for i, (src, (lab, ckey)) in enumerate(STRUCT_SOURCES.items()):
        s = d[d.phi_source == src]
        if not len(s):
            continue
        m = [_ms(s[s.K_cfg == k], "value")[0] for k in Ks]
        e = [_ms(s[s.K_cfg == k], "value")[1] for k in Ks]
        x = np.arange(len(Ks)) + (i - (len(STRUCT_SOURCES) - 1) / 2) * w
        ax.bar(x, m, w, yerr=e, capsize=2, label=lab, color=th.METHOD_COLORS[ckey])
    ax.set_xticks(np.arange(len(Ks)))
    ax.set_xticklabels([f"K={k}" for k in Ks])
    ax.set_ylabel(metric.replace("_", " "))
    th.despine(ax)


# ── whole-figure drafts ──────────────────────────────────────────────────────
# All Figure 2 inputs live in ../data/ -- see code/collect_inputs.py, which copies them in from
# $WD/SIMULATION and ~/LeafletFA/experiments and records provenance in data/INPUTS.json. Reading
# them by absolute path from outside the repo meant the figures built only on this filesystem.
SIM_RECOVER_DIR = _HERE


def load_recover10() -> pd.DataFrame:
    """Over-specification / minibatch runs (2026-07-26) on a K_true = 10 simulation.

    IMPORTANT: use this — not the `batching` column of sweep_results.csv — for any minibatch
    comparison. That sweep predates the Pyro-plate rewrite, so its 'chunked' arm was produced by the
    implementation in which subsampling was silently inactive; it does not describe the current
    minibatch trainer. These runs use the fixed plate implementation with early stopping.
    """
    frames = []
    for km in (10, 20, 30):
        f = f"{SIM_RECOVER_DIR}/sim_recover10_Kmodel{km}.csv"
        if os.path.exists(f):
            frames.append(pd.read_csv(f))
    if not frames:
        raise FileNotFoundError(f"no sim_recover10_Kmodel*.csv in {SIM_RECOVER_DIR}")
    return pd.concat(frames, ignore_index=True)


def panel_minibatch_vs_full(ax, rec: pd.DataFrame) -> None:
    """Does minibatch training cost anything, across fitted K? (fixed plate implementation)"""
    Ks = sorted(rec.initial_K.unique())
    ref = rec[(rec.likelihood == "betabinomial") & (rec.junc_prior)]
    for val, lab, ckey in [("full", "full data", "LeafletFA"), ("mb8192", "minibatch B=8192", "NMF")]:
        s = ref[ref.batch == val]
        m = [_ms(s[s.initial_K == k], "recovered_true")[0] for k in Ks]
        e = [_ms(s[s.initial_K == k], "recovered_true")[1] for k in Ks]
        ax.errorbar(Ks, m, yerr=e, fmt="o-", ms=4, capsize=2, label=lab,
                    color=th.METHOD_COLORS[ckey])
    # Recovery is a correlation, so 1.0 is a hard ceiling — brackets must not push the axis past it.
    # The differences here are ~0.4% (0.999 vs 0.995): significant, but tiny. Annotate the star and
    # the actual gap as text between the two series instead of stacking brackets above them.
    # NO significance stars here, deliberately. Full-batch recovery is reproducible to +/-0.00001,
    # so a 0.4% gap sits ~25 standard deviations out and is "highly significant" while being
    # practically nil -- every value is >= 0.995. Reporting the largest raw difference states what
    # the panel actually shows: minibatch training costs essentially nothing.
    gaps = []
    for k in Ks:
        a = _vals(ref[(ref.batch == "full") & (ref.initial_K == k)], "recovered_true")
        b = _vals(ref[(ref.batch == "mb8192") & (ref.initial_K == k)], "recovered_true")
        if len(a) and len(b):
            gaps.append(a.mean() - b.mean())
    if gaps:
        ax.text(0.5, 0.90, f"largest difference {max(gaps):.3f}", transform=ax.transAxes,
                ha="center", fontsize=th.PT["annot"], color=th.INK)
    ax.set_xticks(Ks); ax.set_xlabel("fitted K")
    ax.set_ylabel("recovery of 10 true programs")
    # Auto-scaling put the axis on [0.995, 1.000], which turns a 0.4% gap into a visual chasm.
    # A wider range shows what is true: every configuration recovers the programs essentially
    # perfectly, and full-batch vs minibatch is indistinguishable in practice.
    ax.set_ylim(0.95, 1.002)
    th.despine(ax)


def panel_pruning(ax, rec: pd.DataFrame) -> None:
    """Over-specified K: how many programs survive pruning when only 10 are real?"""
    Ks = sorted(rec.initial_K.unique())
    ref = rec[(rec.batch == "full") & (rec.likelihood == "betabinomial") & (rec.junc_prior)]
    ax.plot(Ks, Ks, ls="--", lw=0.9, color=th.INK, label="fitted K (no pruning)")
    for col, lab, ckey in [("pruned_K_pi", "retained ($\\pi$ > 0.01)", "null"),
                           ("pruned_K_phi", "retained ($\\phi$ usage)", "LeafletFA")]:
        m = [_ms(ref[ref.initial_K == k], col)[0] for k in Ks]
        e = [_ms(ref[ref.initial_K == k], col)[1] for k in Ks]
        ax.errorbar(Ks, m, yerr=e, fmt="o-", ms=4, capsize=2, label=lab,
                    color=th.METHOD_COLORS[ckey])
    ax.axhline(10, ls=":", lw=1.0, color=th.METHOD_COLORS["ablation"], label="true K = 10")
    ax.set_xticks(Ks); ax.set_xlabel("fitted K"); ax.set_ylabel("programs retained")
    th.despine(ax)


def load_panel_data() -> dict:
    """ATSE illustration + UMAP embeddings precomputed by prep_panels.py."""
    f = f"{_HERE}/panel_data.npz"
    if not os.path.exists(f):
        raise FileNotFoundError(f"{f} missing — run `python prep_panels.py` first")
    z = np.load(f, allow_pickle=True)
    return {k: np.asarray(z[k]) for k in z.files}


def figure_sim(save: bool = True):
    """Fig 2: what was simulated (A, B), what each method recovered (C, D), and how well (E, F)."""
    th.apply_theme()
    bench = load_benchmark()
    data = load_panel_data()
    fig = plt.figure(figsize=th.figsize("double", 4.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.68, wspace=0.42,
                          top=0.90, bottom=0.10, left=0.085, right=0.985)

    # top row: the headline comparison, then how it scales with the amount of signal
    ax_rec = fig.add_subplot(gs[0, :2])
    panel_recovery_combined(ax_rec, bench)
    th.panel_label(ax_rec, "A", dx=-0.09)

    ax_sil = fig.add_subplot(gs[0, 2])
    panel_sim_metric(ax_sil, bench, "sil",
                     title="Do cells separate by their true program\nas more ATSEs are made differential?")
    th.panel_label(ax_sil, "B", dx=-0.24)

    # bottom row: the structure each method recovered, against the structure that was simulated
    ax_u = [fig.add_subplot(gs[1, i]) for i in range(3)]
    panel_umaps(ax_u, data)
    th.panel_label(ax_u[0], "C", dx=-0.12, dy=1.16)

    handles, labels = ax_rec.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, fontsize=th.PT["legend"],
               loc="upper center", bbox_to_anchor=(0.5, 1.0))

    if save:
        th.save(fig, f"{_OUT}/FIG2_simulation_draft")
    return fig


def figure_validation(save: bool = True, K: int = 20):
    """FIGURE 2 — validation: does the model recover truth, and does it hold up on real data?

    Follows the Fig 1 schematic, whose 'Validation' column is simulation AND the imputation /
    downsampling analysis on the 70/30 split. Three rows:

        A  program recovery on simulated cassettes (Psi and effect size), all pairwise tests
        B  cell separation by true program vs how much differential splicing was planted
        C  the simulated structure and what each method recovered (UMAP + latent silhouette)
        D  held-out imputation accuracy vs read coverage
        E  calibration slope vs read coverage (1.0 = calibrated, >1 = predictions compressed)

    Rows A-C are the controlled experiment (known truth); rows D-E are the real atlas. Held-out
    latent structure lives in the supplement — it answers a third question (does the dictionary
    generalise to unseen cells) rather than either of these.
    """
    th.apply_theme()
    bench, data = load_benchmark(), load_panel_data()
    fig = plt.figure(figsize=th.figsize("double", 6.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 1.0, 1.0], hspace=0.85, wspace=0.45,
                          top=0.93, bottom=0.065, left=0.085, right=0.985)

    ax_rec = fig.add_subplot(gs[0, :2])
    panel_recovery_combined(ax_rec, bench)
    th.panel_label(ax_rec, "A", dx=-0.09)

    ax_sil = fig.add_subplot(gs[0, 2])
    panel_sim_metric(ax_sil, bench, "sil",
                     title="Do cells separate by their true program\nas more ATSEs are made differential?")
    th.panel_label(ax_sil, "B", dx=-0.26)

    ax_u = [fig.add_subplot(gs[1, i]) for i in range(3)]
    panel_umaps(ax_u, data)
    th.panel_label(ax_u[0], "C", dx=-0.12, dy=1.20)

    try:
        imp = load_imputation()
        n_cov = imp.downsampling_pct.nunique()
        ax_imp = fig.add_subplot(gs[2, 0])
        panel_imputation_vs_coverage(ax_imp, imp, K=K)
        ax_imp.legend(frameon=False, fontsize=th.PT["legend"])
        th.panel_label(ax_imp, "D", dx=-0.26)
        ax_cal = fig.add_subplot(gs[2, 1])
        panel_calibration_vs_coverage(ax_cal, imp, K=K)
        th.panel_label(ax_cal, "E", dx=-0.26)
        if n_cov < 2:
            print(f"  !! only {n_cov} coverage level available — D/E are single points until "
                  f"04e_allcov.csv lands")
    except FileNotFoundError as e:
        print(f"  (imputation rows skipped: {e})")

    handles, labels = ax_rec.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, fontsize=th.PT["legend"],
               loc="upper center", bbox_to_anchor=(0.5, 1.0))

    if save:
        th.save(fig, f"{_OUT}/FIG2_validation")
    return fig



def figure_simulation(save: bool = True):
    """FIGURE 2 (simulation half) — one row of quantitative panels.

    The UMAPs moved to the supplement: they are illustrative rather than quantitative, and the same
    information is carried numerically by the silhouettes in B. Panels are plotted against the size
    of the simulated splicing difference (median true |delta Psi|), the axis recovery actually
    responds to.

    A  effect-size recovery   -- the headline: can the method recover HOW MUCH splicing differs?
    B  cell separation        -- does the learned space group cells by their true program?
    C  Psi recovery           -- see the caption: at small effects a model predicting junction means
                                 scores well here while having learned nothing.
    D  likelihood x overdispersion -- why the Beta-Binomial is used.
    """
    th.apply_theme()
    eff = load_effect_sweep()
    fig = plt.figure(figsize=th.figsize("double", 2.9))
    gs = fig.add_gridspec(1, 4, wspace=0.58, top=0.80, bottom=0.26, left=0.070, right=0.965)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_effect_sweep(ax_a, eff, "spearman_delta")
    th.panel_label(ax_a, "A", dx=-0.34)

    ax_b = fig.add_subplot(gs[0, 1])
    panel_effect_sweep(ax_b, eff, "sil")
    th.panel_label(ax_b, "B", dx=-0.34)

    ax_c = fig.add_subplot(gs[0, 2])
    panel_effect_sweep(ax_c, eff, "psi_r")
    th.panel_label(ax_c, "C", dx=-0.34)

    ax_d = fig.add_subplot(gs[0, 3])
    try:
        panel_overdispersion(ax_d, load_overdispersion(), "spearman_delta")
        ax_d.legend(frameon=False, fontsize=th.PT["legend"], loc="upper right", handlelength=1.0)
    except FileNotFoundError as e:
        print(f"  (overdispersion panel skipped: {e})")
        panel_prior_vs_effect(ax_d, eff, "psi_r")
        ax_d.legend(frameon=False, fontsize=th.PT["legend"], loc="lower right")
    th.panel_label(ax_d, "D", dx=-0.34)

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, fontsize=th.PT["legend"],
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    if save:
        th.save(fig, f"{_OUT}/FIG2_simulation")
    return fig


def figure_heldout_structure_supp(save: bool = True):
    """Supplement to Fig 2: does the frozen dictionary generalise to cells never seen in training?"""
    th.apply_theme()
    struc = load_structure()
    fig, ax = plt.subplots(1, 3, figsize=th.figsize("double", 2.7))
    for a, metric, letter in zip(ax,
                                 ("ari_leiden_vs_broad_cell_type", "silhouette_by_leiden",
                                  "leiden_purity_broad_ct"), "ABC"):
        panel_structure(a, struc, metric)
        a.legend(frameon=False, fontsize=th.PT["legend"])
        th.panel_label(a, letter)
    fig.tight_layout()
    if save:
        th.save(fig, f"{_OUT}/FIG2_supp_heldout_structure")
    return fig


def figure_sim_supp(save: bool = True):
    """Fig 2 supplement: what the simulation looks like, the sweep resolved per metric, the prior
    ablation, and the training-choice checks (minibatch, over-specified K)."""
    th.apply_theme()
    bench, data = load_benchmark(), load_panel_data()
    fig = plt.figure(figsize=th.figsize("double", 5.6))
    gs = fig.add_gridspec(3, 4, hspace=0.85, wspace=0.5,
                          top=0.95, bottom=0.07, left=0.075, right=0.985)

    # row 1 — the ground truth: one cassette that differs across programs, one that does not
    ax_d, ax_n = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    panel_atse_illustration([ax_d, ax_n], data)
    th.panel_label(ax_d, "A", dx=-0.42)
    ax_u = [fig.add_subplot(gs[0, i]) for i in (2, 3)]
    panel_umaps(ax_u, data, keys=("emb_true", "emb_lfa"),
                titles=("Simulated (true $\\phi$)", "LeafletFA"))
    th.panel_label(ax_u[0], "B", dx=-0.16, dy=1.16)

    # row 2 — the %-differential sweep, resolved metric by metric
    for i, mt in enumerate(("psi_r", "spearman_delta", "auroc", "ari")):
        a = fig.add_subplot(gs[1, i])
        panel_sim_metric(a, bench, mt)
        th.panel_label(a, "CDEF"[i], dx=-0.42)
        if i == 0:
            a.legend(frameon=False, fontsize=th.PT["legend"], loc="lower left")

    # row 3 — training choices. The minibatch/K panels come from the 2026-07-26 runs on the
    # FIXED plate implementation, not the pre-rewrite `batching` column of sweep_results.csv.
    a_pr = fig.add_subplot(gs[2, 0])
    try:   # the prior only separates under a likelihood that models the noise -- show that directly
        panel_prior_interaction(a_pr, load_prior_test(), "psi_r")
        a_pr.legend(frameon=False, fontsize=th.PT["legend"], loc="upper center",
                    bbox_to_anchor=(0.5, -0.28), handlelength=1.0)
    except FileNotFoundError:
        try:
            panel_prior_vs_effect(a_pr, load_effect_sweep(), "psi_r")
        except FileNotFoundError:
            panel_sim_prior(a_pr, bench, "precision")
    th.panel_label(a_pr, "G", dx=-0.42)
    try:
        rec = load_recover10()
        a_mb = fig.add_subplot(gs[2, 1])
        panel_minibatch_vs_full(a_mb, rec)
        a_mb.legend(frameon=False, fontsize=th.PT["legend"], loc="lower left", handlelength=1.0)
        th.panel_label(a_mb, "H", dx=-0.42)
        a_rep = fig.add_subplot(gs[2, 2])
        try:
            panel_seed_consistency(a_rep, load_seed_consistency())
            th.panel_label(a_rep, "I", dx=-0.42)
        except FileNotFoundError as e:
            print(f"  (seed-consistency panel skipped: {e})")
    except FileNotFoundError as e:
        print(f"  (minibatch panel skipped: {e})")

    if save:
        th.save(fig, f"{_OUT}/FIG2_supp_draft")
    return fig


def figure_imputation(K: int = 20, save: bool = True):
    """Fig 3 draft: imputation accuracy, calibration, held-out structure."""
    th.apply_theme()
    imp, struc = load_imputation(), load_structure()
    fig, ax = plt.subplots(1, 4, figsize=th.figsize("double", 2.7))
    panel_imputation_vs_coverage(ax[0], imp, K=K)
    panel_calibration_vs_coverage(ax[1], imp, K=K)
    # held-out structure, read two ways: agreement with cell-type labels (ARI) and how well separated
    # the clusters are on their own terms (silhouette). ARI can look good while the embedding is
    # diffuse, so both are reported.
    panel_structure(ax[2], struc, "ari_leiden_vs_broad_cell_type")
    panel_structure(ax[3], struc, "silhouette_by_leiden")
    for a, letter in zip(ax, "ABCD"):
        a.legend(frameon=False, fontsize=th.PT["legend"])
        th.panel_label(a, letter)
    fig.tight_layout()
    if save:
        th.save(fig, f"{_OUT}/FIG2_imputation")
    return fig


TASKS_BY_K = {10: (0, 1, 2), 20: (3, 4, 5), 30: (6, 7, 8)}


def load_program_celltype_task(task):
    p = f"{IMPUTE_DIR}/A7_prog_ct_task{task}.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return dict(types=[str(t) for t in z["types"]], train=z["train"], test=z["test"],
                K=int(z["K"]), task=task)


def _align_to(ref, m):
    """Order m's programs to match ref's, by Hungarian matching on the cell-type profiles.

    Only ever used to line up DIFFERENT SEEDS side by side. Programs are exchangeable, so two fits
    label the same program differently and an unpermuted figure would look like disagreement that is
    purely cosmetic. The SAME permutation is applied to that seed's train and test maps, so the
    train-vs-held-out comparison the figure is actually making is untouched by it.
    """
    from scipy.optimize import linear_sum_assignment
    A, B = ref["train"], m["train"]
    C = np.zeros((A.shape[1], B.shape[1]))
    for i in range(A.shape[1]):
        for j in range(B.shape[1]):
            a, b = A[:, i], B[:, j]
            C[i, j] = 0.0 if (a.std() < 1e-12 or b.std() < 1e-12) else np.corrcoef(a, b)[0, 1]
    _, col = linear_sum_assignment(-np.nan_to_num(C))
    return {**m, "train": m["train"][:, col], "test": m["test"][:, col]}


def load_transfer_controls(K=10):
    c = f"{IMPUTE_DIR}/A9_K{K}_controls.csv"
    g = f"{IMPUTE_DIR}/A9_K{K}_gram.npz"
    if not (os.path.exists(c) and os.path.exists(g)):
        raise FileNotFoundError(f"{c} / {g} missing — run A9_transfer_controls.py")
    return pd.read_csv(c), np.load(g, allow_pickle=True)


def panel_transfer_controls(ax, ctl) -> None:
    """Train-vs-held-out agreement against the three reference points that give it a scale.

    A correlation of 0.998 means nothing on its own. The bar that matters is the SAMPLING FLOOR: the
    same model and the same training cells, subsampled to the held-out set's size. Transfer sitting on
    that floor says generalising to unseen cells costs no more than looking at fewer cells. The two
    nulls stop that being vacuous — shuffling the cell-type labels collapses the agreement, and a
    different seed's held-out map (programs matched) scores far lower, so the map is model-specific.
    """
    keys = [("sampling_floor", "Sampling floor", th.PALETTE["grey"]),
            ("transfer", "Train vs held-out", th.PALETTE["teal"]),
            ("cross_model", "Different seed", th.PALETTE["ochre"]),
            ("shuffled", "Shuffled labels", th.PALETTE["rose"])]
    x = np.arange(len(keys))
    m = [ctl[k].mean() for k, _, _ in keys]
    e = [ctl[k].std(ddof=1) if len(ctl) > 1 else 0.0 for k, _, _ in keys]
    ax.bar(x, m, yerr=e, color=[c for _, _, c in keys], width=0.68,
           error_kw=dict(lw=1.0, capsize=3, ecolor=th.INK))
    for xi, (k, _, _) in zip(x, keys):
        for v in ctl[k]:
            ax.plot(xi, v, "o", ms=2.6, mfc="white", mec=th.INK, mew=0.7, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab, _ in keys], rotation=30, ha="right")
    ax.set_ylabel(r"agreement in mean $\phi$ per cell type ($r$)")
    ax.set_ylim(0, 1.08)
    ax.axhline(ctl["sampling_floor"].mean(), color=th.PALETTE["grey"], lw=0.8, ls=":", zorder=0)
    gap = ctl["sampling_floor"].mean() - ctl["transfer"].mean()
    ax.text(0.5, 1.045, f"transfer costs {gap:.4f}", ha="center", fontsize=th.PT["annot"],
            color=th.INK)
    ax.plot([0, 1], [1.028, 1.028], lw=0.8, color=th.INK, clip_on=False)
    # Psi is frozen in the E-step, so it cannot differ between train and held-out; everything on this
    # axis is phi. Cross-seed Psi agreement is a separate analysis (A6).
    ax.text(0.5, -0.42, r"all comparisons are on $\phi$; $\Psi$ is frozen by construction",
            transform=ax.transAxes, ha="center", fontsize=th.PT["annot"] - 1, color=th.PALETTE["grey"])


def panel_celltype_gram(axes, gz, cbar_ax=None) -> None:
    """Cell-type x cell-type similarity, one panel per seed. NO program matching anywhere.

    Programs are exchangeable, so two fits number them differently and any direct comparison needs an
    alignment we would have to choose ourselves. Correlating cell types' loading profiles removes the
    problem rather than papering over it: permuting the programs leaves this matrix exactly unchanged,
    because it only asks whether two cell types use the programs alike. Cell-type order is shared
    across panels (hierarchical clustering of the first seed) — legitimate, because cell types, unlike
    programs, are not arbitrary labels.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    types = [str(t) for t in gz["types"]]
    seeds = list(gz["seeds"])
    tasks = sorted({int(k.split("_")[-1]) for k in gz.files if k.startswith("gram_train_")})
    G0 = gz[f"gram_train_{tasks[0]}"]
    D = np.clip(1 - G0, 0, 2); np.fill_diagonal(D, 0)
    order = leaves_list(linkage(squareform(D, checks=False), method="average"))
    for ax, t, sd in zip(axes, tasks, seeds):
        G = gz[f"gram_train_{t}"][np.ix_(order, order)]     # training cells
        im = ax.imshow(G, cmap=th.CMAP["diverging"], vmin=-1, vmax=1, aspect="equal")
        ax.set_title(f"Seed {sd}, training")
        ax.set_xticks([]);  ax.set_yticks(range(len(types)))
        ax.set_yticklabels([_ct_label(types[i]) for i in order] if ax is axes[0] else [],
                           fontsize=th.PT["annot"] - 2)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
    if cbar_ax is not None:
        cb = axes[0].figure.colorbar(im, cax=cbar_ax)
        cb.set_label(r"similarity of $\phi$ profiles ($r$)", fontsize=th.PT["annot"])
        cb.ax.tick_params(labelsize=th.PT["annot"])


def figure_transfer_controls(K: int = 10, save: bool = True):
    """Does the representation transfer to unseen cells, and do independent fits agree?

    Answers the two objections the raw heatmaps cannot: that train and held-out look identical because
    nothing is being tested, and that seeds cannot be compared because their programs are ordered
    differently.
    """
    th.apply_theme()
    ctl, gz = load_transfer_controls(K)
    fig = plt.figure(figsize=th.figsize("double", 6.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.35], width_ratios=[1, 1, 1, 0.05],
                          hspace=0.42, wspace=0.12, left=0.20, right=0.93)
    ax_a = fig.add_subplot(gs[0, :2])
    panel_transfer_controls(ax_a, ctl)
    th.panel_label(ax_a, "A", dx=-0.20)

    axes_b = [fig.add_subplot(gs[1, i]) for i in range(3)]
    panel_celltype_gram(axes_b, gz, cbar_ax=fig.add_subplot(gs[1, 3]))
    th.panel_label(axes_b[0], "B", dx=-0.62)

    gr = pd.read_csv(f"{IMPUTE_DIR}/A9_K{K}_gram.csv")
    across = gr[gr.comparison.str.contains("_vs_seed")]["gram_r"]
    within = gr[gr.comparison.str.contains("train_vs_test")]["gram_r"]
    axes_b[1].text(0.5, -0.10,
                   f"between seeds r = {across.min():.2f}–{across.max():.2f}   ·   "
                   f"train vs held-out r = {within.min():.3f}–{within.max():.3f}",
                   transform=axes_b[1].transAxes, ha="center", fontsize=th.PT["annot"],
                   color=th.INK)
    if save:
        th.save(fig, f"{_OUT}/FIG2_supp_transfer_controls_K{K}")
    return fig


# ── phi consistency across seeds, in simulation (A15) ────────────────────────
SIM_DIR = _HERE
A15_CSV = f"{SIM_DIR}/A15_seed_consistency.csv"
DISP_ORDER = ["no overdispersion", "mild (c=20)", "atlas-like (c=2)"]


A26_NPZ = f"{SIM_DIR}/A26_cellcell_fig.npz"   # reduced by collect_inputs.py
A26_LEVELS = [("binom", "no overdispersion"), ("bb20", "mild (c=20)"), ("bb2", "atlas-like (c=2)")]


def figure_cellcell_consistency(save: bool = True, n_show: int = 400):
    """Do independently seeded fits group the same cells together? (simulation, truth known)

    Cell-cell similarity S[i,j] = corr(phi_i, phi_j) is invariant to relabelling the programs or to
    dividing the same variation among them differently, so it sidesteps the matching problem that
    makes every per-program comparison ambiguous -- on the atlas at K=30, 62% of programs match a
    second partner within 0.02 of the one Hungarian picks.

    Rows are overdispersion levels. First two columns are two independent seeds; the third is the
    TRUTH. Cells are ordered by their true program, so a reproducible fit shows the same block
    structure in all three. Reading across a row separates "the seeds agree" from "the seeds are
    right": an initialisation that made every run agree on a wrong grouping would match in columns 1-2
    and differ from column 3.
    """
    if not os.path.exists(A26_NPZ):
        raise FileNotFoundError(f"{A26_NPZ} missing — run A26_cellcell_consistency.py")
    th.apply_theme()
    z = np.load(A26_NPZ, allow_pickle=True)
    summ = pd.DataFrame(z["summary"]) if "summary" in z.files else None
    levels = [(t, l) for t, l in A26_LEVELS if f"S_{t}_seed0" in z.files]
    fig, axes = plt.subplots(len(levels), 3, figsize=th.figsize("onehalf", 1.9 * len(levels) + 0.9),
                             gridspec_kw=dict(wspace=0.08, hspace=0.30, left=0.16, right=0.88))
    axes = np.atleast_2d(axes)
    for r, (tag, lab) in enumerate(levels):
        order = np.argsort(np.asarray(z[f"order_{tag}"]))[:n_show]
        panels = [(f"S_{tag}_seed0", "Seed 0"), (f"S_{tag}_seed1", "Seed 1"),
                  (f"S_{tag}_true", "True $\\phi$")]
        for c, (key, ttl) in enumerate(panels):
            ax = axes[r, c]
            M = np.asarray(z[key])[np.ix_(order, order)]
            im = ax.imshow(M, cmap=th.CMAP["diverging"], vmin=-1, vmax=1, aspect="equal",
                           interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ("top", "right", "bottom", "left"):
                ax.spines[sp].set_visible(False)
            if r == 0:
                ax.set_title(ttl, fontsize=th.PT["annot"])
            if c == 0:
                ax.text(-0.10, 0.5, lab, transform=ax.transAxes, rotation=90, va="center",
                        ha="center", fontsize=th.PT["annot"])
        if summ is not None and tag in list(summ["dispersion"].astype(str)):
            row = summ[summ["dispersion"].astype(str) == tag].iloc[0]
            axes[r, 1].text(0.5, -0.09,
                            f"between seeds r = {float(row['between_seeds']):.3f}   ·   "
                            f"vs truth r = {float(row['vs_truth']):.3f}",
                            transform=axes[r, 1].transAxes, ha="center",
                            fontsize=th.PT["annot"] - 1, color=th.INK)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label(r"cell-cell similarity in $\phi$", fontsize=th.PT["annot"])
    cb.ax.tick_params(labelsize=th.PT["annot"])
    th.panel_label(axes[0, 0], "A", dx=-0.22)
    if save:
        th.save(fig, f"{_OUT}/FIG2_supp_cellcell_consistency")
    return fig


def figure_seed_consistency_sim(save: bool = True):
    """How consistent is phi across random initialisations, and why -- with the truth available.

    The atlas can show that seeds disagree but not whether they are wrong to, because it has no ground
    truth. Simulation has both. Panels A and B measure agreement BETWEEN fits; panel C measures each
    fit's accuracy AGAINST the known phi, and must be read with them: an intervention that makes every
    run agree on the same wrong answer would look like a success in A and B alone.
    """
    if not os.path.exists(A15_CSV):
        raise FileNotFoundError(f"{A15_CSV} missing — run A15_seed_consistency_sim.py")
    th.apply_theme()
    d = pd.read_csv(A15_CSV)
    d["dispersion_label"] = pd.Categorical(d["dispersion_label"], DISP_ORDER, ordered=True)
    d = d.sort_values("dispersion_label")
    inits = [i for i in ("random", "deterministic") if i in set(d["init"])]
    cols = {"random": th.PALETTE["teal"], "deterministic": th.PALETTE["ochre"]}

    panels = [("mean_canon_corr", "Agreement between seeds\n(mean canonical corr)", None),
              ("frac", "Reproducible directions\n(fraction of K)", (0, 1.05)),
              ("phi_recovery", r"Accuracy vs true $\phi$" + "\n(matched r)", (0, 1.05))]
    fig, axes = plt.subplots(1, 3, figsize=th.figsize("double", 3.1))
    x = np.arange(len(DISP_ORDER))
    for ax, (col, ylab, ylim) in zip(axes, panels):
        for j, init in enumerate(inits):
            sub = d[d.init == init]
            m = [sub[sub.dispersion_label == L][col].mean() for L in DISP_ORDER]
            e = [sub[sub.dispersion_label == L][col].sem() for L in DISP_ORDER]
            off = (j - (len(inits) - 1) / 2) * 0.16
            ax.errorbar(x + off, m, yerr=e, marker="o", ms=5, lw=1.6, capsize=3,
                        color=cols.get(init, th.PALETTE["grey"]),
                        label=th.sentence_case(init) + " init")
        ax.set_xticks(x)
        ax.set_xticklabels([L.replace(" (", "\n(") for L in DISP_ORDER], fontsize=th.PT["annot"])
        ax.set_ylabel(ylab)
        ax.set_xlabel("simulated overdispersion")
        if ylim:
            ax.set_ylim(*ylim)
    axes[0].legend(frameon=False, fontsize=th.PT["annot"] - 1, loc="lower left")
    for ax, L in zip(axes, "ABC"):
        th.panel_label(ax, L)
    # the point of panel C, stated rather than left to be inferred
    axes[2].text(0.5, -0.42, "read C with A and B: agreeing on a wrong answer is not consistency",
                 transform=axes[2].transAxes, ha="center", fontsize=th.PT["annot"] - 1,
                 color=th.PALETTE["grey"])
    if save:
        th.save(fig, f"{_OUT}/FIG2_supp_seed_consistency_sim")
    return fig


def figure_transfer_all_K(save: bool = True, Ks=(10, 20, 30)):
    """Everything about transfer and seed-agreement in one figure, across every K and seed.

    A  the train-vs-held-out agreement at each K, against the three reference points that give it a
       scale (sampling floor, a different seed's held-out map, shuffled labels).
    B  how much the seeds agree on cell-type structure, per K -- permutation-invariant, so no program
       matching is involved anywhere.
    C  the cell-type similarity matrices themselves, one row per K and one column per seed.

    Everything shown is phi. Psi is frozen during the E-step, so it is identical between training and
    held-out cells by construction and cannot appear on any of these axes.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    th.apply_theme()
    have = []
    for K in Ks:
        try:
            have.append((K,) + load_transfer_controls(K))
        except FileNotFoundError:
            print(f"  (K={K} not ready — skipped)")
    if not have:
        raise FileNotFoundError("no A9 outputs — run A9_transfer_controls.py")

    nK = len(have)
    # the theme caps figure height at the page limit; scale the per-K rows to fit rather than
    # letting figsize() raise
    h = min(th.MAX_HEIGHT_IN - 0.05, 2.3 + 2.05 * nK)
    fig = plt.figure(figsize=th.figsize("double", h))
    gs = fig.add_gridspec(1 + nK, 4, height_ratios=[1.05] + [1.5] * nK,
                          width_ratios=[1, 1, 1, 0.045], hspace=0.34, wspace=0.42,
                          left=0.19, right=0.93, top=0.945, bottom=0.035)

    # ---- A: controls, grouped by K ----
    ax = fig.add_subplot(gs[0, :2])
    keys = [("sampling_floor", "Sampling floor", th.PALETTE["grey"]),
            ("transfer", "Train vs held-out", th.PALETTE["teal"]),
            ("cross_model", "Different seed", th.PALETTE["ochre"]),
            ("shuffled", "Shuffled labels", th.PALETTE["rose"])]
    w = 0.8 / len(keys)
    for j, (k, lab, col) in enumerate(keys):
        xs = np.arange(nK) + (j - (len(keys) - 1) / 2) * w
        ys = [c[k].mean() for _, c, _ in have]
        es = [c[k].std(ddof=1) if len(c) > 1 else 0.0 for _, c, _ in have]
        ax.bar(xs, ys, width=w * 0.92, color=col, label=lab, yerr=es,
               error_kw=dict(lw=0.8, capsize=2, ecolor=th.INK))
    ax.set_xticks(range(nK)); ax.set_xticklabels([f"K = {K}" for K, _, _ in have])
    ax.set_ylabel(r"agreement in mean $\phi$ per cell type ($r$)")
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, fontsize=th.PT["annot"] - 1, ncol=2, loc="lower left",
              bbox_to_anchor=(0.0, 1.02))
    th.panel_label(ax, "A", dx=-0.16)

    # ---- B: seed agreement on cell-type structure, per K ----
    axb = fig.add_subplot(gs[0, 2])
    for i, (K, _, _) in enumerate(have):
        gr = pd.read_csv(f"{IMPUTE_DIR}/A9_K{K}_gram.csv")
        a = gr[gr.comparison.str.contains("_vs_seed")]["gram_r"].values
        b = gr[gr.comparison.str.contains("train_vs_test")]["gram_r"].values
        for xoff, v, col, lab in ((-0.13, a, th.PALETTE["indigo"], "between seeds"),
                                  (0.13, b, th.PALETTE["teal"], "train vs held-out")):
            axb.plot(np.full(len(v), i + xoff), v, "o", ms=4, color=col, alpha=0.85,
                     label=lab if i == 0 else None)
            axb.plot([i + xoff - 0.07, i + xoff + 0.07], [np.mean(v)] * 2, "-", lw=1.4, color=col)
    axb.set_xticks(range(nK)); axb.set_xticklabels([f"K = {K}" for K, _, _ in have])
    axb.set_ylabel(r"cell-type structure ($r$)")
    axb.set_title("permutation-invariant:\nno program matching", fontsize=th.PT["annot"] - 1,
                  color=th.PALETTE["grey"])
    axb.set_ylim(0, 1.05)
    axb.legend(frameon=False, fontsize=th.PT["annot"] - 1, loc="lower left")
    th.panel_label(axb, "B", dx=-0.30)

    # ---- C: the matrices, one row per K ----
    cax = fig.add_subplot(gs[1:, 3])
    for rI, (K, _, gz) in enumerate(have):
        types = [str(t) for t in gz["types"]]
        seeds = list(gz["seeds"])
        tasks = sorted({int(k.split("_")[-1]) for k in gz.files if k.startswith("gram_train_")})
        G0 = gz[f"gram_train_{tasks[0]}"]
        D = np.clip(1 - G0, 0, 2); np.fill_diagonal(D, 0)
        order = leaves_list(linkage(squareform(D, checks=False), method="average"))
        # Two of the three columns are different SEEDS (training cells); the third is the HELD-OUT
        # matrix of the first seed. So the row shows both comparisons the figure is making: across
        # seeds, and across train/held-out — the latter otherwise only visible as a number.
        cols = [("train", tasks[0], f"Seed {seeds[0]}, training"),
                ("train", tasks[1], f"Seed {seeds[1]}, training"),
                ("test", tasks[0], f"Seed {seeds[0]}, held-out")]
        for cI, (which, t, ttl) in enumerate(cols):
            a = fig.add_subplot(gs[1 + rI, cI])
            a.set_anchor("W")
            im = a.imshow(gz[f"gram_{which}_{t}"][np.ix_(order, order)], cmap=th.CMAP["diverging"],
                          vmin=-1, vmax=1, aspect="equal")
            a.set_xticks([])
            a.set_yticks(range(len(types)))
            a.set_yticklabels([_ct_label(types[i]) for i in order] if cI == 0 else [],
                              fontsize=th.PT["annot"] - 4)
            a.set_title(f"K = {K}   ·   {ttl}" if cI == 1 else ttl, fontsize=th.PT["annot"])
            for side in ("top", "right", "bottom", "left"):
                a.spines[side].set_visible(False)
            if cI == 0 and rI == 0:
                th.panel_label(a, "C", dx=-0.68)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"similarity of $\phi$ profiles ($r$)", fontsize=th.PT["annot"])
    cb.ax.tick_params(labelsize=th.PT["annot"])
    if save:
        th.save(fig, f"{_OUT}/FIG2_supp_transfer_all_K")
    return fig


def figure_program_celltype_all(K: int, save: bool = True):
    """Every model and every seed at one K: does each cell type keep its program on unseen cells?

    Columns are independent random initialisations; rows are training cells (top) and held-out cells
    (bottom, loadings from the frozen-globals E-step). Read DOWN a column: if a cell type keeps its
    programs in the lower panel, the representation transferred to cells the model never saw. Read
    ACROSS: whether the seeds agree, after aligning program order (see `_align_to`).
    """
    th.apply_theme()
    ms = [load_program_celltype_task(t) for t in TASKS_BY_K[K]]
    ms = [m for m in ms if m is not None]
    if not ms:
        raise FileNotFoundError(f"no A7_prog_ct_task*.npz for K={K} — run a7_all_models.sh")
    ms = [ms[0]] + [_align_to(ms[0], m) for m in ms[1:]]

    types = ms[0]["types"]
    vmax = max(max(m["train"].max(), m["test"].max()) for m in ms)
    fig, axes = plt.subplots(2, len(ms), figsize=th.figsize("double", 3.2 + 0.13 * len(types)),
                             gridspec_kw=dict(wspace=0.08, hspace=0.10, left=0.22, right=0.90))
    axes = np.atleast_2d(axes)
    if axes.shape[1] != len(ms):
        axes = axes.T
    for c, m in enumerate(ms):
        for r, key in enumerate(("train", "test")):
            ax = axes[r, c]
            im = ax.imshow(m[key], aspect="auto", cmap=th.CMAP["sequential"], vmin=0, vmax=vmax)
            ax.set_xticks(range(m["K"]))
            ax.set_xticklabels(range(1, m["K"] + 1) if r == 1 else [],
                               fontsize=th.PT["annot"] - (2 if m["K"] > 10 else 0))
            ax.set_yticks(range(len(types)))
            ax.set_yticklabels([_ct_label(t) for t in types] if c == 0 else [],
                               fontsize=th.PT["annot"] - 1)
            if r == 0:
                ax.set_title(f"Seed {c}")
            if r == 1:
                ax.set_xlabel("splicing program")
            if c == 0:
                ax.text(-0.62, 0.5, "Training cells" if r == 0 else "Held-out cells",
                        transform=ax.transAxes, rotation=90, va="center", ha="center",
                        fontsize=th.PT["label"])
            for side in ("top", "right", "bottom", "left"):
                ax.spines[side].set_visible(False)
        r_tt = np.corrcoef(m["train"].ravel(), m["test"].ravel())[0, 1]
        top = np.mean(m["train"].argmax(1) == m["test"].argmax(1))
        axes[1, c].text(0.5, -0.16, f"r = {r_tt:.3f} · {top:.0%} same top program",
                        transform=axes[1, c].transAxes, ha="center",
                        fontsize=th.PT["annot"], color=th.INK)
    cb = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02)
    cb.ax.tick_params(labelsize=th.PT["annot"])
    cb.set_label("mean loading", fontsize=th.PT["annot"])
    fig.suptitle(f"K = {K}", fontsize=th.PT["label"], y=0.97)
    if save:
        th.save(fig, f"{_OUT}/FIG2_supp_program_celltype_K{K}")
    return fig


def figure_program_celltype(save: bool = True):
    """Supplementary: how the splicing programs relate to annotated cell identity, and whether that
    relationship holds on cells the model never saw."""
    th.apply_theme()
    pc = load_program_celltype()
    fig, ax = plt.subplots(1, 2, figsize=th.figsize("onehalf", 4.4),
                           gridspec_kw=dict(wspace=0.08, left=0.28, right=0.88))
    panel_program_celltype(ax, pc)
    th.panel_label(ax[0], "A", dx=-0.42)
    th.panel_label(ax[1], "B", dx=-0.06)
    if save:
        th.save(fig, f"{_OUT}/FIG2_supp_program_celltype")
    return fig


# ── Figure 2 E/F: imputation ─────────────────────────────────────────────────
# Two panels, deliberately measuring different things on different entry sets.
#
# 2E asks "does the learned phi carry more per-cell splicing information than the ground-truth
#    cell-type label?" and is drawn on `lost_to_downsampling`, where entries are unobserved BY
#    DEFINITION at every level, so no method can pass an observed value through and the label
#    oracle is a fair comparator all along the axis.
#
# 2F asks "how does accuracy hold up as coverage falls?" and MUST use `fixed_lost_at_hardest`,
#    the only entry set whose membership does not change with coverage. Reading the 2E entry set
#    as a curve would be reading a moving population.
#
# The floor in 2F is `naive` (junction mean broadcast to every cell). NOT `mb`: mb passes observed
# values through, so on the fixed set at high coverage it reads ~0.047 and would sit BELOW
# LeafletFA, looking like a defeat when it is really 93% passthrough.

IDENTITY_CEILING = 0.125   # sqrt(median ANOVA R^2 of PSI on ground-truth cell type) -- A27 / notes


def panel_label_oracle(ax, imp: pd.DataFrame, K: int = 20) -> None:
    """2E -- LeafletFA vs the cell-type-label oracle, across coverage.

    `imp` must already be filtered to entry_set="lost_to_downsampling", dataset="all"
    (load_imputation does this and refuses to return mixed rows).
    """
    d = imp[imp.K_cfg == K]
    cov = sorted(pd.to_numeric(d.downsampling_pct, errors="coerce").dropna().unique())
    x = [100 - c for c in cov]                      # plot against reads RETAINED
    series = [("median_pearson_lf",      "LeafletFA",                     "LeafletFA", "-"),
              ("median_pearson_mb_ct",   "cell-type mean (uses labels)",  "other",     "--"),
              ("median_pearson_nmfm_tr", "NMF transfer (mean-impute)",    "NMF",       "-"),
              ("median_pearson_naive",   "junction mean (floor)",         "null",      ":")]
    for col, lab, ckey, ls in series:
        if col not in d:
            continue
        m = [_ms(d[d.downsampling_pct == c], col)[0] for c in cov]
        e = [_ms(d[d.downsampling_pct == c], col)[1] for c in cov]
        ax.errorbar(x, m, yerr=e, fmt="o", ls=ls, ms=3.5, capsize=2, lw=1.4,
                    label=lab, color=th.METHOD_COLORS[ckey])
    ax.axhline(IDENTITY_CEILING, color=th.GRID, lw=0.9, ls=(0, (4, 3)), zorder=0)
    ax.text(0.99, IDENTITY_CEILING, " what cell identity can explain", transform=
            ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=th.PT["annot"],
            color=th.GRID)
    ax.set_xscale("log"); ax.set_xticks([1, 10, 100]); ax.set_xticklabels(["1", "10", "100"])
    ax.set_xlabel("reads retained (%)")
    ax.set_ylabel("median per-junction $r$\n(predicted vs observed $\\Psi$, across cells)")
    th.despine(ax)


def panel_coverage_robustness(ax, imp: pd.DataFrame, K: int = 20) -> None:
    """2F -- MAE vs coverage on the FIXED entry set, pure-prediction methods only.

    `imp` must be filtered to entry_set="fixed_lost_at_hardest".
    """
    d = imp[imp.K_cfg == K]
    cov = sorted(pd.to_numeric(d.downsampling_pct, errors="coerce").dropna().unique())
    x = [100 - c for c in cov]
    for col, lab, ckey, ls in (("mae_lf",      "LeafletFA",                  "LeafletFA", "-"),
                               ("mae_nmfm_tr", "NMF transfer (mean-impute)", "NMF",       "-"),
                               ("mae_nmf0_tr", "NMF transfer (0-impute)",    "ablation",  "-"),
                               ("mae_naive",   "junction mean (floor)",      "null",      ":")):
        if col not in d:
            continue
        m = [_ms(d[d.downsampling_pct == c], col)[0] for c in cov]
        e = [_ms(d[d.downsampling_pct == c], col)[1] for c in cov]
        ax.errorbar(x, m, yerr=e, fmt="o", ls=ls, ms=3.5, capsize=2, lw=1.4,
                    label=lab, color=th.METHOD_COLORS[ckey])
    ax.set_xscale("log"); ax.set_xticks([1, 10, 100]); ax.set_xticklabels(["1", "10", "100"])
    ax.set_xlabel("reads retained (%)")
    ax.set_ylabel("MAE ($\\Psi$ units), fixed entry set")
    th.despine(ax)


def figure_imputation_panels(save: bool = True, K: int = 20):
    """FIG2_EF -- the two imputation panels, each on the entry set it requires."""
    th.apply_theme()
    e = load_imputation(entry_set="lost_to_downsampling")
    try:
        f = load_imputation(kind="04e_allcov_fixedset", entry_set="fixed_lost_at_hardest")
    except (FileNotFoundError, ValueError) as err:
        f = None
        print(f"  (2F skipped -- the fixed-entry-set run has not landed: {err})")
    fig, axes = plt.subplots(1, 2 if f is not None else 1,
                             figsize=th.figsize("double", 3.1), squeeze=False)
    panel_label_oracle(axes[0][0], e, K=K)
    axes[0][0].legend(frameon=False, fontsize=th.PT["annot"], loc="lower left")
    th.panel_label(axes[0][0], "E")
    if f is not None:
        panel_coverage_robustness(axes[0][1], f, K=K)
        axes[0][1].legend(frameon=False, fontsize=th.PT["annot"], loc="upper left")
        th.panel_label(axes[0][1], "F")
    fig.tight_layout()
    if save:
        th.save(fig, f"{_OUT}/FIG2_EF_imputation")
    return fig
