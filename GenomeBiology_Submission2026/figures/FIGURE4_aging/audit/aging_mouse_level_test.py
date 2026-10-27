#!/usr/bin/env python
"""Mouse-level test of the aging-associated program (pre-registered in manuscript/AGING_CLAIM_OPTIONS.md §6).

TMS age is perfectly confounded with sequencing campaign / processing (no plate spans two ages), so this
test can only ask whether the age groups differ by more than same-age mice differ, and whether that is
specific to this program -- not whether the difference is age rather than batch.

Per seed (atlas runs_final_deltaonly K20 s0-2), unit = mouse x tissue-cell-type (>= 20 cells), value =
mean phi of a program over the unit.
  P1  MixedLM  unit ~ age24 + C(tct) + (1 | mouse), males 3m vs 24m: age coefficient, 95% CI,
      between-mouse SD, ratio |coef| / SD
  P2  exact permutation over the 126 assignments of "24m" to 4 of 9 male mice, on mouse means of
      tct-residualised unit values
  S1  later campaign: males 18m vs 24m (2 vs 4, 15 assignments), and MixedLM on all 8 mice + sex
  S2  P1 ratio for every live program (argmax >= 1 % of TMS cells): rank of the aging program
Writes <out>/aging_mouse_level_{units,results,specificity}.tsv.
"""
import argparse, itertools, os, sys, warnings
import numpy as np, pandas as pd, h5py

PIPE = "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa"
sys.path.insert(0, PIPE)
from _phi_rows import phi_row_of  # noqa: E402

M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
AGING = {0: 6, 1: 2, 2: 9}          # A69's selected program index per seed (SP4 / SP3 / SP8)


def cat(g):
    if hasattr(g, "keys") and "categories" in g:
        c = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]); return c[g["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])


def mixed(units, extra=""):
    import statsmodels.formula.api as smf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm(f"v ~ age24 + C(tct){extra}", units, groups=units["mouse"]).fit(reml=True)
    coef = m.params["age24"]; lo, hi = m.conf_int().loc["age24"]
    sd = float(np.sqrt(max(m.cov_re.iloc[0, 0], 0.0)))
    return coef, lo, hi, sd


def perm(mouse_val, ages, target):
    """exact two-sided permutation p of mean(target) - mean(other) over all label assignments"""
    mice = list(mouse_val.index); n_t = int((ages == target).sum())
    v = mouse_val.values
    obs = v[(ages == target).values].mean() - v[(ages != target).values].mean()
    stats = []
    for idx in itertools.combinations(range(len(mice)), n_t):
        mask = np.zeros(len(mice), bool); mask[list(idx)] = True
        stats.append(v[mask].mean() - v[~mask].mean())
    stats = np.array(stats)
    return obs, float((np.abs(stats) >= abs(obs) - 1e-12).mean()), len(stats)


def resid_mouse(units):
    """residualise unit values on tissue-cell type, average per mouse"""
    u = units.copy()
    u["r"] = u.v - u.groupby("tct").v.transform("mean")
    return u.groupby("mouse").r.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation/aging_mouse_level")
    ap.add_argument("--min-cells", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with h5py.File(f"{M}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad", "r") as f:
        obs = pd.DataFrame({k: cat(f["obs"][k]) for k in ["dataset", "age", "sex", "mouse.id", "tissue_cell_type"]})
    res, spec, all_units = [], [], []
    for seed, kA in AGING.items():
        z = np.load(f"{M}/ATLAS_DM/runs_final_deltaonly/dm_simplex_K20_seed{seed}_b8192/fit.npz", allow_pickle=True)
        phi = np.asarray(z["phi"]); prow = phi_row_of(z, len(obs))
        tms = ((obs.dataset == "tabula_muris_senis") & (prow >= 0)).values
        d = obs[tms].reset_index(drop=True); P = phi[prow[tms]]
        live = np.flatnonzero(np.bincount(P.argmax(1), minlength=P.shape[1]) / len(P) >= 0.01)
        progs = sorted(set(live) | {kA})
        for k in progs:
            d["p"] = P[:, k]
            g = d.groupby(["mouse.id", "tissue_cell_type", "age", "sex"]).p.agg(["mean", "size"]).reset_index()
            g = g[g["size"] >= a.min_cells].rename(columns={"mouse.id": "mouse", "tissue_cell_type": "tct", "mean": "v"})
            males = g[(g.sex == "male") & g.age.isin(["3m", "24m"])].copy(); males["age24"] = (males.age == "24m").astype(int)
            coef, lo, hi, sd = mixed(males)
            ratio = abs(coef) / sd if sd > 0 else np.inf
            row = dict(seed=seed, program=k, is_aging=(k == kA), live=k in live, n_mice_3m=males[males.age == "3m"].mouse.nunique(),
                       n_mice_24m=males[males.age == "24m"].mouse.nunique(), n_units=len(males),
                       coef_24m_minus_3m=coef, ci_lo=lo, ci_hi=hi, between_mouse_sd=sd, ratio=ratio)
            if k == kA:
                mv = resid_mouse(males); ages = males.groupby("mouse").age.first().loc[mv.index]
                obs_d, p, n = perm(mv, ages, "24m")
                row.update(P2_diff=obs_d, P2_p=p, P2_n_assign=n,
                           P2_mouse_values=";".join(f"{m}:{v:+.4f}" for m, v in mv.sort_values().items()))
                late_m = g[(g.sex == "male") & g.age.isin(["18m", "24m"])].copy()
                mv2 = resid_mouse(late_m); ages2 = late_m.groupby("mouse").age.first().loc[mv2.index]
                d2, p2, n2 = perm(mv2, ages2, "24m")
                late = g[g.age.isin(["18m", "24m"])].copy(); late["age24"] = (late.age == "24m").astype(int)
                c3, l3, h3, s3 = mixed(late, extra=" + C(sex)")
                row.update(S1_males_diff=d2, S1_males_p=p2, S1_males_n_assign=n2,
                           S1_all8_coef=c3, S1_all8_ci=f"[{l3:.4f}, {h3:.4f}]", S1_all8_between_mouse_sd=s3)
                all_units.append(g.assign(seed=seed, program=k))
                res.append(row)
            spec.append(row)
    R = pd.DataFrame(res); S = pd.DataFrame(spec)
    S["rank_among_live"] = S[S.live].groupby("seed").ratio.rank(ascending=False)
    R.to_csv(f"{a.out}/aging_mouse_level_results.tsv", sep="\t", index=False)
    S.to_csv(f"{a.out}/aging_mouse_level_specificity.tsv", sep="\t", index=False)
    pd.concat(all_units).to_csv(f"{a.out}/aging_mouse_level_units.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 200)
    print(R.drop(columns=["P2_mouse_values"]).round(4).T.to_string())
    print(R[["seed", "P2_mouse_values"]].to_string(index=False))
    print(S[S.live][["seed", "program", "is_aging", "coef_24m_minus_3m", "between_mouse_sd", "ratio", "rank_among_live"]].round(3).to_string(index=False))
    for seed in AGING:
        s = S[(S.seed == seed) & S.live]
        print(f"seed {seed}: aging rank {int(s[s.is_aging].rank_among_live.iloc[0]) if s.is_aging.any() else 'n/a (not live)'} of {len(s)} live; "
              f"programs with ratio > 2: {int((s.ratio > 2).sum())} of {len(s)}")


if __name__ == "__main__":
    main()
