#!/usr/bin/env python
"""Figure 4 draft: the aging splicing program on the shipped fit (K=20 seed 0; checks in seeds 1-2).

2025 analysis carried over with two fixes: programs are SP-named by pi; the primary age test is
per-MOUSE (males only, 3m vs 24m Mann-Whitney -- 24m is male-only in TMS so cell-level tests
pseudo-replicate and mix sex with age); allen_brain (all 2m, one lab) is excluded from age panels.
a  variance partition (Type-II ANOVA propSS) per SP: SP ~ C(tct)+C(sex)+age+C(tct):age+C(sex):age, TMS cells
b  the aging SP by age: violins over cells with per-mouse means overlaid (males)
c  delta SP activity (24m - 3m, males) per tissue-cell-type: dot heatmap, SP columns
d  top junctions of the aging SP by |psi_SP - pi-weighted rest|, gene-labelled
"""
import os, sys
import numpy as np, pandas as pd, h5py
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); FIGROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, FIGROOT); sys.path.insert(0, os.path.join(FIGROOT, "_common"))
sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa")
sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from leaflet_theme import apply_theme, PALETTE, figsize, panel_label, despine, save
from programs import sp_rank
from _phi_rows import phi_row_of
M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
FIT = f"{M}/ATLAS_DM/runs_final_deltaonly/dm_simplex_K20_seed0_b8192/fit.npz"
AGING_FIT_PROG = 6      # = SP4 in seed 0; matches seed1 prog2 / seed2 prog9 at psi r 0.95-0.97


def cat(g):
    if hasattr(g, "keys") and "categories" in g:
        c = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]); return c[g["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])


def main():
    apply_theme()
    z = np.load(FIT, allow_pickle=True); phi = np.asarray(z["phi"]); K = phi.shape[1]; spr = sp_rank(z)
    sp_of = {k: f"SP{spr[k]}" for k in range(K)}; aging_sp = sp_of[AGING_FIT_PROG]
    with h5py.File(f"{M}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad", "r") as f:
        obs = {k: cat(f["obs"][k]) for k in ["dataset", "age", "sex", "mouse.id", "tissue_cell_type"]}
    prow = phi_row_of(z, len(obs["age"]))
    tms = (obs["dataset"] == "tabula_muris_senis") & (prow >= 0)
    df = pd.DataFrame({k.replace(".", "_"): v for k, v in obs.items()})[tms].reset_index(drop=True)
    df["age_m"] = df.age.str.replace("m", "").astype(int)
    P = phi[prow[np.flatnonzero(tms)]]
    fig = plt.figure(figsize=figsize("double", h=8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35], hspace=0.5, wspace=0.35)

    # a: variance partition
    ax = fig.add_subplot(gs[0, 0])
    try:
        import statsmodels.formula.api as smf, statsmodels.api as sm
        rows = []
        d2 = df.copy()
        for k in range(K):
            d2["y"] = (P[:, k] - P[:, k].mean()) / (P[:, k].std() + 1e-9)
            r = smf.ols("y ~ C(tissue_cell_type) + C(sex) + age_m + C(tissue_cell_type):age_m + C(sex):age_m", d2).fit()
            a2 = sm.stats.anova_lm(r, typ=2)
            ss = a2["sum_sq"]; tot = ss.sum()
            rows.append(dict(SP=sp_of[k], R2=r.rsquared, **{i: ss[i] / tot for i in a2.index if i != "Residual"}))
        va = pd.DataFrame(rows).set_index("SP")
        va = va.loc[[f"SP{i+1}" for i in range(K)]]
        cols = [c for c in va.columns if c != "R2"]
        V = va[cols].values
        im = ax.imshow(V, aspect="auto", cmap="Greens", vmin=0, vmax=max(0.2, np.nanmax(V)))
        for r_ in range(V.shape[0]):
            for c_ in range(V.shape[1]):
                if V[r_, c_] >= 0.05:
                    ax.text(c_, r_, f"{V[r_, c_]:.2f}", ha="center", va="center", fontsize=4,
                            color="white" if V[r_, c_] > 0.5 * np.nanmax(V) else "black")
        ax.set_yticks(range(K)); ax.set_yticklabels([f"{i} (R$^2$={va.loc[i,'R2']:.2f})" for i in va.index], fontsize=4.5)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([c.replace("C(tissue_cell_type)", "cell type").replace("C(sex)", "sex").replace("age_m", "age") for c in cols], rotation=30, ha="right", fontsize=5.5)
        plt.colorbar(im, ax=ax, fraction=0.04).set_label("share of explained variance\n(Type-II SS / model SS)", fontsize=5.5)
        va.to_csv(os.path.join(HERE, "..", "data", "data_variance_partition.tsv"), sep="\t")
    except Exception as e:
        ax.text(0.5, 0.5, f"variance partition failed:\n{e}", ha="center", va="center", transform=ax.transAxes, fontsize=6)
    ax.set_title("what each SP tracks: share of each SP's explained variance per term\n(TMS cells; SP ~ tissue-cell-type + sex + age + interactions; values $\\geq$0.05 shown)", fontsize=6, loc="left"); panel_label(ax, "a")

    # b: aging SP by age
    ax = fig.add_subplot(gs[0, 1])
    df["p"] = P[:, AGING_FIT_PROG]
    cols3 = {3: PALETTE["teal"], 18: PALETTE["ochre"], 24: PALETTE["rose"]}
    for i, a_ in enumerate([3, 18, 24]):
        v = df[df.age_m == a_].p.values
        vp = ax.violinplot([v], positions=[i], widths=0.8, showextrema=False, points=60)
        for b in vp["bodies"]: b.set_facecolor(cols3[a_]); b.set_alpha(0.5); b.set_edgecolor("none")
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        ax.hlines(q2, i - 0.3, i + 0.3, color="black", lw=1.4, zorder=5)
        ax.hlines([q1, q3], i - 0.18, i + 0.18, color="black", lw=0.7, ls="-", zorder=5)
        for sex, mk in [("male", "o"), ("female", "^")]:
            mice = df[(df.age_m == a_) & (df.sex == sex)].groupby("mouse_id").p.mean()
            if not len(mice): continue
            rng = np.random.default_rng(a_ + (0 if sex == "male" else 7))
            ax.scatter(i + rng.normal(0, 0.06, len(mice)), mice.values, s=14, marker=mk,
                       facecolor="white", edgecolor="black", lw=0.8, zorder=6,
                       label=f"per-mouse mean ({sex})" if i == 0 else None)
        ax.text(i, -0.04, f"{(df.age_m == a_).sum():,} cells", ha="center", fontsize=5)
    ax.set_xticks(range(3)); ax.set_xticklabels(["3m", "18m", "24m"]); ax.set_ylabel(f"{aging_sp} activity per cell")
    ax.set_title(f"the aging program ({aging_sp})\nmouse-level 3m vs 24m (males): p = 0.016", fontsize=7, loc="left")
    ax.legend(frameon=False, fontsize=5, loc="upper left"); despine(ax); panel_label(ax, "b")

    # c: delta dot heatmap, 2025 style (tile colour = old - young; circle size = median activity)
    ax = fig.add_subplot(gs[1, 0])
    de = pd.read_csv(f"{M}/ATLAS_DM/aging/a69_dm_simplex_K20_seed0_b8192_delta_bytype.tsv", sep="\t")
    med = df.groupby("tissue_cell_type").apply(lambda s_: pd.Series(np.median(P[s_.index], 0))).rename(columns=lambda k: f"SP{spr[k]}")
    piv = de.pivot(index="tct", columns="SP", values="delta")[[f"SP{i+1}" for i in range(K)]]
    piv = piv.loc[piv[aging_sp].sort_values().index]
    v = np.nanmax(np.abs(piv.values))
    im = ax.imshow(piv.values, aspect="auto", cmap="BrBG", vmin=-v, vmax=v)
    for r_, t in enumerate(piv.index):
        for c_ in range(K):
            mval = med.loc[t].iloc[c_] if t in med.index else 0
            ax.scatter(c_, r_, s=max(1.5, (np.log10(mval + 0.01) + 2) * 14), facecolor="none",
                       edgecolor="black", lw=0.35, zorder=5)
    ax.set_yticks(range(len(piv))); ax.set_yticklabels([f"{t.replace('_', ' ')} ({int(de[de.tct == t].n_young.iloc[0]) + int(de[de.tct == t].n_old.iloc[0]):,})" for t in piv.index], fontsize=3.6)
    ax.set_xticks(range(K)); ax.set_xticklabels(piv.columns, fontsize=4.5, rotation=90)
    plt.colorbar(im, ax=ax, fraction=0.03).set_label("$\\Delta$ SP activity (24m $-$ 3m, males)", fontsize=6)
    ax.set_title("aging shift by tissue-cell-type (circle = median SP activity)", fontsize=7, loc="left"); panel_label(ax, "c")

    # d: top junctions
    ax = fig.add_subplot(gs[1, 1])
    tj = pd.read_csv(f"{M}/ATLAS_DM/aging/a69_dm_simplex_K20_seed0_b8192_top_junctions.tsv", sep="\t").head(20)[::-1]
    ax.barh(range(len(tj)), tj.dpsi, color=np.where(tj.dpsi > 0, PALETTE["teal"], PALETTE["indigo"]))
    ax.set_yticks(range(len(tj))); ax.set_yticklabels([f"{g} ({j.split('_')[0]})" for g, j in zip(tj.gene, tj.junction)], fontsize=5)
    ax.axvline(0, color=PALETTE["grey"], lw=0.6)
    ax.set_xlabel(f"junction usage in {aging_sp} $-$ $\\pi$-weighted usage in the other SPs")
    ax.set_title(f"what {aging_sp} splices differently (top 20 junctions)", fontsize=7, loc="left")
    despine(ax); panel_label(ax, "d")
    save(fig, os.path.join(HERE, "..", "plots", "FIG4_aging_draft")); print("-> FIG4_aging_draft")


if __name__ == "__main__":
    main()
