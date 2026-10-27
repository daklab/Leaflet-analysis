#!/usr/bin/env python
"""Supplementary Fig. SX — a partially reproducible program (replaces the former aging figure).

    python build_supp_partial_program.py

Atlas fits K20 seeds 0-2 (runs_final_deltaonly). The seed-0 program SP4 (fit index 6) is matched to its
counterparts by Hungarian assignment on centered psi (each fit's psi minus its pi-weighted mean
composition): seed 1 index 5 (SP6, cosine 0.645), seed 2 index 9 (SP8, 0.650). In seed 2 the program
used by the same cell types (activity-profile match) is a different one, index 14 (SP15, cosine 0.537), so
seed 2 is shown twice: matched by usage and matched by composition. Build notes:
../BUILD_NOTES_partial_program.md.
  a  mean activity per tissue-cell type of the matched program in each fit (top 20 types, >= 20 cells)
  b  minimum centered-psi cosine to the two other fits for every seed-0 program in use (max cell-type
     mean phi >= 0.05, A70's rule); this program highlighted; 0.8 threshold; reproduced programs labeled
  c  signed departure of each shown program from the pi-weighted mean of its fit's other programs, for
     the union of their top-8 junctions; dot = in the top 8 of at least 3 of the 4 programs
Output: figures/supplemental/partial_program/SX_partial_program.{pdf,png} + SX_partial_program_data.tsv
"""
import os, sys
import numpy as np, pandas as pd, h5py
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(FIG, "_common"))
sys.path.insert(0, os.path.join(os.path.dirname(FIG), "pipelines", "05_leafletfa"))
from leaflet_theme import apply_theme, PALETTE, CMAP, figsize, panel_label, save, symmetric_norm  # noqa: E402
from programs import sp_rank  # noqa: E402
from _phi_rows import phi_row_of  # noqa: E402

M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
GRID = f"{M}/ATLAS_DM/runs_final_deltaonly"
REF_SEED, REF_IDX, TOP_TYPES, TOP_J, MIN_CELLS, IN_USE = 0, 6, 20, 8, 20, 0.05
OUT = os.path.join(FIG, "supplemental", "partial_program", "SX_partial_program")


def cat(g):
    if hasattr(g, "keys") and "categories" in g:
        c = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]); return c[g["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])


def centered(z):
    P = z["psi_mean"].astype(np.float64)
    pi = np.asarray(z["trace_pi"][-1] if "trace_pi" in z.files else z["pi"], np.float64); pi = pi / pi.sum()
    C = P - pi @ P
    return C / np.linalg.norm(C, axis=1, keepdims=True).clip(1e-12), P, pi


def departure(P, pi, k):
    w = np.delete(pi, k); w = w / w.sum()
    return P[k] - w @ np.delete(P, k, axis=0)


def short(t):
    t = t.replace("_", " ")
    return t[0].upper() + t[1:].lower() if t else t


def main():
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    apply_theme()
    with h5py.File(f"{M}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad", "r") as f:
        tct = cat(f["obs"]["tissue_cell_type"]); gene = cat(f["var"]["gene_name"])
    Z = {s: np.load(f"{GRID}/dm_simplex_K20_seed{s}_b8192/fit.npz", allow_pickle=True) for s in (0, 1, 2)}
    C = {s: centered(Z[s]) for s in Z}
    # matched programs (centered psi, Hungarian) and per-fit activity profiles
    match = {REF_SEED: REF_IDX}
    for s in (1, 2):
        R = C[0][0] @ C[s][0].T; i, j = linear_sum_assignment(-R); match[s] = int(dict(zip(i, j))[REF_IDX])
    prof = {}
    for s in Z:
        phi = np.asarray(Z[s]["phi"]); prow = phi_row_of(Z[s], len(tct)); ok = prow >= 0
        d = pd.DataFrame(phi[prow[ok]]); d["tct"] = tct[ok]
        n = d.tct.value_counts(); keep = n[n >= MIN_CELLS].index
        prof[s] = d[d.tct.isin(keep)].groupby("tct").mean()
        prof[s]["n"] = n.loc[prof[s].index]
    # seed 2: the program used by the same cell types (best activity-profile r to seed-0 SP4)
    p0 = prof[0][REF_IDX]
    use2 = int(prof[2].drop(columns="n").apply(lambda c: np.corrcoef(p0, c.loc[p0.index])[0, 1]).idxmax())
    cols = [(0, match[0], "by ψ"), (1, match[1], "by ψ and use"), (2, use2, "by use"), (2, match[2], "by ψ")]
    cols[0] = (0, match[0], "reference")
    lab = {(s, k): f"seed {s} SP{sp_rank(Z[s])[k]} ({how})" for s, k, how in cols}
    A = pd.DataFrame({lab[(s, k)]: prof[s][k] for s, k, _ in cols}).dropna()
    A = A.loc[A.mean(1).sort_values(ascending=False).index[:TOP_TYPES]]

    # panel b: min centered cosine of every in-use seed-0 program to seeds 1 and 2
    rk0 = sp_rank(Z[0]); rows = []
    for k in range(C[0][0].shape[0]):
        if prof[0][k].max() < IN_USE:
            continue
        cs = []
        for s in (1, 2):
            R = C[0][0] @ C[s][0].T; i, j = linear_sum_assignment(-R); cs.append(R[k, dict(zip(i, j))[k]])
        rows.append(dict(SP=f"SP{rk0[k]}", idx=k, min_cos=min(cs),
                         top_type=prof[0][k].idxmax().split("_")[-1].lower()))
    B = pd.DataFrame(rows).sort_values("min_cos")

    # panel c: signed departures, union of each fit's top junctions
    dep = {(s, k): departure(C[s][1], C[s][2], k) for s, k, _ in cols}
    tops = {key: set(np.argsort(-np.abs(v))[:TOP_J]) for key, v in dep.items()}
    union = sorted(set.union(*tops.values()), key=lambda j: -np.abs(dep[(0, REF_IDX)][j]))
    Cm = pd.DataFrame({lab[key]: dep[key][union] for key in dep}, index=[gene[j] for j in union])
    in_all = [sum(j in t for t in tops.values()) >= 3 for j in union]   # in >= 3 of the 4 top lists

    fig = plt.figure(figsize=figsize("double", h=4.6))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.3, 0.85, 1.0], wspace=0.95)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(A.values, aspect="auto", cmap=CMAP["sequential"], vmin=0, vmax=float(A.values.max()))
    ax.set_xticks(range(A.shape[1])); ax.set_xticklabels(A.columns, rotation=40, ha="right")
    ax.set_yticks(range(A.shape[0]))
    ax.set_yticklabels([f"{short(t)} ({int(prof[0].loc[t, 'n']):,})" for t in A.index])
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.28); cb.set_label("Mean activity (φ)")
    ax.set_title("The program in each fit")

    ax = fig.add_subplot(gs[1])
    col = [PALETTE["teal"] if r.idx == REF_IDX else (PALETTE["indigo"] if r.min_cos >= 0.8 else PALETTE["grey"])
           for r in B.itertuples()]
    ax.barh(range(len(B)), B.min_cos, color=col, height=0.7)
    ax.axvline(0.8, color=PALETTE["grey"], lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks(range(len(B))); ax.set_yticklabels([f"{r.SP} {r.top_type}" for r in B.itertuples()])
    ax.set_xlabel("Centered-ψ cosine (min over other fits)"); ax.set_xlim(0, 1)
    ax.set_title("Reproducibility threshold 0.8")

    ax = fig.add_subplot(gs[2])
    vmin, vmax, cm = symmetric_norm(Cm.values)
    im = ax.imshow(Cm.values, aspect="auto", cmap=cm, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(Cm.shape[1])); ax.set_xticklabels(Cm.columns, rotation=40, ha="right")
    ax.set_yticks(range(Cm.shape[0])); ax.set_yticklabels(Cm.index, fontstyle="italic")
    for r, a in enumerate(in_all):
        if a:
            ax.plot(Cm.shape[1] - 0.5 + 0.35, r, "o", ms=2.5, color=PALETTE["teal"], clip_on=False)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.28); cb.set_label("Δψ vs other programs")
    ax.set_title("Top junctions per fit")
    for a_, l in zip([fig.axes[0], fig.axes[2], fig.axes[3]], "abc"):
        panel_label(a_, l)
    save(fig, OUT)
    data = pd.concat({"a_activity": A.reset_index().rename(columns={"index": "tissue_cell_type"}),
                      "b_cosine": B, "c_departure": Cm.assign(in_top15_all_fits=in_all).reset_index()
                      .rename(columns={"index": "gene"})}, names=["panel"]).reset_index(level=0)
    data.to_csv(OUT + "_data.tsv", sep="\t", index=False)
    print("columns:", list(lab.values()))
    print(B.round(3).to_string(index=False))
    print(f"junctions: {len(union)} in the union; {sum(in_all)} in >= 3 of the 4 top-{TOP_J} lists")
    print("saved", OUT + ".pdf/.png")


if __name__ == "__main__":
    main()
