#!/usr/bin/env python
"""Fig 5 first analysis: are the splicing programs conserved between mouse and human brain?

Uses the 2026 junction map (00_) restricted to junctions in events with >= 2 matched junctions
(so the within-ATSE composition is comparable), and the K=20 fits of the mouse-AB-only and
human-AB grids. For every (mouse program, human program) pair: Pearson of psi over the matched
junctions; Hungarian one-to-one assignment; report each mouse program's best human partner and r,
repeated across the 3 x 3 seed pairs so conservation carries a seed spread. Programs SP-named per
fit. Output: program_match_mouse_human.tsv + a summary print.
"""
import itertools, os, sys
import numpy as np, pandas as pd, h5py, torch
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/_common")
sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa")
from leafletfa.dirichlet_multinomial import segment_softmax
from programs import sp_rank
W = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD"
MAP = f"{W}/CROSS_SPECIES_2026/junction_map_mouse_human_2026.tsv.gz"


def cat(g):
    if hasattr(g, "keys") and "categories" in g:
        c = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]); return c[g["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])


_LABELS = {}
def _labels(h5, field="tissue_cell_type"):
    if h5 not in _LABELS:
        with h5py.File(h5, "r") as f:
            _LABELS[h5] = cat(f["obs"][field])
    return _LABELS[h5]


def live_of(fit, h5, thresh=0.05):
    """live = some cell type (>= 50 cells) actually uses the program: max cell-type mean activity
    >= thresh. A global-pi cutoff kills rare-cell-type programs (mouse SP13: endothelial mean 0.97,
    tiny pi); an activity cutoff keeps them while excluding the truly dead ones (0.00 everywhere,
    whose near-prior psi manufactures cross-species correlation)."""
    from _phi_rows import phi_row_of
    z = np.load(fit, allow_pickle=True); phi = np.asarray(z["phi"])
    lab = _labels(h5); prow = phi_row_of(z, len(lab)); tr = prow >= 0
    types = [t for t in np.unique(lab[tr]) if (lab[tr] == t).sum() >= 50]
    Mx = np.vstack([phi[prow[(lab == t) & tr]].mean(0) for t in types])
    return np.flatnonzero(Mx.max(0) >= thresh)


def psi_of(fit, jidx):
    z = np.load(fit, allow_pickle=True)
    inv = torch.from_numpy(np.asarray(z["inv_ev"]).astype(np.int64))
    psi = segment_softmax(torch.from_numpy(np.asarray(z["psi_loc"], np.float32)), inv, len(z["event_id"])).numpy()
    pi = np.asarray(z["trace_pi"])[-1]
    dev = psi - (pi / pi.sum()) @ psi          # deviation from the species' pi-weighted mean composition
    return psi[:, jidx], dev[:, jidx], sp_rank(z)


def main():
    mp = pd.read_csv(MAP, sep="\t")
    ev_n = mp.groupby("mouse_event").mouse_junction.transform("size")
    mp = mp[ev_n >= 2].reset_index(drop=True)
    with h5py.File(f"{W}/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad", "r") as f:
        mj = {j: i for i, j in enumerate(cat(f["var"]["junction_id"]))}
    with h5py.File(f"{W}/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/20260617/human_splice_model_ready_2026-06-17.h5ad", "r") as f:
        hj = {j: i for i, j in enumerate(cat(f["var"]["junction_id"]))}
    mp = mp[mp.mouse_junction.isin(mj) & mp.human_junction.isin(hj)]
    mi = mp.mouse_junction.map(mj).values; hi = mp.human_junction.map(hj).values
    print(f"{len(mp):,} matched junctions in {mp.mouse_event.nunique():,} events used", flush=True)
    rows = []
    for sm, sh in itertools.product(range(3), range(3)):
        pm, dm_, rm = psi_of(f"{W}/MOUSE_SPLICING_FOUNDATION/ATLAS_DM/runs_ABonly/dm_simplex_K20_seed{sm}_b8192/fit.npz", mi)
        ph, dh, rh = psi_of(f"{W}/HUMAN_SPLICING_FOUNDATION/LEAFLETFA_MODELS/DM_shipped_2026-08-17/dm_simplex_K20_seed{sh}_b8192/fit.npz", hi)
        base = np.corrcoef(pm.mean(0), ph.mean(0))[0, 1]      # junction-mean baseline: any-program vs any-program floor
        # LIVE programs only (pi > 0.01). Dead programs sit near the uniform prior in BOTH species, so
        # their deviations from the data-driven mean correlate trivially -- matching them manufactures
        # "conservation" (measured: dead-dead pairs at r 0.57-0.67 with zero activity anywhere).
        lm = live_of(f"{W}/MOUSE_SPLICING_FOUNDATION/ATLAS_DM/runs_ABonly/dm_simplex_K20_seed{sm}_b8192/fit.npz",
                     f"{W}/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad")
        lh = live_of(f"{W}/HUMAN_SPLICING_FOUNDATION/LEAFLETFA_MODELS/DM_shipped_2026-08-17/dm_simplex_K20_seed{sh}_b8192/fit.npz",
                     f"{W}/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/20260617/human_splice_model_ready_2026-06-17.h5ad")
        C = np.corrcoef(dm_[lm], dh[lh])[:len(lm), len(lm):]
        r_, c_ = linear_sum_assignment(-C)
        for a, b in zip(r_, c_):
            rows.append(dict(mouse_seed=sm, human_seed=sh, mouse_SP=f"SP{rm[lm[a]]}", human_SP=f"SP{rh[lh[b]]}",
                             r=C[a, b], r_raw=np.corrcoef(pm[lm[a]], ph[lh[b]])[0, 1], baseline=base,
                             n_live_mouse=len(lm), n_live_human=len(lh)))
    d = pd.DataFrame(rows)
    d.to_csv(f"{W}/CROSS_SPECIES_2026/program_match_mouse_human.tsv", sep="\t", index=False)
    # CEILING: the same statistic within mouse, across seeds -- what "perfectly conserved" could reach
    ceil = []
    for s1, s2 in [(0, 1), (0, 2), (1, 2)]:
        p1, d1, r1 = psi_of(f"{W}/MOUSE_SPLICING_FOUNDATION/ATLAS_DM/runs_ABonly/dm_simplex_K20_seed{s1}_b8192/fit.npz", mi)
        p2, d2, r2 = psi_of(f"{W}/MOUSE_SPLICING_FOUNDATION/ATLAS_DM/runs_ABonly/dm_simplex_K20_seed{s2}_b8192/fit.npz", mi)
        l1 = live_of(f"{W}/MOUSE_SPLICING_FOUNDATION/ATLAS_DM/runs_ABonly/dm_simplex_K20_seed{s1}_b8192/fit.npz",
                     f"{W}/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad")
        l2 = live_of(f"{W}/MOUSE_SPLICING_FOUNDATION/ATLAS_DM/runs_ABonly/dm_simplex_K20_seed{s2}_b8192/fit.npz",
                     f"{W}/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad")
        C = np.corrcoef(d1[l1], d2[l2])[:len(l1), len(l1):]
        rr, cc = linear_sum_assignment(-C); ceil.extend(C[rr, cc])
    print(f"WITHIN-mouse across-seed ceiling (live-live, same junctions): median {np.median(ceil):.3f}, "
          f"IQR {np.percentile(ceil, 25):.3f}-{np.percentile(ceil, 75):.3f}")
    g = d.groupby("mouse_SP").r.agg(["median", "min", "max"]).sort_values("median", ascending=False)
    pd.set_option("display.width", 160)
    print(g.round(3).to_string())
    print(f"junction-mean baseline (mean mouse psi vs mean human psi): {d.baseline.iloc[0]:.3f}")
    print(f"\nmouse programs matched at r >= 0.7 in ALL 9 seed pairs: "
          f"{int((d.groupby('mouse_SP').r.min() >= 0.7).sum())}/20; at r >= 0.5: "
          f"{int((d.groupby('mouse_SP').r.min() >= 0.5).sum())}/20")


if __name__ == "__main__":
    main()
