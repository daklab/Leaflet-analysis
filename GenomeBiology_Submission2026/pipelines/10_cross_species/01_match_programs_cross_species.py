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
from leafletfa.dirichlet_multinomial import segment_softmax
from programs import sp_rank
W = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD"
MAP = f"{W}/CROSS_SPECIES_2026/junction_map_mouse_human_2026.tsv.gz"


def cat(g):
    if hasattr(g, "keys") and "categories" in g:
        c = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]); return c[g["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])


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
        C = np.corrcoef(dm_, dh)[:20, 20:]                    # DEVIATION correlation: shared junction-mean removed
        r_, c_ = linear_sum_assignment(-C)
        for a, b in zip(r_, c_):
            rows.append(dict(mouse_seed=sm, human_seed=sh, mouse_SP=f"SP{rm[a]}", human_SP=f"SP{rh[b]}",
                             r=C[a, b], r_raw=np.corrcoef(pm[a], ph[b])[0, 1], baseline=base))
    d = pd.DataFrame(rows)
    d.to_csv(f"{W}/CROSS_SPECIES_2026/program_match_mouse_human.tsv", sep="\t", index=False)
    g = d.groupby("mouse_SP").r.agg(["median", "min", "max"]).sort_values("median", ascending=False)
    pd.set_option("display.width", 160)
    print(g.round(3).to_string())
    print(f"junction-mean baseline (mean mouse psi vs mean human psi): {d.baseline.iloc[0]:.3f}")
    print(f"\nmouse programs matched at r >= 0.7 in ALL 9 seed pairs: "
          f"{int((d.groupby('mouse_SP').r.min() >= 0.7).sum())}/20; at r >= 0.5: "
          f"{int((d.groupby('mouse_SP').r.min() >= 0.5).sum())}/20")


if __name__ == "__main__":
    main()
