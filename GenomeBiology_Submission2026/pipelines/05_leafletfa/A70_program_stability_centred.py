#!/usr/bin/env python
"""A70: program stability across seeds on MEAN-CENTRED psi (Karin 2026-09-27).
Raw psi cosine is carried by the shared junction-level usage (every program ~0.98 alike), so each
seed's psi is centred by its pi-weighted mean composition before Hungarian matching on cosine.
Reference = seed 0 of each grid; a seed-0 program is STABLE if its matched cosine is >= 0.8 in BOTH
other seeds. Programs named SP1..SPK by seed-0 pi. Output: per-program table + per-grid summary."""
import os, sys, itertools
import numpy as np, pandas as pd, torch
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/_common")
from leafletfa.dirichlet_multinomial import segment_softmax
from programs import sp_rank
W = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD"
GRIDS = {"atlas": f"{W}/MOUSE_SPLICING_FOUNDATION/ATLAS_DM/runs_final_deltaonly",
         "atlas70": f"{W}/MOUSE_SPLICING_FOUNDATION/ATLAS_DM/runs_heldout70",
         "TMS_only": f"{W}/MOUSE_SPLICING_FOUNDATION/ATLAS_DM/runs_TMS70",
         "AB_only": f"{W}/MOUSE_SPLICING_FOUNDATION/ATLAS_DM/runs_ABonly",
         "human_AB": f"{W}/HUMAN_SPLICING_FOUNDATION/LEAFLETFA_MODELS/DM_shipped_2026-08-17"}
THR = 0.8


def centred(fit):
    z = np.load(fit, allow_pickle=True)
    inv = torch.from_numpy(np.asarray(z["inv_ev"]).astype(np.int64))
    psi = segment_softmax(torch.from_numpy(np.asarray(z["psi_loc"], np.float32)), inv, len(z["event_id"])).numpy()
    pi = np.asarray(z["trace_pi"])[-1]; pi = pi / pi.sum()
    c = psi - pi @ psi
    return c / np.linalg.norm(c, axis=1, keepdims=True), sp_rank(z), pi


def main():
    out, summ = [], []
    for grid, root in GRIDS.items():
        for K in (10, 20, 30):
            fs = [f"{root}/dm_simplex_K{K}_seed{s}_b8192/fit.npz" for s in (0, 1, 2)]
            if not all(os.path.exists(f) for f in fs): continue
            C = [centred(f) for f in fs]; ref, rk, pi0 = C[0]
            best = np.full((K, 2), np.nan)
            for i, s in enumerate((1, 2)):
                S = ref @ C[s][0].T; r, c = linear_sum_assignment(-S); best[r, i] = S[r, c]
            mn = best.min(1)
            for k in range(K):
                out.append(dict(grid=grid, K=K, SP=f"SP{rk[k]}", pi_seed0=pi0[k], cos_seed1=best[k, 0], cos_seed2=best[k, 1],
                                min_cos=mn[k], stable=bool(mn[k] >= THR)))
            summ.append(dict(grid=grid, K=K, stable=int((mn >= THR).sum()), of=K, median_min_cos=float(np.median(mn)),
                             stable_SPs=",".join(sorted([f"SP{rk[k]}" for k in range(K) if mn[k] >= THR], key=lambda x: int(x[2:])))))
            print(summ[-1], flush=True)
    os.makedirs(f"{W}/MOUSE_SPLICING_FOUNDATION/ATLAS_DM/stability", exist_ok=True)
    pd.DataFrame(out).to_csv(f"{W}/MOUSE_SPLICING_FOUNDATION/ATLAS_DM/stability/a70_program_stability_centred.tsv", sep="\t", index=False)
    S = pd.DataFrame(summ); S.to_csv(f"{W}/MOUSE_SPLICING_FOUNDATION/ATLAS_DM/stability/a70_summary.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 150); print(S.to_string(index=False))


if __name__ == "__main__":
    main()
