#!/usr/bin/env python
"""A69b: find the Fig-4 worked example -- a junction from the aging program whose OBSERVED usage
shifts with age within a tissue-cell-type while the gene's expression does not.

For every junction in the aging program's top-200 (A69) x every TMS tissue-cell-type with >= 25
covered cells in each of 3m/18m/24m: pooled observed PSI per age, Spearman of per-cell PSI vs age,
and the gene's mean log-normalised expression per age with its age Spearman. Rank by |PSI(24m) -
PSI(3m)| among candidates with monotone PSI, |PSI rho| substantial, and flat GE (|rho_GE| < 0.1).
"""
import os, sys
import numpy as np, pandas as pd, h5py
from scipy.stats import spearmanr
M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"


def cat(g):
    if hasattr(g, "keys") and "categories" in g:
        c = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]); return c[g["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])


def main():
    top = pd.read_csv(f"{M}/ATLAS_DM/aging/a69_dm_simplex_K20_seed0_b8192_top_junctions.tsv", sep="\t")
    with h5py.File(f"{M}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad", "r") as f:
        jid = cat(f["var"]["junction_id"]); ev = cat(f["var"]["event_id"])
        obs = {k: cat(f["obs"][k]) for k in ["dataset", "age", "tissue_cell_type"]}
        jpos = {j: i for i, j in enumerate(jid)}
        cols = np.array([jpos[j] for j in top.junction if j in jpos])
        # event totals for the same columns
        gl = f["layers"]["cell_by_junction_matrix"]; cl = f["layers"]["cell_by_cluster_matrix"]
        ipj = gl["indptr"][:]; N = len(ipj) - 1
        Y = np.zeros((N, len(cols)), np.float32); C = np.zeros((N, len(cols)), np.float32)
        cset = {c_: k for k, c_ in enumerate(cols)}
        for g, out in [(gl, Y), (cl, C)]:
            ip = g["indptr"][:]
            for s0 in range(0, N, 20000):
                s1 = min(N, s0 + 20000); a, b = int(ip[s0]), int(ip[s1])
                idx = g["indices"][a:b]; dat = g["data"][a:b]
                rowid = np.repeat(np.arange(s0, s1), np.diff(ip[s0:s1 + 1]))
                m = np.isin(idx, cols)
                if m.any():
                    kk = np.array([cset[c_] for c_ in idx[m]])
                    np.add.at(out, (rowid[m], kk), dat[m])
    with h5py.File(f"{M}/MODEL_INPUT/20260526/ge_adata_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5ad", "r") as f:
        gname = cat(f["var"]["gene_name"]); gpos = {g_: i for i, g_ in enumerate(gname)}
        genes = [g_ for g_ in pd.unique(top.gene) if g_ in gpos]
        gcols = np.array([gpos[g_] for g_ in genes])
        X = f["layers"]["log_norm"]
        if hasattr(X, "keys"):
            import scipy.sparse as sp
            GE = sp.csr_matrix((X["data"][:], X["indices"][:], X["indptr"][:]))[:, gcols].toarray()
        else:
            GE = X[:, np.sort(gcols)][:, np.argsort(np.argsort(gcols))]
    gof = {g_: k for k, g_ in enumerate(genes)}
    tms = obs["dataset"] == "tabula_muris_senis"
    age = pd.Series(obs["age"]).str.replace("m", "").astype(int).values
    tct = obs["tissue_cell_type"]
    rows = []
    keep_j = [j for j in top.junction if j in jpos]
    for t in pd.unique(tct[tms]):
        sel = tms & (tct == t)
        if sel.sum() < 75: continue
        a_ = age[sel]
        if not all(((a_ == x).sum() >= 25) for x in (3, 18, 24)): continue
        Ys, Cs = Y[sel], C[sel]; GEs = GE[sel]
        for k, j in enumerate(keep_j):
            covd = Cs[:, k] >= 1
            if not all((covd & (a_ == x)).sum() >= 25 for x in (3, 18, 24)): continue
            psi = Ys[covd, k] / Cs[covd, k]; aa = a_[covd]
            pooled = {x: Ys[covd & False | (covd), k][aa == x].sum() / Cs[covd, k][aa == x].sum() for x in (3, 18, 24)}
            rho = spearmanr(aa, psi).correlation
            g_ = top.gene.iloc[k]
            if g_ not in gof: continue
            ge = GEs[:, gof[g_]]
            rho_ge = spearmanr(a_, ge).correlation
            rows.append(dict(tct=t, junction=j, gene=g_, event=top.event.iloc[k],
                             psi3=pooled[3], psi18=pooled[18], psi24=pooled[24],
                             d=pooled[24] - pooled[3], rho_psi=rho, rho_ge=rho_ge,
                             ncov3=int((covd & (a_ == 3)).sum()), ncov18=int((covd & (a_ == 18)).sum()), ncov24=int((covd & (a_ == 24)).sum())))
    d = pd.DataFrame(rows)
    d["mono"] = ((d.psi18 - d.psi3) * (d.psi24 - d.psi18) >= 0)
    d["score"] = d.d.abs() * d.mono * (d.rho_ge.abs() < 0.1) * (d.rho_psi.abs() > 0.1)
    d = d.sort_values("score", ascending=False)
    d.to_csv(f"{M}/ATLAS_DM/aging/a69b_example_candidates.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(d.head(25).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
