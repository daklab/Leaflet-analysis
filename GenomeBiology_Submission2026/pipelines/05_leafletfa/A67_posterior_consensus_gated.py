#!/usr/bin/env python
"""A67: the gated posterior consensus table, as a script (it was three inline steps on 2026-08-16).

Inputs: three A51 outputs (one per model seed, same delta), the LeafCutter results, and per
(contrast, ATSE) covered-cell counts. Output: one row per (contrast, ATSE) with per-seed posterior
TV and q, votes (# seeds with q <= 0.05), consensus (>= 2 of 3), LeafCutter q and TV, lc_sig at
LeafCutter's matched-realised-FDP threshold (q <= 1.39e-3), covered cells in each arm and the
>= 25-cell eligibility gate. Coverage is fit-independent, so it is copied from --coverage-from
(the K=20 table) when given, otherwise recomputed from the h5ad.
"""
import argparse, os, sys, re
import numpy as np, pandas as pd, h5py, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from A50_compare_lrt_to_leafcutter import load_leafcutter, norm
from A63_posterior_vs_leafcutter_matched_delta import leafcutter_tv

M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
LC_Q = 1.39e-3


def coverage_table(h5, contrasts_keys):
    with h5py.File(h5, "r") as f:
        n = f["obs"]["tissue_cell_type"]; c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]]); ct = c[n["codes"][:]]
        n = f["var"]["event_id"]; c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]]); ev = c[n["codes"][:]]
        g = f["layers"]["cell_by_cluster_matrix"]; J = len(ev); ip = g["indptr"][:]
        rep = pd.Series(np.arange(J)).groupby(ev).first(); repcols = rep.values
        cov = sp.lil_matrix((len(ct), len(repcols)), dtype=np.int32)
        for a in range(0, len(ct), 8000):
            b = min(a + 8000, len(ct)); s, e = int(ip[a]), int(ip[b])
            blk = sp.csr_matrix((g["data"][s:e], g["indices"][s:e], ip[a:b + 1] - ip[a]), shape=(b - a, J))[:, repcols]
            blk.data = (blk.data > 0).astype(np.int32); cov[a:b] = blk
        cov = cov.tocsr()
    ctn = np.array([norm(x) for x in ct]); ev_of = {e: i for i, e in enumerate(rep.index)}
    out = {}
    for ck in contrasts_keys:
        a, b = ck.split("_VS_"); r1 = np.flatnonzero(ctn == a); r2 = np.flatnonzero(ctn == b)
        if not len(r1) or not len(r2): continue
        out[ck] = (np.asarray(cov[r1].sum(0)).ravel(), np.asarray(cov[r2].sum(0)).ravel(), len(r1), len(r2), ev_of)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a51", nargs=3, required=True, help="a51_group_ds_posterior.tsv for seeds 0,1,2")
    ap.add_argument("--lc", default=f"{M}/LEAFCUTTER_DS_2026-08")
    ap.add_argument("--h5", default=f"{M}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad")
    ap.add_argument("--coverage-from", default=f"{M}/LEAFCUTTER_CMP_DELTAONLY/analysis_matched/posterior_consensus_gated.tsv.gz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    D = [pd.read_csv(p, sep="\t", usecols=["contrast", "cluster", "tv_post_mean", "q"]) for p in args.a51]
    m = D[0].merge(D[1], on=["contrast", "cluster"], suffixes=("_0", "_1")).merge(
        D[2].rename(columns={"tv_post_mean": "tv_2", "q": "q_2"}), on=["contrast", "cluster"])
    m["votes"] = (m.q_0 <= .05).astype(int) + (m.q_1 <= .05).astype(int) + (m.q_2 <= .05).astype(int)
    m["tv"] = m[["tv_post_mean_0", "tv_post_mean_1", "tv_2"]].mean(1)
    m["ckey"] = m.contrast.map(norm)
    lc = load_leafcutter(args.lc); lctv = leafcutter_tv(args.lc)
    with h5py.File(args.h5, "r") as f:
        def cat(n):
            g = f["var"][n]; c = np.array([x.decode() if isinstance(x, bytes) else x for x in g["categories"][:]]); return c[g["codes"][:]]
        ev = cat("event_id"); gn = cat("gene_name"); nj = pd.Series(ev).value_counts()
    e2g = dict(zip(ev, gn))
    rows = []
    for ck, sub in m.groupby("ckey"):
        if ck not in lc: continue
        x = sub.merge(lc[ck], on="cluster", how="left").merge(lctv[ck][["cluster", "tv_lc"]], on="cluster", how="left")
        x["lc_sig"] = x["p.adjust"] <= LC_Q; x["lfa_cons"] = x.votes >= 2; rows.append(x)
    X = pd.concat(rows); X["gene"] = X.cluster.map(e2g); X["m"] = X.cluster.map(nj)
    if args.coverage_from and os.path.exists(args.coverage_from):
        cv = pd.read_csv(args.coverage_from, sep="\t", usecols=["contrast", "cluster", "ncov1", "ncov2", "n1", "n2"]).drop_duplicates(["contrast", "cluster"])
        X = X.merge(cv, on=["contrast", "cluster"], how="left")
    else:
        cov = coverage_table(args.h5, X.ckey.unique()); out = []
        for ck, sub in X.groupby("ckey"):
            if ck not in cov: continue
            c1, c2, n1, n2, ev_of = cov[ck]; ci = sub.cluster.map(ev_of).values
            out.append(sub.assign(ncov1=c1[ci], ncov2=c2[ci], n1=n1, n2=n2))
        X = pd.concat(out)
    X["mincov"] = X[["ncov1", "ncov2"]].min(1); X["eligible"] = X.mincov >= 25
    X.to_csv(args.out, sep="\t", index=False)
    el = X[X.eligible]; a, b = el.lfa_cons, el.lc_sig
    print(f"{len(X):,} rows; eligible {len(el):,}; consensus calls {int(a.sum()):,}; both {int((a&b).sum()):,}; "
          f"LFA-only {int((a&~b).sum()):,}; LC-only {int((~a&b).sum()):,}; contrasts with 0 calls "
          f"{int((el.groupby('contrast').lfa_cons.sum()==0).sum())} of {el.contrast.nunique()}")


if __name__ == "__main__":
    main()
