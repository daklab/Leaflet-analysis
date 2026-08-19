#!/usr/bin/env python
"""A69: which splicing program tracks age, on the shipped atlas fit (TMS cells only).

Design notes (from the atlas composition):
- allen_brain is all 2m and one lab -> EXCLUDED; ages are TMS 3m/18m/24m.
- 24m is male-only in TMS, so the primary test is MALES ONLY, per-MOUSE (not per-cell): mean phi_k
  per mouse, Mann-Whitney 3m vs 24m. Cell-level tests would pseudo-replicate.
- Secondary: per tissue_cell_type (>= 20 cells in each of >= 2 ages), Spearman of phi_k vs age in
  months across cells -> how broadly the trend holds; and the old-minus-young (24m-3m, males) delta
  of mean phi per tissue_cell_type for the dot heatmap.
- Program content: junctions where the aging program's psi differs most from the pi-weighted mean
  of the other programs (|dpsi|), reported with gene names; the honest "what is this program".

Outputs per fit: a69_<run>_program_age.tsv (per-program stats), a69_<run>_delta_bytype.tsv,
a69_<run>_top_junctions.tsv. Run over the 3 shipped seeds; SP numbering per fit via programs.py.
"""
import argparse, os, sys
import numpy as np, pandas as pd, h5py, torch
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/_common")
from _phi_rows import phi_row_of
from programs import sp_rank
from leafletfa.dirichlet_multinomial import segment_softmax

M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"


def cat(g):
    if hasattr(g, "keys") and "categories" in g:
        c = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]); return c[g["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--h5", default=f"{M}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad")
    ap.add_argument("--out", default=f"{M}/ATLAS_DM/aging")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    run = os.path.basename(os.path.dirname(args.fit))
    z = np.load(args.fit, allow_pickle=True)
    phi = np.asarray(z["phi"]); K = phi.shape[1]; spr = sp_rank(z)
    with h5py.File(args.h5, "r") as f:
        obs = {k: cat(f["obs"][k]) for k in ["dataset", "age", "sex", "mouse.id", "tissue_cell_type", "tissue"]}
        gene = cat(f["var"]["gene_name"]); ev = cat(f["var"]["event_id"]); jid = cat(f["var"]["junction_id"])
    prow = phi_row_of(z, len(obs["age"]))
    tms = (obs["dataset"] == "tabula_muris_senis") & (prow >= 0)
    age_m = pd.Series(obs["age"]).str.replace("m", "").astype(float).values
    df = pd.DataFrame(dict(age=obs["age"], age_m=age_m, sex=obs["sex"], mouse=obs["mouse.id"],
                           tct=obs["tissue_cell_type"]))[tms].reset_index(drop=True)
    P = phi[prow[np.flatnonzero(tms)]]
    rows = []
    male = df.sex == "male"
    for k in range(K):
        df["p"] = P[:, k]
        mice = df[male & df.age.isin(["3m", "24m"])].groupby(["mouse", "age"]).p.mean().reset_index()
        a = mice[mice.age == "3m"].p.values; b = mice[mice.age == "24m"].p.values
        pv = mannwhitneyu(a, b).pvalue if len(a) >= 3 and len(b) >= 3 else np.nan
        # breadth: per tissue_cell_type Spearman across cells (all sexes, 3 ages)
        rs = []
        for t, s in df.groupby("tct"):
            if s.age.nunique() >= 2 and (s.groupby("age").size() >= 20).sum() >= 2:
                r = spearmanr(s.age_m, s.p).correlation
                if np.isfinite(r): rs.append(r)
        rows.append(dict(fit=run, prog_fit=k, SP=f"SP{spr[k]}", p_mouse_3v24_male=pv,
                         mean_3m=np.mean(a) if len(a) else np.nan, mean_24m=np.mean(b) if len(b) else np.nan,
                         n_types=len(rs), frac_types_up=np.mean(np.array(rs) > 0) if rs else np.nan,
                         median_type_rho=np.median(rs) if rs else np.nan))
    res = pd.DataFrame(rows).sort_values("p_mouse_3v24_male")
    res.to_csv(f"{args.out}/a69_{run}_program_age.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220); print(res.round(4).to_string(index=False))
    # delta heatmap input: old (24m male) - young (3m male) mean phi per tissue_cell_type
    top = int(res.iloc[0].prog_fit)
    d24 = df[male & (df.age == "24m")].assign(**{f"p{k}": P[male.values & (df.age == "24m").values][:, k] for k in range(K)})
    out = []
    for t in sorted(df.tct.unique()):
        y = df[male & (df.age == "3m") & (df.tct == t)]; o = df[male & (df.age == "24m") & (df.tct == t)]
        if len(y) >= 20 and len(o) >= 20:
            dy = P[(male & (df.age == "3m") & (df.tct == t)).values].mean(0)
            do = P[(male & (df.age == "24m") & (df.tct == t)).values].mean(0)
            for k in range(K):
                out.append(dict(tct=t, SP=f"SP{spr[k]}", prog_fit=k, delta=do[k] - dy[k], n_young=len(y), n_old=len(o)))
    pd.DataFrame(out).to_csv(f"{args.out}/a69_{run}_delta_bytype.tsv", sep="\t", index=False)
    # content of the top program: |psi_top - pi-weighted mean of others| per junction
    inv = np.asarray(z["inv_ev"]).astype(np.int64)
    psi = segment_softmax(torch.from_numpy(np.asarray(z["psi_loc"], np.float32)), torch.from_numpy(inv), len(np.asarray(z["event_id"]))).numpy()
    pi = np.asarray(z["trace_pi"])[-1]; w = pi.copy(); w[top] = 0; w /= w.sum()
    dpsi = psi[top] - w @ psi
    o = np.argsort(-np.abs(dpsi))[:200]
    pd.DataFrame(dict(junction=jid[o], gene=gene[o], event=ev[o], psi_prog=psi[top, o].round(4),
                      psi_rest=(w @ psi)[o].round(4), dpsi=dpsi[o].round(4))).to_csv(
        f"{args.out}/a69_{run}_top_junctions.tsv", sep="\t", index=False)
    print(f"\ntop program by mouse-level 3m-vs-24m: fit prog {top} = SP{spr[top]}; "
          f"top genes: {', '.join(pd.unique(gene[o])[:20])}")


if __name__ == "__main__":
    main()
