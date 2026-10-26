#!/usr/bin/env python
# Run once on 2026-08-20 (outputs written 01:30). Result: mouse vs human dPSI Pearson r 0.25-0.62 across the four contrasts
# (exc-inh 0.42, exc-astro 0.62, inh-astro 0.62, astro-oligo 0.25); 27-49 % of mouse DS junctions conserved in the same direction.
# Outputs: /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/CROSS_SPECIES_2026/conserved_DS_summary.tsv and
#          conserved_DS_per_junction.tsv.gz (same folder). Not used by any figure as of 2026-10-25 (exploratory).
"""Are the ATSEs that are differentially spliced between two cell classes in MOUSE also DS -- in the
same direction -- between the comparable classes in HUMAN?

Comparable classes (both species' Allen Brain data): excitatory neurons (mouse EXCITATORY NEURON /
HIPPOCAMPAL NEURON; human IT, L4 IT, L5 ET, L5/6 IT Car3, L5/6 NP, L6 CT, L6b), inhibitory neurons
(mouse INHIBITORY NEURON; human PVALB, SST, VIP, LAMP5, PAX6), astrocytes, oligodendrocyte lineage
(oligodendrocyte + OPC). Contrasts: exc-inh, exc-astro, inh-astro, astro-oligo.

Model-free, data-level test on the matched junctions (00_): per species and contrast, pooled observed
junction usage per class (junction reads / event reads, summed over cells), keeping junctions whose
event has >= 200 reads in BOTH classes of BOTH species. dPSI = usage_A - usage_B per junction;
conservation = correlation of mouse dPSI vs human dPSI across junctions, plus per-junction classes
(|dPSI| >= 0.15 in: both same direction / both opposite / mouse only / human only / neither).
Output: per-contrast summary + per-junction table for example picking.
"""
import os, sys
import numpy as np, pandas as pd, h5py
from scipy.stats import spearmanr
W = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD"

MOUSE_CLASS = {
    "exc": lambda t: ("EXCITATORY NEURON" in t) or ("HIPPOCAMPAL NEURON" in t),
    "inh": lambda t: "INHIBITORY NEURON" in t,
    "astro": lambda t: "ASTROCYTE" in t,
    "oligo": lambda t: ("OLIGODENDROCYTE" in t) or t.endswith("_OPC"),
}
HUMAN_CLASS = {
    "exc": lambda t: any(t.endswith("_" + k) for k in ("IT", "L4 IT", "L5 ET", "L5/6 IT Car3", "L5/6 NP", "L6 CT", "L6b")),
    "inh": lambda t: any(t.endswith("_" + k) for k in ("PVALB", "SST", "VIP", "LAMP5", "PAX6")),
    "astro": lambda t: t.endswith("_Astrocyte"),
    "oligo": lambda t: t.endswith("_Oligodendrocyte") or t.endswith("_OPC"),
}
CONTRASTS = [("exc", "inh"), ("exc", "astro"), ("inh", "astro"), ("astro", "oligo")]


def cat(g):
    if hasattr(g, "keys") and "categories" in g:
        c = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]); return c[g["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])


def pooled(h5, jcols, class_of, dataset_field=None, dataset_keep=None):
    """per class: (sum junction reads, sum event reads) per matched junction column"""
    with h5py.File(h5, "r") as f:
        tct = cat(f["obs"]["tissue_cell_type"])
        keep = np.ones(len(tct), bool)
        if dataset_field:
            keep = cat(f["obs"][dataset_field]) == dataset_keep
        cls = np.full(len(tct), "", object)
        for name, fn in class_of.items():
            cls[[fn(t) for t in tct]] = np.where(keep[[fn(t) for t in tct]], name, "")
        out = {}
        for name in class_of:
            rows = np.flatnonzero(cls == name)
            Y = np.zeros(len(jcols)); C = np.zeros(len(jcols))
            for lay, acc in [("cell_by_junction_matrix", Y), ("cell_by_cluster_matrix", C)]:
                g = f["layers"][lay]; ip = g["indptr"][:]
                cset = {c_: k for k, c_ in enumerate(jcols)}
                for r in rows:
                    s, e = int(ip[r]), int(ip[r + 1]); idx = g["indices"][s:e]; dat = g["data"][s:e]
                    m = np.isin(idx, jcols)
                    if m.any():
                        kk = np.array([cset[c_] for c_ in idx[m]]); np.add.at(acc, kk, dat[m])
            out[name] = (Y, C, len(rows))
            print(f"  {os.path.basename(h5)[:20]} {name}: {len(rows):,} cells", flush=True)
    return out


def main():
    mp = pd.read_csv(f"{W}/CROSS_SPECIES_2026/junction_map_mouse_human_2026.tsv.gz", sep="\t")
    with h5py.File(f"{W}/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad", "r") as f:
        mj = {j: i for i, j in enumerate(cat(f["var"]["junction_id"]))}
    with h5py.File(f"{W}/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/20260617/human_splice_model_ready_2026-06-17.h5ad", "r") as f:
        hj = {j: i for i, j in enumerate(cat(f["var"]["junction_id"]))}
    mp = mp[mp.mouse_junction.isin(mj) & mp.human_junction.isin(hj)].reset_index(drop=True)
    mcols = mp.mouse_junction.map(mj).values; hcols = mp.human_junction.map(hj).values
    P_m = pooled(f"{W}/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad",
                 mcols, MOUSE_CLASS, "dataset", "allen_brain")
    P_h = pooled(f"{W}/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/20260617/human_splice_model_ready_2026-06-17.h5ad",
                 hcols, HUMAN_CLASS)
    rows = []; per = []
    for a, b in CONTRASTS:
        Ya, Ca, _ = P_m[a]; Yb, Cb, _ = P_m[b]; Xa, Da, _ = P_h[a]; Xb, Db, _ = P_h[b]
        ok = (Ca >= 200) & (Cb >= 200) & (Da >= 200) & (Db >= 200)
        dm = Ya[ok] / Ca[ok] - Yb[ok] / Cb[ok]; dh = Xa[ok] / Da[ok] - Xb[ok] / Db[ok]
        r = np.corrcoef(dm, dh)[0, 1]; rho = spearmanr(dm, dh).correlation
        big_m = np.abs(dm) >= 0.15; big_h = np.abs(dh) >= 0.15; same = np.sign(dm) == np.sign(dh)
        n_cons = int((big_m & big_h & same).sum()); n_opp = int((big_m & big_h & ~same).sum())
        rows.append(dict(contrast=f"{a}_vs_{b}", n_junctions=int(ok.sum()), pearson=r, spearman=rho,
                         mouse_DS=int(big_m.sum()), human_DS=int(big_h.sum()),
                         conserved_same_dir=n_cons, both_DS_opposite=n_opp,
                         mouse_only=int((big_m & ~big_h).sum()), human_only=int((big_h & ~big_m).sum()),
                         frac_mouse_DS_conserved=round(n_cons / max(big_m.sum(), 1), 3)))
        sub = mp[ok].copy(); sub["contrast"] = f"{a}_vs_{b}"; sub["dpsi_mouse"] = dm; sub["dpsi_human"] = dh
        per.append(sub)
    S = pd.DataFrame(rows); pd.set_option("display.width", 220)
    print(S.to_string(index=False))
    S.to_csv(f"{W}/CROSS_SPECIES_2026/conserved_DS_summary.tsv", sep="\t", index=False)
    pd.concat(per).to_csv(f"{W}/CROSS_SPECIES_2026/conserved_DS_per_junction.tsv.gz", sep="\t", index=False)


if __name__ == "__main__":
    main()
