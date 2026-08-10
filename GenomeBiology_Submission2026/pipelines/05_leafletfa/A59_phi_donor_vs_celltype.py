#!/usr/bin/env python
"""A59: does phi encode CELL STATE, or does it encode which mouse the cell came from?

Why ask
-------
A58 found that which junction a cell commits to associates with `mouse.id` at 10x the null rate
(0.104 of ATSEs at p<0.01, against ~0.01 expected), while coverage sits at the null (0.018). So
there is real donor-level structure in the splicing data. If phi absorbs it, a program that is
really "mouse 7" would look like biology on a UMAP, and any cell-type interpretation built on it
would be partly an artefact.

The unit of replication, which is the whole point
-------------------------------------------------
`mouse.id`, `age` and `sex` are properties of the DONOR, not of the cell. With 132,706 cells,
treating cells as independent replicates makes any donor-level association overwhelmingly
"significant" -- that is pseudo-replication, not evidence. So everything here aggregates phi to
(mouse, cell type) means FIRST and uses MICE as the replicates.

Age is confounded with dataset, measured
----------------------------------------
    age   allen_brain   tabula_muris_senis   n mice
    2m         54,571                    0      519
    3m              0               32,618       10
    18m             0               24,009        4
    24m             0               21,508        4

Every 2m cell is allen_brain and every older cell is tabula_muris_senis, so a young-vs-old contrast
that includes 2m is a LAB contrast wearing an age label. Age is therefore tested only within TMS,
where the replication is 10 / 4 / 4 mice. That is the honest sample size for an aging claim.
"""

import argparse, os, sys

import h5py
import numpy as np
import pandas as pd
from scipy.stats import kruskal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _phi_rows import phi_row_of                                    # noqa: E402

WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
H5 = f"{WD}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad"


def cat(f, name):
    x = f["obs"][name]
    if isinstance(x, h5py.Group):
        c = np.array([y.decode() if isinstance(y, bytes) else y for y in x["categories"][:]])
        return c[x["codes"][:]]
    return np.array([y.decode() if isinstance(y, bytes) else y for y in x[:]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--min-cells", type=int, default=25,
                    help="cells required for a (mouse, cell type) mean to be used")
    ap.add_argument("--min-mice", type=int, default=3)
    ap.add_argument("--out", default=f"{WD}/ATLAS_DM/a59_phi_donor_vs_celltype.tsv")
    args = ap.parse_args()

    z = np.load(args.fit, allow_pickle=True)
    phi = np.asarray(z["phi"], float)
    with h5py.File(H5, "r") as f:
        meta = pd.DataFrame({k: cat(f, k) for k in
                             ["mouse.id", "age", "sex", "dataset", "tissue_cell_type",
                              "broad_cell_type"]})
    N = len(meta)
    PROW = phi_row_of(z, N)
    keep = PROW >= 0
    meta = meta[keep].reset_index(drop=True)
    P = phi[PROW[keep]]
    K = P.shape[1]
    print(f"phi {P.shape}, {meta['mouse.id'].nunique()} mice, "
          f"{meta['tissue_cell_type'].nunique()} cell types\n", flush=True)

    # ---- (mouse, cell type) means: the unit everything below uses ------------------------
    meta["_i"] = np.arange(len(meta))
    grp = meta.groupby(["tissue_cell_type", "mouse.id"])["_i"].apply(np.array)
    rows = []
    for (ct, mouse), idx in grp.items():
        if len(idx) < args.min_cells:
            continue
        r = dict(cell_type=ct, mouse=mouse, n_cells=len(idx),
                 age=meta.age.values[idx[0]], dataset=meta.dataset.values[idx[0]],
                 sex=meta.sex.values[idx[0]])
        for k in range(K):
            r[f"phi{k}"] = float(P[idx, k].mean())
        rows.append(r)
    D = pd.DataFrame(rows)
    print(f"(mouse, cell type) groups with >= {args.min_cells} cells: {len(D):,}", flush=True)
    if D.empty:
        print("nothing to test"); return 1

    pcols = [f"phi{k}" for k in range(K)]

    # ---- 1. variance decomposition: cell type vs donor-within-cell-type ------------------
    # Both computed on the SAME (mouse, cell type) means, so they are directly comparable.
    tot, between_ct, within_ct_between_mouse = 0.0, 0.0, 0.0
    for c in pcols:
        v = D[c].values
        gm = v.mean()
        tot += float(((v - gm) ** 2).sum())
        ct_means = D.groupby("cell_type")[c].transform("mean").values
        between_ct += float(((ct_means - gm) ** 2).sum())
        within_ct_between_mouse += float(((v - ct_means) ** 2).sum())
    print("\n=== where does phi's variance live? (on mouse x cell-type means) ===")
    print(f"  between CELL TYPES              {100*between_ct/tot:5.1f}%")
    print(f"  between MICE within a cell type {100*within_ct_between_mouse/tot:5.1f}%")
    print("  a program encoding donor rather than state would put most of it in the second row")

    # ---- 2. age, within TMS, with MICE as replicates -------------------------------------
    print("\n=== age, tested within tabula_muris_senis, MICE as replicates ===")
    print("(2m is excluded: every 2m cell is allen_brain, so including it tests LAB, not age)")
    T = D[D.dataset == "tabula_muris_senis"]
    out = []
    for ct, d in T.groupby("cell_type"):
        ages = d.groupby("age")["mouse"].nunique()
        usable = ages[ages >= args.min_mice].index.tolist()
        if len(usable) < 2:
            continue
        dd = d[d.age.isin(usable)]
        for k in range(K):
            groups = [g[f"phi{k}"].values for _, g in dd.groupby("age")]
            if min(len(x) for x in groups) < args.min_mice:
                continue
            try:
                p = float(kruskal(*groups).pvalue)
            except ValueError:
                continue
            out.append(dict(cell_type=ct, program=k + 1, p=p,   # 1-indexed for reporting
                            n_mice=int(dd["mouse"].nunique()),
                            ages="/".join(usable)))
    A = pd.DataFrame(out)
    if A.empty:
        print("  no cell type has >= %d mice in >= 2 age groups -- the aging claim is not"
              % args.min_mice)
        print("  testable at donor-level replication on this data")
    else:
        A["q"] = np.minimum(1, A.p * len(A))                      # Bonferroni, deliberately blunt
        print(f"  {len(A)} (cell type, program) tests over "
              f"{A.cell_type.nunique()} cell types, median {A.n_mice.median():.0f} mice each")
        print(f"  significant at q<0.05: {int((A.q < 0.05).sum())}")
        print(A.sort_values("p").head(10).to_string(index=False))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    D.to_csv(args.out, sep="\t", index=False)
    if not A.empty:
        A.to_csv(args.out.replace(".tsv", "_age.tsv"), sep="\t", index=False)
    print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
