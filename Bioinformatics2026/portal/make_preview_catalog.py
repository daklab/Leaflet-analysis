"""PREVIEW catalog for building and testing the portal before check 10 exists.

    python make_preview_catalog.py [--genes 100] [--out preview_catalog.tsv.gz]

Writes the catalog table in the export schema of check 10.5 (one row per event x cell type x junction), but the values
are RAW TMS TRAIN pseudobulk, not LeafletFA: psi = sum y / sum N; interval = Beta(y + 0.5, N - y + 0.5) 2.5-97.5 %;
psi_rest = pseudobulk of all other cells; specific = TV(psi, psi_rest) > 0.2 with >= 20 reads and >= 3 covering cells.
Every row carries source = "PREVIEW raw pseudobulk". Genes: Fgfr2, Ptprc and a seeded random sample of the rest.
"""
import argparse, os, sys
import numpy as np, pandas as pd, h5py, scipy.sparse as sp
from scipy.stats import beta

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA/scripts/validation")
import vlib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(HERE, "preview_catalog.tsv.gz"))
    a = ap.parse_args()
    P = vlib.paths("TMS")["train"]
    with h5py.File(P, "r") as f:
        o = f["obs"]
        tct = vlib._strings(o["tissue_cell_type"]).astype(str)
        tissue = vlib._strings(o["tissue"]).astype(str)
        broad = vlib._strings(o["broad_cell_type"]).astype(str)
        v = vlib._splicing_root(f)["var"]
        jid = vlib._strings(v["junction_id"]).astype(str); ev = vlib._strings(v["event_id"]).astype(str)
        gene = vlib._strings(v["gene_name"]).astype(str)
        ann = vlib._strings(v["annotation_status"]).astype(str) if "annotation_status" in v else np.array(["NA"] * len(jid))
    Y = vlib.read_counts(P)
    rng = np.random.default_rng(0)
    ug = np.unique(gene)
    pick = set(["Fgfr2", "Ptprc"]) | set(rng.choice(ug, a.genes - 2, replace=False))
    cols = np.flatnonzero(np.isin(gene, list(pick)))
    Ys = sp.csc_matrix(Y)[:, cols].tocsr()
    evs = ev[cols]; ue, inv = np.unique(evs, return_inverse=True)
    E = sp.csr_matrix((np.ones(len(cols)), (np.arange(len(cols)), inv)), shape=(len(cols), len(ue)))
    Ncell = (Ys @ E).tocsr()                                         # cells x events
    types, ti = np.unique(tct, return_inverse=True)
    T = sp.csr_matrix((np.ones(len(ti)), (ti, np.arange(len(ti)))), shape=(len(types), len(ti)))
    y_t = np.asarray((T @ Ys).todense()); N_t = np.asarray((T @ Ncell).todense())[:, inv]
    cov_t = np.asarray((T @ (Ncell > 0).astype(float)).todense())[:, inv]
    y_all, N_all = y_t.sum(0), N_t.sum(0)
    ncell = np.bincount(ti)
    meta = pd.DataFrame({"cell_type": types}).merge(
        pd.DataFrame(dict(cell_type=tct, tissue=tissue, lineage=broad)).drop_duplicates("cell_type"), on="cell_type")
    rows = []
    for k, t in enumerate(types):
        if ncell[k] < 10:
            continue
        y, N = y_t[k], N_t[k]
        with np.errstate(invalid="ignore", divide="ignore"):
            psi = y / N; rest = (y_all - y) / (N_all - N)
        lo = beta.ppf(0.025, y + 0.5, np.maximum(N - y, 0) + 0.5); hi = beta.ppf(0.975, y + 0.5, np.maximum(N - y, 0) + 0.5)
        d = pd.DataFrame(dict(event_id=evs, junction_id=jid[cols], gene=gene[cols], annotation_status=ann[cols],
                              cell_type=t, psi=psi, psi_lo=lo, psi_hi=hi, psi_rest=rest, reads=N, cells=int(ncell[k]),
                              cells_covering=cov_t[k]))
        d = d[d.reads > 0]
        tv = (d.psi - d.psi_rest).abs().groupby(d.event_id).sum() * 0.5
        d["tv"] = d.event_id.map(tv)
        d["specific"] = (d.tv > 0.2) & (d.reads >= 20) & (d.cells_covering >= 3)
        rows.append(d)
    C = pd.concat(rows).merge(meta, on="cell_type")
    C.insert(0, "dataset", "TMS"); C["source"] = "PREVIEW raw pseudobulk"
    C.to_csv(a.out, sep="\t", index=False, compression="gzip")
    print(f"{len(C):,} rows, {C.gene.nunique()} genes, {C.event_id.nunique()} events, {C.cell_type.nunique()} cell types -> {a.out}")


if __name__ == "__main__":
    main()
