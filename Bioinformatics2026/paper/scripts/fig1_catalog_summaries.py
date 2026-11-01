#!/usr/bin/env python3
"""fig1_catalog_summaries.py -- the numbers behind Figure 1 b-d, from the June 2026 build (CATALOG_NUMBERS.md).

    python scripts/fig1_catalog_summaries.py [--config paper_config.yaml]

Per dataset (TMS: 78,135 cells; Allen: 54,571 nuclei; TRAIN + TEST of the per-atlas splits, which together are all cells):
  cells; junctions, ATSEs and genes with >= 1 read in that dataset;
  per cell: junctions with reads, ATSEs with reads; per observed (cell, ATSE): reads (histogram on log2 bins + quantiles).
Catalog-wide (the shared var, 24,475 ATSEs): event type per ATSE (rules in DEFINITIONS.md 1.6) and junctions per ATSE.
Writes to <output_dir>/data/: fig1b_catalog.tsv, fig1c_event_types.tsv, fig1c_junctions_per_atse.tsv,
fig1d_per_cell.tsv.gz, fig1d_reads_per_event.tsv, fig1d_quantiles.tsv.
"""
import argparse, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp, yaml, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA/scripts/validation")
import vlib  # noqa: E402


def event_type(g):
    m = len(g); D, A, S, T = g.don.values, g.acc.values, g.s.values, g.t.values
    if m == 2:
        return "alt 3' site" if D[0] == D[1] else ("alt 5' site" if A[0] == A[1] else "complex")
    if m == 3:
        for i in range(3):
            o = [k for k in range(3) if k != i]
            if all(S[i] <= S[k] and T[k] <= T[i] for k in o):
                a, b = sorted(o, key=lambda k: S[k])
                if S[a] == S[i] and T[b] == T[i] and T[a] < S[b]:
                    return "skipped exon"
    return "complex"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default=os.path.join(HERE, "..", "paper_config.yaml"))
    cfg = yaml.safe_load(open(ap.parse_args().config))
    out = os.path.join(cfg["output_dir"], "data"); os.makedirs(out, exist_ok=True)
    P = vlib.paths("TMS")
    with h5py.File(P["train"], "r") as f:
        v = vlib._splicing_root(f)["var"]
        jid = vlib._strings(v["junction_id"]).astype(str); ev = vlib._strings(v["event_id"]).astype(str)
        gene = vlib._strings(v["gene_name"]).astype(str)
    d = pd.DataFrame(dict(j=jid, e=ev)); p = d.j.str.rsplit("_", n=3, expand=True)
    d["s"], d["t"], st = p[1].astype(int), p[2].astype(int), p[3]
    d["don"] = np.where(st == "+", d.s, d.t); d["acc"] = np.where(st == "+", d.t, d.s)
    et = d.groupby("e").apply(event_type)
    order = ["alt 3' site", "alt 5' site", "skipped exon", "complex"]
    E = et.value_counts().reindex(order).rename_axis("event_type").reset_index(name="atses")
    E["share"] = E.atses / E.atses.sum(); E.to_csv(os.path.join(out, "fig1c_event_types.tsv"), sep="\t", index=False)
    m = d.groupby("e").size().value_counts().sort_index().rename_axis("junctions_per_atse").reset_index(name="atses")
    m.to_csv(os.path.join(out, "fig1c_junctions_per_atse.tsv"), sep="\t", index=False)
    ue, inv = np.unique(ev, return_inverse=True)
    Emat = sp.csr_matrix((np.ones(len(inv)), (np.arange(len(inv)), inv)), shape=(len(inv), len(ue)))
    cat, cells_rows, hist, quant = [], [], [], []
    bins = 2.0 ** np.arange(0, 21)
    for ds in ("TMS", "Allen"):
        Pd = vlib.paths(ds)
        Y = sp.vstack([vlib.read_counts(Pd["train"]), vlib.read_counts(Pd["test"])]).tocsr()
        N = (Y @ Emat).tocsr()
        jcols = np.asarray((Y > 0).sum(0)).ravel() > 0; ecols = np.asarray((N > 0).sum(0)).ravel() > 0
        cat.append(dict(dataset=ds, cells=Y.shape[0], junctions=int(jcols.sum()), atses=int(ecols.sum()),
                        genes=len(np.unique(gene[jcols]))))
        jpc = np.diff(Y.indptr); epc = np.diff((N > 0).astype(np.int8).tocsr().indptr)
        cells_rows.append(pd.DataFrame(dict(dataset=ds, junctions_with_reads=jpc, atses_with_reads=epc)))
        r = N.data[N.data > 0]
        h, _ = np.histogram(r, bins=np.append(bins, np.inf))
        hist.append(pd.DataFrame(dict(dataset=ds, reads_from=bins.astype(int), observed_events=h)))
        for name, x in (("junctions_with_reads_per_cell", jpc), ("atses_with_reads_per_cell", epc), ("reads_per_observed_atse", r)):
            q = np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9])
            quant.append(dict(dataset=ds, quantity=name, p10=q[0], q25=q[1], median=q[2], q75=q[3], p90=q[4], n=len(x)))
        print(cat[-1], flush=True)
    pd.DataFrame(cat).to_csv(os.path.join(out, "fig1b_catalog.tsv"), sep="\t", index=False)
    pd.concat(cells_rows).to_csv(os.path.join(out, "fig1d_per_cell.tsv.gz"), sep="\t", index=False)
    pd.concat(hist).to_csv(os.path.join(out, "fig1d_reads_per_event.tsv"), sep="\t", index=False)
    pd.DataFrame(quant).to_csv(os.path.join(out, "fig1d_quantiles.tsv"), sep="\t", index=False)
    print(pd.DataFrame(quant).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
