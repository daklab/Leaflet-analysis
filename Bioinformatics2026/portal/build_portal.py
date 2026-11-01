"""Build the static splicing-catalog portal from catalog export tables.

    python build_portal.py CATALOG.tsv.gz [CATALOG2.tsv.gz ...] [--out site] [--zenodo URL]
    python -m http.server -d site 8000        # local preview (fetch() needs http, not file://)

Input: the catalog table of check 10.5, one row per (dataset, event, cell type, junction) with columns
dataset, event_id, junction_id, gene, annotation_status, cell_type, tissue, lineage, psi, psi_lo, psi_hi, psi_rest,
reads, cells, cells_covering, tv, specific, source.
Output (all static):
  site/index.html, app.js, style.css             the app (copied from web/)
  site/data/index.json                           datasets, genes, cell types (tissue, lineage, cells), source labels
  site/data/genes/<dataset>/<gene>.json          per gene: events, junctions, per cell type psi / interval / flags
  site/data/celltypes/<dataset>/<slug>.json      per cell type: every event flagged specific (for the cell-type view)
Prints the build report: build time, total size, file count, largest file, and a Parquet size estimate.
"""
import argparse, json, os, re, shutil, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def r3(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 3)


def junction(jid):
    c, s, e, st = jid.rsplit("_", 3)
    return dict(id=jid, chrom=c, start=int(s), end=int(e), strand=st)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("catalogs", nargs="+")
    ap.add_argument("--out", default=os.path.join(HERE, "site"))
    ap.add_argument("--zenodo", default="", help="Zenodo record URL for the download links (empty until the release)")
    a = ap.parse_args()
    t0 = time.time()
    C = pd.concat([pd.read_csv(p, sep="\t") for p in a.catalogs], ignore_index=True)
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    shutil.copytree(os.path.join(HERE, "web"), a.out)
    index = dict(datasets={}, zenodo=a.zenodo, built=time.strftime("%Y-%m-%d"))
    for ds, D in C.groupby("dataset"):
        ct = (D.groupby("cell_type").agg(tissue=("tissue", "first"), lineage=("lineage", "first"), cells=("cells", "max"))
              .reset_index())
        ct["slug"] = ct.cell_type.map(slug)
        index["datasets"][ds] = dict(source=sorted(D.source.unique().tolist()), genes=sorted(D.gene.unique().tolist()),
                                     cell_types=ct.to_dict("records"))
        for g, G in D.groupby("gene"):
            events = []
            for e, Ev in G.groupby("event_id"):
                js = sorted(Ev.junction_id.unique(), key=lambda j: junction(j)["start"])
                jpos = {j: i for i, j in enumerate(js)}
                per = []
                for t, X in Ev.groupby("cell_type"):
                    X = X.set_index("junction_id").reindex(js)
                    per.append(dict(ct=t, cells=int(X.cells.max()), cov=int(np.nanmax(X.cells_covering)),
                                    reads=int(np.nanmax(X.reads)), tv=r3(X.tv.max()), specific=bool(X.specific.fillna(False).any()),
                                    psi=[r3(v) for v in X.psi], lo=[r3(v) for v in X.psi_lo], hi=[r3(v) for v in X.psi_hi],
                                    rest=[r3(v) for v in X.psi_rest]))
                events.append(dict(event_id=e, annotation=Ev.annotation_status.iloc[0],
                                   junctions=[junction(j) for j in js], cell_types=per))
            write_json(os.path.join(a.out, "data", "genes", ds, f"{slug(g)}.json"), dict(gene=g, dataset=ds, events=events))
        S = D[D.specific.astype(bool)].copy()
        S["delta"] = (S.psi - S.psi_rest).abs()
        S = (S.sort_values("delta", ascending=False).groupby(["cell_type", "event_id"]).first().reset_index())
        for t, X in S.groupby("cell_type"):
            write_json(os.path.join(a.out, "data", "celltypes", ds, f"{slug(t)}.json"),
                       [dict(gene=r.gene, event_id=r.event_id, junction=r.junction_id, delta=r3(r.delta), tv=r3(r.tv),
                             psi=r3(r.psi), rest=r3(r.psi_rest), reads=int(r.reads), cov=int(r.cells_covering))
                        for r in X.itertuples()])
    write_json(os.path.join(a.out, "data", "index.json"), index)
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(a.out) for f in fs]
    sizes = {f: os.path.getsize(f) for f in files}
    big = max(sizes, key=sizes.get)
    pq = os.path.join(HERE, "_size_probe.parquet")
    try:
        C.to_parquet(pq, index=False); pq_mb = os.path.getsize(pq) / 1e6; os.remove(pq)
    except Exception as ex:  # noqa: BLE001
        pq_mb = f"n/a ({type(ex).__name__})"
    print(f"build time {time.time() - t0:.1f} s; {len(files)} files; total {sum(sizes.values()) / 1e6:.2f} MB; "
          f"largest {os.path.relpath(big, a.out)} {sizes[big] / 1e3:.1f} kB; "
          f"catalog rows {len(C):,}; Parquet of these rows {pq_mb if isinstance(pq_mb, str) else f'{pq_mb:.2f} MB'}")


if __name__ == "__main__":
    main()
