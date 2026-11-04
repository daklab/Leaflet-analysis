"""Precompute the tables behind the splicing browser (run on Slurm; see run_precompute.sbatch). Read only on all inputs.

    python precompute.py catalog        # junction / ATSE metadata, event classes, exon structure (GENCODE vM19), md5s
    python precompute.py counts TMS     # TRAIN pseudobulk per type: junction reads, ATSE reads, cells and mice with reads
    python precompute.py counts Allen

Output goes to $V/portal_browser/precomputed/ (not in git); build_portal.py turns it into the static site.
Coordinates: a catalog junction id is chrom_start_end_strand with start = last base of the genomically left exon and
end = first base of the genomically right exon - 1, so it touches GTF exons (a, b) with b == start or a == end + 1.
"""
import gzip, hashlib, json, os, re, sys, time
from collections import defaultdict
import numpy as np, pandas as pd, scipy.sparse as sp, h5py

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/LeafletFA/scripts/validation"))
import vlib, c10_catalog as C  # noqa: E402

V = vlib.V
OUT = os.path.join(V, "portal_browser", "precomputed")
GTF = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/genome_files/gencode.vM19/genes/genes.gtf"
ATSE_FILE = ("/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper/ATSE_files/"
             "MOUSE_SMARTSEQ2_ATSE_FILE_unanno_also_2026-05-12_03-43-52.txt.gz")
META = "/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/METADATA/metadata.csv"
SUBCLASS = os.path.join(V, "tsv", "c10_Allen_group_subclass.tsv")
MAX_INTERNAL_EXONS = 15      # annotated exons inside an ATSE span that no catalog junction touches, most-supported first


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return dict(path=path, bytes=os.path.getsize(path), md5=h.hexdigest())


def read_var(ds):
    with h5py.File(vlib.paths(ds)["train"], "r") as f:
        v = vlib._splicing_root(f)["var"]
        return pd.DataFrame({k: vlib._strings(v[k]).astype(str) for k in
                             ("junction_id", "event_id", "gene_id", "gene_name", "chromosome", "strand")})


def classify(g):
    """Rules of DEFINITIONS.md section 1.6 (2026-11-01). g: columns a, b (genomic), strand of one ATSE."""
    a, b, st = g.a.values, g.b.values, g.strand.iloc[0]
    if len(g) == 2:
        donor_shared = (a[0] == a[1]) if st == "+" else (b[0] == b[1])
        accept_shared = (b[0] == b[1]) if st == "+" else (a[0] == a[1])
        return "alt3" if donor_shared else "alt5" if accept_shared else "complex"
    if len(g) == 3:
        for i in range(3):
            o = [k for k in range(3) if k != i]
            x, y = (o[0], o[1]) if a[o[0]] == a[i] else (o[1], o[0])
            if a[x] == a[i] and b[y] == b[i] and b[x] < a[y] and x != y and b[x] != b[i] and a[y] != a[i]:
                return "skipped_exon"
    return "complex"


def mxe_pairs(v):
    """STAGE5 s177 definition (SpliceVI-utils stage5_s177_mxe_pairs.py; DEFINITIONS.md 1.6), geometry only:
    four distinct junctions (L, b1), (a1, R), (L, b2), (a2, R) of a gene and strand, exon 1 [b1+1, a1] before exon 2 [b2+1, a2]."""
    v = v.assign(g=v.gene_id + "|" + v.strand)
    atse = {(g, a, b): e for g, a, b, e in zip(v.g, v.a, v.b, v.event_id)}
    linked, split, npairs = set(), defaultdict(set), [0, 0]
    for g, d in v.groupby("g"):
        byL = d.groupby("a").b.apply(list); byR = d.groupby("b").a.apply(list)
        for L, bs in byL.items():
            if len(bs) < 2:
                continue
            for R, as_ in byR.items():
                if len(as_) < 2 or R <= L:
                    continue
                ex = sorted((b, a) for b in bs for a in as_ if b < a and b < R and a > L)
                for (b1, a1) in ex:
                    for (b2, a2) in ex:
                        if a1 > b2:
                            continue
                        l, r = atse[(g, L, b1)], atse[(g, a1, R)]
                        if l == r:
                            linked.add(l); npairs[0] += 1
                        else:
                            split[l].add(r); split[r].add(l); npairs[1] += 1
    return linked, split, npairs


def read_exons(genes):
    """gene_id -> {(a, b): number of transcripts with that exon}, plus transcripts per gene"""
    ex, tx = defaultdict(lambda: defaultdict(int)), defaultdict(set)
    gid = re.compile(r'gene_id "([^"]+)"'); tid = re.compile(r'transcript_id "([^"]+)"')
    with open(GTF) as fh:
        for line in fh:
            if line[0] == "#":
                continue
            p = line.split("\t", 8)
            if p[2] != "exon":
                continue
            g = gid.search(p[8]).group(1)
            if g in genes:
                ex[g][(int(p[3]), int(p[4]))] += 1
                tx[g].add(tid.search(p[8]).group(1))
    return ex, {g: len(s) for g, s in tx.items()}


def structure(jn, exons):
    """Exons drawn for one ATSE. kind 0: touches a junction end (one per site, most transcripts first);
    kind 1: inside the span with no junction touching it (non-overlapping, most transcripts first, capped)."""
    lo, hi = jn.a.min(), jn.b.max()
    out, seen = [], set()
    left_sites, right_sites = set(jn.a), set(jn.b + 1)
    by_end, by_start = defaultdict(list), defaultdict(list)
    for (a, b), n in exons.items():
        if b in left_sites:
            by_end[b].append((n, a, b))
        if a in right_sites:
            by_start[a].append((n, a, b))
    for cands in list(by_end.values()) + list(by_start.values()):
        n, a, b = max(cands, key=lambda x: (x[0], -(x[2] - x[1])))
        if (a, b) not in seen:
            seen.add((a, b)); out.append([int(a), int(b), int(n), 0])
    touched = len(out)
    internal = sorted(((n, a, b) for (a, b), n in exons.items() if a > lo and b <= hi and (a, b) not in seen), key=lambda x: (-x[0], x[1]))
    kept = []
    for n, a, b in internal:
        if b in left_sites or a in right_sites:      # an alternative boundary variant of a site already drawn
            continue
        if any(a <= x[1] and b >= x[0] for x in out):
            continue
        out.append([int(a), int(b), int(n), 1]); kept.append(a)
        if len(kept) >= MAX_INTERNAL_EXONS:
            break
    out.sort()
    return out, touched


def run_catalog():
    A, B = read_var("TMS"), read_var("Allen")
    assert A.equals(B), "TMS and Allen var differ"
    v = A
    p = v.junction_id.str.rsplit("_", n=3, expand=True)
    v["a"], v["b"] = p[1].astype(int), p[2].astype(int)
    assert (p[0] == v.chromosome).all() and (p[3] == v.strand).all() and (v.a < v.b).all()
    z = np.load(os.path.join(C.fit_dir("TMS", 0), "fit.npz"), allow_pickle=True)
    z2 = np.load(os.path.join(C.fit_dir("Allen", 0), "fit.npz"), allow_pickle=True)
    jid, ev, inv = z["junction_id"].astype(str), z["event_id"].astype(str), np.asarray(z["inv_ev"], np.int64)
    assert np.array_equal(jid, z2["junction_id"].astype(str)) and np.array_equal(ev, z2["event_id"].astype(str))
    assert np.array_equal(inv, np.asarray(z2["inv_ev"], np.int64))
    assert np.array_equal(jid, v.junction_id.values) and np.array_equal(ev[inv], v.event_id.values)
    log(f"{len(jid):,} junctions, {len(ev):,} ATSEs; var identical in TMS and Allen and equal to fit.npz")
    A = pd.read_csv(ATSE_FILE, sep="\t", usecols=["junction_id", "annotation_status"], keep_default_na=False).drop_duplicates("junction_id")
    v["annotation_status"] = v.junction_id.map(dict(zip(A.junction_id, A.annotation_status)))
    assert v.annotation_status.notna().all()
    ann = v.annotation_status.value_counts().to_dict(); log(f"annotation status {ann}")
    v["event_idx"] = inv
    cls = {e: classify(g) for e, g in v.groupby("event_id")}
    geom = pd.Series(cls).value_counts().to_dict(); log(f"geometric classes (DEFINITIONS.md 1.6 expects alt3 5810, alt5 4867, SE 5708, complex 8090): {geom}")
    linked, split, np_ = mxe_pairs(v)
    log(f"MXE pairs linked {np_[0]}, split {np_[1]}; ATSEs with linked pair {len(linked)} (expected 4265, 303, 1034); ATSEs in split pairs {len(split)} (expected 265)")
    assert all(cls[e] == "complex" for e in linked)
    final = {e: ("mxe_linked" if e in linked else c) for e, c in cls.items()}
    fin = pd.Series(final).value_counts().to_dict(); log(f"final classes {fin}")
    genes = set(v.gene_id)
    log("reading GTF"); exons, ntx = read_exons(genes); log(f"exons for {len(exons):,} of {len(genes):,} genes")
    rows, struct = [], {}
    nsite = nsite_ok = 0
    for e, g in v.groupby("event_id", sort=False):
        ex, _ = structure(g, exons.get(g.gene_id.iloc[0], {}))
        struct[e] = ex
        ends = {x[1] for x in ex if x[3] == 0}; starts = {x[0] for x in ex if x[3] == 0}
        for s in set(g.a):
            nsite += 1; nsite_ok += s in ends
        for s in set(g.b + 1):
            nsite += 1; nsite_ok += s in starts
    log(f"junction sites touching a GENCODE vM19 exon boundary with the documented coordinate rule: {nsite_ok:,} of {nsite:,} ({100 * nsite_ok / nsite:.1f} %)")
    v[["junction_id", "event_id", "event_idx", "gene_id", "gene_name", "chromosome", "strand", "a", "b", "annotation_status"]].to_parquet(os.path.join(OUT, "catalog_junctions.parquet"))
    json.dump(dict(event_id=ev.tolist(), classes=final, split_partners={k: sorted(s) for k, s in split.items()},
                   structure=struct, transcripts_per_gene=ntx), open(os.path.join(OUT, "catalog_events.json"), "w"), separators=(",", ":"))
    rec = dict(annotation_status=ann, geometric_classes=geom, final_classes=fin, mxe_pairs_linked=np_[0], mxe_pairs_split=np_[1],
               atses_linked=len(linked), atses_in_split_pairs=len(split), sites=nsite, sites_touching_exon=int(nsite_ok))
    log("md5s")
    rec["md5"] = {k: md5(x) for k, x in dict(gtf=GTF, atse_file=ATSE_FILE, allen_metadata=META, allen_group_subclass=SUBCLASS).items()}
    json.dump(rec, open(os.path.join(OUT, "catalog_record.json"), "w"), indent=1, default=int)
    log("done")


def run_counts(ds):
    P = vlib.paths(ds)["train"]
    E = np.load(os.path.join(V, "c10", f"{ds}_s0_estimates.npz"), allow_pickle=True)
    types = E["types"].astype(str)
    z = np.load(os.path.join(C.fit_dir(ds, 0), "fit.npz"), allow_pickle=True)
    inv, n_atse = np.asarray(z["inv_ev"], np.int64), len(z["event_id"])
    cells, t, mouse = C.labels(ds, P)
    assert np.array_equal(cells, z["cell_id"].astype(str))
    log(f"{len(cells):,} TRAIN cells, {len(np.unique(t))} labels, {len(types)} types in the estimates")
    Y = vlib.read_counts(P).tocsr(); log(f"counts {Y.shape}, nnz {Y.nnz:,}")
    assert Y.shape == (len(cells), len(inv))
    code = {x: i for i, x in enumerate(types)}
    g = np.array([code.get(x, -1) for x in t]); keep = np.flatnonzero(g >= 0); g = g[keep]
    dropped = pd.Series(t[np.isin(t, types, invert=True)]).value_counts().to_dict(); log(f"labels not in the estimates: {dropped}")
    G = len(types); O = C.onehot(g, G)
    Yk = Y[keep]
    jreads = np.asarray((O @ Yk).todense(), dtype=np.float64)
    Em = sp.csr_matrix((np.ones(len(inv)), (np.arange(len(inv)), inv)), shape=(len(inv), n_atse))
    N = (Yk @ Em).tocsr(); N.eliminate_zeros(); nc = N.tocoo()
    ereads = np.asarray((O @ N).todense(), dtype=np.float64)
    cov = np.zeros((G, n_atse), np.int64); np.add.at(cov, (g[nc.row], nc.col), 1)
    _, mi = np.unique(mouse[keep], return_inverse=True)
    M = mi.max() + 1
    key = np.unique(g[nc.row].astype(np.int64) * n_atse * M + nc.col.astype(np.int64) * M + mi[nc.row])
    tc = key // M
    mice = np.zeros((G, n_atse), np.int64); np.add.at(mice, (tc // n_atse, tc % n_atse), 1)
    # checks against the saved check 10 arrays
    assert np.array_equal(ereads, np.asarray(E["reads"], np.float64)), "ATSE reads differ from the saved estimates"
    assert np.array_equal(cov, np.asarray(E["cov"], np.int64)), "cells with reads differ from the saved estimates"
    assert np.array_equal(np.bincount(g, minlength=G), np.asarray(E["cells"], np.int64))
    nm = np.array([len(np.unique(mouse[keep][g == k])) for k in range(G)])
    assert np.array_equal(nm, E["n_mice"])
    log("reads, cells with reads, cells per type, mice per type all equal the saved estimates")
    with h5py.File(P, "r") as f:
        o = f["obs"]; tissue = vlib._strings(o["tissue"]).astype(str); broad = vlib._strings(o["broad_cell_type"]).astype(str)
        onto = vlib._strings(o["cell_ontology_class"]).astype(str)
    meta = []
    if ds == "Allen":
        m = pd.read_csv(META, usecols=["cluster_label", "class_label"], low_memory=False).drop_duplicates()
        c2c = dict(zip(m.cluster_label, m.class_label)); S = pd.read_csv(SUBCLASS, sep="\t").set_index("cell_group")
        cl = pd.Series(onto[keep]).map(c2c).values
    for k, x in enumerate(types):
        r = g == k
        d = dict(type=x, cells=int(r.sum()), n_mice=int(nm[k]), eligible=bool(E["eligible"][k]))
        if ds == "TMS":
            ts, bs = pd.Series(tissue[keep][r]).value_counts(), pd.Series(broad[keep][r]).value_counts()
            assert len(ts) == 1 and len(bs) == 1 and x == f"{ts.index[0]}_{bs.index[0]}", x
            d.update(tissue=ts.index[0], cell_type=bs.index[0])
        else:
            c = pd.Series(cl[r]).value_counts(dropna=False)
            d.update(allen_class=c.index[0], allen_class_fraction=float(c.iloc[0] / c.sum()), subclass=S.official_subclass.get(x),
                     subclass_fraction=float(S.official_subclass_fraction.get(x)))
        meta.append(d)
    np.savez_compressed(os.path.join(OUT, f"{ds}_train_tables.npz"), types=types, jreads=jreads.astype(np.uint32),
                        ereads=ereads.astype(np.uint32), cov=cov.astype(np.uint16), mice=mice.astype(np.uint16))
    # cross-check the export against the estimates; do not ship its calls
    X = pd.read_csv(os.path.join(V, "c10", "export", f"catalog_{ds}_s0.tsv.gz"), sep="\t",
                    usecols=["event_id", "junction_id", "cell_type", "psi", "psi_lo", "psi_hi", "reads", "cells_covering"])
    jidx = {j: i for i, j in enumerate(z["junction_id"].astype(str))}
    ti, ji = X.cell_type.map(code).values, X.junction_id.map(jidx).values
    assert np.isfinite(ti).all() and np.isfinite(ji).all()
    ti, ji = ti.astype(int), ji.astype(int)
    dmax = {k: float(np.nanmax(np.abs(X[c].values - E[k][ti, ji]))) for c, k in (("psi", "point"), ("psi_lo", "lo"), ("psi_hi", "hi"))}
    rmax = float(np.abs(X.reads.values - ereads[ti, inv[ji]]).max())
    log(f"export vs estimates: {len(X):,} rows (full grid {G * len(inv):,}); max abs diff {dmax}; reads {rmax}")
    rec = dict(dataset=ds, train_cells=len(cells), types=len(types), labels_not_in_estimates=dropped, types_meta=meta,
               export_rows=len(X), export_vs_estimates_max_abs_diff=dmax, export_reads_max_abs_diff=rmax)
    log("md5s")
    rec["md5"] = dict(train=md5(P), estimates=md5(os.path.join(V, "c10", f"{ds}_s0_estimates.npz")),
                      export=md5(os.path.join(V, "c10", "export", f"catalog_{ds}_s0.tsv.gz")),
                      fit=md5(os.path.join(C.fit_dir(ds, 0), "fit.npz")), run_record=md5(os.path.join(os.path.dirname(C.fit_dir(ds, 0)), "run_record.json")))
    rr = json.load(open(os.path.join(os.path.dirname(C.fit_dir(ds, 0)), "run_record.json")))
    assert rr["inputs_md5"]["train"]["md5"] == rec["md5"]["train"]["md5"], "train h5mu md5 differs from the fit's run_record"
    rec["fit_record"] = dict(leafletfa_git=rr["leafletfa_git"], slurm_job=rr["slurm_job"], started=rr["started"], finished=rr["finished"],
                             K=rr["settings"]["K"], seed=rr["settings"]["seed"], resumed_from_step=rr["stages"]["fit"]["resumed_from_step"],
                             steps_run=rr["stages"]["fit"]["steps_run"], train_md5_matches_run_record=True)
    json.dump(rec, open(os.path.join(OUT, f"{ds}_record.json"), "w"), indent=1, default=str)
    log("done")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    run_catalog() if sys.argv[1] == "catalog" else run_counts(sys.argv[2])
