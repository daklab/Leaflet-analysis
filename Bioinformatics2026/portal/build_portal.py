"""Build the static cell-type splicing browser from the precomputed tables and the saved check 10 estimates.

    python precompute.py catalog; python precompute.py counts TMS; python precompute.py counts Allen   # Slurm, see README
    python build_portal.py all       # data shards + web files (Slurm; about 5 min)
    python build_portal.py web       # only re-copy web/ into the site (seconds, no data read)
    python -m http.server -d $V/portal_browser/site 8000

The check 10 `specific`, `tv` and `p_tv` arrays are never read or written. See README.md for the file layout.
"""
import argparse, glob, hashlib, json, os, shutil, sys, time
import numpy as np, pandas as pd

V = "/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation"
HERE = os.path.dirname(os.path.abspath(__file__))
PRE = os.path.join(V, "portal_browser", "precomputed")
VERSION = "preview-2026-11-06"
BUCKET = 64                     # ATSEs per event shard
RANGE_READS, RANGE_CELLS = 20, 3  # support required to enter the "range in other types" columns (fixed at build)
NA = 65535
CLASSES = ["alt3", "alt5", "skipped_exon", "mxe_linked", "complex"]
ANN = ["both", "five_prime", "three_prime", "unannotated"]
TYPE_LAYOUT = [["reads", "u32", "E"], ["cov", "u16", "E"], ["mice", "u16", "E"], ["psi", "u16", "J"], ["lo", "u16", "J"], ["hi", "u16", "J"], ["raw", "u16", "J"],
               ["rest", "u16", "J"], ["rest_lo", "u16", "J"], ["rest_hi", "u16", "J"], ["oth_min", "u16", "J"], ["oth_max", "u16", "J"]]
ALLEN_CLASS = {"Glutamatergic": "excitatory", "GABAergic": "inhibitory"}          # anything else: non-neuronal
SS4 = {"Nrxn1": ("ENSMUSG00000024109.18_atse_2", 90992115, 90992144), "Nrxn2": ("ENSMUSG00000033768.16_atse_1", 6448255, 6448284),
       "Nrxn3": ("ENSMUSG00000066392.11_atse_2", 88850279, 88850308)}             # c10_known_biology_fix.py


def enc(x):
    x = np.asarray(x, np.float64)
    out = np.full(x.shape, NA, np.uint16)
    ok = np.isfinite(x)
    out[ok] = np.round(np.clip(x[ok], 0, 1) * 10000).astype(np.uint16)
    return out


def wj(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, separators=(",", ":"))


def wb(path, arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        for a in arrays:
            fh.write(a.tobytes())


def loo_extreme(P, mask):
    """per junction: lowest and highest of P over masked types, leaving each type out. P, mask: T x J."""
    Q = np.where(mask, P, np.inf)
    idx = np.argmin(Q, 0); s = np.sort(Q, 0)[:2]
    Qx = np.where(mask, P, -np.inf)
    idy = np.argmax(Qx, 0); sx = -np.sort(-Qx, 0)[:2]
    T, J = P.shape
    mn = np.tile(s[0], (T, 1)); mx = np.tile(sx[0], (T, 1))
    col = np.arange(J)
    mn[idx, col] = s[1]; mx[idy, col] = sx[1]
    mn[~np.isfinite(mn)] = np.nan; mx[~np.isfinite(mx)] = np.nan
    return mn, mx


def featured(v, cat):
    """Each entry is checked against the catalog here; the build fails rather than ship a link or a sentence the data do not support."""
    ev = lambda e: v[v.event_id == e].sort_values(["a", "b"])
    F = []
    G = "ENSMUSG00000033981.14_atse_3"; g = ev(G)
    assert (g.gene_name == "Gria2").all() and cat["classes"][G] == "mxe_linked" and len(g) == 7
    flip_flop = {(80690518, 80691319), (80689351, 80691319), (80690518, 80692284), (80689351, 80690403)}   # 115-nt exons 80690404-80690518, 80691320-80691434
    assert flip_flop <= set(zip(g.a, g.b)) and (80691434, 80692284) in set(zip(g.a, g.b))
    F.append(dict(gene="Gria2", title="flip/flop", dataset="Allen", events=[G],
                  text="The two 115-nt exons are one ATSE (MXE linked, 7 junctions). The catalog also holds a junction joining the two exons."))
    F3, F4 = "ENSMUSG00000030849.18_atse_3", "ENSMUSG00000030849.18_atse_4"
    assert (ev(F3).gene_name == "Fgfr2").all() and F4 in cat["split_partners"][F3] and len(ev(F3)) == 2 and len(ev(F4)) == 2
    F.append(dict(gene="Fgfr2", title="IIIb/IIIc", dataset="TMS", events=[F3, F4],
                  text="Split into two ATSEs, one for the junctions on each side of the IIIb and IIIc exons. No junction links the two, so each half has its own ψ."))
    for gene, (e, a, b) in SS4.items():
        x = ev(e); assert (x.gene_name == gene).all()
        into, out_, span = int((x.b == a - 1).sum()), int((x.a == b).sum()), int(((x.a < a) & (x.b > b)).sum())
        plus = x.strand.iloc[0] == "+"
        if into == 0 and span == 0:
            assert out_ > 0 and plus
            text = f"SS4 exon (30 nt, {a:,}-{b:,}). The catalog has no junction ending at the exon's start or spanning it, so this ATSE compares the exon's downstream splice choices ({out_} junctions), not exon inclusion."
        else:
            assert into > 0 and out_ > 0 and span > 0
            text = f"SS4 exon (30 nt, {a:,}-{b:,}). Catalog junctions: {into} ending at one side of the exon, {out_} at the other, {span} spanning it ({len(x)} in this ATSE)."
        F.append(dict(gene=gene, title="SS4", dataset="Allen", events=[e], text=text))
    p = v[v.gene_name == "Ptprc"].event_id.unique().tolist()
    assert p
    n = {e: len(ev(e)) for e in p}
    F.append(dict(gene="Ptprc", title="", dataset="TMS", events=sorted(p), text="All Ptprc ATSEs in the catalog (" + ", ".join(f"{e.split('_')[-1]}: {k} junctions" for e, k in sorted(n.items())) + "). TMS has B and T cell types across tissues."))
    return F


def build_data(site):
    v = pd.read_parquet(os.path.join(PRE, "catalog_junctions.parquet"))
    cat = json.load(open(os.path.join(PRE, "catalog_events.json")))
    crec = json.load(open(os.path.join(PRE, "catalog_record.json")))
    ids = cat["event_id"]; n_e, n_j = len(ids), len(v)
    inv = v.event_idx.values
    perm = np.lexsort((v.b.values, v.a.values, inv))                 # site junction order: by event, then genomic
    vs = v.iloc[perm].reset_index(drop=True)
    inv_s = inv[perm]
    jo = np.searchsorted(inv_s, np.arange(n_e + 1))
    assert (np.diff(jo) >= 2).all()
    first = vs.groupby("event_idx", sort=True).first()
    genes = sorted(vs.gene_name.unique())
    gi = {g: i for i, g in enumerate(genes)}
    gene_id_by_name = vs.groupby("gene_name").gene_id.first()
    # a gene name can map to more than one gene id (11,233 names, 11,236 ids): search by name, list ids on the page
    gid_all = vs.groupby("gene_name").gene_id.agg(lambda s: ", ".join(sorted(set(s))))
    chroms = sorted(vs.chromosome.unique()); ci = {c: i for i, c in enumerate(chroms)}
    wj(os.path.join(site, "data", "catalog", "events.json"), dict(
        ids=ids, genes=genes, gene_ids=[gid_all[g] for g in genes], gene_idx=[gi[g] for g in first.gene_name], classes=CLASSES,
        cls=[CLASSES.index(cat["classes"][e]) for e in ids], chrom=chroms, chrom_idx=[ci[c] for c in first.chromosome],
        strand=[int(s == "-") for s in first.strand], jo=jo.tolist(), partners=cat["split_partners"]))
    st = vs.annotation_status.map({a: i for i, a in enumerate(ANN)}).values
    assert not np.isnan(st).any()
    wb(os.path.join(site, "data", "catalog", "junctions.bin"), [vs.a.values.astype(np.uint32), vs.b.values.astype(np.uint32), st.astype(np.uint8)])
    for b in range(0, n_e, BUCKET):
        wj(os.path.join(site, "data", "catalog", "struct", f"{b // BUCKET}.json"), {e: cat["structure"][e] for e in ids[b:b + BUCKET]})
    index = dict(version=VERSION, built=time.strftime("%Y-%m-%d %H:%M"), bucket_size=BUCKET, type_layout=TYPE_LAYOUT, range_min_reads=RANGE_READS,
                 range_min_cells=RANGE_CELLS, datasets={}, featured=featured(vs, cat), example_compare=dict(a="L2/3 IT CTX", b="Pvalb"))
    manifest = dict(version=VERSION, built=index["built"], site_note="LeafletFA estimates are seed 0, K = 20; raw pseudobulk, reads, cells and mice are TRAIN cells only.",
                    classes=crec, datasets={})
    for ds in ("TMS", "Allen"):
        t0 = time.time()
        E = np.load(os.path.join(V, "c10", f"{ds}_s0_estimates.npz"), allow_pickle=True)
        C = np.load(os.path.join(PRE, f"{ds}_train_tables.npz"))
        rec = json.load(open(os.path.join(PRE, f"{ds}_record.json")))
        types = E["types"].astype(str); assert np.array_equal(types, C["types"].astype(str))
        T = len(types)
        psi = E["point"][:, perm]; lo = E["lo"][:, perm]; hi = E["hi"][:, perm]; rest = E["rest"][:, perm]; rlo = E["lo_rest"][:, perm]; rhi = E["hi_rest"][:, perm]
        jr = C["jreads"][:, perm]; er = C["ereads"]; cov = C["cov"]; mice = C["mice"]
        tot = er[:, inv_s].astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            raw = np.where(tot > 0, jr / tot, np.nan)
        elig = E["eligible"].astype(bool); n_mice = E["n_mice"]
        support = (er >= RANGE_READS) & (cov >= RANGE_CELLS) & elig[:, None]
        mn, mx = loo_extreme(psi, support[:, inv_s])
        ep, el, eh, er_, er2, er3, emn, emx, eraw = enc(psi), enc(lo), enc(hi), enc(rest), enc(rlo), enc(rhi), enc(mn), enc(mx), enc(raw)
        dsdir = os.path.join(site, "data", ds)
        for k in range(T):
            wb(os.path.join(dsdir, "types", f"{k}.bin"), [er[k], cov[k], mice[k], ep[k], el[k], eh[k], eraw[k], er_[k], er2[k], er3[k], emn[k], emx[k]])
        for b in range(0, n_e, BUCKET):
            e1 = min(n_e, b + BUCKET); j0, j1 = jo[b], jo[e1]
            wb(os.path.join(dsdir, "events", f"{b // BUCKET}.bin"), [jr[:, j0:j1], er[:, b:e1], ep[:, j0:j1], el[:, j0:j1], eh[:, j0:j1], eraw[:, j0:j1], cov[:, b:e1], mice[:, b:e1]])
        meta = rec["types_meta"]; tlist = []
        for k, m in enumerate(meta):
            assert m["type"] == types[k]
            d = dict(i=k, name=m["type"], cells=m["cells"], n_mice=m["n_mice"], eligible=m["eligible"])
            if ds == "TMS":
                d.update(label=f"{m['tissue']} | {m['cell_type']}", group1=m["tissue"], group2=m["cell_type"])
            else:
                cl = ALLEN_CLASS.get(m["allen_class"], "non-neuronal")
                d.update(label=m["type"], group1=cl, group2=m["subclass"], allen_class_label=m["allen_class"], allen_class_fraction=m["allen_class_fraction"])
            tlist.append(d)
        fr = rec["fit_record"]; md = rec["md5"]
        src = dict(seed=fr["seed"], K=fr["K"], fit_md5=md["fit"]["md5"], fit_commit=fr["leafletfa_git"]["commit"], fit_dirty=fr["leafletfa_git"]["dirty"], estimates_md5=md["estimates"]["md5"])
        index["datasets"][ds] = dict(types=tlist, train_cells=rec["train_cells"], source=src, splicevi_ds=None)
        lt2 = [t["name"] for t in tlist if t["n_mice"] < 2]; inel = [t["name"] for t in tlist if not t["eligible"]]
        manifest["datasets"][ds] = dict(types=T, train_cells=rec["train_cells"], labels_not_in_estimates=rec["labels_not_in_estimates"], fit_record=fr, inputs_md5=md,
                                        fewer_than_2_mice=lt2, frozen_ineligible=[dict(type=t["name"], n_mice=t["n_mice"]) for t in tlist if not t["eligible"]],
                                        export_rows=rec["export_rows"], export_vs_estimates_max_abs_diff=rec["export_vs_estimates_max_abs_diff"], export_reads_max_abs_diff=rec["export_reads_max_abs_diff"])
        print(f"{ds}: {T} types, {time.time() - t0:.0f} s; n_mice<2 {len(lt2)}, frozen ineligible {len(inel)}", flush=True)
    wj(os.path.join(site, "data", "index.json"), index)
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(site, "data")) for f in fs if f != "SOURCE.json"]
    sz = {f: os.path.getsize(f) for f in files}; big = max(sz, key=sz.get)
    manifest["size"] = dict(files=len(files), total_bytes=sum(sz.values()), largest=os.path.relpath(big, site), largest_bytes=sz[big])
    manifest["size"]["by_dir"] = {d: sum(s for f, s in sz.items() if os.path.relpath(f, os.path.join(site, "data")).startswith(d)) for d in ("catalog", "TMS/types", "TMS/events", "Allen/types", "Allen/events")}
    for p in (os.path.join(site, "data", "SOURCE.json"), os.path.join(HERE, "SOURCE.json")):
        json.dump(manifest, open(p, "w"), indent=1)
    print(json.dumps(manifest["size"], indent=1))


def build_web(site):
    os.makedirs(site, exist_ok=True)
    for f in glob.glob(os.path.join(HERE, "web", "*")):
        shutil.copy(f, site)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["all", "web"])
    ap.add_argument("--out", default=os.path.join(V, "portal_browser", "site"))
    a = ap.parse_args()
    if a.what == "all":
        build_data(a.out)
    build_web(a.out)
    print("site:", a.out)
