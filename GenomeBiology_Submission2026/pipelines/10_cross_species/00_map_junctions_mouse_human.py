#!/usr/bin/env python
"""Map mouse (mm10) atlas junctions onto human Allen Brain (hg38) junctions, for Figure 5.

Scripted re-implementation of the 2025 mapping (Archive .../joint_model_train/00_compare_ATSEs.ipynb),
which used the UCSC web liftOver by hand. Same three gates, now reproducible:
  1. gene gate  : Ensembl one-to-one orthologs (mart export), gene shared by both junction sets
  2. liftOver   : pyliftover with mm10ToHg38.over.chain, both junction ends; drop if either end
                  fails to lift or lands on different chromosomes
  3. coordinate : match to a human junction of the SAME lifted strand with |dstart| <= window and
                  |dend| <= window (default 100 bp, the manuscript's threshold); best match =
                  minimal dstart + dend; duplicates collapsed to one-to-one by that distance
Junction coordinates come from the 2026 model-ready h5ads' var (junction_id = chr_start_end_strand),
so the map is keyed to exactly the junctions the models were trained on. Sequence validation of
splice-site context (2025's step 4) is deliberately deferred; the output keeps the distances so a
stricter cut can be applied later.

Output: junction_map_mouse_human_2026.tsv.gz with mouse/human junction_id, event_id, gene, distances
-- one row per one-to-one matched junction pair -- plus a paired-event table (mouse_event, human_event,
n_matched_junctions, m_mouse, m_human).
"""
import argparse, gzip, os, re
import numpy as np, pandas as pd, h5py
from pyliftover import LiftOver

W = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD"
CHAIN = "/gpfs/commons/home/kisaev/Leaflet-analysis/Archive/LeafletFA_Submission2025/Multi_Species_Splicing_Foundation/pyliftover/mm10ToHg38.over.chain"


def var_table(h5, species):
    with h5py.File(h5, "r") as f:
        def cat(k):
            g = f["var"][k]
            if hasattr(g, "keys") and "categories" in g:
                c = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]); return c[g["codes"][:]]
            return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])
        d = pd.DataFrame(dict(junction_id=cat("junction_id"), event_id=cat("event_id"),
                              gene_id=cat("gene_id"), gene_name=cat("gene_name")))
    p = d.junction_id.str.extract(r"^(?P<chrom>[^_]+)_(?P<start>\d+)_(?P<end>\d+)_(?P<strand>[+-])$")
    d = pd.concat([d, p], axis=1); d["start"] = d.start.astype(int); d["end"] = d.end.astype(int)
    d["gene_stable"] = d.gene_id.str.replace(r"\.\d+$", "", regex=True)
    d["species"] = species
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse-h5", default=f"{W}/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad")
    ap.add_argument("--human-h5", default=f"{W}/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/20260617/human_splice_model_ready_2026-06-17.h5ad")
    ap.add_argument("--orthologs", default=f"{W}/human_mouse_ensemble_orthos_mart_export.txt.gz")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--out", default=f"{W}/CROSS_SPECIES_2026/junction_map_mouse_human_2026.tsv.gz")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    mo = var_table(args.mouse_h5, "mouse"); hu = var_table(args.human_h5, "human")
    print(f"mouse {len(mo):,} junctions / {mo.event_id.nunique():,} ATSEs; human {len(hu):,} / {hu.event_id.nunique():,}", flush=True)
    orth = pd.read_csv(args.orthologs, sep="\t")
    orth.columns = [re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_") for c in orth.columns]
    gcol = [c for c in orth.columns if "homology_type" in c]
    mcol = [c for c in orth.columns if c.startswith("mouse_gene_stable_id")][:1]
    hcol = [c for c in orth.columns if c == "gene_stable_id" or c.startswith("gene_stable_id")][:1]
    print("ortholog columns used:", mcol, hcol, gcol, flush=True)
    o = orth[orth[gcol[0]] == "ortholog_one2one"][[mcol[0], hcol[0]]].drop_duplicates()
    o.columns = ["mouse_gene", "human_gene"]
    m2h = dict(zip(o.mouse_gene, o.human_gene))
    mo = mo[mo.gene_stable.isin(m2h)].copy(); mo["human_gene"] = mo.gene_stable.map(m2h)
    hu_keep = hu[hu.gene_stable.isin(set(mo.human_gene))].copy()
    print(f"after one2one-ortholog gate: mouse {len(mo):,} junctions, human {len(hu_keep):,}", flush=True)
    lo = LiftOver(CHAIN)
    def lift(chrom, pos, strand):
        r = lo.convert_coordinate(chrom, pos, strand)
        return r[0] if r else None
    lifted = []
    for r in mo.itertuples():
        a = lift(r.chrom, r.start, r.strand); b = lift(r.chrom, r.end, r.strand)
        if a is None or b is None or a[0] != b[0]:
            continue
        s, e = sorted((a[1], b[1]))
        lifted.append((r.junction_id, r.event_id, r.gene_stable, r.gene_name, r.human_gene, a[0], s, e, a[2]))
    L = pd.DataFrame(lifted, columns=["m_junction", "m_event", "m_gene", "m_gene_name", "human_gene", "h_chrom", "h_start", "h_end", "h_strand"])
    print(f"lifted cleanly: {len(L):,} of {len(mo):,}", flush=True)
    # match within the same human gene (ortholog gate already applied), same strand, both ends within window
    hu_by_gene = {g: s for g, s in hu_keep.groupby("gene_stable")}
    rows = []
    for r in L.itertuples():
        s = hu_by_gene.get(r.human_gene)
        if s is None: continue
        s = s[(s.strand == r.h_strand) & (s.chrom == r.h_chrom)]
        if not len(s): continue
        ds = (s.start - r.h_start).abs(); de = (s.end - r.h_end).abs()
        ok = (ds <= args.window) & (de <= args.window)
        if not ok.any(): continue
        d = (ds + de)[ok]; j = d.idxmin()
        rows.append(dict(mouse_junction=r.m_junction, mouse_event=r.m_event, mouse_gene=r.m_gene_name,
                         human_junction=s.loc[j, "junction_id"], human_event=s.loc[j, "event_id"],
                         human_gene_name=s.loc[j, "gene_name"], dstart=int(ds.loc[j]), dend=int(de.loc[j])))
    Mt = pd.DataFrame(rows)
    # one-to-one: a human junction may be hit by several mouse junctions -- keep the closest
    Mt["dist"] = Mt.dstart + Mt.dend
    Mt = Mt.sort_values("dist").drop_duplicates("human_junction").drop_duplicates("mouse_junction")
    Mt.to_csv(args.out, sep="\t", index=False)
    ev = Mt.groupby(["mouse_event", "human_event"]).size().rename("n_matched").reset_index()
    mm = mo.groupby("event_id").size(); hm = hu.groupby("event_id").size()
    ev["m_mouse"] = ev.mouse_event.map(mm); ev["m_human"] = ev.human_event.map(hm)
    ev.to_csv(args.out.replace("junction_map", "event_map"), sep="\t", index=False)
    print(f"matched one-to-one: {len(Mt):,} junction pairs "
          f"({(Mt.dist == 0).mean()*100:.0f}% exact); event pairs: {len(ev):,}; "
          f"events with >= 2 matched junctions: {(ev.n_matched >= 2).sum():,}", flush=True)


if __name__ == "__main__":
    main()
