#!/usr/bin/env python
"""
Supplementary figure: how many junctions / ATSEs does a gene carry, and what scales it?

Runs on any dataset: rna_atlas (bulk), mouse_smartseq2 (SC), human_allen_brain (SC).

Two candidate drivers, and they are confounded:
  * exon count  — more exons => more splice sites => more chances for an ATSE
  * read depth  — more reads => more power to detect any junction at all
A gene with many exons is also typically longer and more highly expressed, so the naive
"exons vs ATSEs" correlation is partly just depth. We report BOTH marginal correlations AND
partial Spearman (each controlling for the other), which is what licenses the claim that
exon count matters on its own.

────────────────────────────────────────────────────────────────────────────────────────
FOUR THINGS THAT WILL SILENTLY CORRUPT THIS ANALYSIS IF YOU GET THEM WRONG
────────────────────────────────────────────────────────────────────────────────────────
1. THIS FIGURE DESCRIBES THE ANALYSED SUBSET ONLY — genes that already have an ATSE.
   We deliberately use the SAME 09b-aligned/filtered GE as the splice object it pairs with,
   so every file in the figure comes from one pipeline run. A consequence: 09b keeps only
   genes with surviving junctions, so ~100% of analysed genes have an ATSE and the question
   "WHICH genes have one" cannot be asked here. The figure therefore reports HOW MANY
   junctions/ATSEs a gene carries, and how that scales with exon count and read depth.

2. EXON COUNT MUST BE "MAX EXONS OVER ALL TRANSCRIPTS", NOT A REPRESENTATIVE ISOFORM.
   What determines whether a gene CAN have an ATSE is whether it has ANY intron — i.e. the
   maximum exon count across its transcripts. Using a "representative" isoform breaks this:
   e.g. mouse ENSMUSG00000053574.4 has a 1-exon transcript tagged `basic` and a 2-exon
   untagged one; an appris/basic-priority rule picks the 1-exon isoform, so the gene appears
   "single-exon" yet has 4 junctions and an ATSE — biologically impossible. With rule=appris
   this produced 20/26 "single-exon genes with ATSEs" in mouse (vs 0.7% in the human atlas,
   where MANE_Select is well curated and hid the problem).
   Rule is ALSO not portable: vM19 (mouse) has NO MANE_Select and NO Ensembl_canonical, so a
   MANE-based rule silently degrades for mouse only and corrupts mouse-vs-human comparison.
   DEFAULT = `max_exons`: species-neutral, tag-free, and correctly defines splicing capacity.
   Caveat to state: max-exons is sensitive to annotation depth (v45 is better annotated than
   vM19), so absolute exon counts are not perfectly comparable across species — the
   correlations and the single-exon claim are.

3. EVERY DATASET KEYS GENES DIFFERENTLY — and one has no gene_id at all.
   atlas : var_names ARE versioned gene_ids            -> join on gene_id
   mouse : var_names are SYMBOLS, var['gene_id'] exists -> join on gene_id
   human : var_names are JUNK ('3.8-1.2'), var has ONLY gene_name (no gene_id!)
                                                       -> join on gene_name
   Per-dataset `gene_key` below. Getting this wrong yields a silent zero/partial join.

3a. THE HUMAN GE HAS TWO DIFFERENT INTEGER MATRICES.
   `.X` (== `layers['raw_counts']`, max 174,878) and `layers['counts']` (max 366,649) are NOT
   the same matrix. Reading 'counts' for human while reading `.X` for mouse silently compares
   two different quantities. All datasets now read `.X`.

3b. ALL DEPTH IS RAW COUNTS, READ FROM `.X`.
   Verified integer-valued for every dataset. The normalised layers (`length_norm`,
   `log_norm`) are never used for the depth axis.

4. GE AND SPLICE CELL IDS ARE IN DIFFERENT CONVENTIONS (mouse/human SC).
   The splice object's obs_names are reset to integers and its real id is obs['cell_id'],
   ALREADY CLEANED by align_splice_ge.py. The raw GE keeps original ids
   (TMS: 'US-1250273_E1_S37'; AB: exp_component_name). A naive obs_names join gives ZERO
   overlap. We apply the pipeline's own `clean_ge_cell_ids()` + the combined metadata, which
   recovers all 132,706/132,706 mouse cells. Depth is then summed over the SHARED cells only,
   so it is the depth the splice pipeline actually saw.
   (For the bulk atlas both objects are keyed by bare SRR, so no cleaning is needed.)

Outputs: supp_exons_vs_atses_<dataset>.{png,pdf,csv}

Usage:
    python supp_exons_vs_atses.py --dataset rna_atlas
    python supp_exons_vs_atses.py --dataset mouse_smartseq2  --min-frac 0.05
    python supp_exons_vs_atses.py --dataset human_allen_brain --min-frac 0.05
"""

import argparse
import os
import sys
import re
from collections import defaultdict

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr, rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_common"))
from leaflet_theme import (apply_theme, PALETTE, CMAP, PT, INK,     # noqa: E402
                           figsize, panel_label, save)

ATLAS = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/RNA_ATLAS/GSE138734"
MSF   = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
HSF   = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION"
GTF_HUMAN = ("/gpfs/commons/datasets/controlled/BRAIN_NeMO/human-reference/gencode/"
             "gencode.v45.primary_assembly.annotation.gtf")
GTF_MOUSE = ("/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/"
             "genome_files/gencode.vM19/genes/genes.gtf")

DEFAULTS = {
    # bulk: GE obs_names are bare SRR (== splice cell ids); var_names ARE versioned gene_ids
    "rna_atlas": dict(
        splice=f"{ATLAS}/MODEL_INPUT/20260709/"
               f"splice_adata_aligned_filtered_minCells50_minJuncs500_2026-07-11.h5ad",
        ge=f"{ATLAS}/MODEL_INPUT/20260709/"
           f"ge_adata_aligned_filtered_minCells50_minJuncs500_2026-07-11.h5ad",
        gtf=GTF_HUMAN, min_frac=0.10, label="RNA atlas (bulk)",
        gene_key="gene_id", counts_layer=None, clean_ge=False),
    # SC: 06-17 files supersede 06-13 (see WORKFLOW). GE var_names are SYMBOLS but gene_id exists.
    # Raw GE cell ids need the pipeline's cleaner to match splice obs['cell_id'] (note 4).
    "mouse_smartseq2": dict(
        splice=f"{MSF}/MODEL_INPUT/20260526/"
               f"splice_adata_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5ad",
        ge=f"{MSF}/MODEL_INPUT/20260526/"
           f"ge_adata_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5ad",
        gtf=GTF_MOUSE, min_frac=0.05, label="Mouse SmartSeq2 (SC)",
        gene_key="gene_id", counts_layer=None,
        clean_ge=False),
    # The mouse object POOLS Tabula Muris Senis whole cells with Allen Brain nuclei.
    # These two entries split it so like is compared with like (mouse nuclei vs human
    # nuclei). `subset` restricts the CELLS, and junction/ATSE detection is then
    # recomputed within that subset — otherwise the per-gene counts would be identical
    # to the pooled object and only the depth axis would change.
    "mouse_tms": dict(
        splice=f"{MSF}/MODEL_INPUT/20260526/"
               f"splice_adata_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5ad",
        ge=f"{MSF}/MODEL_INPUT/20260526/"
           f"ge_adata_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5ad",
        gtf=GTF_MOUSE, min_frac=0.05, label="Mouse Tabula Muris Senis (single cells)",
        gene_key="gene_id", counts_layer=None,
        clean_ge=False,
        subset=("dataset", "tabula_muris_senis")),
    "mouse_allen_brain": dict(
        splice=f"{MSF}/MODEL_INPUT/20260526/"
               f"splice_adata_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5ad",
        ge=f"{MSF}/MODEL_INPUT/20260526/"
           f"ge_adata_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5ad",
        gtf=GTF_MOUSE, min_frac=0.05, label="Mouse Allen Brain (single nuclei)",
        gene_key="gene_id", counts_layer=None,
        clean_ge=False,
        subset=("dataset", "allen_brain")),
    # Human AB: GE obs_names ALREADY == splice cell_id (sample_name) -> NO cleaner needed.
    # GE var has NO gene_id (var_names are junk) -> join on gene_name.
    # GE .X is NORMALIZED -> depth MUST come from layers['counts'].
    "human_allen_brain": dict(
        splice=f"{HSF}/MODEL_INPUT/20260617/"
               f"splice_adata_aligned_filtered_minCells25_minJuncs500_2026-06-17.h5ad",
        ge=f"{HSF}/MODEL_INPUT/20260617/"
           f"ge_adata_aligned_filtered_minCells25_minJuncs500_2026-06-17.h5ad",
        gtf=GTF_HUMAN, min_frac=0.05, label="Human Allen Brain (SC)",
        gene_key="gene_name", counts_layer=None, clean_ge=False),   # .X == raw_counts
}


def ge_cell_ids(ge, cfg):
    """Cell ids for the GE object, in the SAME convention as splice obs['cell_id'].

    Bulk atlas: obs_names are already bare SRR -> use as-is.
    Mouse/human SC: raw GE ids must go through the pipeline's own clean_ge_cell_ids()
    (align_splice_ge.py) + combined metadata, else the join yields ZERO overlap.
    """
    if not cfg.get("clean_ge"):
        # The ALIGNED GE objects also carry integer obs_names with the real id in
        # obs['cell_id'] (same convention as splice). Prefer that column when present,
        # otherwise obs_names (bulk atlas, where obs_names are bare SRR accessions).
        if "cell_id" in ge.obs.columns:
            return pd.Series(ge.obs["cell_id"].astype(str).values, index=range(ge.n_obs))
        return pd.Series(ge.obs_names.astype(str).values, index=range(ge.n_obs))
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "..", "..", "pipelines", "04_align"))
    from align_splice_ge import clean_ge_cell_ids           # noqa: E402
    meta = pd.read_csv(cfg["metadata"])
    cleaned = pd.Series(clean_ge_cell_ids(ge, meta)).astype(str)
    print(f"  applied clean_ge_cell_ids() -> e.g. {list(cleaned[:2])}")
    return cleaned

EXON_BINS   = [0, 1, 2, 4, 8, 16, 32, 64, 10_000]
EXON_LABELS = ["1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65+"]
BLUE, RED = PALETTE["indigo"], PALETTE["rose"]   # theme palette, not ad-hoc hex

_gid = re.compile(r'gene_id "([^"]+)"')
_tid = re.compile(r'transcript_id "([^"]+)"')
_gname = re.compile(r'gene_name "([^"]+)"')
_appris = re.compile(r'tag "appris_principal_(\d)"')


def representative_exon_counts(gtf_path, rule="max_exons", gene_key="gene_id"):
    """gene_id -> exon count used as the gene's splicing capacity.

    rule='max_exons' (DEFAULT): max exons over ALL transcripts. Species-neutral, tag-free.
                     A gene with max 1 exon genuinely has NO intron and therefore cannot
                     form an ATSE — which is the claim the figure rests on.
    rule='appris'  : appris_principal_N > basic > most exons.  ⚠ BROKEN for this purpose —
                     can pick a 1-exon `basic` isoform for a gene that is really spliced
                     (see module docstring, mouse ENSMUSG00000053574.4).
    rule='mane'    : MANE_Select > Ensembl_canonical > most exons. HUMAN ONLY (vM19 lacks
                     both tags), so do NOT use for cross-species comparisons.

    Returns (gene_exons, tier_counts).
    """
    tx_exons, tx_gene, tx_key = defaultdict(int), {}, {}
    with open(gtf_path) as fh:
        for line in fh:
            if line[0] == "#":
                continue
            p = line.split("\t", 8)
            if len(p) < 9:
                continue
            feat, attr = p[2], p[8]
            if feat == "transcript":
                # key the map by gene_id or gene_name depending on what the GE object has
                g = (_gid.search(attr) if gene_key == "gene_id" else _gname.search(attr))
                t = _tid.search(attr)
                if not (g and t):
                    continue
                tx_gene[t.group(1)] = g.group(1)
                if rule == "mane":
                    tier = (0 if "MANE_Select" in attr else
                            1 if "Ensembl_canonical" in attr else 3)
                elif rule == "appris":
                    m = _appris.search(attr)
                    tier = (int(m.group(1)) if m else            # 1..5 principal
                            2.5 if 'tag "basic"' in attr else 3)
                else:  # max_exons — no tag preference; the -exons key below picks the max
                    tier = 3
                tx_key[t.group(1)] = tier
            elif feat == "exon":
                t = _tid.search(attr)
                if t:
                    tx_exons[t.group(1)] += 1

    best, best_tier = {}, {}
    for tid, gid in tx_gene.items():
        k = (tx_key.get(tid, 3), -tx_exons.get(tid, 0))
        if gid not in best or k < best[gid][0]:
            best[gid] = (k, tx_exons.get(tid, 0))
            best_tier[gid] = tx_key.get(tid, 3)
    tiers = pd.Series(best_tier).value_counts().sort_index()
    return {g: v[1] for g, v in best.items()}, tiers


def partial_spearman(x, y, z):
    """Spearman of x,y controlling for z: correlate residuals of rank(x)~rank(z), rank(y)~rank(z)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.c_[np.ones_like(rz), rz]
    res_x = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    res_y = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(res_x, res_y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rna_atlas", choices=list(DEFAULTS))
    ap.add_argument("--splice", default=None)
    ap.add_argument("--ge", default=None, help="FULL GE h5ad — NOT the 09b-filtered one")
    ap.add_argument("--gtf", default=None)
    ap.add_argument("--rep-rule", default="max_exons", choices=["max_exons", "appris", "mane"],
                    help="exon-count rule (default max_exons = species-neutral, tag-free)")
    ap.add_argument("--min-frac", type=float, default=None,
                    help="gene is 'expressed' if detected in >= this fraction of cells")
    ap.add_argument("--outdir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
    args = ap.parse_args()

    d = DEFAULTS[args.dataset]
    splice_p = args.splice or d["splice"]
    ge_p     = args.ge or d["ge"]
    gtf_p    = args.gtf or d["gtf"]
    min_frac = args.min_frac if args.min_frac is not None else d["min_frac"]
    os.makedirs(args.outdir, exist_ok=True)

    print(f"=== {d['label']}  [{args.dataset}] ===")
    print(f"GTF    : {gtf_p}\nsplice : {splice_p}\nGE     : {ge_p}")
    print(f"rule   : {args.rep_rule}   min_frac: {min_frac}\n")

    print(f"parsing GTF (representative transcript, rule={args.rep_rule}) ...", flush=True)
    gene_exons, tiers = representative_exon_counts(gtf_p, args.rep_rule, d["gene_key"])
    print(f"  {len(gene_exons):,} genes")
    print(f"  representative-transcript tier counts (lower = more principal):\n"
          f"{tiers.to_string()}")
    if args.rep_rule == "mane" and 0 not in tiers.index:
        print("  ⚠ rule='mane' but NO MANE_Select in this GTF — rule degraded to 'most exons'. "
              "Do NOT compare across species with this.")
    print(flush=True)

    # ── splice: junctions / ATSEs per gene ────────────────────────────────────
    splice = ad.read_h5ad(splice_p, backed="r")
    gkey = d["gene_key"]
    var = splice.var[[gkey, "event_id"]].astype(str).rename(columns={gkey: "gene_id"})
    # obs_names may be reset integers — the real id is obs['cell_id'] (already cleaned)
    splice_cells = set(splice.obs["cell_id"].astype(str) if "cell_id" in splice.obs.columns
                       else splice.obs_names.astype(str))
    obs_full = splice.obs.copy()          # capture BEFORE closing the file handle
    n_vars_full = splice.n_vars
    splice.file.close()
    sub = d.get("subset")
    if sub:
        col, val = sub
        if col not in obs_full.columns:
            raise SystemExit(f"ERROR: subset column {col!r} not in obs. "
                             f"Available: {list(obs_full.columns)}")
        mask = (obs_full[col].astype(str) == val).values
        print(f"  subset {col}=={val!r}: {mask.sum():,} / {len(mask):,} cells")
        if mask.sum() == 0:
            raise SystemExit(f"ERROR: subset {col}=={val!r} selected no cells")
        # junction/ATSE counts per gene MUST be recomputed inside the subset, otherwise
        # every subset would report the pooled object's numbers.
        import sys as _s
        _s.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "_shared"))
        import _h5ad_scan as _scan
        import h5py as _h5py
        with _h5py.File(splice_p, "r") as _f:
            _r = _scan.scan_csr(_f, "layers/cell_by_junction_matrix", n_vars_full,
                                chunk_rows=20_000, row_mask=mask)
        detected = _r["col_nnz"] > 0
        print(f"  junctions detected in subset: {detected.sum():,} / {len(detected):,}")
        var = var.loc[detected]
        splice_cells = set(obs_full.loc[mask, "cell_id"].astype(str)) \
            if "cell_id" in obs_full.columns else set(obs_full.index[mask].astype(str))
    n_junc = var.groupby("gene_id").size()
    n_atse = var.groupby("gene_id")["event_id"].nunique()
    print(f"splice : {len(splice_cells):,} cells, {len(var):,} junctions, "
          f"{n_atse.size:,} genes with ATSEs, {var['event_id'].nunique():,} ATSEs\n")

    # ── GE: gene ids, depth over SHARED cells only ───────────────────────────
    ge = ad.read_h5ad(ge_p)
    # (3) var_names may be SYMBOLS — prefer an explicit gene_id column
    if gkey in ge.var.columns:
        ge_ids = ge.var[gkey].astype(str).values
        print(f"GE genes keyed by var['{gkey}']")
    else:
        ge_ids = ge.var_names.astype(str).values
        print(f"GE genes keyed by var_names (no '{gkey}' column)")

    # (4) match cells — GE ids may need the pipeline's cleaner
    ge_cell = ge_cell_ids(ge, d).values
    keep = np.isin(ge_cell, list(splice_cells))
    print(f"GE     : {ge.n_obs:,} cells x {ge.n_vars:,} genes | matched to splice: "
          f"{keep.sum():,} / {len(splice_cells):,}")
    if keep.sum() == 0:
        raise SystemExit(
            "ERROR: GE and splice share no cells. The splice cell_ids are CLEANED "
            "(align_splice_ge.py) while the raw GE ids are not — set clean_ge=True and give a "
            "metadata csv for this dataset.")
    if keep.sum() < len(splice_cells):
        print(f"  note: {len(splice_cells) - keep.sum():,} splice cells absent from GE")
    ge = ge[keep].copy()

    cl = d.get("counts_layer")
    M = ge.layers[cl] if cl else ge.X
    if cl:
        print(f"depth from layers['{cl}'] (.X is normalized for this dataset)")
    X = M.tocsc() if sp.issparse(M) else sp.csc_matrix(M)
    n_det = np.diff(X.indptr)                       # cells detecting each gene
    exonic_reads = np.asarray(X.sum(axis=0)).ravel()
    min_cells = int(min_frac * ge.n_obs)
    expressed = n_det >= min_cells
    print(f"expressed genes (detected in >= {min_cells:,} / {ge.n_obs:,} cells): "
          f"{expressed.sum():,} of {ge.n_vars:,}\n", flush=True)

    # ── gene_id join sanity (versions!) ──────────────────────────────────────
    ge_expr_ids = ge_ids[expressed]
    ov = len(set(ge_expr_ids) & set(gene_exons))
    print(f"gene_id join: {ov:,}/{len(ge_expr_ids):,} expressed GE genes found in GTF")
    if ov < 0.5 * len(ge_expr_ids):
        print("  ⚠ poor overlap — trying version-stripped ids")
        gene_exons = {k.split(".")[0]: v for k, v in gene_exons.items()}
        ge_expr_ids = np.array([g.split(".")[0] for g in ge_expr_ids])
        n_junc.index = [i.split(".")[0] for i in n_junc.index]
        n_atse.index = [i.split(".")[0] for i in n_atse.index]
        ov = len(set(ge_expr_ids) & set(gene_exons))
        print(f"  after stripping: {ov:,}/{len(ge_expr_ids):,}")
    print(flush=True)

    df = pd.DataFrame(dict(gene_id=ge_expr_ids, exonic_reads=exonic_reads[expressed]))
    df["n_exons"] = df.gene_id.map(gene_exons)
    df = df.dropna(subset=["n_exons"])
    df["n_exons"]     = df.n_exons.astype(int)
    df["n_junctions"] = df.gene_id.map(n_junc).fillna(0).astype(int)
    df["n_atses"]     = df.gene_id.map(n_atse).fillna(0).astype(int)
    df["total_reads"] = df.exonic_reads
    df["has_atse"]    = df.n_atses > 0
    df["exon_bin"]    = pd.cut(df.n_exons, EXON_BINS, labels=EXON_LABELS)

    # ── stats ────────────────────────────────────────────────────────────────
    n_without = int((df.n_atses == 0).sum())
    if False:   # every analysed gene has an ATSE by construction; nothing to warn about
        print(f"\n  !! Only {n_without} of {len(df):,} analysed genes LACK an ATSE.\n"
              f"     The GE object is gene-matched to the splice object, so 'which genes\n"
              f"     have an ATSE' is ~100% by construction. Panels a and f are therefore\n"
              f"     NOT interpretable for this dataset; panels b-e (how MANY events per\n"
              f"     gene, among genes that have them) remain valid.\n")
    single, multi = df[df.n_exons == 1], df[df.n_exons > 1]
    print(f"Expressed genes analysed : {len(df):,}  "
          f"({df.has_atse.sum():,} with an ATSE, {(~df.has_atse).sum():,} without)")
    if len(single):
        print(f"  single-exon : {len(single):,}  -> with ATSE {int(single.has_atse.sum())} "
              f"({100*single.has_atse.mean():.1f}%)   <- can never form an ATSE")
    print(f"  multi-exon  : {len(multi):,}  -> with ATSE {int(multi.has_atse.sum()):,} "
          f"({100*multi.has_atse.mean():.1f}%)\n")

    w = df[df.has_atse]
    if len(w) < 50:
        raise SystemExit("ERROR: too few genes with ATSEs — check the gene_id join.")
    lr = np.log10(w.total_reads + 1)
    marg = dict(
        ex_junc=spearmanr(w.n_exons, w.n_junctions).statistic,
        ex_atse=spearmanr(w.n_exons, w.n_atses).statistic,
        rd_junc=spearmanr(lr, w.n_junctions).statistic,
        rd_atse=spearmanr(lr, w.n_atses).statistic,
        ex_rd=spearmanr(w.n_exons, lr).statistic)
    p_ex_atse = partial_spearman(w.n_exons, w.n_atses, lr)
    p_rd_atse = partial_spearman(lr, w.n_atses, w.n_exons)
    p_ex_junc = partial_spearman(w.n_exons, w.n_junctions, lr)
    p_rd_junc = partial_spearman(lr, w.n_junctions, w.n_exons)

    print(f"Marginal Spearman (genes with >=1 ATSE, n={len(w):,}):")
    print(f"  exons vs #junctions        rho = {marg['ex_junc']:+.3f}")
    print(f"  exons vs #ATSEs            rho = {marg['ex_atse']:+.3f}")
    print(f"  log10(reads) vs #junctions rho = {marg['rd_junc']:+.3f}")
    print(f"  log10(reads) vs #ATSEs     rho = {marg['rd_atse']:+.3f}")
    print(f"  exons vs log10(reads)      rho = {marg['ex_rd']:+.3f}   <- the confound")
    print("\nPartial Spearman (each controlling for the other):")
    print(f"  exons vs #ATSEs      | reads       rho = {p_ex_atse:+.3f}")
    print(f"  log10(reads) vs #ATSEs | exons     rho = {p_rd_atse:+.3f}")
    print(f"  exons vs #junctions  | reads       rho = {p_ex_junc:+.3f}")
    print(f"  log10(reads) vs #junctions | exons rho = {p_rd_junc:+.3f}")

    tab = df.groupby("exon_bin", observed=True).agg(
        n_genes=("gene_id", "size"), pct_with_atse=("has_atse", lambda x: 100 * x.mean()),
        med_junctions=("n_junctions", "median"), med_atses=("n_atses", "median"),
        med_reads=("total_reads", "median"))
    print("\nDetection vs exon count:")
    print(tab.round(1).to_string())

    # ── figure ───────────────────────────────────────────────────────────────
    apply_theme()
    fig = plt.figure(figsize=figsize("double", h=5.4))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.42,
                          left=0.075, right=0.965, top=0.93, bottom=0.10)
    panel_axes = []

    ax = fig.add_subplot(gs[0, 0]); panel_axes.append(ax)
    # Every analysed gene HAS an ATSE (the aligned GE is gene-matched to the splice
    # object), so "% with >=1 ATSE" would be 100% by construction. The question that the
    # data can answer is HOW MANY events a gene carries.
    med = df.groupby("exon_bin", observed=True)["n_atses"].median()
    q1 = df.groupby("exon_bin", observed=True)["n_atses"].quantile(.25)
    q3 = df.groupby("exon_bin", observed=True)["n_atses"].quantile(.75)
    xs = np.arange(len(med))
    ax.bar(xs, med.values, color=BLUE, width=0.75)
    ax.errorbar(xs, med.values, yerr=[med.values - q1.values, q3.values - med.values],
                fmt="none", ecolor=INK, elinewidth=0.8, capsize=0)
    for i, (n, top) in enumerate(zip(tab.n_genes, q3.values)):
        ax.text(i, top, f"  {n:,}", ha="center", va="bottom", fontsize=PT["annot"],
                color=PALETTE["grey"], rotation=90)
    ax.set_xticks(xs); ax.set_xticklabels([str(v) for v in med.index])
    ax.set_ylim(0, q3.max() * 1.45)
    ax.set_xlabel("exons (representative transcript)")
    ax.set_ylabel("ATSEs detected per gene\n(median, IQR)")
    ax.set_title("ATSEs per gene vs exon count", fontsize=PT["title"], loc="left")
    ax.tick_params(axis="x", rotation=45)

    for j, (col, lab, rho, prho) in enumerate([
            ("n_junctions", "junctions detected", marg["ex_junc"], p_ex_junc),
            ("n_atses", "ATSEs detected", marg["ex_atse"], p_ex_atse)]):
        ax = fig.add_subplot(gs[0, j + 1]); panel_axes.append(ax)
        ax.scatter(w.n_exons, w[col], s=3, alpha=0.15, color=BLUE, rasterized=True)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("exons (representative transcript)"); ax.set_ylabel(lab)
        ax.set_title(f"exons vs {lab}\nρ = {rho:.2f}  (partial | reads: {prho:.2f})",
                     fontsize=PT["title"], loc="left")

    for j, (col, lab, rho, prho) in enumerate([
            ("n_junctions", "junctions detected", marg["rd_junc"], p_rd_junc),
            ("n_atses", "ATSEs detected", marg["rd_atse"], p_rd_atse)]):
        ax = fig.add_subplot(gs[1, j]); panel_axes.append(ax)
        ax.scatter(w.total_reads + 1, w[col], s=3, alpha=0.15, color=RED, rasterized=True)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("gene read depth (exonic counts)"); ax.set_ylabel(lab)
        ax.set_title(f"depth vs {lab}\nρ = {rho:.2f}  (partial | exons: {prho:.2f})",
                     fontsize=PT["title"], loc="left")

    ax = fig.add_subplot(gs[1, 2]); panel_axes.append(ax)
    df["depth_q"] = pd.qcut(df.total_reads, 5,
                            labels=["Q1\nlow", "Q2", "Q3", "Q4", "Q5\nhigh"], duplicates="drop")
    piv = (df.groupby(["exon_bin", "depth_q"], observed=True)["n_atses"].median().unstack())
    im = ax.imshow(piv.values, aspect="auto", cmap=CMAP["sequential"], origin="lower")
    # Print the value in each tile — the colour alone is hard to read off, and the row/column
    # trends (depth at fixed exons, exons at fixed depth) are the whole point of the panel.
    vals = piv.values.astype(float)
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if not np.isfinite(v):
                continue
            frac = (v - lo) / (hi - lo) if hi > lo else 0.5
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=PT["annot"],
                    color=INK if frac > 0.55 else "white")   # mako is dark at low values
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, fontsize=PT["tick"])
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=PT["tick"])
    ax.set_xlabel("gene read depth (quintile)"); ax.set_ylabel("exons")
    ax.set_title("ATSEs per gene\n(exon count × depth)", fontsize=PT["title"], loc="left")
    fig.colorbar(im, ax=ax, label="median ATSEs per gene", fraction=0.046)

    # no suptitle in final artwork (FIGURE_THEME §2) — dataset + n belong in the caption
    for ax_, letter in zip(panel_axes, "abcdef"):
        panel_label(ax_, letter)
    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.join(os.path.abspath(args.outdir),
                        f"supp_exons_vs_atses_{args.dataset}")
    for pth in save(fig, stem):
        print(f"\nfigure -> {pth}")
    csv = os.path.join(args.outdir, f"supp_exons_vs_atses_{args.dataset}.csv")
    df.drop(columns=["exon_bin", "depth_q"]).to_csv(csv, index=False)
    print(f"table  -> {csv}")


if __name__ == "__main__":
    main()
