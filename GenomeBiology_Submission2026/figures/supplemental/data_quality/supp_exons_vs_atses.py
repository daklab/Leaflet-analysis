#!/usr/bin/env python
"""
Supplementary figure: what predicts how many junctions / ATSEs we detect in a gene?

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
1. USE THE *FULL* GE OBJECT, NOT THE 09b-FILTERED ONE.
   09b restricts GE to genes that have surviving junctions — i.e. exactly the genes with
   ATSEs. Using it would make "genes with ATSEs" == "all genes" and the result circular.

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

3b. GE .X IS NOT ALWAYS RAW COUNTS.
   atlas/mouse .X = raw integer counts. human .X is NORMALIZED (min 0.005, non-integer);
   its raw counts live in layers['counts'] (max 121,123). Using .X for the human "depth"
   axis would silently produce a meaningless covariate. Per-dataset `counts_layer` below.

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
        ge=f"{ATLAS}/gene_expression/ge_adata_rna_atlas.h5ad",          # FULL GE
        gtf=GTF_HUMAN, min_frac=0.10, label="RNA atlas (bulk)",
        gene_key="gene_id", counts_layer=None, clean_ge=False),
    # SC: 06-17 files supersede 06-13 (see WORKFLOW). GE var_names are SYMBOLS but gene_id exists.
    # Raw GE cell ids need the pipeline's cleaner to match splice obs['cell_id'] (note 4).
    "mouse_smartseq2": dict(
        splice=f"{MSF}/MODEL_INPUT/20260526/"
               f"splice_adata_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5ad",
        ge=f"{MSF}/processed_data/tms_ab_exons_combo_ge_adata_2026-05-24.h5ad",  # FULL GE
        gtf=GTF_MOUSE, min_frac=0.05, label="Mouse SmartSeq2 (SC)",
        gene_key="gene_id", counts_layer=None,
        metadata=f"{MSF}/metadata_mouse_metadata_combined.csv", clean_ge=True),
    # Human AB: GE obs_names ALREADY == splice cell_id (sample_name) -> NO cleaner needed.
    # GE var has NO gene_id (var_names are junk) -> join on gene_name.
    # GE .X is NORMALIZED -> depth MUST come from layers['counts'].
    "human_allen_brain": dict(
        splice=f"{HSF}/MODEL_INPUT/20260617/"
               f"splice_adata_aligned_filtered_minCells25_minJuncs500_2026-06-17.h5ad",
        ge="/gpfs/commons/datasets/controlled/BRAIN_NeMO/lein-human-cortex/GeneExpression/"
           "adata_2025-01-07_HVG_only_False.h5ad",                       # FULL GE
        gtf=GTF_HUMAN, min_frac=0.05, label="Human Allen Brain (SC)",
        gene_key="gene_name", counts_layer="counts", clean_ge=False),
}


def ge_cell_ids(ge, cfg):
    """Cell ids for the GE object, in the SAME convention as splice obs['cell_id'].

    Bulk atlas: obs_names are already bare SRR -> use as-is.
    Mouse/human SC: raw GE ids must go through the pipeline's own clean_ge_cell_ids()
    (align_splice_ge.py) + combined metadata, else the join yields ZERO overlap.
    """
    if not cfg.get("clean_ge"):
        return pd.Series(ge.obs_names.astype(str).values, index=range(ge.n_obs))
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "..", "pipelines", "04_align"))
    from align_splice_ge import clean_ge_cell_ids           # noqa: E402
    meta = pd.read_csv(cfg["metadata"])
    cleaned = pd.Series(clean_ge_cell_ids(ge, meta)).astype(str)
    print(f"  applied clean_ge_cell_ids() -> e.g. {list(cleaned[:2])}")
    return cleaned

EXON_BINS   = [0, 1, 2, 4, 8, 16, 32, 64, 10_000]
EXON_LABELS = ["1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65+"]
BLUE, RED = "#4C72B0", "#C44E52"

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
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
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
    splice.file.close()
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
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    frac = df.groupby("exon_bin", observed=True)["has_atse"].mean() * 100
    ax.bar(frac.index.astype(str), frac.values, color=BLUE)
    for i, (n, val) in enumerate(zip(tab.n_genes, frac.values)):
        ax.text(i, val + 1.5, f"{n:,}", ha="center", fontsize=6, color="grey")
    ax.set_xlabel("exons (representative transcript)")
    ax.set_ylabel("% expressed genes with ≥1 ATSE")
    ax.set_title("A. ATSE detection vs exon count")
    ax.tick_params(axis="x", rotation=45)

    for j, (col, lab, rho, prho) in enumerate([
            ("n_junctions", "junctions detected", marg["ex_junc"], p_ex_junc),
            ("n_atses", "ATSEs detected", marg["ex_atse"], p_ex_atse)]):
        ax = fig.add_subplot(gs[0, j + 1])
        ax.scatter(w.n_exons, w[col], s=3, alpha=0.15, color=BLUE, rasterized=True)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("exons (representative transcript)"); ax.set_ylabel(lab)
        ax.set_title(f"{'B' if j == 0 else 'C'}. exons vs {lab}\n"
                     f"ρ = {rho:.2f}  (partial | reads: {prho:.2f})")

    for j, (col, lab, rho, prho) in enumerate([
            ("n_junctions", "junctions detected", marg["rd_junc"], p_rd_junc),
            ("n_atses", "ATSEs detected", marg["rd_atse"], p_rd_atse)]):
        ax = fig.add_subplot(gs[1, j])
        ax.scatter(w.total_reads + 1, w[col], s=3, alpha=0.15, color=RED, rasterized=True)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("gene read depth (exonic counts)"); ax.set_ylabel(lab)
        ax.set_title(f"{'D' if j == 0 else 'E'}. depth vs {lab}\n"
                     f"ρ = {rho:.2f}  (partial | exons: {prho:.2f})")

    ax = fig.add_subplot(gs[1, 2])
    df["depth_q"] = pd.qcut(df.total_reads, 5,
                            labels=["Q1\nlow", "Q2", "Q3", "Q4", "Q5\nhigh"], duplicates="drop")
    piv = (df.groupby(["exon_bin", "depth_q"], observed=True)["has_atse"].mean().unstack() * 100)
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=0, vmax=100, origin="lower")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, fontsize=7)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=7)
    ax.set_xlabel("gene read depth (quintile)"); ax.set_ylabel("exons")
    ax.set_title("F. % with ≥1 ATSE\n(exon count and depth act together)")
    fig.colorbar(im, ax=ax, label="% genes with ≥1 ATSE", fraction=0.046)

    fig.suptitle(f"What predicts detected splicing events per gene? — {d['label']}   "
                 f"({len(df):,} expressed genes, rule={args.rep_rule})", y=0.98, fontsize=12)
    for ext in ("png", "pdf"):
        p = os.path.join(args.outdir, f"supp_exons_vs_atses_{args.dataset}.{ext}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"\nfigure -> {p}")
    csv = os.path.join(args.outdir, f"supp_exons_vs_atses_{args.dataset}.csv")
    df.drop(columns=["exon_bin", "depth_q"]).to_csv(csv, index=False)
    print(f"table  -> {csv}")


if __name__ == "__main__":
    main()
