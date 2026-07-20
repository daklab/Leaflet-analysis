#!/usr/bin/env python
"""
Step 8a: Load raw gene expression counts, compute gene lengths, length-normalize,
and produce a combined (multi-source) log-normalized AnnData.

Mouse sources: Tabula Muris Senis FACS + Allen Brain (exons + introns).
Human sources: Tabula Sapiens + Allen Brain Human MTG.

Outputs (in cfg['gene_expression']['output_dir']):
  - tms_ab_exons_combo_ge_adata_YYYY-MM-DD.h5ad   (mouse)
  - ab_adata_introns_YYYY-MM-DD.h5ad              (mouse introns, kept separate)

Usage:
    python pipelines/03_gene_expression/01_prepare_counts.py config/mouse_smartseq2.yaml
"""

import sys
import os
import json
import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.sparse import csr_matrix
import gffutils
from tqdm import tqdm
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── GTF-based gene length computation ─────────────────────────────────────────

def build_gene_length_table(gtf_file: str, db_file: str) -> pd.DataFrame:
    """Compute mean transcript and intron length per gene from GTF."""
    print("Building gene length table from GTF...")
    if os.path.exists(db_file):
        db = gffutils.FeatureDB(db_file, keep_order=True)
    else:
        print("  Creating gffutils DB (one-time, may take a few minutes)...")
        db = gffutils.create_db(gtf_file, db_file, force=True, keep_order=True,
                                disable_infer_transcripts=False, disable_infer_genes=True)

    # Collect gene-level biotype from gene features (gene_type attribute in GENCODE GTF)
    gene_biotype = {}
    for gene in db.features_of_type("gene"):
        gid = gene.attributes.get("gene_id", [None])[0]
        if gid:
            gene_biotype[gid] = gene.attributes.get("gene_type", ["unknown"])[0]

    exon_lens, intron_lens, gene_names, tx_counts = (
        defaultdict(list), defaultdict(list), {}, defaultdict(int)
    )

    for tx in tqdm(db.features_of_type("transcript"), desc="Transcripts"):
        gene_id   = tx.attributes.get("gene_id",   [None])[0]
        gene_name = tx.attributes.get("gene_name", ["unknown"])[0]
        if not gene_id:
            continue
        exons = list(db.children(tx, featuretype="exon", order_by="start"))
        if not exons:
            continue
        el = sum(e.end - e.start + 1 for e in exons)
        span = max(e.end for e in exons) - min(e.start for e in exons) + 1
        exon_lens[gene_id].append(el)
        intron_lens[gene_id].append(max(0, span - el))
        gene_names[gene_id] = gene_name
        tx_counts[gene_id] += 1

    gene_ids = list(exon_lens)
    df = pd.DataFrame({
        "gene_id":                gene_ids,
        "gene_name":              [gene_names[g] for g in gene_ids],
        "gene_type":              [gene_biotype.get(g, "unknown") for g in gene_ids],
        "mean_transcript_length": [np.mean(exon_lens[g]) for g in gene_ids],
        "mean_intron_length":     [np.mean(intron_lens[g]) for g in gene_ids],
        "num_transcripts":        [tx_counts[g] for g in gene_ids],
    })
    print(f"  {len(df)} genes extracted")
    return df


# GENCODE protein-coding and lncRNA biotype values (underscore form used in GTF)
KEEP_BIOTYPES = {
    "protein_coding",
    "lncRNA", "lincRNA", "antisense", "sense_intronic", "sense_overlapping",
    "processed_transcript", "3prime_overlapping_ncRNA", "non_coding",
    "macro_lncRNA", "bidirectional_promoter_lncRNA",
}


def annotate_genes(gene_df: pd.DataFrame, species=None) -> pd.DataFrame:
    """Restrict to protein-coding + lncRNA genes using GTF gene_type."""
    before = len(gene_df)
    gene_df = gene_df[gene_df["gene_type"].isin(KEEP_BIOTYPES)].copy()
    counts = gene_df["gene_type"].value_counts()
    print(f"  Biotype filter: {before} → {len(gene_df)} genes")
    print(f"  Retained: {counts.to_dict()}")
    return gene_df


# ── Normalization helpers ──────────────────────────────────────────────────────

def length_normalize(adata: ad.AnnData) -> None:
    """Divide counts by mean transcript length (stored in adata.var)."""
    lengths = adata.var["mean_transcript_length"].values.astype(float)
    if hasattr(adata.X, "toarray"):
        X = adata.X.toarray()
    else:
        X = adata.X.copy()
    X = X / lengths[np.newaxis, :]
    adata.layers["length_norm"] = csr_matrix(X)


def normalize_and_log(adata: ad.AnnData) -> None:
    """Library-size normalize from length_norm layer, then log1p."""
    sc.pp.normalize_total(adata, layer="length_norm", target_sum=1e4)
    X = adata.layers["length_norm"]
    if hasattr(X, "toarray"):
        X = X.toarray()
    adata.layers["log_norm"] = csr_matrix(np.log1p(X))


# ── Dataset loading ────────────────────────────────────────────────────────────

def _keep_cols(adata: ad.AnnData, cols: list) -> ad.AnnData:
    for c in cols:
        if c not in adata.obs.columns:
            adata.obs[c] = "unknown"
    adata.obs = adata.obs[cols].copy()
    return adata


def load_mouse_datasets(cfg: dict, gene_df: pd.DataFrame):
    src = cfg["sources"]
    tms_src = src["tabula_muris_senis"]
    ab_src = src["allen_brain"]
    keep = ["cell_id", "age", "cell_ontology_class", "mouse.id", "sex", "subtissue", "tissue", "dataset", "meta_sample_name"]

    # ── TMS ────────────────────────────────────────────────────────────────────
    print("Loading Tabula Muris Senis...")
    tms = sc.read_h5ad(tms_src["ge_h5ad"]).copy()
    tms.obs["dataset"] = "tabula_muris_senis"
    tms = _keep_cols(tms, keep)
    if not isinstance(tms.X, csr_matrix):
        tms.X = csr_matrix(tms.X)

    # ── Allen Brain (exons + introns) ──────────────────────────────────────────
    print("Loading Allen Brain exons...")
    ab_exons = sc.read_csv(ab_src["ge_exons"]).copy()
    ab_exons.obs["sample_name"] = ab_exons.obs.index
    meta_ab = pd.read_csv(ab_src["metadata_csv"], low_memory=False)

    # Include sample_name from metadata as meta_sample_name to avoid join conflict
    # with the sample_name column already set from the adata index above
    meta_ab_slim = meta_ab[["exp_component_name", "cell_type_alias_label",
                             "external_donor_name_label", "donor_sex_label",
                             "region_label", "sample_name"]].rename(
        columns={"sample_name": "meta_sample_name"}).set_index("exp_component_name")

    def _annotate_ab(adata, label):
        adata.obs = adata.obs.join(meta_ab_slim, how="left")
        adata.obs["cell_id"]             = adata.obs.index
        adata.obs["cell_ontology_class"] = adata.obs.get("cell_type_alias_label", "unknown")
        adata.obs["tissue"]              = adata.obs.index
        adata.obs["age"]                 = ab_src.get("age", "2m")
        adata.obs["mouse.id"]            = adata.obs.get("external_donor_name_label", "unknown")
        adata.obs["subtissue"]           = adata.obs.get("region_label", "unknown")
        adata.obs["sex"]                 = adata.obs.get("donor_sex_label", "unknown")
        adata.obs["dataset"]             = label
        adata = _keep_cols(adata, keep)
        if not isinstance(adata.X, csr_matrix):
            adata.X = csr_matrix(adata.X)
        return adata

    ab_exons = _annotate_ab(ab_exons, "allen_brain_exons")

    print("Loading Allen Brain introns...")
    ab_introns = sc.read_csv(ab_src["ge_introns"]).copy()
    ab_introns.obs["sample_name"] = ab_introns.obs.index
    ab_introns = _annotate_ab(ab_introns, "allen_brain_introns")

    return tms, ab_exons, ab_introns


def _add_gene_lengths(adata: ad.AnnData, gene_df: pd.DataFrame, name: str) -> ad.AnnData:
    """Inner join on gene_name to attach length info."""
    gname_col = "gene_symbol" if "gene_symbol" in adata.var.columns else None
    if gname_col:
        adata.var["gene_name"] = adata.var[gname_col]
    else:
        adata.var["gene_name"] = adata.var.index

    before = adata.shape[1]
    merged = adata.var.reset_index().merge(gene_df, on="gene_name", how="inner").set_index("index")
    adata = adata[:, merged.index].copy()
    adata.var = merged
    print(f"  {name}: {before} → {adata.shape[1]} genes after gene-length join")
    return adata


def process_mouse(cfg: dict) -> None:
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    ge_cfg = cfg["gene_expression"]
    out_dir = ge_cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    species = ge_cfg.get("species", "mouse")

    # Gene lengths
    gene_df = build_gene_length_table(cfg["gtf_file"], cfg["db_file"])
    gene_df = annotate_genes(gene_df, species=None)

    # Drop single-exon genes (mean_intron_length == 0) upfront so exon and intron
    # files share the same gene set — required for sc/sn correction
    n_before = len(gene_df)
    gene_df = gene_df[gene_df["mean_intron_length"] > 0].copy()
    print(f"  Dropped {n_before - len(gene_df)} single-exon genes (no introns), {len(gene_df)} remain")

    # Load raw datasets
    tms, ab_exons, ab_introns = load_mouse_datasets(cfg, gene_df)

    # Attach gene lengths
    tms      = _add_gene_lengths(tms,       gene_df, "TMS")
    ab_exons = _add_gene_lengths(ab_exons,  gene_df, "AB exons")
    ab_introns = _add_gene_lengths(ab_introns, gene_df, "AB introns")

    # Common genes
    common = set(tms.var["gene_name"]) & set(ab_exons.var["gene_name"]) & set(ab_introns.var["gene_name"])
    tms      = tms[:,      tms.var["gene_name"].isin(common)].copy()
    ab_exons = ab_exons[:, ab_exons.var["gene_name"].isin(common)].copy()
    ab_introns = ab_introns[:, ab_introns.var["gene_name"].isin(common)].copy()
    print(f"Common genes (protein-coding + lncRNA): {len(common)}")

    # Deduplicate gene names, sort, and set var.index to gene_name so concat works
    def _dedup_sort(adata, label):
        dupes = adata.var["gene_name"].duplicated(keep="first")
        if dupes.any():
            dupe_names = adata.var["gene_name"][dupes].tolist()
            print(f"  {label}: dropping {dupes.sum()} duplicate gene names: {dupe_names}")
        else:
            print(f"  {label}: no duplicate gene names")
        adata = adata[:, ~dupes].copy()
        adata = adata[:, adata.var.sort_values("gene_name").index].copy()
        adata.var.index = adata.var["gene_name"]
        adata.var.index.name = "gene_name"
        return adata

    tms        = _dedup_sort(tms, "TMS")
    ab_exons   = _dedup_sort(ab_exons, "AB exons")
    ab_introns = _dedup_sort(ab_introns, "AB introns")

    # Normalize exon-based datasets by mean transcript (exon) length
    for ds, label in [(tms, "TMS"), (ab_exons, "AB exons")]:
        length_normalize(ds)
        normalize_and_log(ds)
        print(f"  {label}: normalized")

    # Normalize introns by mean intron length
    ab_introns.var["mean_transcript_length"] = ab_introns.var["mean_intron_length"].copy()
    length_normalize(ab_introns)
    normalize_and_log(ab_introns)
    print("  AB introns: normalized")

    # Combine TMS + AB exons
    combined = ad.concat(
        [ab_exons, tms], axis=0, join="inner",
        label="batch", keys=["allen_brain_exons", "tabula_muris_senis"],
        index_unique=None,
    )
    combined.var = ab_exons.var.copy()
    combined.layers["raw_counts"] = combined.X.copy()  # preserve raw before any further ops

    # Add broad_cell_type from cell type map
    ct_map_path = cfg.get("cell_type_map")
    if ct_map_path:
        map_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                ct_map_path)
        if os.path.exists(map_file):
            with open(map_file) as f:
                ct_map = json.load(f)
            combined.obs["broad_cell_type"] = (
                combined.obs["cell_ontology_class"].map(ct_map)
                .fillna(combined.obs["cell_ontology_class"])
            )
            n_unmapped = combined.obs["broad_cell_type"].isnull().sum()
            if n_unmapped:
                print(f"  WARNING: {n_unmapped} cells have no broad_cell_type")
            print("  broad_cell_type distribution:")
            print(combined.obs["broad_cell_type"].value_counts().to_string())
        else:
            print(f"  WARNING: cell_type_map not found at {map_file}")

    # Save
    combo_path = os.path.join(out_dir, f"tms_ab_exons_combo_ge_adata_{today}.h5ad")
    intron_path = os.path.join(out_dir, f"ab_adata_introns_{today}.h5ad")
    combined.write_h5ad(combo_path, compression="lzf")
    ab_introns.write_h5ad(intron_path, compression="lzf")
    print(f"\nSaved combined ({combined.shape}) → {combo_path}")
    print(f"Saved introns  ({ab_introns.shape}) → {intron_path}")


def process_human_ab(cfg: dict) -> None:
    """
    Build GE AnnData for human Allen Brain (lein-human-cortex, single source).

    Source: lein-human-cortex/GeneExpression/adata_2025-01-07_HVG_only_False.h5ad
      - obs_names: sample_name format (F2S4_160113_027_A01) — matches splice obs_names
      - layer: 'counts' (raw integer counts)
      - 49,417 cells × 46,907 genes

    Outlier cells (outlier_call==True in INFO/metadata.csv) are removed.
    Gene processing: Gencode v45 GTF → length table → biotype filter → drop single-exon
    → length normalize → library-size normalize → log1p.

    Output: processed_data/ab_adata_exons_YYYY-MM-DD.h5ad
    """
    today   = datetime.datetime.now().strftime("%Y-%m-%d")
    ge_cfg  = cfg["gene_expression"]
    out_dir = ge_cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Gene lengths from Gencode v45 GTF
    gene_df = build_gene_length_table(cfg["gtf_file"], cfg["db_file"])
    gene_df = annotate_genes(gene_df)
    n_before = len(gene_df)
    gene_df = gene_df[gene_df["mean_intron_length"] > 0].copy()
    print(f"  Dropped {n_before - len(gene_df)} single-exon genes, {len(gene_df)} remain")

    # ── Load raw GE h5ad ──────────────────────────────────────────────────────
    # obs_names = sample_name (F1S4_... format) — already matches splice obs_names
    ge_h5ad = ge_cfg["ge_h5ad"]
    print(f"Loading {ge_h5ad} ...")
    adata = sc.read_h5ad(ge_h5ad).copy()
    print(f"  Loaded {adata.shape}")
    print(f"  obs_names example: {adata.obs_names[0]}")

    # Set X to raw counts (stored in 'counts' layer)
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    adata.layers["raw_counts"] = csr_matrix(adata.X)
    if not isinstance(adata.X, csr_matrix):
        adata.X = csr_matrix(adata.X)

    # ── Filter outlier cells ──────────────────────────────────────────────────
    src = cfg["sources"]["allen_brain_human"]
    meta_csv = src.get("metadata_csv",
        "/gpfs/commons/datasets/controlled/BRAIN_NeMO/lein-human-cortex/INFO/metadata.csv")
    meta = pd.read_csv(meta_csv, low_memory=False)
    outlier_ids = set(meta.loc[meta["outlier_call"] == True, "sample_name"].astype(str))
    n_before = adata.shape[0]
    adata = adata[~adata.obs_names.isin(outlier_ids)].copy()
    print(f"  Outlier filter: {n_before} → {adata.shape[0]} cells ({n_before - adata.shape[0]} removed)")

    # ── Gene length join ──────────────────────────────────────────────────────
    adata.var["gene_name"] = adata.var.index
    before = adata.shape[1]
    merged = adata.var.reset_index().merge(gene_df, on="gene_name", how="inner").set_index("index")
    adata = adata[:, merged.index].copy()
    adata.var = merged
    print(f"  Gene-length join: {before} → {adata.shape[1]} genes")

    # Deduplicate gene names, sort, reindex to gene_name
    dupes = adata.var["gene_name"].duplicated(keep="first")
    if dupes.any():
        print(f"  Dropping {dupes.sum()} duplicate gene names")
        adata = adata[:, ~dupes].copy()
    adata = adata[:, adata.var.sort_values("gene_name").index].copy()
    adata.var.index = adata.var["gene_name"]
    adata.var.index.name = "gene_name"

    # ── Normalize ─────────────────────────────────────────────────────────────
    length_normalize(adata)
    normalize_and_log(adata)
    print(f"  Normalized: {adata.shape}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(out_dir, f"ab_adata_exons_{today}.h5ad")
    adata.write_h5ad(out_path, compression="lzf")
    print(f"\nSaved ({adata.shape}) → {out_path}")
    print(f"Layers: {list(adata.layers.keys())}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python 01_prepare_counts.py <config.yaml>")
        sys.exit(1)
    cfg = load_config(sys.argv[1])
    species = cfg.get("species", "mouse")
    if species == "mouse":
        process_mouse(cfg)
    elif species == "human":
        process_human_ab(cfg)
    else:
        raise NotImplementedError(f"species={species} not yet implemented")


if __name__ == "__main__":
    main()
