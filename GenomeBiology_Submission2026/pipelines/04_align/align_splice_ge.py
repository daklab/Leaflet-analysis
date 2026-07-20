#!/usr/bin/env python
"""
align_splice_ge.py — align splicing and gene-expression AnnDatas cell-by-cell.

Cell ID formats in this dataset:
  Splice AnnData (obs_names are already canonical):
    AB cells  → SRR run IDs (e.g. SRR16389944)
    TMS cells → cleaned junction-filename IDs (e.g. A10_B000497_B009023_S10,
                or 3m format A1-B000126-3_39_F-1-1)

  GE AnnData (obs_names need cleaning):
    AB cells  → exp_component_name (e.g. US-1250273_E1_S37 or LS-15348_S38_E1-50)
                which matches the `tissue` column in the combined metadata CSV,
                allowing us to reverse-map to SRR via metadata[tissue → cell_id].
    TMS cells → raw h5ad IDs that need platform-suffix stripping:
                non-3m: A10_B000497_B009023_S10.mm10-plus-0-0 → A10_B000497_B009023_S10
                3m:     A1.B000126.3_39_F.1.1-1-1 → A1-B000126-3_39_F-1-1
                (same transformation as format_cell_id in metadata_prep.py)

  After cleaning, both modalities share the same canonical ID space and can be
  intersected directly.

Usage:
    python pipelines/04_align/align_splice_ge.py \\
        --config  config/mouse_smartseq2.yaml \\
        --splice  /path/to/merged_anndata.h5ad \\
        --ge      /path/to/tms_ab_exons_combo_ge_adata.h5ad \\
        --output-dir /path/to/MODEL_INPUT/YYYYMMDD

SLURM:
    sbatch --mem=150G -p cpu \\
        --wrap "python pipelines/04_align/align_splice_ge.py --config ... --splice ... --ge ... --output-dir ..."
"""

import argparse
import datetime
import json
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def format_tms_cell_id(index: str, age: str) -> str:
    """
    Convert a raw TMS GE obs_name to the canonical cell ID used in the metadata CSV
    and in the splice AnnData obs_names. Mirrors format_cell_id() in metadata_prep.py.

    3m cells:
      raw  → A1.B000126.3_39_F.1.1-1-1
      canonical → A1-B000126-3_39_F-1-1

    non-3m cells (any other age):
      raw  → A10_B000497_B009023_S10.mm10-plus-0-0
      canonical → A10_B000497_B009023_S10   (strip at first '.')
    """
    if age == "3m":
        try:
            parts = index.replace(".", "-", 1).replace("_", "-", 1).split(".")
            if len(parts) < 2:
                return index
            corrected = parts[1].replace("-", "_", 1)
            return parts[0] + "-" + corrected + "-1-1"
        except Exception:
            return index
    return index.split(".")[0]


def clean_ge_cell_ids(ge: ad.AnnData, metadata: pd.DataFrame) -> pd.Series:
    """
    Map GE obs_names to canonical cell IDs that match splice obs_names.

    AB cells (dataset=='allen_brain_exons'):
        obs_name is exp_component_name; map to SRR via metadata tissue→cell_id.
        ~94% of GE AB cells have a match; unmapped cells are dropped at intersection.

    TMS cells (dataset=='tabula_muris_senis'):
        apply format_tms_cell_id(obs_name, age) — strips platform suffix or handles
        the 3m special format.

    Returns a Series aligned with ge.obs.index.
    """
    ab_meta = metadata[metadata["age"] == "2m"]
    tissue_to_srr = dict(zip(ab_meta["tissue"], ab_meta["cell_id"]))

    obs_names = ge.obs_names.tolist()
    datasets  = ge.obs["dataset"].tolist()
    ages      = ge.obs["age"].tolist() if "age" in ge.obs.columns else ["18m"] * len(obs_names)

    cleaned = []
    for obs_name, dataset, age in zip(obs_names, datasets, ages):
        if dataset == "allen_brain_exons":
            cleaned.append(tissue_to_srr.get(obs_name, obs_name))
        else:
            cleaned.append(format_tms_cell_id(obs_name, age))

    return pd.Series(cleaned, index=ge.obs.index)


def add_splice_metadata(splice: ad.AnnData, metadata: pd.DataFrame) -> ad.AnnData:
    """
    After alignment the splice AnnData obs_names are canonical cell IDs.
    Attach age, cell type, and other metadata columns from the combined CSV.
    Also set a dataset column (allen_brain / tabula_muris_senis).
    """
    meta_idx = metadata.set_index("cell_id")
    meta_sub = meta_idx.reindex(splice.obs_names)

    # cell_id from metadata index — same as obs_names at this point, but explicit
    splice.obs["cell_id"] = meta_sub.index.tolist()

    for col in ["age", "cell_ontology_class", "mouse.id", "sex", "subtissue", "tissue"]:
        if col in meta_sub.columns:
            splice.obs[col] = meta_sub[col].values

    splice.obs["dataset"] = "tabula_muris_senis"
    if "age" in splice.obs.columns:
        splice.obs.loc[splice.obs["age"] == "2m", "dataset"] = "allen_brain"

    # Normalize sex values: M/male → male, F/female → female
    if "sex" in splice.obs.columns:
        sex_map = {"M": "male", "F": "female", "male": "male", "female": "female"}
        splice.obs["sex"] = splice.obs["sex"].map(sex_map)
        n_unknown_sex = splice.obs["sex"].isna().sum()
        if n_unknown_sex:
            print(f"  WARNING: {n_unknown_sex} cells have unrecognized sex value — set to NaN")

    n_miss = splice.obs.get("age", pd.Series([None])).isna().sum()
    if n_miss:
        print(f"  WARNING: {n_miss} aligned cells are missing metadata — "
              f"they may have been excluded from the metadata CSV (e.g. 21m)")
    return splice


def apply_cell_type_map(splice: ad.AnnData, map_path: str) -> ad.AnnData:
    """Load {fine_label: broad_label} JSON and add broad_cell_type to splice obs."""
    with open(map_path) as f:
        ct_map = json.load(f)

    if "cell_ontology_class" not in splice.obs.columns:
        print("  WARNING: no cell_ontology_class in splice obs — skipping broad_cell_type")
        return splice

    splice.obs["broad_cell_type"] = (
        splice.obs["cell_ontology_class"]
        .map(ct_map)
        .fillna(splice.obs["cell_ontology_class"])
    )
    n_unmapped = splice.obs["broad_cell_type"].isnull().sum()
    if n_unmapped:
        print(f"  WARNING: {n_unmapped} cells have no broad_cell_type")
    else:
        print("  ✓ broad_cell_type distribution (top 10):")
        top = splice.obs["broad_cell_type"].value_counts().head(10)
        for bt, n in top.items():
            print(f"    {bt}: {n:,}")
    return splice


def mad_upper_bound(arr: np.ndarray, n_mads: float = 5.0) -> float:
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    return median + n_mads * mad


def apply_cell_qc(
    splice: ad.AnnData,
    ge: ad.AnnData,
    min_junctions: int = 5,
    max_junctions_nmads: float = 5.0,
    min_genes: int = 200,
    max_genes_nmads: float = 5.0,
) -> tuple[ad.AnnData, ad.AnnData]:
    """
    Filter cells jointly on splice and GE QC metrics.
    Adaptive upper bounds use median + n_mads * MAD to catch doublets.
    Both AnnDatas are filtered to the same cells so they stay aligned.
    """
    print("\nApplying cell QC filters...")

    # ── Splice: detected junctions and total reads ────────────────────────────
    junc_layer = splice.layers.get("cell_by_junction_matrix", splice.X)
    if sp.issparse(junc_layer):
        junc_layer = junc_layer.tocsr()
    junc_detected  = np.asarray((junc_layer > 0).sum(axis=1)).ravel()
    junc_reads     = np.asarray(junc_layer.sum(axis=1)).ravel()

    max_junctions  = mad_upper_bound(junc_detected, max_junctions_nmads)
    max_reads      = mad_upper_bound(junc_reads,    max_junctions_nmads)

    splice_pass = (
        (junc_detected >= min_junctions) &
        (junc_detected <= max_junctions) &
        (junc_reads    <= max_reads)
    )

    # ── GE: detected genes and total counts ───────────────────────────────────
    ge_X = ge.X
    if sp.issparse(ge_X):
        ge_X = ge_X.tocsr()
    genes_detected = np.asarray((ge_X > 0).sum(axis=1)).ravel()
    ge_counts      = np.asarray(ge_X.sum(axis=1)).ravel()

    max_genes      = mad_upper_bound(genes_detected, max_genes_nmads)
    max_ge_counts  = mad_upper_bound(ge_counts,      max_genes_nmads)

    ge_pass = (
        (genes_detected >= min_genes) &
        (genes_detected <= max_genes) &
        (ge_counts      <= max_ge_counts)
    )

    # ── Combined filter ───────────────────────────────────────────────────────
    keep = splice_pass & ge_pass
    n_before = splice.n_obs

    removed_min_junc  = (junc_detected  < min_junctions).sum()
    removed_max_junc  = (junc_detected  > max_junctions).sum()
    removed_max_reads = (junc_reads     > max_reads).sum()
    removed_min_gene  = (genes_detected < min_genes).sum()
    removed_max_gene  = (genes_detected > max_genes).sum()
    removed_max_ge    = (ge_counts      > max_ge_counts).sum()

    print(f"  Thresholds (splice):")
    print(f"    min junctions:  {min_junctions}  "
          f"max junctions: {max_junctions:.0f} (median+{max_junctions_nmads}×MAD)  "
          f"max reads: {max_reads:.0f}")
    print(f"  Thresholds (GE):")
    print(f"    min genes: {min_genes}  "
          f"max genes: {max_genes:.0f} (median+{max_genes_nmads}×MAD)  "
          f"max counts: {max_ge_counts:.0f}")
    print(f"  Removed breakdown:")
    print(f"    < {min_junctions} junctions:          {removed_min_junc:>6,}")
    print(f"    > {max_junctions:.0f} junctions (doublets): {removed_max_junc:>6,}")
    print(f"    > {max_reads:.0f} junction reads:    {removed_max_reads:>6,}")
    print(f"    < {min_genes} genes detected:      {removed_min_gene:>6,}")
    print(f"    > {max_genes:.0f} genes detected:    {removed_max_gene:>6,}")
    print(f"    > {max_ge_counts:.0f} GE counts:        {removed_max_ge:>6,}")
    print(f"  Cells before QC: {n_before:,}  →  after: {keep.sum():,}  "
          f"(removed {(~keep).sum():,})")

    splice = splice[keep].copy()
    ge     = ge[keep].copy()
    return splice, ge


def describe(arr: np.ndarray, label: str) -> list[str]:
    """Return summary stat lines for a 1-D array."""
    return [
        f"  {label}:",
        f"    min={arr.min():.0f}  p5={np.percentile(arr,5):.0f}  median={np.median(arr):.0f}"
        f"  mean={arr.mean():.1f}  p95={np.percentile(arr,95):.0f}  max={arr.max():.0f}",
    ]


def print_and_save_summary(splice: ad.AnnData, ge: ad.AnnData, out_path: str) -> None:
    """
    Print and save alignment summary statistics to out_path (text file).

    Covers:
      - Final gene / junction / ATSE counts (must match across modalities)
      - Junction total read count distribution
      - Junctions per ATSE distribution
      - ATSEs per gene distribution
      - Per-cell: total junction reads and detected junction count (min + distribution)
    """
    lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    junc_layer = splice.layers.get("cell_by_junction_matrix")
    if junc_layer is None:
        p("WARNING: cell_by_junction_matrix layer not found — skipping count summaries")
        junc_layer = splice.X

    if sp.issparse(junc_layer):
        junc_layer = junc_layer.tocsr()

    # ── Counts ────────────────────────────────────────────────────────────────
    n_cells    = splice.n_obs
    n_junctions = splice.n_vars
    n_atses    = splice.var["event_id"].nunique() if "event_id" in splice.var.columns else "N/A"
    n_genes    = splice.var["gene_name"].nunique() if "gene_name" in splice.var.columns else "N/A"
    n_ge_genes = ge.n_vars

    p("=" * 60)
    p("ALIGNMENT SUMMARY")
    p("=" * 60)
    p(f"Cells:          {n_cells:>10,}")
    p(f"Junctions:      {n_junctions:>10,}")
    p(f"ATSEs:          {n_atses:>10,}" if isinstance(n_atses, str) else f"ATSEs:          {n_atses:>10,}")
    p(f"Genes (splice): {n_genes:>10,}" if isinstance(n_genes, str) else f"Genes (splice): {n_genes:>10,}")
    p(f"Genes (GE):     {n_ge_genes:>10,}")
    if isinstance(n_genes, int) and n_genes == n_ge_genes:
        p("✓ Splice and GE gene counts match")
    else:
        p(f"NOTE: splice unique genes={n_genes}, GE vars={n_ge_genes} "
          f"(GE filtered to common set, splice has multi-junction genes)")

    # ── Junction total read counts (summed across all cells) ──────────────────
    p()
    p("Junction total read counts (sum across all cells per junction):")
    junc_totals = np.asarray(junc_layer.sum(axis=0)).ravel()
    for line in describe(junc_totals, "reads per junction"): p(line)

    # ── Per-cell stats ────────────────────────────────────────────────────────
    p()
    p("Per-cell statistics:")
    cell_total_reads = np.asarray(junc_layer.sum(axis=1)).ravel()
    cell_detected    = np.diff(junc_layer.indptr) if sp.isspmatrix_csr(junc_layer) \
                       else np.asarray((junc_layer > 0).sum(axis=1)).ravel()
    for line in describe(cell_total_reads, "total junction reads per cell"): p(line)
    for line in describe(cell_detected,    "detected junctions per cell"):   p(line)

    # ── Junctions per ATSE ────────────────────────────────────────────────────
    if "event_id" in splice.var.columns:
        p()
        p("Junctions per ATSE:")
        juncs_per_atse = splice.var.groupby("event_id", observed=True).size().values
        for line in describe(juncs_per_atse, "junctions per ATSE"): p(line)

    # ── ATSEs per gene ────────────────────────────────────────────────────────
    if "event_id" in splice.var.columns and "gene_name" in splice.var.columns:
        p()
        p("ATSEs per gene:")
        atses_per_gene = (
            splice.var[["gene_name", "event_id"]]
            .drop_duplicates()
            .groupby("gene_name", observed=True)
            .size()
            .values
        )
        for line in describe(atses_per_gene, "ATSEs per gene"): p(line)

    # ── Column existence checks ───────────────────────────────────────────────
    p()
    p("Column checks:")
    for col, obj, label in [
        ("cell_id",     splice.obs, "splice .obs"),
        ("junction_id", splice.var, "splice .var"),
        ("cell_id",     ge.obs,     "GE .obs"),
    ]:
        present = col in obj.columns
        p(f"  {'✓' if present else '✗ MISSING'} {label}['{col}']")

    # ── Obs metadata breakdowns ───────────────────────────────────────────────
    for col in ["dataset", "age", "sex", "tissue_cell_type"]:
        if col not in splice.obs.columns:
            p()
            p(f"WARNING: '{col}' not found in splice .obs")
            continue
        p()
        counts = splice.obs[col].value_counts(dropna=False)
        p(f"{col} breakdown ({counts.nunique()} unique values):")
        for val, n in counts.items():
            pct = 100 * n / splice.n_obs
            p(f"  {str(val):<45s}  {n:>7,}  ({pct:.1f}%)")

    p()
    p("=" * 60)

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--splice",     required=True, help="Merged splice AnnData (.h5ad)")
    parser.add_argument("--ge",         required=True, help="GE AnnData from 01_prepare_counts.py")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cfg     = load_config(args.config)
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    os.makedirs(args.output_dir, exist_ok=True)
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    print(f"Dataset: {cfg.get('dataset_label', 'unknown')}")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\nLoading splice AnnData...")
    splice = ad.read_h5ad(args.splice)
    print(f"  {splice.n_obs:,} cells × {splice.n_vars:,} features")
    print(f"  obs columns: {list(splice.obs.columns)}")

    # ── Filter cells with zero junction reads ─────────────────────────────────
    junc_layer = splice.layers.get("cell_by_junction_matrix", splice.X)
    cell_total_reads = np.asarray(junc_layer.sum(axis=1)).ravel()
    zero_mask = cell_total_reads == 0
    n_zero = zero_mask.sum()
    if n_zero:
        print(f"  Dropping {n_zero:,} cells with zero junction reads before alignment")
        splice = splice[~zero_mask].copy()
        print(f"  Splice after QC filter: {splice.n_obs:,} cells")
    else:
        print("  ✓ No zero-read cells found")

    print("Loading GE AnnData...")
    ge = ad.read_h5ad(args.ge)
    print(f"  {ge.n_obs:,} cells × {ge.n_vars:,} features")

    # ── Load combined metadata ────────────────────────────────────────────────
    meta_path = cfg.get("metadata_csv")
    if not meta_path:
        print("ERROR: 'metadata_csv' not set in config.")
        print("  Add it pointing to the combined cell metadata CSV "
              "(cell_id, age, cell_ontology_class, mouse.id, sex, subtissue, tissue).")
        sys.exit(1)
    if not os.path.isabs(meta_path):
        meta_path = os.path.join(cfg_dir, meta_path)
    print(f"\nLoading metadata: {meta_path}")
    metadata = pd.read_csv(meta_path)
    print(f"  {len(metadata):,} cells  |  ages: {dict(metadata['age'].value_counts())}")

    # ── Clean GE cell IDs ─────────────────────────────────────────────────────
    print("\nCleaning GE cell IDs...")
    ge_clean = clean_ge_cell_ids(ge, metadata)

    # Show a few examples
    ab_mask  = ge.obs["dataset"] == "allen_brain_exons"
    tms_mask = ge.obs["dataset"] == "tabula_muris_senis"
    if ab_mask.any():
        idx0 = ge.obs[ab_mask].index[0]
        print(f"  AB  example: {idx0}  →  {ge_clean[idx0]}")
    if tms_mask.any():
        idx0 = ge.obs[tms_mask].index[0]
        print(f"  TMS example: {idx0}  →  {ge_clean[idx0]}")

    # Splice obs_names are already canonical — no cleaning needed
    splice_clean = pd.Series(splice.obs_names.tolist(), index=splice.obs.index)

    # Duplicate check
    for name, clean in [("splice", splice_clean), ("ge", ge_clean)]:
        n_dup = clean.duplicated().sum()
        if n_dup:
            print(f"  WARNING: {n_dup} duplicate cleaned IDs in {name} — investigate!")
        else:
            print(f"  ✓ {name}: no duplicate cleaned IDs")

    # ── Intersect + align ─────────────────────────────────────────────────────
    common = np.intersect1d(splice_clean.values, ge_clean.values)
    pct_splice = 100 * len(common) / splice.n_obs
    pct_ge     = 100 * len(common) / ge.n_obs
    print(f"\nCommon cells: {len(common):,}")
    print(f"  Splice: {splice.n_obs:,} → {pct_splice:.1f}% retained")
    print(f"  GE:     {ge.n_obs:,} → {pct_ge:.1f}% retained")

    if len(common) == 0:
        print("ERROR: No common cells — cell ID cleaning has failed.")
        sys.exit(1)
    if pct_splice < 70 or pct_ge < 70:
        print("WARNING: <70% of cells matched — check cell ID formats")

    splice.obs_names = pd.Index(splice_clean.values)
    ge.obs_names     = pd.Index(ge_clean.values)

    splice = splice[common].copy()
    ge     = ge[common].copy()

    assert (splice.obs_names == ge.obs_names).all(), \
        "FATAL: splice and GE obs_names differ after intersection"
    print("✓ Cell order verified: splice and GE obs_names identical")

    # ── Add metadata + cell types to splice ───────────────────────────────────
    print("\nAttaching metadata to splice obs...")
    splice = add_splice_metadata(splice, metadata)

    ct_map_path = cfg.get("cell_type_map")
    if ct_map_path:
        # cell_type_map path is relative to repo root (one level above config/)
        repo_root = os.path.dirname(cfg_dir)
        map_file = os.path.join(repo_root, ct_map_path)
        if os.path.exists(map_file):
            print("Applying cell type map to splice...")
            splice = apply_cell_type_map(splice, map_file)
        else:
            print(f"WARNING: cell_type_map not found at {map_file}")

    # ── Recode AB tissue → brain region categories ────────────────────────────
    # AB tissue col = exp_component_name (ID mapping artefact, not a tissue name).
    # Recode using subtissue (Allen CCF region) → broad anatomical category so
    # tissue_cell_type is directly comparable between TMS and AB.
    AB_REGION_MAP = {
        # Visual cortex
        "VISp": "Brain_Isocortex_Visual",
        "VIS":  "Brain_Isocortex_Visual",
        # Motor cortex
        "MOp":  "Brain_Isocortex_Motor",
        "ALM":  "Brain_Isocortex_Motor",
        # Somatosensory cortex
        "SSp":  "Brain_Isocortex_Somatosensory",
        "SSs":  "Brain_Isocortex_Somatosensory",
        # Hippocampal formation
        "HIP":          "Brain_Hippocampus",
        "SUB-ProS":     "Brain_Hippocampus",
        "PAR-POST-PRE": "Brain_Hippocampus",
        "ENTl":         "Brain_Hippocampus",
        "ENTm":         "Brain_Hippocampus",
        # Association / prefrontal cortex
        "ACA":   "Brain_Isocortex_Association",
        "RSP":   "Brain_Isocortex_Association",
        "RSPv":  "Brain_Isocortex_Association",
        "PL-ILA":"Brain_Isocortex_Association",
        "ORB":   "Brain_Isocortex_Association",
        "AI":    "Brain_Isocortex_Association",
        "GU":    "Brain_Isocortex_Association",
        "CLA":   "Brain_Isocortex_Association",
        # Temporal / parietal cortex
        "TEa-PERI-ECT": "Brain_Isocortex_Temporal",
        "PTLp":         "Brain_Isocortex_Temporal",
        "AUD":          "Brain_Isocortex_Temporal",
    }
    if "dataset" in splice.obs.columns and "subtissue" in splice.obs.columns:
        ab_mask = splice.obs["dataset"] == "allen_brain"
        mapped = splice.obs.loc[ab_mask, "subtissue"].map(AB_REGION_MAP).fillna("Brain")
        n_unmapped = mapped.eq("Brain").sum()
        if n_unmapped:
            unmapped_vals = splice.obs.loc[ab_mask, "subtissue"][mapped.eq("Brain")].unique()
            print(f"  WARNING: {n_unmapped} AB cells have unmapped subtissue → 'Brain': {list(unmapped_vals)}")
        splice.obs.loc[ab_mask, "tissue"] = mapped.values
        print(f"\nAB tissue_label distribution after remapping:")
        for val, n in splice.obs.loc[ab_mask, "tissue"].value_counts().items():
            print(f"  {val}: {n:,}")

    # ── Add tissue_cell_type (tissue + broad_cell_type, uniform for TMS and AB) ─
    if "broad_cell_type" in splice.obs.columns:
        splice.obs["tissue_cell_type"] = splice.obs["tissue"] + "_" + splice.obs["broad_cell_type"]

        n_null = splice.obs["tissue_cell_type"].isna().sum()
        print(f"\ntissue_cell_type: {splice.obs['tissue_cell_type'].nunique():,} unique values")
        if n_null:
            print(f"  WARNING: {n_null:,} cells have null tissue_cell_type")
        else:
            print(f"  ✓ All cells have tissue_cell_type (no nulls)")
        print("  Top 10:")
        for val, n in splice.obs["tissue_cell_type"].value_counts().head(10).items():
            print(f"    {val}: {n:,}")

    # Dataset breakdown
    print("\nAligned dataset breakdown:")
    for modality, adata in [("splice", splice), ("ge", ge)]:
        if "dataset" in adata.obs.columns:
            dist = adata.obs["dataset"].value_counts()
            print(f"  {modality}: {dict(dist)}")

    # ── Filter to common genes between splice junctions and GE vars ───────────
    print("\nFiltering to common genes between splice and GE...")
    splice_genes = set(splice.var["gene_name"].dropna().unique())
    ge_genes     = set(ge.var_names)
    common_genes = splice_genes & ge_genes
    print(f"  Splice junction genes: {len(splice_genes):,}")
    print(f"  GE genes:              {len(ge_genes):,}")
    print(f"  Common genes:          {len(common_genes):,}")

    splice_mask = splice.var["gene_name"].isin(common_genes)
    ge_mask     = ge.var_names.isin(common_genes)
    splice = splice[:, splice_mask].copy()
    ge     = ge[:, ge_mask].copy()
    print(f"  Splice after filter: {splice.n_obs:,} cells × {splice.n_vars:,} junctions")
    print(f"  GE     after filter: {ge.n_obs:,} cells × {ge.n_vars:,} genes")

    # Drop any cells that became zero-read after gene filtering
    junc_layer_post = splice.layers.get("cell_by_junction_matrix", splice.X)
    cell_reads_post = np.asarray(junc_layer_post.sum(axis=1)).ravel()
    zero_post = cell_reads_post == 0
    n_zero_post = zero_post.sum()
    if n_zero_post:
        print(f"  Dropping {n_zero_post:,} cells with zero reads after gene filter")
        keep_cells = ~zero_post
        splice = splice[keep_cells].copy()
        ge     = ge[keep_cells].copy()
        print(f"  Cells after zero-read filter: {splice.n_obs:,}")
    else:
        print("  ✓ No zero-read cells after gene filter")

    # ── Cell QC filters (min/max junctions and genes) ─────────────────────────
    qc_cfg = cfg.get("cell_qc", {})
    splice, ge = apply_cell_qc(
        splice, ge,
        min_junctions       = qc_cfg.get("min_junctions",       5),
        max_junctions_nmads = qc_cfg.get("max_junctions_nmads", 5.0),
        min_genes           = qc_cfg.get("min_genes",           200),
        max_genes_nmads     = qc_cfg.get("max_genes_nmads",     5.0),
    )

    # ── Add junction_id as explicit var column (var_names are junction coords) ──
    splice.var["junction_id"] = splice.var_names.tolist()

    # ── Sync obs columns so both AnnDatas have the same metadata ─────────────
    # Propagate tissue_cell_type from splice → GE (tissue already synced above)
    for col in ["tissue_cell_type"]:
        if col in splice.obs.columns:
            ge.obs[col] = splice.obs[col].values
    # Propagate batch and meta_sample_name from GE → splice (GE-processing columns)
    for col in ["batch", "meta_sample_name"]:
        if col in ge.obs.columns:
            splice.obs[col] = ge.obs[col].values

    # ── Propagate cell_id to GE (already set in splice obs from metadata) ──────
    ge.obs["cell_id"] = splice.obs["cell_id"].values

    # ── Reset to integer obs_names for model compatibility ────────────────────
    splice.obs_names = pd.Index(map(str, range(splice.n_obs)))
    ge.obs_names     = pd.Index(map(str, range(ge.n_obs)))

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_out = os.path.join(args.output_dir, f"alignment_summary_{today}.txt")
    print_and_save_summary(splice, ge, summary_out)

    # ── Save ──────────────────────────────────────────────────────────────────
    splice_out = os.path.join(args.output_dir, f"splice_adata_aligned_{today}.h5ad")
    ge_out     = os.path.join(args.output_dir, f"ge_adata_aligned_{today}.h5ad")

    print(f"\nSaving splice {splice.shape} → {splice_out}")
    splice.write_h5ad(splice_out, compression="lzf")

    print(f"Saving GE    {ge.shape} → {ge_out}")
    ge.write_h5ad(ge_out, compression="lzf")

    print("\nDone.")


if __name__ == "__main__":
    main()
