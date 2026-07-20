#!/usr/bin/env python
"""
align_splice_ge_human.py — align human Allen Brain splicing and GE AnnDatas.

Human case is simpler than mouse (single source, no SRR/TMS mapping):
  Splice obs_names = sample_name (from BED filenames)  → already canonical
  GE obs_names    = sample_name (01_prepare_counts.py remapped from exp_component_name)
  → direct intersection, no ID cleaning needed

Metadata from metadata_human_allen_brain_combined.csv (output of 00_prepare_human_ab_metadata.py)
  Columns attached to splice obs: donor, sex, age, tissue, cortical_layer,
    cell_type_ontology_class, broad_cell_type, class_label, tissue_cell_type

Usage:
    python pipelines/04_align/align_splice_ge_human.py \\
        --config  config/human_allen_brain.yaml \\
        --splice  /path/to/merged_anndata.h5ad \\
        --ge      /path/to/ab_adata_exons_YYYY-MM-DD.h5ad \\
        --output-dir /path/to/MODEL_INPUT/YYYYMMDD

SLURM (100–150G sufficient — human ~47K cells):
    PYTHON=/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python
    sbatch --mem=150G -p cpu \\
        --output=${OUT_DIR}/align_%j.out \\
        --wrap "export PYTHONUNBUFFERED=1; $PYTHON pipelines/04_align/align_splice_ge_human.py \\
            --config $CONFIG --splice $SPLICE --ge $GE --output-dir $OUT_DIR"
"""

import argparse
import datetime
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
) -> tuple:
    print("\nApplying cell QC filters...")

    junc_layer = splice.layers.get("cell_by_junction_matrix", splice.X)
    if sp.issparse(junc_layer):
        junc_layer = junc_layer.tocsr()
    junc_detected = np.asarray((junc_layer > 0).sum(axis=1)).ravel()
    junc_reads    = np.asarray(junc_layer.sum(axis=1)).ravel()

    max_junctions = mad_upper_bound(junc_detected, max_junctions_nmads)
    max_reads     = mad_upper_bound(junc_reads,    max_junctions_nmads)

    ge_X = ge.X
    if sp.issparse(ge_X):
        ge_X = ge_X.tocsr()
    genes_detected = np.asarray((ge_X > 0).sum(axis=1)).ravel()
    ge_counts      = np.asarray(ge_X.sum(axis=1)).ravel()

    max_genes     = mad_upper_bound(genes_detected, max_genes_nmads)
    max_ge_counts = mad_upper_bound(ge_counts,      max_genes_nmads)

    keep = (
        (junc_detected >= min_junctions) &
        (junc_detected <= max_junctions) &
        (junc_reads    <= max_reads) &
        (genes_detected >= min_genes) &
        (genes_detected <= max_genes) &
        (ge_counts     <= max_ge_counts)
    )

    print(f"  Splice thresholds:  junc ≥ {min_junctions}, junc ≤ {max_junctions:.0f} "
          f"(median+{max_junctions_nmads}×MAD),  reads ≤ {max_reads:.0f}")
    print(f"  GE thresholds:      genes ≥ {min_genes}, genes ≤ {max_genes:.0f}, "
          f"counts ≤ {max_ge_counts:.0f}")
    print(f"  Removed breakdown:")
    print(f"    < {min_junctions} junctions:          {(junc_detected < min_junctions).sum():>6,}")
    print(f"    > {max_junctions:.0f} junctions (doublets): {(junc_detected > max_junctions).sum():>6,}")
    print(f"    > {max_reads:.0f} junction reads:    {(junc_reads > max_reads).sum():>6,}")
    print(f"    < {min_genes} genes:               {(genes_detected < min_genes).sum():>6,}")
    print(f"    > {max_genes:.0f} genes (doublets):   {(genes_detected > max_genes).sum():>6,}")
    print(f"  Cells: {splice.n_obs:,} → {keep.sum():,}  (removed {(~keep).sum():,})")

    return splice[keep].copy(), ge[keep].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--splice",     required=True, help="Merged splice AnnData (.h5ad)")
    parser.add_argument("--ge",         required=True, help="GE AnnData from 01_prepare_counts.py")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cfg     = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    today   = datetime.datetime.now().strftime("%Y-%m-%d")

    print(f"Dataset: {cfg.get('dataset_label', 'unknown')}")

    # ── Load splice ───────────────────────────────────────────────────────────
    print("\nLoading splice AnnData...")
    splice = ad.read_h5ad(args.splice)
    print(f"  {splice.n_obs:,} cells × {splice.n_vars:,} junctions")
    print(f"  Layers: {list(splice.layers.keys())}")
    print(f"  obs_names example: {splice.obs_names[0]}")

    # Drop cells with zero junction reads
    junc_layer = splice.layers.get("cell_by_junction_matrix", splice.X)
    cell_reads = np.asarray(junc_layer.sum(axis=1)).ravel()
    zero_mask  = cell_reads == 0
    if zero_mask.any():
        print(f"  Dropping {zero_mask.sum():,} zero-read cells")
        splice = splice[~zero_mask].copy()
        print(f"  Splice after zero-read filter: {splice.n_obs:,} cells")
    else:
        print("  ✓ No zero-read cells")

    # ── Load GE ───────────────────────────────────────────────────────────────
    print("\nLoading GE AnnData...")
    ge = ad.read_h5ad(args.ge)
    print(f"  {ge.n_obs:,} cells × {ge.n_vars:,} genes")
    print(f"  Layers: {list(ge.layers.keys())}")
    print(f"  obs_names example: {ge.obs_names[0]}")

    # ── Load metadata ─────────────────────────────────────────────────────────
    meta_path = cfg.get("metadata_csv")
    if not meta_path:
        print("ERROR: 'metadata_csv' not set in config.")
        sys.exit(1)
    print(f"\nLoading metadata: {meta_path}")
    metadata = pd.read_csv(meta_path)
    print(f"  {len(metadata):,} rows | "
          f"{metadata['tissue'].nunique()} tissues | "
          f"{metadata['broad_cell_type'].nunique()} cell types")

    # ── Check for duplicate cell IDs ──────────────────────────────────────────
    for name, ids in [("splice", splice.obs_names), ("ge", ge.obs_names)]:
        n_dup = pd.Series(ids).duplicated().sum()
        if n_dup:
            print(f"  WARNING: {n_dup} duplicate obs_names in {name}")
        else:
            print(f"  ✓ {name}: no duplicate obs_names")

    # ── Direct intersection (both already use sample_name format) ─────────────
    print("\nIntersecting on sample_name obs_names...")
    common = np.intersect1d(splice.obs_names, ge.obs_names)
    pct_splice = 100 * len(common) / splice.n_obs
    pct_ge     = 100 * len(common) / ge.n_obs
    print(f"  Common: {len(common):,}  "
          f"(splice {pct_splice:.1f}% / GE {pct_ge:.1f}% retained)")

    if len(common) == 0:
        print("ERROR: No common cells. Check that GE obs_names are in sample_name "
              "format (01_prepare_counts.py must have run successfully).")
        sys.exit(1)
    if pct_splice < 85 or pct_ge < 85:
        print("WARNING: <85% cells matched — investigate obs_names format mismatch")
        print(f"  Splice examples: {splice.obs_names[:3].tolist()}")
        print(f"  GE examples:     {ge.obs_names[:3].tolist()}")

    splice = splice[common].copy()
    ge     = ge[common].copy()

    assert (splice.obs_names == ge.obs_names).all(), \
        "FATAL: splice and GE obs_names differ after intersection"
    print("  ✓ Cell order verified: splice and GE obs_names identical")

    # ── Attach metadata to splice obs ─────────────────────────────────────────
    print("\nAttaching metadata to splice obs...")
    meta_idx = metadata.set_index("cell_id")
    meta_sub = meta_idx.reindex(splice.obs_names)

    n_miss = meta_sub.isnull().all(axis=1).sum()
    if n_miss:
        print(f"  WARNING: {n_miss} aligned cells missing from metadata CSV")
    else:
        print(f"  ✓ All {splice.n_obs:,} cells found in metadata CSV")

    for col in ["donor", "sex", "age", "tissue", "cortical_layer",
                "cell_type_ontology_class", "broad_cell_type", "class_label",
                "tissue_cell_type", "exp_component_name"]:
        if col in meta_sub.columns:
            splice.obs[col] = meta_sub[col].values

    splice.obs["cell_id"] = splice.obs_names.tolist()
    splice.obs["dataset"] = "allen_brain_human"

    # Propagate minimal metadata to GE obs
    ge.obs["cell_id"]          = splice.obs["cell_id"].values
    ge.obs["dataset"]          = "allen_brain_human"
    ge.obs["broad_cell_type"]  = splice.obs.get("broad_cell_type", pd.Series(dtype=str)).values
    ge.obs["tissue_cell_type"] = splice.obs.get("tissue_cell_type", pd.Series(dtype=str)).values

    print("\n  broad_cell_type distribution:")
    for val, n in splice.obs["broad_cell_type"].value_counts().items():
        print(f"    {val}: {n:,}")

    print("\n  tissue distribution:")
    for val, n in splice.obs["tissue"].value_counts().items():
        print(f"    {val}: {n:,}")

    # ── Filter to common genes ────────────────────────────────────────────────
    print("\nFiltering to common genes between splice and GE...")
    if "gene_name" in splice.var.columns:
        splice_genes = set(splice.var["gene_name"].dropna().unique())
    else:
        splice_genes = set(splice.var_names)
    ge_genes     = set(ge.var_names)
    common_genes = splice_genes & ge_genes
    print(f"  Splice genes: {len(splice_genes):,}")
    print(f"  GE genes:     {len(ge_genes):,}")
    print(f"  Common:       {len(common_genes):,}")

    if "gene_name" in splice.var.columns:
        splice_mask = splice.var["gene_name"].isin(common_genes)
    else:
        splice_mask = splice.var_names.isin(common_genes)
    ge_mask = ge.var_names.isin(common_genes)

    splice = splice[:, splice_mask].copy()
    ge     = ge[:, ge_mask].copy()
    print(f"  Splice after filter: {splice.n_obs:,} × {splice.n_vars:,} junctions")
    print(f"  GE     after filter: {ge.n_obs:,} × {ge.n_vars:,} genes")

    # Drop cells with zero reads after gene filter
    junc_post  = splice.layers.get("cell_by_junction_matrix", splice.X)
    reads_post = np.asarray(junc_post.sum(axis=1)).ravel()
    zero_post  = reads_post == 0
    if zero_post.any():
        print(f"  Dropping {zero_post.sum():,} cells with zero reads after gene filter")
        splice = splice[~zero_post].copy()
        ge     = ge[~zero_post].copy()
        print(f"  Cells after zero-read filter: {splice.n_obs:,}")
    else:
        print("  ✓ No zero-read cells after gene filter")

    # ── Cell QC ───────────────────────────────────────────────────────────────
    qc_cfg = cfg.get("cell_qc", {})
    splice, ge = apply_cell_qc(
        splice, ge,
        min_junctions       = qc_cfg.get("min_junctions",       5),
        max_junctions_nmads = qc_cfg.get("max_junctions_nmads", 5.0),
        min_genes           = qc_cfg.get("min_genes",           200),
        max_genes_nmads     = qc_cfg.get("max_genes_nmads",     5.0),
    )

    # ── Add junction_id var column ────────────────────────────────────────────
    splice.var["junction_id"] = splice.var_names.tolist()

    # ── Reset obs_names to integers (model compatibility) ─────────────────────
    splice.obs_names = pd.Index(map(str, range(splice.n_obs)))
    ge.obs_names     = pd.Index(map(str, range(ge.n_obs)))

    # ── Summary ───────────────────────────────────────────────────────────────
    n_atses = splice.var["event_id"].nunique() if "event_id" in splice.var.columns else "N/A"
    n_genes = splice.var["gene_name"].nunique() if "gene_name" in splice.var.columns else "N/A"
    print("\n" + "=" * 60)
    print("ALIGNMENT SUMMARY")
    print("=" * 60)
    print(f"Cells:          {splice.n_obs:>8,}")
    print(f"Junctions:      {splice.n_vars:>8,}")
    print(f"ATSEs:          {n_atses!s:>8}")
    print(f"Genes (splice): {n_genes!s:>8}")
    print(f"Genes (GE):     {ge.n_vars:>8,}")
    print()
    print("tissue_cell_type distribution (top 15):")
    for val, n in splice.obs["tissue_cell_type"].value_counts().head(15).items():
        pct = 100 * n / splice.n_obs
        print(f"  {str(val):<45s}  {n:>6,}  ({pct:.1f}%)")
    print("=" * 60)

    # ── Save ──────────────────────────────────────────────────────────────────
    splice_out = os.path.join(args.output_dir, f"splice_adata_aligned_{today}.h5ad")
    ge_out     = os.path.join(args.output_dir, f"ge_adata_aligned_{today}.h5ad")

    print(f"\nSaving splice {splice.shape} → {splice_out}")
    splice.write_h5ad(splice_out, compression="lzf")

    print(f"Saving GE     {ge.shape} → {ge_out}")
    ge.write_h5ad(ge_out, compression="lzf")

    print("\nDone.")


if __name__ == "__main__":
    main()
