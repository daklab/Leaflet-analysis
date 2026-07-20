#!/usr/bin/env python
"""
09b_filter_aligned.py — filter aligned splice + GE AnnDatas before training.

Sits between Step 9 (align_splice_ge.py) and Step 10 (prep_input / extract_train).
The original aligned objects are never modified — filtered copies are written with
a '_filtered_YYYYMMDD' suffix.

Two filters applied in order:
  1. Junction filter: drop junctions detected in fewer than --min-cells-per-junction cells.
     All counts computed from the sparse matrix directly (no precomputed column required).
  2. Cell filter: drop cells with fewer than --min-junctions-per-cell junctions *after*
     the junction filter (so cells aren't penalised for junctions we just removed).

ATSE consistency:
  After the junction filter, cell_by_cluster_matrix (ATSE totals used as PSI denominator)
  is recomputed as the sum of surviving junction reads within each ATSE. This ensures
  PSI = junction_reads / atse_total remains correct even when some junctions from an
  ATSE are filtered out. ATSEs whose every junction is removed vanish automatically.
  ATSEs that lose some but not all junctions have their denominator adjusted accordingly.

The matching GE AnnData is subset to the same cells.

Use --scan first to see how many junctions / cells are retained at each threshold.

Usage:
    # Scan thresholds (no files written):
    python pipelines/04_align/09b_filter_aligned.py \\
        --splice splice_adata_aligned_2026-05-27.h5ad \\
        --ge     ge_adata_aligned_2026-05-27.h5ad \\
        --output-dir /path/to/MODEL_INPUT/20260526 \\
        --scan

    # Apply filters and write output:
    python pipelines/04_align/09b_filter_aligned.py \\
        --splice splice_adata_aligned_2026-05-27.h5ad \\
        --ge     ge_adata_aligned_2026-05-27.h5ad \\
        --output-dir /path/to/MODEL_INPUT/20260526 \\
        --min-cells-per-junction 75 \\
        --min-junctions-per-cell 500

SLURM:
    sbatch --mem=150G -p cpu \\
        --wrap "python pipelines/04_align/09b_filter_aligned.py --splice ... --ge ... --output-dir ..."
"""

import argparse
import os
from datetime import datetime
from collections import defaultdict

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


# ── helpers ────────────────────────────────────────────────────────────────────

def load_csr(path: str, label: str) -> ad.AnnData:
    print(f"  Loading {label}: {path}")
    adata = ad.read_h5ad(path)
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    else:
        adata.X = adata.X.tocsr()
    print(f"    shape={adata.shape}  nnz={adata.X.nnz:,}")
    return adata


def percentiles(arr, pcts=(5, 10, 25, 50, 75, 90, 95)):
    vals = np.percentile(arr, pcts)
    return "  ".join(f"p{p}={v:.0f}" for p, v in zip(pcts, vals))


# ── threshold scan ─────────────────────────────────────────────────────────────

def scan_thresholds(splice: ad.AnnData) -> None:
    X = splice.X.tocsr()
    n_cells_per_junc = X.getnnz(axis=0)
    n_juncs_per_cell = X.getnnz(axis=1)
    n_junc = splice.n_vars
    n_cell = splice.n_obs

    # ATSE-level impact at each threshold
    has_event = "event_id" in splice.var.columns
    if has_event:
        event_ids = splice.var["event_id"].values
        n_atses_total = len(set(event_ids))

    print("\n=== Junction filter (min cells detecting a junction) ===")
    print(f"  Current: {n_junc:,} junctions  |  per-junction cell counts: {percentiles(n_cells_per_junc)}")
    if has_event:
        print(f"  Current: {n_atses_total:,} ATSEs")
    header = f"  {'threshold':>10}  {'junctions kept':>16}  {'%':>6}"
    if has_event:
        header += f"  {'ATSEs fully kept':>17}  {'ATSEs partial':>15}  {'ATSEs lost':>12}"
    print(header)
    for t in [5, 10, 15, 20, 25, 30, 50, 75, 100, 150, 200]:
        mask = n_cells_per_junc >= t
        kept = int(mask.sum())
        row = f"  {t:>10}  {kept:>16,}  {100*kept/n_junc:>5.1f}%"
        if has_event:
            kept_events = set(event_ids[mask])
            lost_events = set(event_ids[~mask])
            full_lost   = lost_events - kept_events          # all junctions filtered
            partial     = kept_events & lost_events           # some filtered, some kept
            full_kept   = kept_events - partial               # no junctions filtered
            row += f"  {len(full_kept):>17,}  {len(partial):>15,}  {len(full_lost):>12,}"
        print(row)

    print("\n=== Cell filter (min junctions per cell, AFTER junction filter) ===")
    print(f"  Current: {n_cell:,} cells  |  per-cell junction counts: {percentiles(n_juncs_per_cell)}")
    print("  (shown before junction filter — counts will shift after junctions are removed)")
    print(f"  {'threshold':>10}  {'cells kept':>14}  {'%':>6}")
    for t in [5, 100, 250, 500, 750, 1000, 1500, 2000]:
        kept = int((n_juncs_per_cell >= t).sum())
        print(f"  {t:>10}  {kept:>14,}  {100*kept/n_cell:>5.1f}%")


# ── ATSE-consistent cluster matrix recomputation ───────────────────────────────

def recompute_cluster_matrix(splice: ad.AnnData) -> ad.AnnData:
    """
    Recompute cell_by_cluster_matrix so that each junction's ATSE total reflects
    only the surviving junctions in that ATSE.

    Without this, PSI = junc_reads / atse_total is wrong for any ATSE that lost
    some (but not all) junctions: the denominator still includes filtered junction
    reads, making PSI artificially low.

    Algorithm: two sparse matrix multiplications instead of a Python loop.
      M  = junction-to-ATSE indicator  (J × A, float32)
      atse_sums = jc @ M               (N × A) — one ATSE total per cell
      new_at    = atse_sums @ M.T      (N × J) — broadcast total back to each junction
    This is O(nnz(jc)) versus the old O(n_ATSEs × n_cells) Python loop.
    """
    if "cell_by_junction_matrix" not in splice.layers:
        print("  WARNING: cell_by_junction_matrix layer not found — skipping cluster matrix recompute.")
        return splice
    if "cell_by_cluster_matrix" not in splice.layers:
        print("  WARNING: cell_by_cluster_matrix layer not found — skipping cluster matrix recompute.")
        return splice

    jc = sp.csr_matrix(splice.layers["cell_by_junction_matrix"])  # (N × J)
    event_ids = splice.var["event_id"].values                      # (J,)

    n_cells, n_juncs = jc.shape

    # Build junction-to-ATSE indicator matrix M  (J × A)
    unique_events, atse_idx = np.unique(event_ids, return_inverse=True)
    A = len(unique_events)
    M = sp.csr_matrix(
        (np.ones(n_juncs, dtype=np.float32), (np.arange(n_juncs), atse_idx)),
        shape=(n_juncs, A)
    )

    # atse_sums[i, a] = total reads in ATSE a for cell i  (N × A)
    atse_sums = (jc @ M).tocsr()  # (N × A), float64

    # Broadcast ATSE total back to ALL junction positions in each observed ATSE (N × J).
    # Process in cell batches: each batch looks up all junction columns intact (no ATSE split).
    # nnz(cluster) >= nnz(junction): zero-read junctions in observed ATSEs get stored entries.
    CHUNK = 5000  # cells per batch
    n_batches = (n_cells + CHUNK - 1) // CHUNK
    chunks = []
    for i, start in enumerate(range(0, n_cells, CHUNK)):
        end = min(start + CHUNK, n_cells)
        batch = atse_sums[start:end, :].tocsc()[:, atse_idx].tocsr().astype(np.int64)
        chunks.append(batch)
        print(f"  batch {i+1}/{n_batches} ({end:,}/{n_cells:,} cells)", flush=True)
    new_at = sp.vstack(chunks, format="csr")

    # Critical: every observed junction must have a non-zero ATSE denominator.
    # Check only at non-zero junction positions (avoids dense N×J boolean matrix).
    jc_rows, jc_cols = jc.nonzero()
    atse_at_junc = np.asarray(new_at[jc_rows, jc_cols]).ravel()
    orphan_nnz = int((atse_at_junc == 0).sum())
    assert orphan_nnz == 0, (
        f"CRITICAL: {orphan_nnz:,} junction entries have zero ATSE total — PSI denominator = 0."
    )
    print(f"  nnz — junction: {jc.nnz:,}  cluster: {new_at.nnz:,}")

    splice.layers["cell_by_cluster_matrix"] = new_at
    print(f"  Recomputed cell_by_cluster_matrix for {A:,} ATSEs ({n_juncs:,} junctions)")
    return splice


# ── filtering ──────────────────────────────────────────────────────────────────

def _regroup_atses(splice: ad.AnnData, output_dir: str = None) -> ad.AnnData:
    """
    Re-derive ATSE groupings on surviving junctions using ATSEmapper's
    ATSEAnalyzer. Builds a networkx splice graph (nodes = splice sites,
    edges = junctions) then finds connected components via find_atse_groups().
    Event IDs use the original format: gene_id_atse_N.

    min_splice_site_usage=0.0 skips the usage filter — junctions already passed
    the cell-count filter so no additional removal is applied here.

    The original ATSEmapper event_id is preserved in var["original_event_id"].
    ATSEAnalyzer writes singleton_junctions.tsv / filtered_low_usage_junctions.tsv
    as side effects; written to output_dir (or discarded if None).

    Requires gene_id or gene_name in splice.var for gene-based grouping.
    Junction var_names: chrN_start_end_strand; rsplit(n=3) handles scaffold names.
    """
    import os
    import tempfile
    from atsemapper.atsemapper.event_detection import ATSEAnalyzer

    split = pd.Series(splice.var_names).str.rsplit("_", n=3, expand=True)
    chroms  = split[0].values
    starts  = split[1].astype(int).values
    ends    = split[2].astype(int).values
    strands = split[3].values

    if "gene_id" in splice.var.columns:
        raw_gene_ids = splice.var["gene_id"].values
    elif "gene_name" in splice.var.columns:
        raw_gene_ids = splice.var["gene_name"].values
    else:
        raise ValueError("splice.var needs 'gene_id' or 'gene_name' for ATSEAnalyzer grouping")

    junc_layer   = splice.layers.get("cell_by_junction_matrix", splice.X)
    total_scores = np.asarray(junc_layer.sum(axis=0)).ravel().astype(int)

    junctions: dict = {}
    for i, junc_id in enumerate(splice.var_names):
        gid = raw_gene_ids[i]
        junctions[junc_id] = {
            "gene_ids":    [gid] if (pd.notna(gid) and str(gid) != "nan") else [],
            "chrom":       chroms[i],
            "start":       int(starts[i]),
            "end":         int(ends[i]),
            "strand":      strands[i],
            "total_score": max(1, int(total_scores[i])),
        }

    analyzer = ATSEAnalyzer()
    gene_graphs, _ = analyzer.build_splice_graph(junctions)

    # find_atse_groups writes TSV files to cwd; redirect to output_dir or temp
    orig_cwd = os.getcwd()
    tmp_ctx  = tempfile.TemporaryDirectory() if output_dir is None else None
    work_dir = output_dir if output_dir else tmp_ctx.name
    os.chdir(work_dir)
    try:
        atse_groups = analyzer.find_atse_groups(gene_graphs, min_splice_site_usage=0.0)
    finally:
        os.chdir(orig_cwd)
        if tmp_ctx:
            tmp_ctx.cleanup()

    junc_to_event: dict = {}
    for event_id, group in atse_groups.items():
        for junc_id in group["junction_ids"]:
            junc_to_event[junc_id] = event_id

    if "event_id" in splice.var.columns:
        splice.var["original_event_id"] = splice.var["event_id"].copy()

    new_event_ids = np.array([junc_to_event.get(j, "") for j in splice.var_names], dtype=object)
    keep_mask     = new_event_ids != ""
    n_singletons  = int((~keep_mask).sum())

    print(f"\n  ATSE re-grouping via ATSEAnalyzer:")
    print(f"    {len(atse_groups):,} ATSEs from {keep_mask.sum():,} junctions")
    print(f"    {n_singletons:,} singleton junctions dropped")
    print(f"    {splice.n_vars:,} -> {keep_mask.sum():,} junctions retained")

    # Print 5 example ATSEs so output can be spot-checked visually
    print(f"\n  Example ATSEs from re-mapping:")
    shown = 0
    for event_id, group in atse_groups.items():
        if shown >= 5:
            break
        juncs = group["junction_ids"]
        gene  = group.get("gene_id", "?")
        strand = group.get("strand", "?")
        print(f"    {event_id}  gene={gene}  strand={strand}  n_juncs={len(juncs)}")
        for jid in juncs:
            jd = junctions[jid]
            print(f"      {jid:<50s}  {jd['chrom']}:{jd['start']}-{jd['end']} ({jd['strand']})")
        shown += 1

    splice.var["event_id"] = new_event_ids
    splice = splice[:, keep_mask].copy()
    return splice


def filter_splice(splice: ad.AnnData, min_cells_per_junc: int, min_juncs_per_cell: int,
                  output_dir: str = None) -> ad.AnnData:
    has_event   = "event_id" in splice.var.columns
    has_layers  = ("cell_by_junction_matrix" in splice.layers and
                   "cell_by_cluster_matrix"  in splice.layers)

    X = splice.X.tocsr()
    n_cells_per_junc = X.getnnz(axis=0)

    # ── 1. Junction filter ─────────────────────────────────────────────────────
    junc_mask = n_cells_per_junc >= min_cells_per_junc
    print(f"\n  Junction filter (min_cells_per_junction={min_cells_per_junc}):")
    print(f"    {splice.n_vars:,} → {junc_mask.sum():,} junctions  ({(~junc_mask).sum():,} dropped)")

    if has_event:
        event_ids    = splice.var["event_id"].values
        kept_evts    = set(event_ids[junc_mask])
        lost_evts    = set(event_ids[~junc_mask])
        full_lost    = lost_evts - kept_evts
        partial      = kept_evts & lost_evts
        fully_intact = kept_evts - partial
        print(f"    ATSEs fully removed: {len(full_lost):,}")
        print(f"    ATSEs partially filtered: {len(partial):,}  (will be re-derived below)")
        print(f"    ATSEs fully intact: {len(fully_intact):,}")

    splice = splice[:, junc_mask].copy()

    # ── 2. Re-derive ATSE groupings from surviving junctions ──────────────────
    # Rather than patching the original ATSEmapper IDs, we re-run ATSE grouping
    # (union-find on shared splice sites) on the filtered junction set. This
    # correctly handles partial ATSEs, disconnected components, and singletons
    # in one step. Original IDs are preserved in var["original_event_id"].
    if has_event:
        splice = _regroup_atses(splice, output_dir=output_dir)

    # ── 3. Recompute ATSE totals for PSI correctness ───────────────────────────
    if has_layers and has_event:
        print(f"\n  Recomputing cell_by_cluster_matrix after junction filter ...")
        splice = recompute_cluster_matrix(splice)
    elif not has_layers:
        print("  NOTE: cell_by_junction_matrix / cell_by_cluster_matrix layers not present "
              "— no cluster matrix recompute needed.")

    # ── 4. Cell filter (post junction filter + ATSE re-grouping) ─────────────
    X2 = splice.X.tocsr()
    n_juncs_per_cell = X2.getnnz(axis=1)
    cell_mask = n_juncs_per_cell >= min_juncs_per_cell
    print(f"\n  Cell filter (min_junctions_per_cell={min_juncs_per_cell}, post junction filter):")
    print(f"    {splice.n_obs:,} → {cell_mask.sum():,} cells  ({(~cell_mask).sum():,} dropped)")
    splice = splice[cell_mask, :].copy()

    # ── Add n_junctions_in_atse column to var ─────────────────────────────────
    if has_event:
        junc_counts_per_atse = splice.var["event_id"].map(
            splice.var["event_id"].value_counts()
        )
        splice.var["n_junctions_in_atse"] = junc_counts_per_atse.astype(int)

    # ── Summary ────────────────────────────────────────────────────────────────
    X3 = splice.X.tocsr()
    jpc = X3.getnnz(axis=1)
    print(f"\n  Final splice: {splice.shape}  nnz={X3.nnz:,}")
    print(f"  Junctions/cell: {percentiles(jpc)}")
    if has_event:
        n_atses_final = splice.var["event_id"].nunique()
        print(f"  ATSEs retained: {n_atses_final:,}")
        jpa = splice.var["n_junctions_in_atse"]
        print(f"  Junctions/ATSE: min={jpa.min()}  median={jpa.median():.0f}  max={jpa.max()}")

    return splice


def filter_ge(ge: ad.AnnData, splice_filt: ad.AnnData) -> ad.AnnData:
    # ── cell filter: keep only cells in filtered splice ────────────────────────
    cell_mask = ge.obs_names.isin(splice_filt.obs_names)
    ge = ge[cell_mask, :].copy()
    print(f"\n  GE cells: {cell_mask.shape[0]:,} → {cell_mask.sum():,}")

    # ── gene filter: keep only genes still represented in filtered splice ──────
    # Use gene_name to match (present in splice .var and GE .var / var_names).
    # Falls back to gene_id (strip version suffix) if gene_name unavailable.
    if "gene_name" in splice_filt.var.columns:
        splice_genes = set(splice_filt.var["gene_name"].dropna().astype(str))
        key = "gene_name"
    elif "gene_id" in splice_filt.var.columns:
        splice_genes = set(splice_filt.var["gene_id"].str.split(".").str[0])
        key = "gene_id (version-stripped)"
    else:
        print("  WARNING: no gene_name/gene_id in splice .var — skipping GE gene filter.")
        return ge

    # Match against GE var_names (gene symbols) or gene_name column
    if "gene_name" in ge.var.columns:
        ge_gene_vals = ge.var["gene_name"].astype(str).values
    else:
        ge_gene_vals = ge.var_names.astype(str).values

    gene_mask = np.array([g in splice_genes for g in ge_gene_vals])
    n_before  = ge.n_vars
    ge = ge[:, gene_mask].copy()
    print(f"  GE genes: {n_before:,} → {ge.n_vars:,}  "
          f"({n_before - ge.n_vars:,} dropped — no surviving junctions in filtered splice)")
    print(f"  (matched on splice .var['{key}'] vs GE var_names)")
    return ge


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splice",      required=True)
    parser.add_argument("--ge",          required=True)
    parser.add_argument("--output-dir",  required=True)
    parser.add_argument("--min-cells-per-junction", type=int, default=25)
    parser.add_argument("--min-junctions-per-cell", type=int, default=500)
    parser.add_argument("--scan", action="store_true",
                        help="Print threshold scan (including ATSE impact) and exit without writing")
    args = parser.parse_args()

    print("=== Loading aligned AnnDatas ===")
    splice = load_csr(args.splice, "splice")
    ge     = load_csr(args.ge,     "ge")

    if args.scan:
        print("\n=== Threshold scan (no files will be written) ===")
        scan_thresholds(splice)
        return

    if not splice.obs_names.equals(ge.obs_names):
        print("WARNING: splice and GE obs_names don't match exactly — will use intersection.")

    print(f"\n=== Applying filters ===")
    splice_filt = filter_splice(splice, args.min_cells_per_junction, args.min_junctions_per_cell,
                                output_dir=args.output_dir)
    ge_filt     = filter_ge(ge, splice_filt)

    # Assert: gene overlap is exactly splice genes ∩ original GE genes — no more, no less.
    if "gene_name" in splice_filt.var.columns:
        splice_gene_set = set(splice_filt.var["gene_name"].dropna().astype(str))
        orig_ge_genes   = set(ge.var["gene_name"].astype(str)) if "gene_name" in ge.var.columns \
                          else set(ge.var_names.astype(str))
        ge_filt_genes   = set(ge_filt.var["gene_name"].astype(str)) if "gene_name" in ge_filt.var.columns \
                          else set(ge_filt.var_names.astype(str))
        expected_genes  = splice_gene_set & orig_ge_genes
        assert ge_filt_genes == expected_genes, (
            f"CRITICAL: GE gene overlap mismatch — "
            f"expected {len(expected_genes):,} genes (splice ∩ original GE), "
            f"got {len(ge_filt_genes):,}  "
            f"[extra in GE: {len(ge_filt_genes - expected_genes):,}  "
            f"missing from GE: {len(expected_genes - ge_filt_genes):,}]"
        )
        print(f"\n  ✓ Gene overlap assert: {len(ge_filt_genes):,} GE genes == splice ∩ original GE")

    # Assert cells are in identical order in both filtered objects.
    assert list(splice_filt.obs_names) == list(ge_filt.obs_names), (
        f"CRITICAL: cell order mismatch between filtered splice and GE — "
        f"{(splice_filt.obs_names != ge_filt.obs_names).sum():,} positions differ"
    )
    print(f"\n  ✓ Cell order assert: {splice_filt.n_obs:,} cells in identical order in splice and GE")

    # Spot-check: sample ATSEs with 2, 3, 5, and max junctions + one random,
    # verify all junctions in each ATSE get the same cluster total for sampled cells.
    print(f"\n=== Spot-check: random ATSE cluster matrix sanity ===")
    jc_sp = sp.csr_matrix(splice_filt.layers["cell_by_junction_matrix"])
    at_sp = sp.csr_matrix(splice_filt.layers["cell_by_cluster_matrix"])
    np.random.seed(42)

    jpa = splice_filt.var["event_id"].value_counts()
    def _pick_atse(n_juncs):
        candidates = jpa[jpa == n_juncs].index
        return np.random.choice(candidates) if len(candidates) else None

    max_junc = int(jpa.max())
    sample_atses = []
    for target in [2, 3, 5, max_junc]:
        a = _pick_atse(target)
        if a is not None:
            sample_atses.append(a)
    # add one fully random
    sample_atses.append(np.random.choice(jpa[jpa >= 2].index))
    sample_atses = list(dict.fromkeys(sample_atses))  # deduplicate, preserve order

    cell_idx = np.random.choice(splice_filt.n_obs, size=5, replace=False)
    all_ok = True
    for atse_id in sample_atses:
        junc_idx = np.where(splice_filt.var["event_id"] == atse_id)[0]
        print(f"\n  ATSE: {atse_id}  ({len(junc_idx)} junctions)")
        for ci in cell_idx:
            junc_counts = np.asarray(jc_sp[ci, junc_idx].todense()).ravel()
            atse_counts = np.asarray(at_sp[ci, junc_idx].todense()).ravel()
            total       = int(junc_counts.sum())
            match = bool(np.all(atse_counts == total)) if total > 0 else bool(np.all(atse_counts == 0))
            if not match:
                all_ok = False
            print(f"    cell {ci}: junc={junc_counts}  sum={total}  atse={atse_counts}  {'✓' if match else '✗ MISMATCH'}")
    assert all_ok, "CRITICAL: cluster matrix spot-check failed — ATSE totals inconsistent."
    print(f"\n  ✓ Spot-check passed")

    # Print .obs summaries for both filtered objects.
    print(f"\n=== Filtered splice .obs ({splice_filt.n_obs:,} cells × {len(splice_filt.obs.columns)} columns) ===")
    print(f"  Columns: {list(splice_filt.obs.columns)}")
    print(splice_filt.obs.head())

    print(f"\n=== Filtered GE .obs ({ge_filt.n_obs:,} cells × {len(ge_filt.obs.columns)} columns) ===")
    print(f"  Columns: {list(ge_filt.obs.columns)}")
    print(ge_filt.obs.head())

    today = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(args.output_dir, exist_ok=True)
    tag        = f"minCells{args.min_cells_per_junction}_minJuncs{args.min_junctions_per_cell}_{today}"
    splice_out = os.path.join(args.output_dir, f"splice_adata_aligned_filtered_{tag}.h5ad")
    ge_out     = os.path.join(args.output_dir, f"ge_adata_aligned_filtered_{tag}.h5ad")

    # BED file: one row per junction with new event_id for easy IGV / grep inspection
    bed_out = os.path.join(args.output_dir, f"junctions_filtered_{tag}.bed")
    print(f"\n=== Writing junction BED file ===")
    print(f"  {bed_out}")
    bed_split = pd.Series(splice_filt.var_names).str.rsplit("_", n=3, expand=True)
    bed_df = pd.DataFrame({
        "chrom":     bed_split[0].values,
        "start":     bed_split[1].astype(int).values,
        "end":       bed_split[2].astype(int).values,
        "junction_id": splice_filt.var_names,
        "score":     0,
        "strand":    bed_split[3].values,
        "event_id":  splice_filt.var["event_id"].values,
        "original_event_id": splice_filt.var.get("original_event_id", pd.Series([""] * splice_filt.n_vars, index=splice_filt.var_names)).values,
        "gene_id":   splice_filt.var["gene_id"].values if "gene_id" in splice_filt.var.columns else "",
        "gene_name": splice_filt.var["gene_name"].values if "gene_name" in splice_filt.var.columns else "",
    })
    bed_df.to_csv(bed_out, sep="\t", index=False, header=True)
    print(f"  {len(bed_df):,} junctions written")

    print(f"\n=== Writing filtered objects ===")
    print(f"  Splice → {splice_out}")
    splice_filt.write_h5ad(splice_out)
    print(f"  GE     → {ge_out}")
    ge_filt.write_h5ad(ge_out)

    print(f"\nDone. Feed these into Step 10-pre (00_extract_train_splice.py) via the h5mu.")
    print(f"Original aligned h5ads are unchanged.")


if __name__ == "__main__":
    main()
