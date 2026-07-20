#!/usr/bin/env python
"""
Step 08-01: Aggregate single-cell junction counts into pseudobulk samples
           and export as a LeafCutter DS counts matrix.

Background
----------
LeafCutter DS requires a gzip-compressed TSV counts matrix:
  - Rows: one per junction, named chr:start:end:ATSE_ID_strand
  - Columns: one per sample (pseudobulk)
  - Values: integer junction read counts

Zero-fill rule
--------------
If a junction has 0 reads in ALL pseudobulk samples → drop that row entirely.
If a junction has 0 reads in sample S but reads in sample T → write 0 for
sample S. This is critical: LeafCutter computes the denominator for a cluster
in sample S as the SUM of all junction counts in that cluster for sample S.
If junction B is missing from sample S's column but junction A (same ATSE) has
reads, LeafCutter would undercount the denominator. We avoid this by converting
the active-junction submatrix to dense, which fills zeros automatically.

Pseudobulk design
-----------------
Cells are aggregated by (mouse.id × tissue_cell_type). Each unique combination
becomes one sample (column) in the LeafCutter counts matrix. Groups with fewer
than MIN_CELLS cells are excluded — too few cells produce unreliable PSI estimates
and low read depth, which LeafCutter DS will filter anyway at the cluster level.

The age of each pseudobulk sample is recorded in sample_metadata.csv for use
in the groups file when running LeafCutter DS (01_run_leafcutter_ds.py).

Memory strategy
---------------
The full cell_by_junction_matrix (132,706 cells × 79,768 junctions, 603M nnz)
is ~5-7 GB as a sparse matrix. We load it in cell-batches of CHUNK_SIZE rows,
accumulate group sums into a (n_groups × n_junctions) sparse matrix, then
convert only the active-junction rows to dense for writing.

Usage:
    python 01_pseudobulk_counts.py \\
        --splice-h5ad   <splice_adata_2026-06-17.h5ad> \\
        --junc-map      <junction_atse_map.tsv> \\
        --output-dir    <output_directory> \\
        --min-cells     10

Outputs (all in output_dir):
    leafcutter_counts.tsv.gz    LeafCutter DS input counts matrix
    sample_metadata.csv         pseudobulk sample metadata (mouse.id, tissue_cell_type, age, n_cells)
"""

import argparse
import os
import gzip
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad


CHUNK_SIZE = 5000   # cells per batch when loading the junction matrix


def load_obs(splice_h5ad):
    adata = ad.read_h5ad(splice_h5ad, backed="r")
    obs = adata.obs.copy()
    adata.file.close()
    return obs


def read_sparse_layer_chunk(h5ad_path, layer_name, row_start, row_end):
    """Read a row slice of a CSR sparse layer from h5ad without loading the full matrix."""
    with h5py.File(h5ad_path, "r") as f:
        grp = f[f"layers/{layer_name}"]
        indptr  = grp["indptr"][row_start:row_end + 1]
        col_start = int(indptr[0])
        col_end   = int(indptr[-1])
        data    = grp["data"][col_start:col_end]
        indices = grp["indices"][col_start:col_end]
        indptr  = indptr - indptr[0]
        n_cols  = int(grp.attrs["shape"][1]) if "shape" in grp.attrs \
                  else int(grp["indices"][:].max()) + 1
        return sp.csr_matrix(
            (data, indices, indptr),
            shape=(row_end - row_start, n_cols)
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splice-h5ad", required=True,
                    help="Filtered mouse splice h5ad (cell_by_junction_matrix layer)")
    ap.add_argument("--junc-map", required=True,
                    help="junction_atse_map.tsv from 00_export_junction_atse_map.py")
    ap.add_argument("--output-dir", required=True,
                    help="Directory for output files")
    ap.add_argument("--min-cells", type=int, default=10,
                    help="Minimum cells per pseudobulk group (default: 10)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_counts = os.path.join(args.output_dir, "leafcutter_counts.tsv.gz")
    out_meta   = os.path.join(args.output_dir, "sample_metadata.csv")

    # ── 1. Load obs and define pseudobulk groups ──────────────────────────────
    print("[1] Loading obs and defining pseudobulk groups...", flush=True)
    obs = load_obs(args.splice_h5ad)
    n_cells_total = len(obs)
    print(f"  {n_cells_total:,} cells total", flush=True)

    # Group key: donor × cell type
    obs["group_key"] = obs["mouse.id"].astype(str) + "||" + obs["tissue_cell_type"].astype(str)
    group_sizes = obs["group_key"].value_counts()
    valid_groups = group_sizes[group_sizes >= args.min_cells].index
    print(f"  Groups with >= {args.min_cells} cells: {len(valid_groups):,} / {len(group_sizes):,}", flush=True)

    obs_valid = obs[obs["group_key"].isin(valid_groups)].copy()
    print(f"  Cells in valid groups: {len(obs_valid):,}", flush=True)

    # Assign integer group index to each cell (for sparse accumulation)
    group_list = sorted(valid_groups)
    group_to_idx = {g: i for i, g in enumerate(group_list)}
    n_groups = len(group_list)

    # Build metadata for valid groups
    meta = (
        obs_valid
        .groupby("group_key", observed=True)
        .agg(
            mouse_id        = ("mouse.id", "first"),
            tissue_cell_type= ("tissue_cell_type", "first"),
            tissue          = ("tissue", "first"),
            broad_cell_type = ("broad_cell_type", "first"),
            age             = ("age", "first"),
            sex             = ("sex", "first"),
            dataset         = ("dataset", "first"),
            n_cells         = ("group_key", "count"),
        )
        .loc[group_list]
        .reset_index()
    )
    # Sample name = group_key with || replaced by __ for LeafCutter compatibility
    meta["sample_name"] = meta["group_key"].str.replace("||", "__", regex=False)
    meta.to_csv(out_meta, index=False)
    print(f"  Sample metadata saved → {out_meta}", flush=True)

    sample_names = meta["sample_name"].tolist()

    # ── 2. Load junction map (row names for output) ───────────────────────────
    print("\n[2] Loading junction → LeafCutter ID map...", flush=True)
    junc_map = pd.read_csv(args.junc_map, sep="\t")
    J = len(junc_map)
    print(f"  {J:,} junctions", flush=True)

    # ── 3. Accumulate pseudobulk counts by chunked cell loading ───────────────
    # Result: pb_counts (n_groups × n_junctions) sparse int64 matrix
    print(f"\n[3] Accumulating pseudobulk counts (chunk size = {CHUNK_SIZE} cells)...", flush=True)

    # Map each cell's original row index in the full h5ad to its group index (-1 = excluded)
    cell_to_group = np.full(n_cells_total, -1, dtype=np.int32)
    for orig_idx, grp in zip(obs_valid.index, obs_valid["group_key"]):
        cell_to_group[obs.index.get_loc(orig_idx)] = group_to_idx[grp]

    # Accumulate into a list of COO triplets to build final sparse matrix
    pb_counts = sp.csr_matrix((n_groups, J), dtype=np.int64)

    n_chunks = (n_cells_total + CHUNK_SIZE - 1) // CHUNK_SIZE
    for chunk_idx in range(n_chunks):
        row_start = chunk_idx * CHUNK_SIZE
        row_end   = min(row_start + CHUNK_SIZE, n_cells_total)

        grp_idx = cell_to_group[row_start:row_end]
        valid_mask = grp_idx >= 0
        if not valid_mask.any():
            continue

        jc_chunk = read_sparse_layer_chunk(args.splice_h5ad, "cell_by_junction_matrix",
                                           row_start, row_end)
        jc_chunk = jc_chunk[valid_mask].astype(np.int64)
        grp_idx_valid = grp_idx[valid_mask]

        # Sum rows by group: indicator matrix (n_groups × n_valid_cells) @ jc_chunk
        n_valid = jc_chunk.shape[0]
        indicator = sp.csr_matrix(
            (np.ones(n_valid, dtype=np.float32),
             (grp_idx_valid, np.arange(n_valid))),
            shape=(n_groups, n_valid)
        )
        pb_counts = pb_counts + (indicator @ jc_chunk).astype(np.int64)

        if (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
            print(f"  chunk {chunk_idx + 1}/{n_chunks} (cells {row_end:,}/{n_cells_total:,})",
                  flush=True)

    print(f"  pb_counts: {pb_counts.shape}  nnz={pb_counts.nnz:,}", flush=True)

    # ── 4. Sanity checks ──────────────────────────────────────────────────────
    print("\n[4] Sanity checks...", flush=True)

    # 4a. Cell-level vs pseudobulk sum verification
    # Pick 3 junctions with high total reads and verify that summing individual
    # cell counts for one pseudobulk group matches pb_counts exactly.
    print("\n  [4a] Cell-level vs pseudobulk sum for 3 example junctions:", flush=True)
    col_sums_check = np.asarray(pb_counts.sum(axis=0)).ravel()
    # Pick junctions with highest total reads (most likely to have readable examples)
    top_junc_idxs = np.argsort(col_sums_check)[-3:][::-1]

    for junc_idx in top_junc_idxs:
        junc_id    = junc_map["junction_id"].iloc[junc_idx]
        atse_id    = junc_map["event_id"].iloc[junc_idx]
        gene_name  = junc_map["gene_name"].iloc[junc_idx]

        # Find the pseudobulk group with the most reads for this junction
        junc_col = np.asarray(pb_counts[:, junc_idx].todense()).ravel()
        best_grp  = int(np.argmax(junc_col))
        pb_count  = int(junc_col[best_grp])
        grp_name  = group_list[best_grp]
        n_cells_grp = int(meta.loc[meta["group_key"] == grp_name, "n_cells"].iloc[0])

        # Find the row indices of cells in this group
        cell_row_idxs = np.where(cell_to_group == best_grp)[0]

        # Reload junction counts for those cells directly from h5ad
        # (load one chunk at a time in case cells span multiple chunks)
        cell_junc_counts = []
        with h5py.File(args.splice_h5ad, "r") as f:
            grp_h5 = f["layers/cell_by_junction_matrix"]
            for ci in cell_row_idxs:
                start = int(grp_h5["indptr"][ci])
                end   = int(grp_h5["indptr"][ci + 1])
                indices = grp_h5["indices"][start:end]
                data    = grp_h5["data"][start:end]
                match = np.where(indices == junc_idx)[0]
                cell_junc_counts.append(int(data[match[0]]) if len(match) else 0)

        cell_sum = sum(cell_junc_counts)
        nonzero_cells = [c for c in cell_junc_counts if c > 0]
        match_str = "✓ MATCH" if cell_sum == pb_count else "✗ MISMATCH"

        print(f"\n    Junction: {junc_id}  gene={gene_name}  ATSE={atse_id}", flush=True)
        print(f"    Pseudobulk group: {grp_name}  ({n_cells_grp} cells)", flush=True)
        print(f"    pb_counts value:      {pb_count}", flush=True)
        print(f"    Sum of cell counts:   {cell_sum}  {match_str}", flush=True)
        print(f"    Cells with reads:     {len(nonzero_cells)} / {len(cell_junc_counts)}", flush=True)
        if nonzero_cells:
            print(f"    Non-zero cell counts: {sorted(nonzero_cells, reverse=True)[:10]}", flush=True)

    # 4b. Within-ATSE zero-fill check
    # For one ATSE, show all junction counts across all pseudobulks to confirm
    # that when junction A has reads in a sample, junction B (same ATSE) is
    # present in the output with explicit 0 (not missing).
    print("\n  [4b] Within-ATSE zero-fill check (2-junction ATSE example):", flush=True)
    atse_sizes_series = junc_map.groupby("event_id", observed=True).size()
    two_junc_atses = atse_sizes_series[atse_sizes_series == 2].index.tolist()
    # Pick one where both junctions have reads
    example_atse = None
    for atse in two_junc_atses[:50]:
        jidxs = junc_map.index[junc_map["event_id"] == atse].tolist()
        counts_a = np.asarray(pb_counts[:, jidxs[0]].todense()).ravel()
        counts_b = np.asarray(pb_counts[:, jidxs[1]].todense()).ravel()
        # Find a sample where A has reads but B has 0 — the classic zero-fill case
        mixed = (counts_a > 0) & (counts_b == 0)
        if mixed.any():
            example_atse = atse
            ex_jidxs = jidxs
            ex_counts_a = counts_a
            ex_counts_b = counts_b
            break
    if example_atse:
        sample_with_only_a = int(np.where((ex_counts_a > 0) & (ex_counts_b == 0))[0][0])
        sample_with_both   = int(np.where((ex_counts_a > 0) & (ex_counts_b > 0))[0][0]) \
                             if ((ex_counts_a > 0) & (ex_counts_b > 0)).any() else None
        print(f"    ATSE: {example_atse}", flush=True)
        print(f"    Junction A: {junc_map['junction_id'].iloc[ex_jidxs[0]]}", flush=True)
        print(f"    Junction B: {junc_map['junction_id'].iloc[ex_jidxs[1]]}", flush=True)
        print(f"    Sample where A has reads, B=0: {group_list[sample_with_only_a]}", flush=True)
        print(f"      A count={int(ex_counts_a[sample_with_only_a])}  "
              f"B count={int(ex_counts_b[sample_with_only_a])}  "
              f"(B will be written as 0 — ✓ denominator preserved)", flush=True)
        if sample_with_both is not None:
            print(f"    Sample where both have reads: {group_list[sample_with_both]}", flush=True)
            print(f"      A count={int(ex_counts_a[sample_with_both])}  "
                  f"B count={int(ex_counts_b[sample_with_both])}", flush=True)
    else:
        print("    No 2-junction ATSE found with mixed zero/nonzero — all balanced.", flush=True)

    # 4c. Zero-count junction examples
    print("\n  [4c] Zero-count junction examples (reads=0 in all pseudobulks):", flush=True)
    zero_junc_idxs = np.where(col_sums_check == 0)[0]
    print(f"    {len(zero_junc_idxs):,} junctions have 0 reads across all samples", flush=True)
    print(f"    These are kept in the output — LeafCutter DS will skip them at cluster level",
          flush=True)
    for zi in zero_junc_idxs[:3]:
        print(f"    Example: {junc_map['junction_id'].iloc[zi]}  "
              f"ATSE={junc_map['event_id'].iloc[zi]}  gene={junc_map['gene_name'].iloc[zi]}",
              flush=True)

    # 4d. Pseudobulk group summary
    print("\n  [4d] Pseudobulk group summary:", flush=True)
    total_reads = int(pb_counts.sum())
    print(f"    Total pseudobulk groups: {n_groups}", flush=True)
    print(f"    Total junction reads across all groups: {total_reads:,}", flush=True)
    group_read_totals = np.asarray(pb_counts.sum(axis=1)).ravel()
    print(f"    Reads per group — min={group_read_totals.min():,}  "
          f"median={int(np.median(group_read_totals)):,}  "
          f"max={group_read_totals.max():,}", flush=True)
    # Top 5 groups by total reads
    top5 = np.argsort(group_read_totals)[-5:][::-1]
    print(f"    Top 5 groups by total reads:", flush=True)
    for gi in top5:
        print(f"      {group_list[gi]}  reads={group_read_totals[gi]:,}  "
              f"cells={int(meta.loc[meta['group_key'] == group_list[gi], 'n_cells'].iloc[0])}",
              flush=True)

    # ── 5. Report junction coverage (no dropping — LeafCutter filters internally) ──
    print("\n[4] Junction coverage summary (writing all junctions)...", flush=True)
    # We write ALL junctions as rows, including those with zero reads across all
    # samples. Zeros within a row are written explicitly (densification handles this).
    # LeafCutter DS applies its own filters (minclureads, mincluratio) internally,
    # so we preserve the full junction set here and let LeafCutter decide what to test.
    col_sums = np.asarray(pb_counts.sum(axis=0)).ravel()
    n_active = int((col_sums > 0).sum())
    print(f"  Junctions with reads in ≥1 sample: {n_active:,} / {J:,}", flush=True)
    print(f"  Junctions with 0 reads in all samples (kept, LeafCutter will skip): "
          f"{J - n_active:,}", flush=True)

    # ── 5. Write gzip TSV ─────────────────────────────────────────────────────
    print(f"\n[5] Writing LeafCutter counts matrix → {out_counts}", flush=True)
    print(f"  Writing all {J:,} junctions with explicit zeros where needed", flush=True)

    with gzip.open(out_counts, "wt") as fout:
        # Header: "chrom" followed by sample names
        fout.write("chrom\t" + "\t".join(sample_names) + "\n")

        # Write all junctions in chunks to avoid densifying the full matrix at once.
        # scipy .todense() fills sparse zeros as 0, so within-ATSE zero-fill is
        # automatic — no special handling needed.
        JCHUNK = 2000
        n_jchunks = (J + JCHUNK - 1) // JCHUNK
        rows_written = 0
        for jc in range(n_jchunks):
            j_start = jc * JCHUNK
            j_end   = min(j_start + JCHUNK, J)
            # Dense block: (n_groups × chunk_junctions), transposed → (chunk_junctions × n_groups)
            block = np.asarray(pb_counts[:, j_start:j_end].todense(), dtype=np.int64).T
            for j_local in range(j_end - j_start):
                row_name = junc_map["leafcutter_junc_id"].iloc[j_start + j_local]
                counts_str = "\t".join(map(str, block[j_local]))
                fout.write(f"{row_name}\t{counts_str}\n")
                rows_written += 1
            if (jc + 1) % 20 == 0 or jc == n_jchunks - 1:
                print(f"  wrote {rows_written:,}/{J:,} junction rows", flush=True)

    print(f"\n=== Done ===", flush=True)
    print(f"  Counts matrix: {out_counts}", flush=True)
    print(f"  Samples: {n_groups}  Junctions: {J:,}", flush=True)
    print(f"  Sample metadata: {out_meta}", flush=True)
    print(f"\nNext step: 02_run_leafcutter_ds.py — build groups files and run LeafCutter DS",
          flush=True)


if __name__ == "__main__":
    main()
