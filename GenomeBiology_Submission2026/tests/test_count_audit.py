#!/usr/bin/env python
"""
test_count_audit.py — verify counts in merged AnnData against raw BED files,
and verify ATSE cluster totals are consistent with junction-level counts.

Two checks:

  1. Junction count audit
     For each audit record (cell_id, junction_id, raw_count) sampled during
     05_make_anndata.py, confirm that cell_by_junction_matrix[cell, junc] == raw_count.

  2. ATSE cluster consistency
     For a random sample of (cell, junction) pairs, confirm that
     cell_by_cluster_matrix[cell, junc] == sum of cell_by_junction_matrix[cell, :]
     across ALL junctions belonging to the same ATSE as junc.

Usage:
    python tests/test_count_audit.py <merged_anndata.h5ad> <anndata_dir>

    merged_anndata.h5ad : output of 06_merge_anndatas.py
    anndata_dir         : directory containing chunk_*_audit.json files

Exit code 0 = all checks pass. Non-zero = failures found.
"""

import argparse
import glob
import json
import os
import random
import sys

import anndata as ad
import numpy as np
import scipy.sparse as sp


def load_audit_records(anndata_dir: str) -> list[dict]:
    files = sorted(glob.glob(os.path.join(anndata_dir, "chunk_*_audit.json")))
    if not files:
        print(f"ERROR: no chunk_*_audit.json files found in {anndata_dir}")
        sys.exit(1)
    records = []
    for f in files:
        with open(f) as fh:
            records.extend(json.load(fh))
    print(f"Loaded {len(records):,} audit records from {len(files)} chunk files")
    return records


def check_1_junction_counts(adata: ad.AnnData, records: list[dict]) -> bool:
    """
    Verify cell_by_junction_matrix[cell, junc] == raw BED count for each audit record.
    Prints a detailed breakdown of observed vs expected counts.
    """
    print("\n── Check 1: junction counts vs raw BED files ──────────────────────")

    layer = adata.layers.get("cell_by_junction_matrix")
    if layer is None:
        print("ERROR: 'cell_by_junction_matrix' layer not found in AnnData")
        return False
    if sp.issparse(layer):
        layer = layer.tocsr()

    cell_index = {cid: i for i, cid in enumerate(adata.obs_names)}
    junc_index = {jid: i for i, jid in enumerate(adata.var_names)}

    n_pass = n_fail = n_skip = 0
    failures = []
    observed_counts = []   # raw_count values for passing records
    example_matches = []   # first 10 passing records for display

    for rec in records:
        cell_id     = rec["cell_id"]
        junction_id = rec["junction_id"]
        expected    = rec["raw_count"]

        ci = cell_index.get(cell_id)
        ji = junc_index.get(junction_id)
        if ci is None or ji is None:
            n_skip += 1
            continue

        got = int(layer[ci, ji])
        if got == expected:
            n_pass += 1
            observed_counts.append(expected)
            if len(example_matches) < 10:
                example_matches.append((cell_id, junction_id, expected))
        else:
            n_fail += 1
            failures.append({
                "cell_id":    cell_id,
                "junction_id": junction_id,
                "expected":   expected,
                "got":        got,
                "bed_file":   rec.get("bed_file", ""),
            })

    n_checked = n_pass + n_fail
    print(f"  Records checked (non-skipped): {n_checked:,}")
    print(f"  PASS:    {n_pass:,}")
    print(f"  FAIL:    {n_fail:,}")
    print(f"  SKIPPED: {n_skip:,}  (cell/junction filtered by QC — not a bug)")

    if n_checked == 0:
        print("  WARNING: all records skipped — possible cell_id format mismatch")
        print(f"  Audit cell_id example:    {records[0]['cell_id']}")
        print(f"  AnnData obs_name example: {adata.obs_names[0]}")
        return False

    # Detailed breakdown of observed counts among passing records
    if observed_counts:
        arr = np.array(observed_counts)
        print(f"\n  Observed junction read counts (passing records):")
        print(f"    n={len(arr):,}  min={arr.min()}  p25={int(np.percentile(arr,25))}"
              f"  median={int(np.median(arr))}  p75={int(np.percentile(arr,75))}"
              f"  max={arr.max()}")
        # Count distribution by read-count bucket
        buckets = [(1,1),(2,2),(3,5),(6,10),(11,50),(51,200),(201,99999999)]
        print(f"    Count distribution:")
        for lo, hi in buckets:
            n = ((arr >= lo) & (arr <= hi)).sum()
            label = f"{lo}" if lo == hi else f"{lo}–{hi}" if hi < 99999999 else f">{lo-1}"
            print(f"      reads {label:>8}: {n:>6,} records  ({100*n/len(arr):.1f}%)")

    # Show example passing records
    print(f"\n  Example matching records (first {len(example_matches)}):")
    for cell_id, junction_id, count in example_matches:
        print(f"    cell={cell_id}  junc={junction_id}  reads={count}")

    # Show failures
    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for f in failures[:10]:
            print(f"    cell={f['cell_id']}  junc={f['junction_id']}")
            print(f"      expected={f['expected']}  got={f['got']}")
            print(f"      bed={f['bed_file']}")
        if len(failures) > 10:
            print(f"  ... and {len(failures)-10} more")
    else:
        print(f"\n  ✓ All {n_pass:,} checked records exactly match raw BED counts")

    return n_fail == 0


def check_2_atse_cluster_totals(adata: ad.AnnData, n_samples: int = 300, seed: int = 42) -> bool:
    """
    For a random sample of (cell, junction) pairs, verify:
      cell_by_cluster_matrix[cell, junc] == sum of cell_by_junction_matrix[cell, j]
      for all junctions j in the same ATSE as junc.
    """
    print("\n── Check 2: ATSE cluster totals vs sum of junction counts ─────────")

    junc_layer = adata.layers.get("cell_by_junction_matrix")
    clus_layer = adata.layers.get("cell_by_cluster_matrix")
    if junc_layer is None or clus_layer is None:
        print("  SKIP: required layers not found")
        return True

    if "event_id" not in adata.var.columns:
        print("  SKIP: 'event_id' column not in adata.var")
        return True

    junc_layer = junc_layer.tocsr() if sp.issparse(junc_layer) else junc_layer
    clus_layer = clus_layer.tocsr() if sp.issparse(clus_layer) else clus_layer

    # Build ATSE → list of junction column indices
    atse_to_cols: dict[str, list[int]] = {}
    for col_idx, event_id in enumerate(adata.var["event_id"]):
        atse_to_cols.setdefault(event_id, []).append(col_idx)

    # Only bother checking ATSEs with >1 junction (cluster total will differ from junction)
    multi_atses = {a: cols for a, cols in atse_to_cols.items() if len(cols) > 1}
    print(f"  ATSEs with >1 junction: {len(multi_atses):,}")

    rng = random.Random(seed)
    sampled_atses = rng.sample(list(multi_atses), min(n_samples, len(multi_atses)))
    sampled_cells = rng.sample(range(adata.n_obs), min(50, adata.n_obs))

    n_pass = n_fail = n_skip = 0
    failures = []

    for event_id in sampled_atses:
        junc_cols = multi_atses[event_id]
        for cell_row in sampled_cells:
            # Expected cluster total = sum of all junction counts for this ATSE
            if sp.issparse(junc_layer):
                junc_counts = np.asarray(junc_layer[cell_row, junc_cols].todense()).ravel()
            else:
                junc_counts = junc_layer[cell_row, junc_cols]

            expected_cluster = int(junc_counts.sum())
            if expected_cluster == 0:
                n_skip += 1
                continue

            # Got: cluster value at first junction in this ATSE (all should be the same)
            first_col = junc_cols[0]
            if sp.issparse(clus_layer):
                got_cluster = int(clus_layer[cell_row, first_col])
            else:
                got_cluster = int(clus_layer[cell_row, first_col])

            if got_cluster == expected_cluster:
                n_pass += 1
            else:
                n_fail += 1
                failures.append({
                    "event_id":  event_id,
                    "cell_row":  cell_row,
                    "expected":  expected_cluster,
                    "got":       got_cluster,
                    "n_juncs":   len(junc_cols),
                })

    n_checked = n_pass + n_fail
    print(f"  PASS:    {n_pass:,}")
    print(f"  FAIL:    {n_fail:,}")
    print(f"  SKIPPED: {n_skip:,}  (zero-count cells — not a bug)")

    for f in failures[:10]:
        print(f"  FAIL: atse={f['event_id']}  cell_row={f['cell_row']}")
        print(f"        expected_cluster={f['expected']}  got={f['got']}  n_junctions={f['n_juncs']}")
    if len(failures) > 10:
        print(f"  ... and {len(failures)-10} more failures")

    if n_fail == 0 and n_checked > 0:
        print(f"  ✓ All {n_pass:,} checked (cell, ATSE) pairs: cluster == sum of junction counts")
    elif n_checked == 0:
        print("  WARNING: all pairs were zero-count — increase n_samples or check data")
        return False
    return n_fail == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("merged_anndata", help="Path to merged_anndata.h5ad")
    parser.add_argument("anndata_dir",    help="Directory with chunk_*_audit.json files")
    parser.add_argument("--n-atse-samples", type=int, default=300,
                        help="Number of ATSEs to sample for cluster check (default 300)")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print("COUNT AUDIT")
    print(f"  AnnData:   {args.merged_anndata}")
    print(f"  Audit dir: {args.anndata_dir}")
    print(f"{'='*65}")

    print("\nLoading merged AnnData...")
    adata = ad.read_h5ad(args.merged_anndata)
    print(f"  Shape:  {adata.n_obs:,} cells × {adata.n_vars:,} junctions")
    print(f"  Layers: {list(adata.layers.keys())}")
    print(f"  Var cols: {list(adata.var.columns)}")

    records = load_audit_records(args.anndata_dir)

    ok1 = check_1_junction_counts(adata, records)
    ok2 = check_2_atse_cluster_totals(adata, n_samples=args.n_atse_samples)

    print(f"\n{'='*65}")
    if ok1 and ok2:
        print("ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("AUDIT FAILED — do not proceed with downstream analysis")
        sys.exit(1)


if __name__ == "__main__":
    main()
