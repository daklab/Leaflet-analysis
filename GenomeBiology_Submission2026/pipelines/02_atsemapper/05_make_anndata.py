#!/usr/bin/env python
"""
05_make_anndata.py — build per-chunk AnnData from raw junction BED files.

For each chunk of cells:
1. Loads the ATSE file to get junction universe and junction→ATSE mapping
2. Reads each cell's BED file to get per-cell junction counts
3. Builds two sparse matrices:
     cell_by_junction_matrix: raw junction read counts per cell
     cell_by_cluster_matrix:  ATSE-level total counts per cell (denominator for PSI),
                              expanded to junction space (same shape as junction matrix)
4. Saves as AnnData h5ad with junctions as var features + both layers + audit trail JSON

AnnData format (matches LeafletFA input requirements):
  var_names  = junction IDs  (e.g. chr1_4784105_4785572_-)
  var columns = junction_id, event_id (ATSE), gene_name, gene_id, num_junctions, ...
  X          = cell_by_junction_matrix (sparse)
  layers["cell_by_junction_matrix"] = raw junction read counts
  layers["cell_by_cluster_matrix"]  = ATSE total counts expanded to junction space

Usage:
    python 05_make_anndata.py \
        --config  config/mouse_smartseq2.yaml \
        --chunk-file  /path/to/chunks/chunk_3.txt \
        --atse-file   /path/to/ATSE_FILE.txt.gz \
        --junctions-pkl /path/to/results/final_junctions.pkl \
        --output-dir  /path/to/anndatas \
        --chunk-id    3
"""

import argparse
import json
import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from tqdm import tqdm

from atsemapper.atsemapper.cell_query import CellJunctionQuery


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def extract_cell_id(bed_file: str) -> str:
    """
    Extract cell ID from BED file path.
    - TMS files are named junctions_with_barcodes.bed → cell ID is the parent directory name
    - Allen Brain files are named SRR*_junctions_with_barcodes.bed → cell ID is the file stem
    """
    p = Path(bed_file)
    if p.name == "junctions_with_barcodes.bed":
        return p.parent.name
    return p.stem.replace("_junctions_with_barcodes", "")


def read_cell_counts(bed_file: str, relevant_junctions: set) -> Tuple[Optional[str], Dict[str, int]]:
    """Read one BED file, return (cell_id, {junction_id: count}) filtered to relevant_junctions."""
    cell_id = extract_cell_id(bed_file)
    df = CellJunctionQuery._load_junction_file(bed_file)
    if df is None:
        return None, {}
    df = df[df["junction_id"].isin(relevant_junctions)]
    if df.empty:
        return cell_id, {}
    counts = df.groupby("junction_id")["score"].sum().astype(int).to_dict()
    return cell_id, counts


def build_matrices(
    junc_files: List[str],
    junction_ids: List[str],
    atse_ids: List[str],
    junc_to_atse: Dict[str, str],
    max_workers: int = 4,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, List[str]]:
    """
    Build cell × junction and cell × ATSE sparse count matrices.
    ATSE counts = sum of all junction counts in the same ATSE group.
    Returns (cell_by_junc, cell_by_atse, cell_ids).
    """
    relevant_junctions = set(junction_ids)
    junc_idx = {j: i for i, j in enumerate(junction_ids)}
    atse_idx = {a: i for i, a in enumerate(atse_ids)}

    cell_ids = []
    junc_rows, junc_cols, junc_vals = [], [], []
    atse_rows, atse_cols, atse_vals = [], [], []

    def process_file(bed_file):
        return read_cell_counts(bed_file, relevant_junctions)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_file, f): f for f in junc_files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Building matrices"):
            cell_id, counts = fut.result()
            if cell_id is None:
                continue
            row = len(cell_ids)
            cell_ids.append(cell_id)
            for junc_id, count in counts.items():
                junc_rows.append(row)
                junc_cols.append(junc_idx[junc_id])
                junc_vals.append(count)
                atse_id = junc_to_atse.get(junc_id)
                if atse_id in atse_idx:
                    atse_rows.append(row)
                    atse_cols.append(atse_idx[atse_id])
                    atse_vals.append(count)

    n_cells = len(cell_ids)
    # scipy.sparse sums duplicate (row, col) entries automatically — correct for ATSE aggregation
    cell_by_junc = sp.csr_matrix(
        (junc_vals, (junc_rows, junc_cols)), shape=(n_cells, len(junction_ids))
    )
    cell_by_atse = sp.csr_matrix(
        (atse_vals, (atse_rows, atse_cols)), shape=(n_cells, len(atse_ids))
    )
    return cell_by_junc, cell_by_atse, cell_ids


def sample_audit_records(
    junc_files: List[str],
    relevant_junctions: set,
    n_samples: int,
    seed: int = 42,
) -> List[dict]:
    """
    Sample (cell_id, junction_id, raw_count) tuples directly from BED files.
    These are used by tests/test_count_audit.py to verify matrix correctness.
    """
    rng = random.Random(seed)
    candidates = rng.sample(junc_files, min(50, len(junc_files)))
    records = []
    for bed_file in candidates:
        cell_id, counts = read_cell_counts(bed_file, relevant_junctions)
        if not counts:
            continue
        sampled = rng.sample(list(counts), min(5, len(counts)))
        for junc_id in sampled:
            records.append({
                "cell_id":     cell_id,
                "junction_id": junc_id,
                "raw_count":   counts[junc_id],
                "bed_file":    bed_file,
            })
    rng.shuffle(records)
    return records[:n_samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        required=True)
    parser.add_argument("--chunk-file",    required=True)
    parser.add_argument("--atse-file",     required=True)
    parser.add_argument("--junctions-pkl", required=True)
    parser.add_argument("--output-dir",    required=True)
    parser.add_argument("--chunk-id",      required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ann_cfg = cfg["anndata"]
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Read chunk file list ──────────────────────────────────────────────────
    with open(args.chunk_file) as f:
        junc_files = [l.strip() for l in f if l.strip()]
    print(f"Chunk {args.chunk_id}: {len(junc_files)} junction files")

    # ── Load ATSE file ────────────────────────────────────────────────────────
    print("Loading ATSE file...")
    atses = pd.read_csv(args.atse_file, sep="\t", compression="gzip")
    atses = atses.drop_duplicates(subset="junction_id", keep="first")
    junc_to_atse = dict(zip(atses["junction_id"], atses["event_id"]))
    print(f"  {atses['junction_id'].nunique():,} junctions, {atses['event_id'].nunique():,} ATSEs")

    # ── Intersect with dataset pkl ────────────────────────────────────────────
    with open(args.junctions_pkl, "rb") as f:
        junction_dict = pickle.load(f)
    dataset_junctions = set(junction_dict.keys())

    junction_ids = [j for j in atses["junction_id"].unique() if j in dataset_junctions]
    relevant_junctions = set(junction_ids)
    print(f"  {len(junction_ids):,} junctions after intersection with dataset pkl")

    # ATSE IDs — only those with at least one surviving junction
    surviving_atses = {junc_to_atse[j] for j in junction_ids if j in junc_to_atse}
    atse_ids = [a for a in atses["event_id"].unique() if a in surviving_atses]
    print(f"  {len(atse_ids):,} ATSEs with at least one junction")

    # ── Audit trail ───────────────────────────────────────────────────────────
    print(f"Sampling {ann_cfg['audit_sample_n']} audit records...")
    audit_records = sample_audit_records(junc_files, relevant_junctions, n_samples=ann_cfg["audit_sample_n"])
    audit_file = os.path.join(args.output_dir, f"chunk_{args.chunk_id}_audit.json")
    with open(audit_file, "w") as f:
        json.dump(audit_records, f, indent=2)
    print(f"  Saved {len(audit_records)} audit records → {audit_file}")

    # ── Build matrices ────────────────────────────────────────────────────────
    print("Building cell × junction and cell × ATSE matrices...")
    cell_by_junc, cell_by_atse, cell_ids = build_matrices(
        junc_files, junction_ids, atse_ids, junc_to_atse,
        max_workers=ann_cfg["max_workers"],
    )
    print(f"  {len(cell_ids)} cells × {len(atse_ids)} ATSEs")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    zero_cells = (np.asarray(cell_by_atse.sum(axis=1)).ravel() == 0).sum()
    pct_zero = 100 * zero_cells / max(len(cell_ids), 1)
    print(f"  Cells with zero ATSE counts: {zero_cells}/{len(cell_ids)} ({pct_zero:.1f}%)")
    if pct_zero > 50:
        print("  WARNING: >50% of cells have zero ATSE counts — check cell ID alignment")
    n_atses_covered = (np.asarray(cell_by_atse.sum(axis=0)).ravel() > 0).sum()
    print(f"  ATSEs with ≥1 count: {n_atses_covered}/{len(atse_ids)}")

    # ── Build junction-level var metadata ─────────────────────────────────────
    # ATSE file has one row per junction — filter to surviving junctions
    var_cols = ["junction_id", "event_id", "gene_name", "gene_id", "num_junctions",
                "chromosome", "strand", "position_off_5_prime", "position_off_3_prime"]
    available_cols = [c for c in var_cols if c in atses.columns]
    junc_meta = (
        atses[atses["junction_id"].isin(relevant_junctions)][available_cols]
        .drop_duplicates(subset="junction_id")
        .set_index("junction_id")
        .loc[junction_ids]
    )

    # ── Build cell_by_cluster_matrix (ATSE totals expanded to junction space) ─
    # For each junction j belonging to ATSE a, cluster count = sum of all reads
    # for that ATSE. Shape matches cell_by_junc: (n_cells, n_junctions).
    atse_idx = {a: i for i, a in enumerate(atse_ids)}
    atse_col_for_junc = [atse_idx[junc_to_atse[j]] for j in junction_ids]
    cell_by_cluster = cell_by_atse[:, atse_col_for_junc]

    # ── Build AnnData ─────────────────────────────────────────────────────────
    adata = anndata.AnnData(
        X=cell_by_junc,
        obs=pd.DataFrame(index=cell_ids),
        var=junc_meta,
    )
    adata.layers["cell_by_junction_matrix"] = cell_by_junc
    adata.layers["cell_by_cluster_matrix"]  = cell_by_cluster

    output_file = os.path.join(args.output_dir, f"chunk_{args.chunk_id}_anndata.h5ad")
    adata.write_h5ad(output_file)
    print(f"Saved AnnData ({adata.shape[0]} cells × {adata.shape[1]} junctions) → {output_file}")
    print(f"Chunk {args.chunk_id} complete.")


if __name__ == "__main__":
    main()
