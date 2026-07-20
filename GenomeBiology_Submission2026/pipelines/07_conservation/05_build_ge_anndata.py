#!/usr/bin/env python
"""
05_build_ge_anndata.py — combine featureCounts batch outputs into a GE AnnData.

Reads counts_batch_*.txt from 04_featurecounts_array.sh, concatenates their sample
columns, and writes an AnnData with:
    obs_names = SRR accession   (matches the splice AnnData obs → trivial alignment)
    var_names = gene_id         (gencode.v45, same annotation as the ATSE mapping)
    var['gene_name']            = gene symbol, parsed from the GTF   ← REQUIRED, see below
    var['gene_length']          = featureCounts Length column (for TPM/RPKM later)
    X / layers['counts']        = raw integer fragment counts (samples × genes)

featureCounts names each sample column by its BAM path; we map path → SRR via the
parent directory name (bams/<SRR>/Aligned.sortedByCoord.out.bam).

Two things this script MUST get right, because 09b_filter_aligned.py fails on them
(one loudly, one SILENTLY):

  1. var['gene_name'] — 09b's filter_ge() matches splice .var['gene_name'] (symbols)
     against GE .var['gene_name'] if present, else GE .var_names. Our var_names are
     VERSIONED gene_ids (ENSG…​.5), so without a gene_name column every gene matches
     nothing and GE is silently filtered to ZERO genes (its assert passes trivially,
     both sides being empty). So we parse gene_id → gene_name from the GTF.

  2. obs order — 09b asserts list(splice.obs_names) == list(ge.obs_names) AFTER filtering,
     but its filter_ge() subsets GE with .isin(), preserving GE's own order. Our GE order
     is manifest order; the splice order is chunk-concat order. Pass --splice to reorder
     GE obs to match the splice AnnData exactly.

Writes via a .tmp + atomic rename so an interrupted write never leaves a truncated h5ad.

Usage:
    python 05_build_ge_anndata.py [--counts-dir <dir>] [--out <ge.h5ad>] [--gtf <gtf>]
                                  [--splice <merged_anndata_with_metadata.h5ad>]

    --splice: report GE↔splice obs overlap AND reorder GE obs to the splice order.
              Strongly recommended — without it, D9 will fail the cell-order assert.
"""

import argparse
import glob
import os
import re
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

DEFAULT_WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/RNA_ATLAS/GSE138734"
DEFAULT_GTF = ("/gpfs/commons/datasets/controlled/BRAIN_NeMO/human-reference/gencode/"
               "gencode.v45.primary_assembly.annotation.gtf")

_GENE_ID_RE   = re.compile(r'gene_id "([^"]+)"')
_GENE_NAME_RE = re.compile(r'gene_name "([^"]+)"')


def srr_from_bam_path(col: str) -> str:
    """bams/<SRR>/Aligned.sortedByCoord.out.bam -> <SRR>"""
    return os.path.basename(os.path.dirname(col))


def gene_id_to_name(gtf_path: str) -> dict:
    """Map versioned gene_id -> gene symbol, scanning only 'gene' feature lines."""
    mapping = {}
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            # field 3 is the feature type; cheap check before regex
            parts = line.split("\t", 3)
            if len(parts) < 3 or parts[2] != "gene":
                continue
            gid = _GENE_ID_RE.search(line)
            gname = _GENE_NAME_RE.search(line)
            if gid:
                mapping[gid.group(1)] = gname.group(1) if gname else gid.group(1)
    print(f"[ge] gene_id → gene_name map: {len(mapping):,} genes from GTF")
    return mapping


def read_batch(path: str) -> pd.DataFrame:
    """featureCounts output: comment line, then Geneid Chr Start End Strand Length <bams...>"""
    df = pd.read_csv(path, sep="\t", comment="#")
    meta_cols = ["Geneid", "Chr", "Start", "End", "Strand", "Length"]
    counts = df.drop(columns=[c for c in meta_cols if c != "Geneid"]).set_index("Geneid")
    counts.columns = [srr_from_bam_path(c) for c in counts.columns]
    lengths = df.set_index("Geneid")["Length"]
    return counts, lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts-dir", default=os.path.join(DEFAULT_WD, "gene_expression", "featurecounts"))
    ap.add_argument("--out", default=os.path.join(DEFAULT_WD, "gene_expression", "ge_adata_rna_atlas.h5ad"))
    ap.add_argument("--gtf", default=DEFAULT_GTF, help="GTF for gene_id → gene_name.")
    ap.add_argument("--splice", default=None,
                    help="Splice h5ad: check obs overlap AND reorder GE obs to match it.")
    args = ap.parse_args()

    batches = sorted(glob.glob(os.path.join(args.counts_dir, "counts_batch_*.txt")))
    if not batches:
        sys.exit(f"ERROR: no counts_batch_*.txt in {args.counts_dir}")
    print(f"[ge] found {len(batches)} batch files")

    frames, lengths = [], None
    for i, b in enumerate(batches, 1):
        counts, ln = read_batch(b)
        if lengths is None:
            lengths = ln
        elif not lengths.index.equals(ln.index):
            sys.exit(f"ERROR: gene order differs in {b} — all batches must share the GTF")
        frames.append(counts)
        print(f"  [{i}/{len(batches)}] {os.path.basename(b)}: {counts.shape[1]} samples")

    mat = pd.concat(frames, axis=1)          # genes × samples
    if mat.columns.duplicated().any():
        dup = mat.columns[mat.columns.duplicated()].unique().tolist()
        sys.exit(f"ERROR: duplicate SRR columns across batches: {dup[:5]}")

    print(f"[ge] combined: {mat.shape[0]:,} genes × {mat.shape[1]:,} samples")

    # gene_name is REQUIRED by 09b's filter_ge (see module docstring) — without it the
    # GE gene filter matches symbols against versioned ENSG ids and silently keeps 0 genes.
    id2name = gene_id_to_name(args.gtf)
    gene_ids = mat.index.astype(str)
    gene_names = np.array([id2name.get(g, g) for g in gene_ids], dtype=object)
    n_unmapped = int(sum(1 for g, n in zip(gene_ids, gene_names) if n == g and g not in id2name))
    if n_unmapped:
        print(f"[ge] WARNING: {n_unmapped:,} gene_ids had no gene_name in the GTF "
              f"(fell back to gene_id)")

    X = sp.csr_matrix(mat.T.values.astype(np.float32))   # samples × genes
    A = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=mat.columns.astype(str)),
        var=pd.DataFrame({"gene_name": gene_names, "gene_length": lengths.values},
                         index=gene_ids),
    )
    A.layers["counts"] = X.copy()

    if args.splice:
        S = ad.read_h5ad(args.splice, backed="r")
        splice_obs = S.obs_names.astype(str).tolist()
        S.file.close()
        s_obs, g_obs = set(splice_obs), set(A.obs_names.astype(str))
        print(f"[ge] splice obs {len(s_obs):,} | GE obs {len(g_obs):,} | "
              f"intersect {len(s_obs & g_obs):,}")
        only_s, only_g = s_obs - g_obs, g_obs - s_obs
        if only_s:
            print(f"  splice-only (no GE): {len(only_s):,} e.g. {sorted(only_s)[:3]}")
        if only_g:
            print(f"  GE-only (no splice): {len(only_g):,} e.g. {sorted(only_g)[:3]}")

        # Reorder GE obs to the SPLICE order. 09b asserts identical obs order after
        # filtering, but its filter_ge() preserves GE's own order — so match it here.
        order = [c for c in splice_obs if c in g_obs]
        A = A[order, :].copy()
        print(f"[ge] reordered GE obs to splice order ({A.n_obs:,} cells)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    A.write_h5ad(tmp)
    os.replace(tmp, args.out)
    print(f"[ge] wrote {A.shape[0]:,} samples × {A.shape[1]:,} genes → {args.out}")


if __name__ == "__main__":
    main()
