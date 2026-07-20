#!/usr/bin/env python3
"""
Parse SraRunTable_RNAAtlas.csv (downloaded from NCBI SRA Run Selector for GSE138734/SRP225193),
filter to human tissue samples with PAIRED-end reads, write a manifest CSV, and print
fasterq-dump commands to stdout.

Usage:
    python 00_download_gse138734.py SraRunTable_RNAAtlas.csv \\
        --outdir /path/to/fastqs \\
        [--sample-types tissue] \\
        [--threads 8]

    Redirect stdout to save download commands:
        python 00_download_gse138734.py SraRunTable_RNAAtlas.csv \\
            --outdir $FASTQ_DIR > $FASTQ_DIR/download_commands.sh

Each sample's FASTQs land at:
    <outdir>/<SRR>/<SRR>_R1.fastq.gz
    <outdir>/<SRR>/<SRR>_R2.fastq.gz

This matches the Snakefile wildcard: {raw_dir}/{sample}/{sample}_R{1,2}.fastq.gz

After downloading, the *_junctions_with_barcodes.bed files produced by the Snakemake
pipeline are fed directly into the ATSEmapper Step 1 pipeline
(pipelines/01_junctions/01_split.sh) via a future config/gse138734_atse.yaml.

Note on total RNA vs polyA: SraRunTable does not have a per-run library-type
column — both library types share the same GSM accession. PAIRED-end runs are
enriched for total RNA (Ribo-Zero Gold) libraries based on the study design, and
for canonical junction detection (ATSEs) both library types perform equivalently.
"""

import argparse
import csv
import os
import sys

import pandas as pd


VALID_SAMPLE_TYPES = {"tissue", "cell type", "cell line", "cancer cell line"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sra_table", help="SraRunTable CSV from NCBI SRA Run Selector")
    ap.add_argument("--outdir", required=True, help="Root directory for downloaded FASTQs")
    ap.add_argument(
        "--sample-types",
        nargs="+",
        default=["tissue"],
        choices=list(VALID_SAMPLE_TYPES),
        metavar="TYPE",
        help="sample_type values to include (default: tissue). "
             "Options: tissue, 'cell type', 'cell line', 'cancer cell line'",
    )
    ap.add_argument("--manifest", help="Where to write manifest CSV (default: <outdir>/manifest.csv)")
    ap.add_argument("--threads", type=int, default=8, help="Threads for fasterq-dump and pigz (default: 8)")
    return ap.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.sra_table)

    # Report what's in the file
    print("# SraRunTable summary:", file=sys.stderr)
    print(f"#   Total rows: {len(df)}", file=sys.stderr)
    print(f"#   LibraryLayout: {df['LibraryLayout'].value_counts().to_dict()}", file=sys.stderr)
    print(f"#   sample_type: {df['sample_type'].value_counts(dropna=False).to_dict()}", file=sys.stderr)

    # Filter: PAIRED-end, selected sample types, human only
    mask = (
        (df["LibraryLayout"] == "PAIRED") &
        (df["sample_type"].isin(args.sample_types)) &
        (df["Organism"] == "Homo sapiens")
    )
    keep = df[mask].copy()

    print(f"\n# After filter (PAIRED + {args.sample_types} + Homo sapiens):", file=sys.stderr)
    print(f"#   {len(keep)} runs across {keep['source_name'].nunique()} unique source names", file=sys.stderr)
    if "organ_system" in keep.columns:
        print(f"#   organ_system counts:", file=sys.stderr)
        for sys_name, cnt in keep["organ_system"].value_counts().items():
            print(f"#     {sys_name}: {cnt}", file=sys.stderr)

    if len(keep) == 0:
        sys.exit("ERROR: 0 samples after filtering. Check --sample-types and LibraryLayout.")

    os.makedirs(args.outdir, exist_ok=True)
    manifest_path = args.manifest or os.path.join(args.outdir, "manifest.csv")

    manifest_cols = {
        "srr": "Run",
        "source_name": "source_name",
        "sample_type": "sample_type",
        "organ_system": "organ_system",
        "gsm": "GEO_Accession (exp)",
    }

    rows = []
    for _, row in keep.iterrows():
        r = {k: row.get(v, "") for k, v in manifest_cols.items()}
        r["fastq_dir"] = os.path.join(args.outdir, r["srr"])
        rows.append(r)

    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_cols.keys()) + ["fastq_dir"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n# Manifest ({len(rows)} runs) written → {manifest_path}", file=sys.stderr)
    print("# Review it before running the download commands below.\n", file=sys.stderr)

    t = args.threads
    for r in rows:
        srr = r["srr"]
        d = r["fastq_dir"]
        print(f"mkdir -p {d} && \\")
        print(f"  fasterq-dump {srr} --outdir {d} --split-files --threads {t} && \\")
        print(f"  mv {d}/{srr}_1.fastq {d}/{srr}_R1.fastq && \\")
        print(f"  mv {d}/{srr}_2.fastq {d}/{srr}_R2.fastq && \\")
        print(f"  pigz -p {t} {d}/{srr}_R1.fastq {d}/{srr}_R2.fastq")
        print()


if __name__ == "__main__":
    main()
