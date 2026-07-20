#!/usr/bin/env python
"""
04_run_atsemapper.py — map ATSEs from merged junction pkl using ATSEmapper v2.

Usage:
    python 04_run_atsemapper.py <config_yaml> <processing_dir>

    config_yaml:    e.g. config/mouse_smartseq2.yaml
    processing_dir: e.g. /gpfs/.../junction_processing_20260511

SLURM submit:
    sbatch --mem=300G --time=2-00:00:00 -J atsemap -p cpu,bigmem \
        --wrap "python pipelines/02_atsemapper/04_run_atsemapper.py $CONFIG $PROC_DIR"
"""

import argparse
import datetime
import os
import pickle
import yaml

from atsemapper.atsemapper.genome_utils import GenomeDB, JunctionAnalyzer
from atsemapper.atsemapper.event_detection import ATSEAnalyzer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to dataset config YAML")
    parser.add_argument("proc_dir", help="Processing directory (contains results/final_junctions.pkl)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    am = cfg["atsemapper"]

    gtf_file   = cfg["gtf_file"]
    db_file    = cfg["db_file"]
    fasta_file = cfg["fasta_file"]
    dataset    = cfg["dataset_label"]

    pkl_path   = os.path.join(args.proc_dir, "results", "final_junctions.pkl")
    output_dir = os.path.join(cfg["wd"], "ATSE_mapper", "ATSE_files")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Dataset:      {dataset}")
    print(f"Junction pkl: {pkl_path}")
    print(f"Output dir:   {output_dir}")

    # ── Load junctions ────────────────────────────────────────────────────────
    print("\nLoading junction pkl...")
    with open(pkl_path, "rb") as f:
        junctions = pickle.load(f)
    print(f"  {len(junctions):,} junctions loaded (already QC-filtered at merge step)")

    # ── Genome database ───────────────────────────────────────────────────────
    print("\nInitialising genome database...")
    genome_db = GenomeDB(db_name=db_file, gtf_file=gtf_file, fasta_file=fasta_file)

    # ── Splice site validation ────────────────────────────────────────────────
    analyzer = JunctionAnalyzer(
        fasta_file=fasta_file, db=genome_db.get_db(), tolerance=1
    )
    print("\nChecking splice sites...")
    junctions = analyzer.check_splice_sites(junctions)
    junctions = analyzer.filter_canonical(junctions, only_canonical=True)
    print(f"  After canonical filter: {len(junctions):,} junctions")

    # ── Annotation ────────────────────────────────────────────────────────────
    print("\nAnnotating junctions against genome...")
    junctions = analyzer.check_junction_annotation(junctions)
    junctions = analyzer.filter_annotated(
        junctions, annotation_status_include=am["annotation_status"]
    )
    print(f"  After annotation filter ({am['annotation_status']}): {len(junctions):,} junctions")

    # ── ATSE detection ────────────────────────────────────────────────────────
    print("\nBuilding splice graphs...")
    atse_analyzer = ATSEAnalyzer()
    gene_graphs, graph_stats = atse_analyzer.build_splice_graph(junctions)

    print("\nFinding ATSE groups...")
    atse_groups = atse_analyzer.find_atse_groups(gene_graphs, min_splice_site_usage=0.01)
    print(f"  {len(atse_groups):,} ATSE groups found")

    # ── Save ──────────────────────────────────────────────────────────────────
    now = datetime.datetime.now()
    outfile = os.path.join(
        output_dir,
        f"{dataset.upper()}_ATSE_FILE_{am['annotation_status']}"
        f"_{now.strftime('%Y-%m-%d_%H-%M-%S')}.txt.gz"
    )
    atse_analyzer.save_atse_file(atse_groups, junctions, outfile)
    print(f"\nATSE file saved: {outfile}")
    print(f"Next: bash pipelines/02_atsemapper/05_make_anndata_slurm.sh $CONFIG $PROC_DIR {outfile}")


if __name__ == "__main__":
    main()
