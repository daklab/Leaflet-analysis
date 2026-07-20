#!/usr/bin/env python
"""
Step 7: Prepare combined cell metadata CSV.

Mouse: merges Tabula Muris Senis FACS + Allen Brain into one CSV.
Human: merges Tabula Sapiens + Allen Brain Human MTG.

Produces: $WD/metadata_combined.csv

Usage:
    python pipelines/03_gene_expression/00_prepare_metadata.py config/mouse_smartseq2.yaml
"""

import json
import sys
import os
import re
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Mouse ─────────────────────────────────────────────────────────────────────

def _format_3m_id(index: str) -> str:
    """3-month TMS cells need . → - replacement and -1-1 suffix."""
    parts = index.replace(".", "-", 1).replace("_", "-", 1).split(".")
    corrected = parts[1].replace("-", "_", 1)
    return parts[0] + "-" + corrected + "-1-1"


def _strip_platform_suffix(index: str) -> str:
    return re.sub(r"\.(mm10-plus-0-0|mus-2-1).*$", "", index).split(".")[0]


def prepare_mouse_metadata(cfg: dict) -> pd.DataFrame:
    src = cfg["sources"]
    tms = src["tabula_muris_senis"]
    ab = src["allen_brain"]

    # ── Tabula Muris Senis ─────────────────────────────────────────────────────
    print("Loading TMS metadata...")
    meta_tms = pd.read_csv(tms["metadata_csv"], low_memory=False)
    meta_tms = meta_tms[meta_tms["method"] == "facs"].copy()

    def fmt(row):
        return _format_3m_id(row["index"]) if row["age"] == "3m" else _strip_platform_suffix(row["index"])

    meta_tms["cell_id"] = meta_tms.apply(fmt, axis=1)
    meta_tms = meta_tms[["cell_id", "age", "cell_ontology_class", "mouse.id", "sex", "subtissue", "tissue"]]
    print(f"  TMS: {len(meta_tms)} FACS cells")

    # ── Allen Brain ────────────────────────────────────────────────────────────
    print("Loading Allen Brain metadata...")
    meta_ab = pd.read_csv(ab["metadata_csv"], low_memory=False)
    sample_group = pd.read_csv(ab["sample_group"])
    sra = pd.read_csv(ab["sra_table"])
    gse = pd.read_csv(ab["gse_metadata"], low_memory=False)
    sra_links = pd.read_csv(ab["sra_s3_links"], sep="\t", header=None, names=["SRR", "expected_R1_fastq"])

    sra_links["filename"] = sra_links["expected_R1_fastq"].str.extract(r"/([^/]+)$")[0]
    sra_links["expected_R1_fastq"] = sra_links["filename"].str.replace(r"\.fastq.*", ".fastq.gz", regex=True)

    gse = gse.merge(sra_links, on="expected_R1_fastq", how="left")
    gse["Run"] = gse["SRR"]
    sra = sra.merge(gse, on="Run")

    ab_df = pd.DataFrame({
        "cell_id":            sra["Run"],
        "age":                ab.get("age", "2m"),
        "cell_ontology_class": sra["cell_type_alias_label"],
        "mouse.id":           sra["external_donor_name_label"],
        "sex":                sra["donor_sex_label"],
        "subtissue":          sra["region_label"],
        "tissue":             sra["sample_name"],
    })
    print(f"  Allen Brain: {len(ab_df)} cells")

    combined = pd.concat([meta_tms, ab_df], ignore_index=True)
    n_dupes = combined["cell_id"].duplicated().sum()
    if n_dupes:
        print(f"  WARNING: {n_dupes} duplicate cell IDs in combined metadata")
    return combined


# ── Human (stub — expand when Tabula Sapiens + Allen Human is ready) ──────────

def prepare_human_metadata(cfg: dict) -> pd.DataFrame:
    raise NotImplementedError("Human metadata preparation not yet implemented")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python 00_prepare_metadata.py <config.yaml>")
        sys.exit(1)

    cfg = load_config(sys.argv[1])
    wd = cfg["wd"]
    species = cfg.get("species", "mouse")

    if species == "mouse":
        combined = prepare_mouse_metadata(cfg)
    elif species == "human":
        combined = prepare_human_metadata(cfg)
    else:
        raise ValueError(f"Unknown species: {species}")

    # Age exclusions (configured in metadata.exclude_ages)
    exclude_ages = cfg.get("metadata", {}).get("exclude_ages", [])
    if exclude_ages:
        before = len(combined)
        combined = combined[~combined["age"].isin(exclude_ages)].copy()
        print(f"\nExcluded ages {exclude_ages}: {before} → {len(combined)} cells")

    # Broad cell type annotation
    celltype_map_path = cfg.get("cell_type_map")
    if celltype_map_path:
        map_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                celltype_map_path)
        if os.path.exists(map_file):
            with open(map_file) as f:
                celltype_map = json.load(f)
            combined["broad_cell_type"] = combined["cell_ontology_class"].map(celltype_map)
            n_unmapped = combined["broad_cell_type"].isnull().sum()
            if n_unmapped:
                print(f"  WARNING: {n_unmapped} cells have unmapped cell_ontology_class")
                print(combined[combined["broad_cell_type"].isnull()]["cell_ontology_class"].value_counts().head(10))
            print(f"  broad_cell_type distribution:")
            print(combined["broad_cell_type"].value_counts().to_string())
        else:
            print(f"  WARNING: cell_type_map not found: {map_file}")

    out = os.path.join(wd, "metadata_combined.csv")
    combined.to_csv(out, index=False)
    print(f"\nSaved {len(combined)} rows to {out}")
    print(combined.head())


if __name__ == "__main__":
    main()
