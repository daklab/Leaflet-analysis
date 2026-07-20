#!/usr/bin/env python
"""
Prepare combined cell metadata CSV for the human dataset.

Merges Tabula Sapiens V2 + Allen Brain Human MTG into one CSV.
Tabula Sapiens cell IDs are synthesized from the h5ad obs metadata to match
the splice AnnData (junction BED file stem format).

Produces: $WD/metadata_combined.csv

Usage:
    python pipelines/03_gene_expression/00_prepare_human_metadata.py config/human_smartseq2.yaml
"""

import os
import re
import sys

import pandas as pd
import scanpy as sc
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Tabula Sapiens ────────────────────────────────────────────────────────────

def _build_ts_cell_id(row) -> str:
    """
    Reconstruct the splice-format cell ID from Tabula Sapiens obs metadata.

    The BED file for a TS cell is named:
      {project_id}_{junc_prefix_core}.multi_star.output_raw.output_raw_per_{old_index}_junctions_with_barcodes.bed

    So extract_cell_id() in 05_make_anndata.py returns the stem:
      {project_id}_{junc_prefix_core}.multi_star.output_raw.output_raw_per_{old_index}

    old_index format (9 underscore-separated fields):
      {sample_num}_{platform}_{donor1}_{plate1}_{donor2}_{plate2}_{tissue}_{subtissue}_{celltype}
    e.g. "1_smartseq2_L144736_G2_B133819_M4_Liver_NA_Hepatocyte"
    """
    project_id = re.match(r"^(TSP\d+)", str(row["sample_id"]))
    if not project_id:
        return str(row["old_index"])
    project_id = project_id.group(1)

    parts = str(row["old_index"]).split("_")
    if len(parts) < 8:
        return str(row["old_index"])

    junc_prefix_core = "_".join([
        parts[0],   # sample number (e.g. "1")
        parts[1],   # platform     (e.g. "smartseq2")
        parts[2],   # donor1       (e.g. "L144736")
        parts[4],   # donor2       (e.g. "B133819")
        parts[6],   # tissue       (e.g. "Liver")
        parts[7],   # subtissue    (e.g. "NA")
    ])

    return (
        project_id + "_" + junc_prefix_core
        + ".multi_star.output_raw.output_raw_per_"
        + str(row["old_index"])
    )


def prepare_tabula_sapiens(src: dict) -> pd.DataFrame:
    print("Loading Tabula Sapiens h5ad...")
    adata = sc.read_h5ad(src["ge_h5ad"])

    required = ["old_index", "sample_id", "donor", "tissue", "age", "sex",
                "cell_ontology_class", "free_annotation"]
    missing = [c for c in required if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"Tabula Sapiens h5ad missing columns: {missing}")

    obs = adata.obs[required].copy()
    obs = obs.reset_index(drop=True)

    # Generate splice-format cell IDs
    obs["cell_id"] = obs.apply(_build_ts_cell_id, axis=1)
    print(f"  Cell ID example: {obs['cell_id'].iloc[0]}")

    # Standardise sex coding
    obs["sex"] = obs["sex"].astype(str).str.strip().str.lower()
    obs["sex"] = obs["sex"].map({"male": "M", "female": "F"}).fillna(obs["sex"])

    # Apply bam_keep_list filter if provided
    keep_list = src.get("bam_keep_list")
    if keep_list and os.path.exists(keep_list):
        with open(keep_list) as f:
            keep_ids = set(l.strip() for l in f if l.strip())
        before = len(obs)
        obs = obs[obs["cell_id"].isin(keep_ids)].copy()
        print(f"  bam_keep_list filtered: {before} → {len(obs)} cells")
    elif keep_list:
        print(f"  WARNING: bam_keep_list not found: {keep_list}")

    result = pd.DataFrame({
        "cell_id":            obs["cell_id"],
        "donor":              obs["donor"].astype(str),
        "sex":                obs["sex"],
        "age":                obs["age"].astype(str),
        "dataset":            "tabula_sapiens",
        "tissue":             obs["tissue"].astype(str),
        "cell_type":          obs["free_annotation"].astype(str),
        "cell_ontology_class": obs["cell_ontology_class"].astype(str),
    })
    print(f"  Tabula Sapiens: {len(result)} cells")
    return result


# ── Allen Brain Human MTG ─────────────────────────────────────────────────────

# Donor ages from Allen Brain MTG 2018 publication (Table 1)
# https://pmc.ncbi.nlm.nih.gov/articles/PMC6919571/table/T1/
_AB_HUMAN_DONOR_AGES = {
    "H200.1030": 54,
    "H200.1023": 43,
    "H200.1025": 50,
}


def prepare_allen_brain_human(src: dict) -> pd.DataFrame:
    print("Loading Allen Brain Human metadata...")
    meta = pd.read_csv(src["metadata_csv"], low_memory=False)
    print(f"  Loaded {len(meta)} rows, columns: {list(meta.columns)}")

    # Cell ID: use 'sample_name' if present (matches SRR accession from BED stem)
    if "sample_name" in meta.columns:
        cell_id_col = "sample_name"
    elif "cell_id" in meta.columns:
        cell_id_col = "cell_id"
    else:
        raise ValueError(f"Allen Brain metadata has no cell_id or sample_name column. "
                         f"Columns: {list(meta.columns)}")

    donor_col  = "external_donor_name_label" if "external_donor_name_label" in meta.columns else None
    sex_col    = "donor_sex_label"            if "donor_sex_label"            in meta.columns else None
    tissue_col = "region_label"               if "region_label"               in meta.columns else None
    type_col   = "subclass_label"             if "subclass_label"             in meta.columns else None

    result = pd.DataFrame({
        "cell_id": meta[cell_id_col].astype(str),
        "donor":   meta[donor_col].astype(str)  if donor_col  else "unknown",
        "sex":     meta[sex_col].astype(str)     if sex_col    else "unknown",
        "tissue":  meta[tissue_col].astype(str)  if tissue_col else "MTG",
        "cell_type": meta[type_col].astype(str)  if type_col   else "unknown",
    })
    result["dataset"] = "allen_brain_human"
    result["cell_ontology_class"] = result["cell_type"]

    # Add age from donor map
    result["age"] = result["donor"].map(
        {k: str(v) for k, v in _AB_HUMAN_DONOR_AGES.items()}
    ).fillna(src.get("age", "adult"))

    # Standardise sex
    result["sex"] = result["sex"].str.lower().map(
        {"male": "M", "female": "F"}
    ).fillna(result["sex"])

    print(f"  Allen Brain Human: {len(result)} cells")
    if donor_col:
        print(f"  Donor distribution:\n{result['donor'].value_counts().to_string()}")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python 00_prepare_human_metadata.py <config.yaml>")
        sys.exit(1)

    cfg = load_config(sys.argv[1])
    if cfg.get("species", "human") != "human":
        raise ValueError(f"This script is for human configs. Got species={cfg.get('species')}")

    wd = cfg["wd"]
    src = cfg["sources"]

    ts_meta = prepare_tabula_sapiens(src["tabula_sapiens"])
    ab_meta = prepare_allen_brain_human(src["allen_brain_human"])

    combined = pd.concat([ts_meta, ab_meta], ignore_index=True)

    n_dupes = combined["cell_id"].duplicated().sum()
    if n_dupes:
        print(f"WARNING: {n_dupes} duplicate cell IDs in combined metadata")

    print(f"\nCombined: {len(combined)} cells")
    print(combined["dataset"].value_counts().to_string())

    out = os.path.join(wd, "metadata_combined.csv")
    combined.to_csv(out, index=False)
    print(f"\nSaved {len(combined)} rows → {out}")
    print(combined.head(3).to_string())


if __name__ == "__main__":
    main()
