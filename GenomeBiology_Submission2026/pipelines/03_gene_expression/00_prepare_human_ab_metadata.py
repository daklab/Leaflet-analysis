#!/usr/bin/env python
"""
Prepare cell metadata CSV for the human Allen Brain dataset.

Single source: lein-human-cortex/INFO/metadata.csv
  - sample_name        → cell_id (matches splice AnnData obs_names)
  - exp_component_name → needed to join GE h5ad obs_names to splice
  - donor_sex_label    → sex
  - external_donor_name_label → donor
  - region_label       → mapped to explicit tissue (Brain_MTG, Brain_A1C, ...)
  - cortical_layer_label → cortical_layer
  - subclass_label     → cell_type_ontology_class (verbatim, fine-grained)
  - subclass_label     → broad_cell_type (via cell_type_maps/human_allen_brain_celltype_map.json)
  - outlier_call==True → REMOVED (1,985 cells)

Donor ages from Hodge et al. 2019 Table 1 (MTG dataset paper).

Produces:
    $WD/metadata_human_allen_brain_combined.csv

Usage:
    python pipelines/03_gene_expression/00_prepare_human_ab_metadata.py config/human_allen_brain.yaml
"""

import os
import sys

import pandas as pd
import yaml


REGION_TO_TISSUE = {
    "MTG":  "Brain_MTG",
    "A1C":  "Brain_A1C",
    "CgG":  "Brain_CgG",
    "M1lm": "Brain_M1",
    "M1ul": "Brain_M1",
    "S1lm": "Brain_S1",
    "S1ul": "Brain_S1",
    "V1C":  "Brain_V1C",
}

# Hodge et al. 2019 Table 1
DONOR_AGES = {
    "H200.1030": 54,
    "H200.1023": 43,
    "H200.1025": 50,
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    if len(sys.argv) < 2:
        print("Usage: python 00_prepare_human_ab_metadata.py <config.yaml>")
        sys.exit(1)

    cfg = load_config(sys.argv[1])
    wd  = cfg["wd"]
    src = cfg["sources"]["allen_brain_human"]

    # ── Load raw metadata ──────────────────────────────────────────────────────
    meta_csv = src.get("metadata_csv",
                       "/gpfs/commons/datasets/controlled/BRAIN_NeMO/lein-human-cortex/INFO/metadata.csv")
    print(f"Loading {meta_csv} ...")
    meta = pd.read_csv(meta_csv, low_memory=False)
    print(f"  Loaded {len(meta)} rows")

    # ── Drop outliers ──────────────────────────────────────────────────────────
    n_before = len(meta)
    meta = meta[meta["outlier_call"] != True].copy()
    print(f"  Outlier filter (outlier_call==True): {n_before} → {len(meta)} cells "
          f"({n_before - len(meta)} removed)")

    # ── Build output columns ───────────────────────────────────────────────────
    out = pd.DataFrame()
    out["cell_id"]              = meta["sample_name"].astype(str)
    out["exp_component_name"]   = meta["exp_component_name"].astype(str)  # GE bridge key
    out["donor"]                = meta["external_donor_name_label"].astype(str)
    out["sex"]                  = meta["donor_sex_label"].astype(str).str.strip()
    out["age"]                  = out["donor"].map(DONOR_AGES).astype("Int64")
    out["dataset"]              = "allen_brain_human"
    out["tissue"]               = meta["region_label"].astype(str).map(REGION_TO_TISSUE).fillna(
                                      meta["region_label"].astype(str))
    out["cortical_layer"]       = meta["cortical_layer_label"].astype(str).fillna("unknown")
    out["cell_type_ontology_class"] = meta["subclass_label"].astype(str)
    # Use subclass_label directly as broad_cell_type (explicit enough for human brain-only dataset)
    # The JSON map (human_allen_brain_celltype_map.json) provides coarser groupings for reference
    out["broad_cell_type"]      = out["cell_type_ontology_class"]
    out["class_label"]          = meta["class_label"].astype(str)   # GABAergic / Glutamatergic / Non-neuronal
    out["tissue_cell_type"]     = out["tissue"] + "_" + out["broad_cell_type"]

    # ── QC report ─────────────────────────────────────────────────────────────
    print(f"\nBroad cell type distribution:\n{out['broad_cell_type'].value_counts().to_string()}")
    print(f"\nTissue distribution:\n{out['tissue'].value_counts().to_string()}")
    print(f"\nDonor distribution:\n{out['donor'].value_counts().to_string()}")
    print(f"\nSex distribution:\n{out['sex'].value_counts().to_string()}")

    n_dupes = out["cell_id"].duplicated().sum()
    if n_dupes:
        print(f"WARNING: {n_dupes} duplicate cell_ids")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(wd, "metadata_human_allen_brain_combined.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out)} rows → {out_path}")
    print(out.head(3).to_string())


if __name__ == "__main__":
    main()
