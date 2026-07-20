"""
Allen Brain Atlas (mouse isocortex/hippocampus 2021) — cell ID cleaning.

Unlike TMS, the Allen Brain cell ID = SRR run accession (e.g. "SRR14748932").
The junction file basename is {SRR}_junctions_with_barcodes.bed so there is
no reformatting needed for the splice AnnData.

The complexity is in metadata assembly: the mapping
  S3 path → filename → SRR ID → GSE metadata → biological cell type
requires 3 separate CSV files and specific regex parsing (see build_metadata).

Extracted from metadata_prep.py and 01_prepare_expression_data.py.
"""

import pandas as pd
import re


def clean_splice_cell_ids(obs: pd.Series) -> pd.Series:
    """
    The splicing AnnData cell_id IS the SRR run ID — no cleaning needed.
    This function exists for API consistency; it returns obs unchanged.
    """
    return obs.copy()


def build_metadata(
    metadata_csv: str,
    sra_table: str,
    gse_metadata_gz: str,
    sample_group_gz: str,
    sra_s3_links_txt: str,
) -> pd.DataFrame:
    """
    Assemble Allen Brain metadata by linking:
      sra_s3_links → filename → expected_R1_fastq → GSE metadata → SRA table

    Returns a DataFrame with columns:
      cell_id, age, cell_ontology_class, mouse.id, sex, subtissue, tissue

    The cell_id is the SRR accession (matches the junction file basename).
    """
    metadata_ab = pd.read_csv(metadata_csv, low_memory=False)

    sample_group = pd.read_csv(sample_group_gz)
    sra_ab = pd.read_csv(sra_table)
    gse_meta = pd.read_csv(gse_metadata_gz, low_memory=False)

    # Extract filename from S3 path, clean to canonical fastq.gz name
    sra_links = pd.read_csv(sra_s3_links_txt, sep="\t", header=None,
                            names=["SRR", "expected_R1_fastq"])
    sra_links["filename"] = sra_links["expected_R1_fastq"].str.extract(r"/([^/]+)$")[0]
    sra_links["expected_R1_fastq"] = sra_links["filename"].str.replace(
        r"\.fastq.*", ".fastq.gz", regex=True
    )

    # Warn on merge failures — any drop here loses cells permanently
    n_before = len(gse_meta)
    gse_meta = gse_meta.merge(sra_links, on="expected_R1_fastq", how="left",
                               indicator=True)
    n_failed = (gse_meta["_merge"] != "both").sum()
    if n_failed:
        print(f"[allen_brain_mouse] WARNING: {n_failed}/{n_before} GSE rows "
              f"could not be linked to an SRR via expected_R1_fastq")
    gse_meta = gse_meta.drop(columns=["_merge"])

    gse_meta["Run"] = gse_meta["SRR"]
    merged = sra_ab.merge(gse_meta, on="Run")

    merged["cell_id"]             = merged["Run"]
    merged["cell_ontology_class"] = merged["cell_type_alias_label"]
    merged["tissue"]              = merged["sample_name"]
    merged["age"]                 = "2m"
    merged["mouse.id"]            = merged["external_donor_name_label"]
    merged["subtissue"]           = merged["region_label"]
    merged["sex"]                 = merged["donor_sex_label"]

    return merged[["cell_id", "age", "cell_ontology_class",
                   "mouse.id", "sex", "subtissue", "tissue"]]
