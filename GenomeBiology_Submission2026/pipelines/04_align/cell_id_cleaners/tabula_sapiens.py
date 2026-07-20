"""
Tabula Sapiens (Smart-seq2) — human cell ID cleaning.

This is the most complex cleaner. The same cell appears under DIFFERENT ID
formats in the splicing AnnData (derived from BAM/junction file names) vs.
the gene-expression AnnData (from the original Tabula Sapiens release).

GE format (obs['cell_id_backup']):
  TSP{sample_id}_{platform}_{donor1}_{plate1}_{donor2}_{plate2}_{tissue}_{subtissue}_{celltype}
  e.g. "TSP1_SS2_L144742_B1_B133819_D2_Liver_NA_Hepatocyte"

Splice format (obs['cell_id']):
  {prefix}_per_{sample_id}_{platform}_{donor1}_{plate1}_{donor2}_{plate2}_{tissue}_{subtissue}_{celltype}
  e.g. "some_prefix_per_1_SS2_L144736_G2_B133819_M4_Liver_NA_Hepatocyte"

Canonical ID (used for joining):
  {sample_id}_donor_{donor1}_{plate1}_{donor2}_{plate2}_{tissue}_{celltype}
  (note: subtissue is deliberately dropped — it's often "NA" and differs between formats)

Extracted from Human_Splicing_Foundation/GeneExpression/06_align_splice_ge_anndatas.py
"""

import pandas as pd


def _parse_ts_fields(series: pd.Series, id_col: str, is_splice: bool) -> pd.Series:
    """
    Parse the 9-field TS cell ID and build a canonical join key.

    is_splice=True  → split on 'per_' first (splice format)
    is_splice=False → strip leading 'TSP' prefix (GE format)
    """
    if is_splice:
        # "some_prefix_per_1_SS2_L144736_G2_B133819_M4_Liver_NA_Hepatocyte"
        core = series.str.split("per_").str[-1]
    else:
        # "TSP1_SS2_L144742_B1_B133819_D2_Liver_NA_Hepatocyte"
        # drop leading "TSP"
        core = series.str.replace("^TSP", "", regex=True)

    parts = core.str.split("_")

    sample_id = parts.str[0]
    # parts[1] = platform (SS2 etc.) — skip
    donor1  = parts.str[2]
    plate1  = parts.str[3]
    donor2  = parts.str[4]
    plate2  = parts.str[5]
    tissue  = parts.str[6]
    # parts[7] = subtissue — deliberately excluded from key (too inconsistent)
    celltype = parts.str[8]

    canonical = (
        sample_id + "_donor_" + donor1 + "_" + plate1
        + "_" + donor2 + "_" + plate2
        + "_" + tissue + "_" + celltype
    )
    return canonical


def clean_splice_cell_ids(obs: pd.DataFrame) -> pd.Series:
    """
    Clean TS cell IDs in the splice AnnData.

    obs must have columns: 'cell_id', 'dataset'
    Non-TS cells (e.g. Allen Brain) are returned unchanged.
    Returns a Series aligned with obs.index.
    """
    result = obs["cell_id"].copy()
    ts_mask = obs["dataset"] == "tabula_sapiens"

    if ts_mask.any():
        result[ts_mask] = _parse_ts_fields(
            obs.loc[ts_mask, "cell_id"], id_col="cell_id", is_splice=True
        )

    return result


def clean_ge_cell_ids(obs: pd.DataFrame) -> pd.Series:
    """
    Clean TS cell IDs in the gene-expression AnnData.

    obs must have column 'cell_id_backup' (the original TS cell ID).
    Non-TS cells fall back to 'cell_id_backup' unchanged.
    Returns a Series aligned with obs.index.
    """
    result = obs["cell_id_backup"].copy()
    ts_mask = obs["cell_id_backup"].str.startswith("TSP")

    if ts_mask.any():
        result[ts_mask] = _parse_ts_fields(
            obs.loc[ts_mask, "cell_id_backup"], id_col="cell_id_backup", is_splice=False
        )

    return result
