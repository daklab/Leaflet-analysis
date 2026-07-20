"""
Tabula Muris Senis (FACS / Smart-seq2) — mouse cell ID cleaning.

The raw TMS cell IDs are inconsistent across age groups:
  - 3m cells:  "AAACGGGCAAGGTTCT.mm10-plus-0-0.1-2_B001-A-000"
               → strip suffix → parse specially → adds '-1-1'
  - other ages: "AAACGGGCAAGGTTCT.mm10-plus-0-0"
               → split('.')[0] gives the barcode directly

The same 3m inconsistency appears BOTH in metadata_prep.py (for the combined
metadata CSV) AND in 01_prepare_expression_data.py (for the GE AnnData).
Both must produce the same canonical ID or cells will be dropped at the join.

DO NOT change either function without updating both.
"""

import pandas as pd
import re


# ── shared suffixes to strip before anything else ──────────────────────────────
_STRIP_PATTERNS = [".mm10-plus-0-0", ".mus-2-1"]


def _strip_platform_suffix(cell_id: str) -> str:
    for pat in _STRIP_PATTERNS:
        cell_id = cell_id.split(pat)[0]
    return cell_id


def _format_3m_id(index: str) -> str:
    """
    Special rule for 3-month-old TMS cells.

    Raw index example: "AAACGGGCAAGGTTCT.1-1_B001-A-000"
    Expected output:   "AAACGGGCAAGGTTCT-1-1_B001-1-1"

    Logic (preserved exactly from metadata_prep.py::format_cell_id):
      1. Replace first '.' with '-', first '_' with '-'
      2. Split on '.'
      3. In part[1], replace first '-' with '_'
      4. Rejoin with '-', append '-1-1'
    """
    index = str(index)
    try:
        modified = index.replace(".", "-", 1).replace("_", "-", 1)
        parts = modified.split(".")
        if len(parts) < 2:
            return index
        corrected = parts[1].replace("-", "_", 1)
        return parts[0] + "-" + corrected + "-1-1"
    except Exception:
        return index


def clean_metadata_cell_ids(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Clean cell IDs in the combined TMS+AB metadata CSV.
    This mirrors the logic in metadata_prep.py.
    Returns a copy with a 'cell_id' column added.
    """
    meta = metadata[metadata["method"] == "facs"].copy()
    meta["cell_id"] = meta.apply(
        lambda row: _format_3m_id(row["index"]) if row["age"] == "3m"
        else str(row["index"]).split(".")[0],
        axis=1,
    )
    return meta


def clean_splice_cell_ids(obs: pd.Series, age_col: pd.Series = None) -> pd.Series:
    """
    Clean cell IDs from the splicing AnnData (obs_names or obs['cell_id']).
    The splicing AnnData derives its cell IDs directly from junction file basenames,
    which mirror the raw TMS BAM file names. So the same suffix-stripping applies.

    obs: Series of raw cell IDs
    age_col: optional Series of age labels (same index) — needed for 3m rule
    """
    cleaned = obs.map(_strip_platform_suffix)
    if age_col is not None:
        mask_3m = age_col == "3m"
        cleaned[mask_3m] = obs[mask_3m].map(_format_3m_id)
    return cleaned


def clean_ge_cell_ids(obs: pd.DataFrame) -> pd.Series:
    """
    Clean cell IDs from the GE AnnData (obs DataFrame, needs 'age' column).
    Mirrors logic in 01_prepare_expression_data.py Step 5.

    Returns a Series of canonical IDs aligned with obs.index.
    """
    # Step 1: strip platform suffix from obs_names
    cell_ids = obs.index.to_series().map(_strip_platform_suffix)

    # Step 2: 3m special rule (replaces '.' with '_' up to 2 times, keep first 2 underscore fields)
    if "age" in obs.columns:
        mask_3m = obs["age"] == "3m"
        if mask_3m.any():
            # From 01_prepare_expression_data.py:
            # tms_adata.obs.loc[mask, 'cell_clean'] = tms_adata.obs.loc[mask, 'cell'].str.replace('.', '_', 2)
            # tms_adata.obs["cell_clean"] = tms_adata.obs["cell_clean"].str.split('_').str[:2].str.join('_')
            raw_3m = obs.loc[mask_3m, "cell"] if "cell" in obs.columns else cell_ids[mask_3m]
            cell_ids[mask_3m] = (
                raw_3m.str.replace(".", "_", regex=False, n=2)
                      .str.split("_").str[:2].str.join("_")
            )

    return cell_ids
