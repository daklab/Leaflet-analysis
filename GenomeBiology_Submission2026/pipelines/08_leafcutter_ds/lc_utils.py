#!/usr/bin/env python
"""
Shared helpers for the LeafCutter DS pipeline (step 08).

The reason this module exists
-----------------------------
LeafCutter2 reads the groups file WHITESPACE-separated, not tab-separated:

    meta = pd.read_table(args.groups_file, header=None, sep='\\s+')
    # LeafCutter2/leafcutter/differential_splicing/leafcutter_ds.py:59

Our labels come from `broad_cell_type` / `tissue_cell_type`, which contain spaces
("ENDOTHELIAL CELL", "B CELL", "HEMATOPOIETIC PROGENITOR"). Writing them tab-separated
looks correct and parses catastrophically:

  * sample names truncate at the first space, so they stop matching the counts columns
    ("18_46_F__Heart_ENDOTHELIAL CELL" -> "18_46_F__Heart_ENDOTHELIAL");
  * distinct groups collapse -- Marrow "B CELL" and "T CELL" BOTH parse to "CELL";
  * rows become ragged (1-word cell type -> 4 fields, 2-word -> 6), so covariates land
    in different columns per row and pandas pads with NaN;
  * the baseline falls back to the nonexistent default "Control".

The 2026-06-20 run hit all four. It failed silently in the sense that it raised
"There are no groups with enough samples to test", which reads like a sample-size
problem. The tell was that the only jobs that COMPLETED were the ones whose cell type
happened to be a single word.

So: every field written into a groups file goes through `safe_name`, and every groups
file is read back through `assert_groups_file_parses` -- which parses it exactly the way
LeafCutter does -- before any job is submitted.
"""

import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Label sanitisation
# ---------------------------------------------------------------------------

def safe_name(s):
    """Collapse anything LeafCutter's `sep='\\s+'` parser would split on.

    Keeps [A-Za-z0-9_.-]; everything else (crucially spaces) becomes '_'. Applied to
    sample names, group labels AND covariate values -- a space in any field shifts every
    later column on that row.
    """
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))


def pseudobulk_sample_name(tissue_cell_type, sex=None, age=None, donor=None):
    """`tissue_cell_type [_sex] [_age] [_donor]`, sanitised.

    Fields are appended only when available for that dataset, rather than filled with a
    placeholder: Tabula Muris Senis has sex/age/donor, Allen Brain has sex only (its
    `mouse.id` is not a usable donor id and its `age` is hardcoded upstream).
    """
    parts = [tissue_cell_type]
    parts += [p for p in (sex, age, donor) if p is not None and str(p) != "" and str(p) != "nan"]
    return "_".join(safe_name(p) for p in parts)


# ---------------------------------------------------------------------------
# The check whose absence cost the June run
# ---------------------------------------------------------------------------

def read_groups_as_leafcutter_does(path):
    """Parse a groups file byte-for-byte the way LeafCutter2 will parse it."""
    return pd.read_table(path, header=None, sep=r"\s+")


def assert_groups_file_parses(path, counts_cols, expected_labels, n_expected_cols=None):
    """Verify LeafCutter will read this groups file the way we wrote it.

    Parameters
    ----------
    path            : groups file to check
    counts_cols     : column names of the counts matrix this file will be run against
    expected_labels : the group labels we intend LeafCutter to see (set-like)
    n_expected_cols : expected field count per row; inferred from the file if None

    Raises AssertionError describing the specific corruption. Returns the parsed frame.
    """
    meta = read_groups_as_leafcutter_does(path)

    # 1. Ragged rows -> NaN padding -> covariates shifted between rows.
    if n_expected_cols is not None:
        assert meta.shape[1] == n_expected_cols, (
            f"{path}: LeafCutter parses {meta.shape[1]} columns, expected {n_expected_cols}. "
            f"A field almost certainly contains whitespace. First row as parsed: "
            f"{meta.iloc[0].tolist()}"
        )
    assert not meta.isna().any().any(), (
        f"{path}: NaN after whitespace parsing -- rows have unequal field counts, so "
        f"covariates are shifted. Rows affected: "
        f"{meta.index[meta.isna().any(axis=1)].tolist()[:5]}"
    )

    # 2. Group labels must survive intact. This is what merged "B CELL"/"T CELL" -> "CELL".
    got_labels = set(meta[1].astype(str))
    want_labels = set(map(str, expected_labels))
    assert got_labels == want_labels, (
        f"{path}: LeafCutter would see groups {sorted(got_labels)}, we intended "
        f"{sorted(want_labels)}. Labels containing spaces get split."
    )

    # 3. Sample names must match the counts columns, or those samples are silently dropped.
    got_samples = set(meta[0].astype(str))
    missing = got_samples - set(map(str, counts_cols))
    assert not missing, (
        f"{path}: {len(missing)} of {len(got_samples)} sample names do not appear in the "
        f"counts matrix columns -- LeafCutter will drop them. Examples: "
        f"{sorted(missing)[:5]}"
    )

    # 4. Duplicate sample names would silently reweight a group.
    dups = meta[0][meta[0].duplicated()].tolist()
    assert not dups, f"{path}: duplicate sample names {sorted(set(dups))[:5]}"

    return meta


# ---------------------------------------------------------------------------
# Per-contrast covariate selection
# ---------------------------------------------------------------------------

def select_covariates(meta, group_col, candidates):
    """Keep only covariates that vary within the contrast and are not aliased with the group.

    LeafCutter2 one-hot encodes every extra column blindly (`OneHotEncoder(drop='first')`),
    so a constant column contributes nothing and a column collinear with the group label
    makes the design rank-deficient -- which corrupts the fit rather than erroring.

    Returns (kept, dropped) where `dropped` is {covariate: reason}, so the manifest can
    record why each was excluded.
    """
    kept, dropped = [], {}

    for cov in candidates:
        if cov not in meta.columns:
            dropped[cov] = "absent"
            continue

        vals = meta[cov].astype(str)
        if vals.nunique() < 2:
            dropped[cov] = f"constant ({vals.iloc[0]})"
            continue

        # Aliased with the group? e.g. every sample of group A is female and every
        # sample of group B is male -- the group effect and the covariate are the same
        # column, and neither is identifiable.
        crosstab = pd.crosstab(meta[group_col].astype(str), vals)
        if (crosstab > 0).sum(axis=1).max() == 1 and (crosstab > 0).sum(axis=0).max() == 1:
            dropped[cov] = f"perfectly aliased with {group_col}"
            continue

        trial = kept + [cov]
        if _design_rank_deficient(meta, group_col, trial):
            dropped[cov] = "rank-deficient with already-kept covariates"
            continue

        kept.append(cov)

    return kept, dropped


def _design_rank_deficient(meta, group_col, covariates):
    """Full-rank check on the design LeafCutter will actually build."""
    parts = [np.ones((len(meta), 1))]
    for col in [group_col] + list(covariates):
        parts.append(pd.get_dummies(meta[col].astype(str), drop_first=True).to_numpy(float))
    design = np.hstack([p for p in parts if p.shape[1] > 0])
    return np.linalg.matrix_rank(design) < design.shape[1]
