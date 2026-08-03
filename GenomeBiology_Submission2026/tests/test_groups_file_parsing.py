#!/usr/bin/env python
"""
test_groups_file_parsing.py — pin the LeafCutter groups-file corruption of 2026-06-20.

What went wrong
---------------
LeafCutter2 reads its groups file whitespace-separated:

    meta = pd.read_table(args.groups_file, header=None, sep='\\s+')
    # LeafCutter2/leafcutter/differential_splicing/leafcutter_ds.py:59

pipelines/08_leafcutter_ds wrote it TAB-separated with labels taken straight from
`broad_cell_type` / `tissue_cell_type`, which contain spaces ("ENDOTHELIAL CELL",
"B CELL", "HEMATOPOIETIC PROGENITOR"). Four things broke at once:

  1. sample names truncated at the first space and stopped matching the counts columns
  2. distinct groups merged — Marrow "B CELL" and "T CELL" BOTH parsed to "CELL"
  3. rows went ragged (1-word cell type -> 4 fields, 2-word -> 6), shifting covariates
  4. the baseline fell back to the nonexistent default "Control"

It surfaced as `ValueError: There are no groups with enough samples to test`, which
reads like a sample-size problem. The tell was that the only jobs that COMPLETED were
those whose cell type happened to be a single word (MICROGLIA, FIBROBLAST,
GRANULOCYTE, KERATINOCYTE) — 4 of 20 by_celltype, 4 of 9 by_tissue.

This test asserts that lc_utils.assert_groups_file_parses catches each failure mode,
and that files written through lc_utils.safe_name survive the round trip.

Usage:
    python tests/test_groups_file_parsing.py

Exit code 0 = all checks pass.
"""

import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "pipelines", "08_leafcutter_ds"))

from lc_utils import (  # noqa: E402
    safe_name,
    pseudobulk_sample_name,
    assert_groups_file_parses,
    select_covariates,
)

# The real Marrow labels — the case where two distinct cell types collapse into one.
MARROW_LABELS = ["B CELL", "GRANULOCYTE", "HEMATOPOIETIC PROGENITOR", "T CELL"]
MARROW_ROWS = [
    ("18_45_M__Marrow_B CELL",                   "B CELL",                   "old",   "male"),
    ("18_46_F__Marrow_B CELL",                   "B CELL",                   "old",   "female"),
    ("18_46_F__Marrow_GRANULOCYTE",              "GRANULOCYTE",              "old",   "female"),
    ("18_46_F__Marrow_HEMATOPOIETIC PROGENITOR", "HEMATOPOIETIC PROGENITOR", "old",   "female"),
    ("18_46_F__Marrow_T CELL",                   "T CELL",                   "old",   "female"),
    ("3_9_M__Marrow_T CELL",                     "T CELL",                   "young", "male"),
]

failures = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except AssertionError as e:
        print(f"  FAIL  {name}\n        {e}")
        failures.append(name)


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write("\t".join(r) + "\n")


def test_raw_labels_are_rejected(tmp):
    """The exact 2026-06-20 file must be rejected, not silently accepted."""
    path = os.path.join(tmp, "raw.txt")
    write(path, MARROW_ROWS)
    counts_cols = [r[0] for r in MARROW_ROWS]
    try:
        assert_groups_file_parses(path, counts_cols, MARROW_LABELS, n_expected_cols=4)
    except AssertionError:
        return                       # correctly rejected
    raise AssertionError("unsanitised labels with spaces were accepted")


def test_label_collapse_is_detected(tmp):
    """B CELL and T CELL both parsing to CELL must be caught even without a column count.

    The collapse comes from the SAMPLE NAME: "18_45_M__Marrow_B CELL" splits after
    "..._B", which shifts "CELL" into the group column. Every row here is two-word, so
    field counts stay equal and the ragged-row check cannot fire — the label-set check
    is the only thing standing between this file and a corrupted run.
    """
    path = os.path.join(tmp, "collapse.txt")
    rows = [("18_45_M__Marrow_B CELL", "B CELL", "male"),
            ("18_46_F__Marrow_B CELL", "B CELL", "female"),
            ("18_46_F__Marrow_T CELL", "T CELL", "female"),
            ("3_9_M__Marrow_T CELL",   "T CELL", "male")]
    write(path, rows)

    parsed = pd.read_table(path, header=None, sep=r"\s+")
    assert not parsed.isna().any().any(), "precondition: rows should not be ragged"
    assert set(parsed[1]) == {"CELL"}, \
        f"precondition: labels should have collapsed to CELL, got {set(parsed[1])}"

    try:
        assert_groups_file_parses(path, [r[0] for r in rows], {"B CELL", "T CELL"})
    except AssertionError as e:
        assert "groups" in str(e), f"caught, but not for the label reason: {e}"
        return
    raise AssertionError("collapsed labels B CELL/T CELL -> CELL were accepted")


def test_sample_name_mismatch_is_detected(tmp):
    """Truncated sample names must be caught, not silently dropped by LeafCutter."""
    path = os.path.join(tmp, "trunc.txt")
    rows = [("18_46_F__Heart_ENDOTHELIAL CELL", "ENDO", "male"),
            ("18_47_F__Heart_FIBROBLAST_x",     "FIBRO", "male")]
    write(path, rows)
    try:
        assert_groups_file_parses(path, [r[0] for r in rows], {"ENDO", "FIBRO"})
    except AssertionError:
        return
    raise AssertionError("truncated sample names were accepted")


def test_sanitised_file_round_trips(tmp):
    """Files written through safe_name must parse exactly as intended."""
    path = os.path.join(tmp, "clean.txt")
    rows = [tuple(safe_name(x) for x in r) for r in MARROW_ROWS]
    write(path, rows)
    labels = {safe_name(l) for l in MARROW_LABELS}
    meta = assert_groups_file_parses(path, [r[0] for r in rows], labels, n_expected_cols=4)
    assert set(meta[1]) == labels, f"labels changed: {set(meta[1])}"
    assert len(meta) == len(rows), "rows lost"


def test_duplicate_sample_names_rejected(tmp):
    path = os.path.join(tmp, "dup.txt")
    rows = [("s1", "A", "male"), ("s1", "B", "male"), ("s2", "A", "male")]
    write(path, rows)
    try:
        assert_groups_file_parses(path, ["s1", "s2"], {"A", "B"})
    except AssertionError as e:
        assert "duplicate" in str(e).lower(), f"wrong reason: {e}"
        return
    raise AssertionError("duplicate sample names were accepted")


def test_sample_name_scheme():
    """tissue_cell_type [_sex] [_age] [_donor], fields present only when available."""
    tms = pseudobulk_sample_name("Heart_FIBROBLAST", "female", "18m", "18_46_F")
    assert tms == "Heart_FIBROBLAST_female_18m_18_46_F", tms

    # Allen has no donor and no measured age — those fields are omitted, not faked.
    allen = pseudobulk_sample_name("Brain_Isocortex_Visual_EXCITATORY NEURON", "male")
    assert allen == "Brain_Isocortex_Visual_EXCITATORY_NEURON_male", allen

    assert " " not in tms and " " not in allen, "whitespace survived sanitisation"


def test_covariate_selection():
    """Constant and group-aliased covariates must be dropped before they corrupt the fit."""
    m = pd.DataFrame({
        "group": ["A"] * 6 + ["B"] * 6,
        "sex":   ["female", "male"] * 6,     # varies within both groups -> keep
        "age":   ["3m"] * 12,                # constant -> drop
        "batch": ["x"] * 6 + ["y"] * 6,      # aliased with group -> drop
    })
    kept, dropped = select_covariates(m, "group", ["sex", "age", "batch"])
    assert kept == ["sex"], f"kept {kept}"
    assert "constant" in dropped["age"], dropped
    assert "aliased" in dropped["batch"], dropped

    # Allen EXC vs INH: sex varies, age is the hardcoded constant.
    a = pd.DataFrame({"group": ["EXC"] * 12 + ["INH"] * 12,
                      "sex": (["female", "male"] * 6) * 2,
                      "age": ["2m"] * 24})
    kept, dropped = select_covariates(a, "group", ["sex", "age"])
    assert kept == ["sex"], f"allen kept {kept}"
    assert "constant" in dropped["age"], dropped


def main():
    print("test_groups_file_parsing.py — LeafCutter groups-file regression suite\n")
    with tempfile.TemporaryDirectory() as tmp:
        check("raw 2026-06-20 labels are rejected",      lambda: test_raw_labels_are_rejected(tmp))
        check("B CELL/T CELL collapse is detected",      lambda: test_label_collapse_is_detected(tmp))
        check("truncated sample names are detected",     lambda: test_sample_name_mismatch_is_detected(tmp))
        check("duplicate sample names are rejected",     lambda: test_duplicate_sample_names_rejected(tmp))
        check("sanitised file round-trips",              lambda: test_sanitised_file_round_trips(tmp))
        check("sample-name scheme",                      test_sample_name_scheme)
        check("per-contrast covariate selection",        test_covariate_selection)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
