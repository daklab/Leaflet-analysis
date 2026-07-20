"""
Allen Brain Atlas (human MTG 2018 / SEA-AD) — cell ID cleaning.

TODO: Fill in when human Allen Brain data is downloaded.
The cell_id format will depend on which human AB dataset is used.
Likely SRR accession (like mouse) but confirm from the SRA table.
"""

import pandas as pd


def clean_splice_cell_ids(obs: pd.Series) -> pd.Series:
    """
    Placeholder. Return obs unchanged until the dataset is confirmed.
    """
    return obs.copy()


def build_metadata(metadata_csv: str, **kwargs) -> pd.DataFrame:
    """
    Placeholder. Fill in from the human AB SRA/metadata files.
    """
    raise NotImplementedError(
        "Human Allen Brain metadata builder not yet implemented. "
        "See allen_brain_mouse.py for the pattern to follow."
    )
