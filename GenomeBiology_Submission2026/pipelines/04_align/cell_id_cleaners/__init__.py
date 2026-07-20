"""
cell_id_cleaners
================
One module per data source. Each module exposes:

    clean_splice_cell_ids(obs: pd.Series) -> pd.Series
    clean_ge_cell_ids(obs: pd.Series) -> pd.Series

Both return a Series of cleaned IDs that can be intersected across modalities.
The logic here was historically the primary cause of cell drop and alignment bugs
between splice and gene-expression AnnDatas — keep it isolated and tested.
"""

from .tms_mouse import clean_splice_cell_ids as tms_mouse_splice
from .tms_mouse import clean_ge_cell_ids as tms_mouse_ge
from .allen_brain_mouse import clean_splice_cell_ids as allen_brain_mouse_splice
from .allen_brain_mouse import build_metadata as allen_brain_mouse_metadata
from .tabula_sapiens import clean_splice_cell_ids as tabula_sapiens_splice
from .tabula_sapiens import clean_ge_cell_ids as tabula_sapiens_ge
from .allen_brain_human import clean_splice_cell_ids as allen_brain_human_splice

CLEANERS = {
    "tms_mouse":         {"splice": tms_mouse_splice,          "ge": tms_mouse_ge},
    "allen_brain_mouse": {"splice": allen_brain_mouse_splice,   "ge": None},
    "tabula_sapiens":    {"splice": tabula_sapiens_splice,      "ge": tabula_sapiens_ge},
    "allen_brain_human": {"splice": allen_brain_human_splice,   "ge": None},
}


def get_cleaner(source_name: str, modality: str):
    """Return the cleaning function for a given source + modality."""
    if source_name not in CLEANERS:
        raise ValueError(f"No cleaner registered for source '{source_name}'. "
                         f"Available: {list(CLEANERS)}")
    fn = CLEANERS[source_name][modality]
    if fn is None:
        raise ValueError(f"No {modality} cleaner for source '{source_name}'")
    return fn
