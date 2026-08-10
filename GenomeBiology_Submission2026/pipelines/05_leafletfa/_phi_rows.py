"""Map a fit's phi rows back onto atlas cell indices.

With `--holdout-frac`, phi covers only the TRAINING cells, so `phi[some_atlas_index]` is wrong --
it silently reads the wrong cell, or raises if a boolean mask of full atlas length is used. Every
consumer must go through this.

Fits written before train_idx was saved are handled by recomputing the split, which is safe because
the split seed is fixed at 12345 in A44 for exactly this reason.
"""
import numpy as np


def train_index(z, n_atlas_cells):
    """Atlas cell indices, in phi row order."""
    if "train_idx" in getattr(z, "files", []):
        ti = np.asarray(z["train_idx"])
        if ti.size:
            return ti.astype(int)
    phi_n = np.asarray(z["phi"]).shape[0]
    if phi_n == n_atlas_cells:
        return np.arange(n_atlas_cells)
    # reconstruct the fixed-seed holdout used by A44
    frac = 1.0 - phi_n / n_atlas_cells
    rng = np.random.default_rng(12345)
    ho = np.sort(rng.choice(n_atlas_cells, int(round(frac * n_atlas_cells)), replace=False))
    keep = np.setdiff1d(np.arange(n_atlas_cells), ho)
    if len(keep) != phi_n:
        raise ValueError(f"cannot reconstruct the split: phi has {phi_n} rows, "
                         f"reconstruction gives {len(keep)} of {n_atlas_cells}")
    return keep


def phi_row_of(z, n_atlas_cells):
    """atlas cell index -> phi row, with -1 where the cell was held out."""
    ti = train_index(z, n_atlas_cells)
    lut = np.full(n_atlas_cells, -1, dtype=np.int64)
    lut[ti] = np.arange(len(ti))
    return lut
