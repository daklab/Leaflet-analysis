"""
MeanBayes: Bayesian PSI imputation baseline using Beta conjugate priors.

Fits a Beta prior from the observed PSI distribution (method of moments)
and computes the conjugate posterior mean for every unobserved position.
Observed PSI values are returned unchanged.

Adapted from Smriti Vaidyanathan's SpliceVI implementation.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

logger = logging.getLogger(__name__)


def _to_dense(mat) -> np.ndarray:
    if sp.issparse(mat):
        return np.asarray(mat.todense(), dtype=np.float64)
    return np.asarray(mat, dtype=np.float64)


def _fit_beta_prior(psi: np.ndarray, observed_mask: np.ndarray) -> tuple[float, float]:
    n_cells, n_junctions = psi.shape
    junction_means = np.empty(n_junctions)
    for j in range(n_junctions):
        observed_vals = psi[:, j][observed_mask[:, j]]
        junction_means[j] = observed_vals.mean() if len(observed_vals) > 0 else 0.5

    global_mean = junction_means.mean()
    global_var  = junction_means.var()

    max_var = global_mean * (1.0 - global_mean)
    if max_var <= 1e-8 or global_var <= 1e-8:
        return 1.0, 1.0

    global_var = np.clip(global_var, 1e-8, max_var - 1e-8)
    common  = (global_mean * (1.0 - global_mean) / global_var) - 1.0
    a_prior = max(global_mean * common, 1e-6)
    b_prior = max((1.0 - global_mean) * common, 1e-6)
    return float(a_prior), float(b_prior)


def _impute_with_prior(psi: np.ndarray, observed_mask: np.ndarray,
                       a_prior: float, b_prior: float) -> np.ndarray:
    psi_obs_only  = psi * observed_mask
    sum_psi = psi_obs_only.sum(axis=0)
    n_obs   = observed_mask.sum(axis=0).astype(float)
    post_mean_per_junction = (a_prior + sum_psi) / (a_prior + b_prior + n_obs)
    return np.where(observed_mask, psi, post_mean_per_junction[np.newaxis, :])


class MeanBayes:
    """
    Bayesian PSI imputation baseline using Beta conjugate prior.

    Parameters
    ----------
    adata : AnnData with layers:
        - junc_ratio_layer ("junc_ratio"): PSI values in [0, 1]
        - psi_mask_layer ("psi_mask"):     binary mask, 1=observed 0=missing
    group_by : optional obs column for per-group priors (e.g. cell type)
    min_group_cells : minimum observed cells per junction to use group prior
    """

    def __init__(
        self,
        adata: AnnData,
        junc_ratio_layer: str = "junc_ratio",
        psi_mask_layer: str = "psi_mask",
        group_by: str | None = None,
        min_group_cells: int = 2,
    ) -> None:
        if junc_ratio_layer not in adata.layers:
            raise KeyError(f"Layer '{junc_ratio_layer}' not in adata.layers: {list(adata.layers)}")
        if psi_mask_layer not in adata.layers:
            raise KeyError(f"Layer '{psi_mask_layer}' not in adata.layers: {list(adata.layers)}")
        if group_by is not None and group_by not in adata.obs.columns:
            raise KeyError(f"group_by '{group_by}' not in adata.obs columns")

        self.adata             = adata
        self.junc_ratio_layer  = junc_ratio_layer
        self.psi_mask_layer    = psi_mask_layer
        self.group_by          = group_by
        self.min_group_cells   = min_group_cells

        self._psi  = _to_dense(adata.layers[junc_ratio_layer])
        self._mask = _to_dense(adata.layers[psi_mask_layer]).astype(bool)

    def _compute_imputed_global(self, psi, mask):
        a, b = _fit_beta_prior(psi, mask)
        return _impute_with_prior(psi, mask, a, b)

    def _compute_imputed_by_group(self, psi, mask, groups):
        n_cells, n_junctions = psi.shape
        imputed = np.empty_like(psi)

        a_global, b_global = _fit_beta_prior(psi, mask)
        sum_psi_global = (psi * mask).sum(axis=0)
        n_obs_global   = mask.sum(axis=0).astype(float)
        post_mean_global = (a_global + sum_psi_global) / (a_global + b_global + n_obs_global)

        for grp in np.unique(groups):
            idx      = np.where(groups == grp)[0]
            psi_g    = psi[idx]
            mask_g   = mask[idx]

            a_g, b_g    = _fit_beta_prior(psi_g, mask_g)
            sum_psi_g   = (psi_g * mask_g).sum(axis=0)
            n_obs_g     = mask_g.sum(axis=0).astype(float)
            post_mean_g = (a_g + sum_psi_g) / (a_g + b_g + n_obs_g)

            sufficient  = n_obs_g >= self.min_group_cells
            post_mean_j = np.where(sufficient, post_mean_g, post_mean_global)

            imputed[idx] = np.where(mask[idx], psi[idx], post_mean_j[np.newaxis, :])

        return imputed

    def get_imputed_splicing(
        self,
        indices: Sequence[int] | None = None,
        junction_list: Sequence[str] | None = None,
        return_numpy: bool = False,
    ):
        psi  = self._psi
        mask = self._mask

        if self.group_by is None:
            imputed = self._compute_imputed_global(psi, mask)
        else:
            groups  = np.asarray(self.adata.obs[self.group_by])
            imputed = self._compute_imputed_by_group(psi, mask, groups)

        cell_names = self.adata.obs_names
        junc_names = self.adata.var_names

        if indices is not None:
            imputed    = imputed[list(indices)]
            cell_names = cell_names[list(indices)]

        if junction_list is not None:
            junc_pos   = [self.adata.var_names.get_loc(j) for j in junction_list]
            imputed    = imputed[:, junc_pos]
            junc_names = list(junction_list)

        if return_numpy:
            return imputed.astype(np.float32)

        return pd.DataFrame(imputed, index=cell_names, columns=junc_names)
