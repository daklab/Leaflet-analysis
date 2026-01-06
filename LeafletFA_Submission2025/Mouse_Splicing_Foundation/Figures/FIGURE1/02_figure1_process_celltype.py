#!/usr/bin/env python
"""
FIGURE 1 - STEP 2: Process single cell type
This script processes one cell type and saves intermediate results
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import issparse
from tqdm import tqdm


class _ATSECache:
    """Cache for masks/indices used across many calls"""
    def __init__(self, adata, groupby_col="tissue_celltype"):
        self.groupby_col = groupby_col
        self.obs_group = adata.obs[groupby_col].to_numpy()
        self.age = adata.obs["age_group"].to_numpy()
        self.event_ids = adata.var["event_id"].to_numpy()
        self.junc_ids = (adata.var["junction_id"].to_numpy()
                         if "junction_id" in adata.var.columns
                         else adata.var_names.to_numpy())
        self.gene_names = (adata.var["gene_name"].to_numpy()
                           if "gene_name" in adata.var.columns else None)
        self._rowmask_by_ct = {}
        self._colidx_by_event = {}
        
        J = adata.layers["cell_by_junction_matrix"]
        A = adata.layers["cell_by_cluster_matrix"]
        PSI = adata.layers["psi"]
        self.J = J.tocsr(copy=False) if issparse(J) else J
        self.A = A.tocsr(copy=False) if issparse(A) else A
        self.PSI = PSI
    
    def rowmask(self, cell_type):
        m = self._rowmask_by_ct.get(cell_type)
        if m is None:
            m = (self.obs_group == cell_type)
            self._rowmask_by_ct[cell_type] = m
        return m
    
    def colidx(self, event_id):
        idx = self._colidx_by_event.get(event_id)
        if idx is None:
            idx = np.where(self.event_ids == event_id)[0]
            self._colidx_by_event[event_id] = idx
        return idx


def extract_atse_data_fast_cached(adata, event_id, cell_type, cache):
    """Vectorized extractor for one (event_id, cell_type) pair"""
    row_mask = cache.rowmask(cell_type)
    if not row_mask.any():
        return pd.DataFrame(columns=["cell_id", "cell_age_group", "junction_id",
                                     "junction_counts", "atse_counts", "psi_value",
                                     "gene_name", "event_id", "cell_type"])
    
    col_idx = cache.colidx(event_id)
    if col_idx.size == 0:
        return pd.DataFrame(columns=["cell_id", "cell_age_group", "junction_id",
                                     "junction_counts", "atse_counts", "psi_value",
                                     "gene_name", "event_id", "cell_type"])
    
    J_sub = cache.J[row_mask][:, col_idx]
    A_sub = cache.A[row_mask][:, col_idx]
    PSI_full = cache.PSI[row_mask][:, col_idx]
    
    if issparse(J_sub):
        Jcoo = J_sub.tocoo(copy=False)
        Acoo = A_sub.tocoo(copy=False)
        r = np.concatenate([Jcoo.row, Acoo.row])
        c = np.concatenate([Jcoo.col, Acoo.col])
        if r.size == 0:
            return pd.DataFrame(columns=["cell_id", "cell_age_group", "junction_id",
                                         "junction_counts", "atse_counts", "psi_value",
                                         "gene_name", "event_id", "cell_type"])
        rc = np.unique(np.stack([r, c], axis=1), axis=0)
        rr, cc = rc[:, 0], rc[:, 1]
        Jcsr = J_sub.tocsr(copy=False)
        Acsr = A_sub.tocsr(copy=False)
        jv = Jcsr[rr, cc].A1
        av = Acsr[rr, cc].A1
        if issparse(PSI_full):
            PSIcsr = PSI_full.tocsr(copy=False)
            pv = PSIcsr[rr, cc].A1
        else:
            PSI_dense = np.asarray(PSI_full)
            pv = PSI_dense[rr, cc]
    else:
        Jd = np.asarray(J_sub)
        Ad = np.asarray(A_sub)
        mask = (Jd > 0) | (Ad > 0)
        rr, cc = np.where(mask)
        if rr.size == 0:
            return pd.DataFrame(columns=["cell_id", "cell_age_group", "junction_id",
                                         "junction_counts", "atse_counts", "psi_value",
                                         "gene_name", "event_id", "cell_type"])
        jv = Jd[rr, cc]
        av = Ad[rr, cc]
        PSId = np.asarray(PSI_full)
        pv = PSId[rr, cc]
    
    cells = adata.obs.index[row_mask].to_numpy()
    junc_ids = cache.junc_ids[col_idx]
    gene_map = None
    if cache.gene_names is not None:
        gene_map = pd.Series(cache.gene_names[col_idx], index=junc_ids)
    
    df = pd.DataFrame({
        "cell_id": cells[rr],
        "cell_age_group": adata.obs.loc[cells[rr], "age_group"].to_numpy(),
        "junction_id": junc_ids[cc],
        "junction_counts": jv.astype(float),
        "atse_counts": av.astype(float),
        "psi_value": np.where(np.isnan(pv), np.nan, pv.astype(float)),
    })
    if gene_map is not None:
        df["gene_name"] = df["junction_id"].map(gene_map)
    df["event_id"] = event_id
    df["cell_type"] = cell_type
    return df


def delta_psi_from_result_df(result_df, young_label="young", old_label="old"):
    """Compute per-junction ΔPSI and ATSE-level coordination"""
    req = {"cell_id", "cell_age_group", "junction_id", "psi_value",
           "junction_counts", "atse_counts", "event_id"}
    missing = req - set(result_df.columns)
    if missing:
        raise ValueError(f"result_df missing required columns: {sorted(missing)}")
    
    df = result_df.copy()
    cell_type = result_df["cell_type"].iloc[0]
    
    # Per-junction per-age PSI stats
    agg = (df.groupby(["junction_id", "cell_age_group"], observed=True)
             .agg(psi_mean=("psi_value", "mean"),
                  psi_median=("psi_value", "median"),
                  n_cells=("psi_value", "count")))
    wide = agg.unstack("cell_age_group")
    
    def getcol(name, age):
        col = (name, age)
        return wide[col] if col in wide.columns else pd.Series(index=wide.index, dtype=float)
    
    psi_mean_y = getcol("psi_mean", young_label)
    psi_mean_o = getcol("psi_mean", old_label)
    psi_med_y = getcol("psi_median", young_label)
    psi_med_o = getcol("psi_median", old_label)
    n_y = getcol("n_cells", young_label).astype("Int64")
    n_o = getcol("n_cells", old_label).astype("Int64")
    
    d_mean = psi_mean_o - psi_mean_y
    d_med = psi_med_o - psi_med_y
    
    # Pseudobulk per junction-age
    sums = (df.groupby(["junction_id", "cell_age_group"], observed=True)
              .agg(junc_sum=("junction_counts", "sum"),
                   atse_sum=("atse_counts", "sum"))).unstack("cell_age_group")
    
    def getsum(name, age):
        col = (name, age)
        return sums[col] if col in sums.columns else pd.Series(index=sums.index, dtype=float)
    
    junc_sum_y = getsum("junc_sum", young_label)
    junc_sum_o = getsum("junc_sum", old_label)
    atse_sum_y = getsum("atse_sum", young_label)
    atse_sum_o = getsum("atse_sum", old_label)
    
    def safe_div(num, den):
        num = num.astype(float)
        den = den.astype(float)
        return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=(den > 0))
    
    pb_y = safe_div(junc_sum_y, atse_sum_y)
    pb_o = safe_div(junc_sum_o, atse_sum_o)
    d_pb = pb_o - pb_y
    
    # Junction-level table
    j_meta = df.drop_duplicates("junction_id").set_index("junction_id")
    junction_df = pd.DataFrame({"event_id": j_meta["event_id"]})
    if "gene_name" in j_meta.columns:
        junction_df["gene_name"] = j_meta["gene_name"]
    
    junction_df["n_cells_young"] = n_y
    junction_df["n_cells_old"] = n_o
    junction_df["psi_mean_young"] = psi_mean_y
    junction_df["psi_mean_old"] = psi_mean_o
    junction_df["delta_psi_mean"] = d_mean
    junction_df["psi_median_young"] = psi_med_y
    junction_df["psi_median_old"] = psi_med_o
    junction_df["delta_psi_median"] = d_med
    junction_df["pb_young"] = pb_y
    junction_df["pb_old"] = pb_o
    junction_df["delta_pb"] = d_pb
    
    junction_df = junction_df.reset_index()
    junction_df.insert(0, "cell_type", cell_type)
    
    # ATSE-level summaries
    def sign_concordance(series):
        s = np.sign(series.to_numpy(dtype=float))
        nz = s[s != 0]
        if nz.size == 0:
            return np.nan
        p_pos = (nz > 0).mean()
        return max(p_pos, 1.0 - p_pos)
    
    def opposite_sign_pair(series):
        s = np.sign(series.to_numpy(dtype=float))
        if s.size != 2:
            return np.nan
        return bool(np.array_equal(np.sort(s), np.array([-1.0, 1.0])))
    
    atse_df = (junction_df
               .groupby("event_id", observed=True)
               .agg(n_junctions=("junction_id", "count"),
                    mean_delta_mean=("delta_psi_mean", "mean"),
                    median_delta_median=("delta_psi_median", "median"),
                    mean_abs_delta=("delta_psi_mean", lambda x: float(np.mean(np.abs(x)))),
                    std_delta=("delta_psi_mean", "std"),
                    sign_concordance=("delta_psi_mean", sign_concordance),
                    mean_delta_pb=("delta_pb", "mean"),
                    median_delta_pb=("delta_pb", "median"),
                    opposite_sign_pair=("delta_psi_mean", opposite_sign_pair))
               .reset_index())
    atse_df.insert(0, "cell_type", cell_type)
    
    return junction_df, atse_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--celltype', required=True, help='Cell type to process')
    parser.add_argument('--today', required=True, help='Date string (YYYYMMDD)')
    args = parser.parse_args()
    
    celltype = args.celltype
    today = args.today
    
    # Paths
    output_dir = "/gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/Mouse_Splicing_Foundation/Figures/FIGURE1"
    run_dir = f"{output_dir}/figure1_{today}"
    intermediate_dir = f"{run_dir}/intermediate"
    
    print(f"\n{'='*60}")
    print(f"Processing cell type: {celltype}")
    print(f"{'='*60}\n")
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    splice_adata = ad.read_h5ad("/gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/Mouse_Splicing_Foundation/Figures/FIGURE1/figure1_20260105/preprocessed_data_20260105.h5ad")
    event_ids_file = "/gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/Mouse_Splicing_Foundation/Figures/FIGURE1/figure1_20260105/event_ids_20260105.txt"
    #data = np.load(event_ids_file, allow_pickle=True)
    data = pd.read_csv(event_ids_file, header=None)
    event_ids = data[0].tolist()

    print(f"Loaded data: {splice_adata.n_obs:,} cells, {splice_adata.n_vars:,} junctions")
    print(f"Processing {len(event_ids):,} event IDs")

    # Build cache
    print("Building cache...")
    cache = _ATSECache(splice_adata, groupby_col="tissue_celltype")
    
    # Process all events for this cell type
    all_junc, all_atse = [], []
    
    print(f"Processing events for {celltype}...")
    for ev in tqdm(event_ids):
        result_df = extract_atse_data_fast_cached(
            adata=splice_adata,
            event_id=ev,
            cell_type=celltype,
            cache=cache
        )
        if result_df.empty:
            continue
        
        jdf, adf = delta_psi_from_result_df(result_df, young_label="young", old_label="old")
        if not jdf.empty:
            all_junc.append(jdf)
        if not adf.empty:
            all_atse.append(adf)
    
    # Concatenate and save results for this cell type
    if all_junc:
        junction_df = pd.concat(all_junc, ignore_index=True)
        atse_df = pd.concat(all_atse, ignore_index=True)
        
        # Sanitize filename (replace special characters)
        safe_celltype = celltype.replace('/', '_').replace(' ', '_')
        
        junction_file = f"{intermediate_dir}/junction_{safe_celltype}.csv"
        atse_file = f"{intermediate_dir}/atse_{safe_celltype}.csv"
        result_file = f"{intermediate_dir}/result_df_{safe_celltype}.csv"
        
        junction_df.to_csv(junction_file, index=False)
        atse_df.to_csv(atse_file, index=False)
        result_df.to_csv(result_file, index=False)
        
        print(f"\nSaved results for {celltype}:")
        print(f"  Junctions: {len(junction_df):,} rows -> {junction_file}")
        print(f"  ATSEs: {len(atse_df):,} rows -> {atse_file}")
    else:
        print(f"\nNo data found for {celltype}")
    
    print(f"\nCompleted processing {celltype}")


if __name__ == "__main__":
    main()