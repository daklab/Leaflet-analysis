#!/usr/bin/env python3
"""Make MuData objects for all metacell sizes, aligning to ss2 foundation reference."""

import gc
import mudata as md
import anndata as ad
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, issparse

base = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA"

metacell_configs = {
    "default": {
        "splice": f"{base}/MetaCells/SpliceVI/20260210/RH/ATSEmap/junction_processing_20260213/anndatas/merged_anndata.h5ad",
        "ge": f"{base}/metacell_RT_anndata.h5ad",
    },
    "500": {
        "splice": f"{base}/MetaCells/SpliceVI_500/ATSEmap/junction_processing_20260318/anndatas/merged_anndata.h5ad",
        "ge": f"{base}/metacell_RT_anndata_500.h5ad",
    },
    "1000": {
        "splice": f"{base}/MetaCells/SpliceVI_1000/ATSEmap/junction_processing_20260318/anndatas/merged_anndata.h5ad",
        "ge": f"{base}/metacell_RT_anndata_1000.h5ad",
    },
    "2000": {
        "splice": f"{base}/MetaCells/SpliceVI_2000/ATSEmap/junction_processing_20260318/anndatas/merged_anndata.h5ad",
        "ge": f"{base}/metacell_RT_anndata_2000.h5ad",
    },
    "4000": {
        "splice": f"{base}/MetaCells/SpliceVI_4000/ATSEmap/junction_processing_20260318/anndatas/merged_anndata.h5ad",
        "ge": f"{base}/metacell_RT_anndata_4000.h5ad",
    },
}


def align_splice_adata_to_mudata(splice_adata, ss2_splicing):
    ss2_junction_ids = ss2_splicing.var["junction_id"].values
    n_obs, n_junctions_ref = splice_adata.n_obs, len(ss2_junction_ids)

    junc_to_col = {}
    for col_idx, jid in enumerate(splice_adata.var["junction_id"].values):
        if jid not in junc_to_col:
            junc_to_col[jid] = col_idx

    n_matched = sum(1 for jid in ss2_junction_ids if jid in junc_to_col)

    def _build_aligned_matrix(orig_X):
        if hasattr(orig_X, "toarray"):
            orig_X = orig_X.toarray()
        out = np.zeros((n_obs, n_junctions_ref), dtype=orig_X.dtype)
        for j in range(n_junctions_ref):
            jid = ss2_junction_ids[j]
            if jid in junc_to_col:
                out[:, j] = orig_X[:, junc_to_col[jid]]
        return csr_matrix(out) if issparse(splice_adata.X) else out

    new_X = _build_aligned_matrix(splice_adata.X)
    layers = {}
    for name, layer in splice_adata.layers.items():
        if layer.shape == (n_obs, splice_adata.n_vars):
            layers[name] = _build_aligned_matrix(layer)

    new_adata = ad.AnnData(
        new_X,
        obs=splice_adata.obs.copy(),
        var=ss2_splicing.var.copy(),
        layers=layers,
    )
    return new_adata, n_matched


def align_ge_adata_to_mudata(ge_adata, ss2_rna):
    ss2_gene_ids = ss2_rna.var["gene_id"].values
    n_obs, n_genes_ref = ge_adata.n_obs, len(ss2_gene_ids)

    if "gene_id" in ge_adata.var.columns:
        gene_ids = ge_adata.var["gene_id"].astype(str).values
    else:
        gene_ids = ge_adata.var_names.astype(str).values
    gene_to_col = {}
    for col_idx, gid in enumerate(gene_ids):
        gid_plain = gid.split(".")[0] if "." in gid else gid
        if gid_plain not in gene_to_col:
            gene_to_col[gid_plain] = col_idx
        if gid not in gene_to_col:
            gene_to_col[gid] = col_idx

    def _get_col(ss2_gid):
        s = str(ss2_gid)
        gid_plain = s.split(".")[0] if "." in s else s
        return gene_to_col.get(gid_plain) or gene_to_col.get(s)

    n_matched = sum(1 for j in range(n_genes_ref) if _get_col(ss2_gene_ids[j]) is not None)

    def _build_aligned_matrix(orig_X):
        if issparse(orig_X):
            orig_X = orig_X.toarray()
        out = np.zeros((n_obs, n_genes_ref), dtype=orig_X.dtype)
        for j in range(n_genes_ref):
            col_idx = _get_col(ss2_gene_ids[j])
            if col_idx is not None:
                out[:, j] = orig_X[:, col_idx]
        return csr_matrix(out) if issparse(ge_adata.X) else out

    new_X = _build_aligned_matrix(ge_adata.X)
    layers = {}
    for name, layer in ge_adata.layers.items():
        if layer.shape == (n_obs, ge_adata.n_vars):
            layers[name] = _build_aligned_matrix(layer)

    new_adata = ad.AnnData(
        new_X,
        obs=ge_adata.obs.copy(),
        var=ss2_rna.var.copy(),
        layers=layers,
    )
    return new_adata, n_matched


def main():
    # Load ss2 reference
    print("Loading ss2 reference mudata...", flush=True)
    ss2 = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/train_70_30_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_UPDATEDOBS.h5mu"
    ss2_mudata = md.read_h5mu(ss2)
    ss2_rna = ss2_mudata["rna"]
    ss2_splicing = ss2_mudata["splicing"]
    print(f"ss2 rna: {ss2_rna.n_vars} genes, ss2 splicing: {ss2_splicing.n_vars} junctions", flush=True)

    # Load junction ID mapping
    print("Loading junction ID mapping...", flush=True)
    INTRON_CLUSTS_FILE = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper/ATSE_files/MOUSE_FOUNDATION_ATSE_FILE_unanno_also_2025-10-01_21-36-40_lifted_mm39.txt.gz"
    atse_df = pd.read_csv(INTRON_CLUSTS_FILE, sep="\t")
    easysci_to_foundation = atse_df.drop_duplicates("junction_id").set_index("junction_id")["mouse_foundation_junction_id"]
    print(f"Loaded {len(easysci_to_foundation)} junction mappings (mm39 -> foundation)", flush=True)

    output_base = base

    for size_name, paths in metacell_configs.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Processing metacell size: {size_name}", flush=True)
        print(f"{'='*60}", flush=True)

        # Load splice and GE anndatas
        splice_adata = ad.read_h5ad(paths["splice"])
        ge_adata = ad.read_h5ad(paths["ge"])
        print(f"  Loaded splice: {splice_adata.shape}, ge: {ge_adata.shape}", flush=True)

        # Convert junction IDs from mm39 to foundation
        splice_adata.var["junction_id_easysci_mm39"] = splice_adata.var["junction_id"].astype(str).copy()
        splice_adata.var["junction_id"] = (
            splice_adata.var["junction_id"].astype(str)
            .map(easysci_to_foundation)
            .fillna(splice_adata.var["junction_id_easysci_mm39"])
        )
        n_mapped = (splice_adata.var["junction_id"] != splice_adata.var["junction_id_easysci_mm39"]).sum()
        print(f"  Converted {n_mapped} junction_ids to foundation; {splice_adata.n_vars - n_mapped} unchanged", flush=True)

        # Align to ss2 reference
        splice_aligned, n_matched_junctions = align_splice_adata_to_mudata(splice_adata, ss2_splicing)
        ge_aligned, n_matched_genes = align_ge_adata_to_mudata(ge_adata, ss2_rna)
        print(f"  Splice aligned: {splice_aligned.shape} ({n_matched_junctions} matched, {ss2_splicing.n_vars - n_matched_junctions} zero-filled)", flush=True)
        print(f"  GE aligned: {ge_aligned.shape} ({n_matched_genes} matched, {ss2_rna.n_vars - n_matched_genes} zero-filled)", flush=True)

        # Set obs names to metacell IDs
        splice_aligned.obs.reset_index(drop=True, inplace=True)
        ge_aligned.obs_names = ge_aligned.obs["Main_cluster_name_wkmeans"].values
        splice_aligned.obs_names = splice_aligned.obs["Main_cluster_name_wkmeans"].values

        # Intersect and align obs
        common_obs = ge_aligned.obs_names.intersection(splice_aligned.obs_names)
        common_ordered = [x for x in ge_aligned.obs_names if x in common_obs]
        ge_aligned = ge_aligned[common_ordered].copy()
        splice_aligned = splice_aligned[common_ordered].copy()
        print(f"  Common cells: {len(common_ordered)}", flush=True)

        # Build MuData
        mudata = md.MuData({"rna": ge_aligned, "splicing": splice_aligned})

        # Save
        suffix = f"_{size_name}" if size_name != "default" else ""
        mudata_path = f"{output_base}/mudata_easysci{suffix}.h5mu"
        mudata.write_h5mu(mudata_path)
        print(f"  Saved: {mudata_path}", flush=True)
        print(f"  MuData: {mudata.n_obs} cells, rna={mudata['rna'].n_vars} genes, splicing={mudata['splicing'].n_vars} junctions", flush=True)

        # Also save individual anndatas
        ge_aligned.write_h5ad(f"{output_base}/SpliceVI_EasySci_input_ge_adata{suffix}.h5ad")
        splice_aligned.write_h5ad(f"{output_base}/SpliceVI_EasySci_input_splice_adata{suffix}.h5ad")

        del splice_adata, ge_adata, splice_aligned, ge_aligned, mudata
        gc.collect()

    print("\nDone! All metacell sizes processed.", flush=True)


if __name__ == "__main__":
    main()
