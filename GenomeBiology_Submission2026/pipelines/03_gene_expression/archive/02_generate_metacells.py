#!/usr/bin/env python
"""
Step 8b: Generate metacells for sc vs. sn correction.

Aggregates cells by (broad_cell_type, dataset) into pseudo-bulk metacells
that are used to train the sc/sn regression correction in the next step.

Usage:
    python pipelines/03_gene_expression/02_generate_metacells.py config/mouse_smartseq2.yaml
"""

import sys
import os
import glob
import datetime
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def latest_file(pattern: str) -> str:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")
    return files[-1]


def make_metacells(adata: ad.AnnData, groupby: list, min_cells: int = 10) -> ad.AnnData:
    """Aggregate log_norm expression by group into metacell means."""
    records = []
    for key, idx in adata.obs.groupby(groupby).groups.items():
        if len(idx) < min_cells:
            continue
        sub = adata[idx]
        X = sub.layers["log_norm"]
        if hasattr(X, "toarray"):
            X = X.toarray()
        mean_expr = X.mean(axis=0)
        obs_row = {g: (key if isinstance(key, str) else key[i]) for i, g in enumerate(groupby)}
        obs_row["n_cells"] = len(idx)
        records.append((obs_row, mean_expr))

    meta_obs = pd.DataFrame([r[0] for r in records])
    meta_X = np.vstack([r[1] for r in records])

    meta = ad.AnnData(
        X=csr_matrix(meta_X),
        obs=meta_obs,
        var=adata.var.copy(),
    )
    meta.layers["log_norm"] = csr_matrix(meta_X)
    print(f"  Generated {len(meta)} metacells from {adata.n_obs} cells")
    return meta


def main():
    if len(sys.argv) < 2:
        print("Usage: python 02_generate_metacells.py <config.yaml>")
        sys.exit(1)

    cfg = load_config(sys.argv[1])
    out_dir = cfg["gene_expression"]["output_dir"]
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    ge_path = latest_file(os.path.join(out_dir, "tms_ab_exons_combo_ge_adata_*.h5ad"))
    print(f"Loading GE AnnData from {ge_path}")
    adata = ad.read_h5ad(ge_path)

    if "log_norm" not in adata.layers:
        raise ValueError("log_norm layer not found — run 01_prepare_counts.py first")

    # Load broad cell type mapping and apply
    map_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        cfg.get("cell_type_map", "cell_type_maps/mouse_celltype_map.json"),
    )
    if os.path.exists(map_path):
        import json
        with open(map_path) as f:
            ct_map = json.load(f)
        adata.obs["broad_cell_type"] = adata.obs["cell_ontology_class"].map(ct_map).fillna(
            adata.obs["cell_ontology_class"]
        )
    else:
        print(f"  WARNING: cell type map not found at {map_path}, skipping broad mapping")
        adata.obs["broad_cell_type"] = adata.obs.get("cell_ontology_class", "unknown")

    meta = make_metacells(adata, groupby=["broad_cell_type", "dataset"])

    out_path = os.path.join(out_dir, f"metacells_{today}.h5ad")
    meta.write_h5ad(out_path, compression="lzf")
    print(f"Saved metacells → {out_path}")


if __name__ == "__main__":
    main()
