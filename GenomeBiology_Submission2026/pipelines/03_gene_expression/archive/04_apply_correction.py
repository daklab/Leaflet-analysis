#!/usr/bin/env python
"""
Step 8d: Apply sc vs. sn regression correction to gene expression data.

Uses a pre-trained linear correction model (from notebook step 8c) to adjust
single-nucleus Allen Brain data toward single-cell TMS expression levels.
The corrected AnnData is saved for scVI training.

Usage:
    python pipelines/03_gene_expression/04_apply_correction.py config/mouse_smartseq2.yaml \
        --model <correction_model.pkl> --ge <ge_adata.h5ad>
"""

import sys
import os
import glob
import datetime
import argparse
import pickle
import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def latest_file(pattern: str) -> str:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching: {pattern}")
    return files[-1]


def apply_correction(ge_adata: ad.AnnData, model: dict) -> ad.AnnData:
    """
    Apply gene-wise linear correction: X_corrected = X - (coef * is_snRNA + intercept).

    model dict expected keys:
      - 'coef': array of shape (n_genes,)
      - 'intercept': array of shape (n_genes,) or scalar
      - 'is_snRNA_col': obs column name indicating single-nucleus origin
    """
    coef      = np.array(model["coef"])
    intercept = np.array(model.get("intercept", 0.0))
    sn_col    = model.get("is_snRNA_col", "dataset")
    sn_value  = model.get("sn_value", "allen_brain_exons")

    is_sn = (ge_adata.obs[sn_col] == sn_value).values.astype(float)

    layer = "log_norm"
    X = ge_adata.layers[layer]
    if hasattr(X, "toarray"):
        X = X.toarray()

    correction = np.outer(is_sn, coef) + intercept
    X_corrected = X - correction
    X_corrected = np.clip(X_corrected, 0, None)

    ge_adata = ge_adata.copy()
    ge_adata.layers["log_norm_corrected"] = csr_matrix(X_corrected)
    # Use corrected as primary log_norm for downstream scVI
    ge_adata.layers["log_norm"] = csr_matrix(X_corrected)
    return ge_adata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--model", required=True, help="Path to correction model .pkl")
    parser.add_argument("--ge",    required=True, help="Path to combined GE AnnData")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    out_dir = cfg["gene_expression"]["output_dir"]
    today   = datetime.datetime.now().strftime("%Y-%m-%d")

    print(f"Loading GE AnnData from {args.ge}")
    adata = ad.read_h5ad(args.ge)

    print(f"Loading correction model from {args.model}")
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    print("Applying correction...")
    adata = apply_correction(adata, model)

    out_path = os.path.join(out_dir, f"ge_adata_corrected_{today}.h5ad")
    adata.write_h5ad(out_path, compression="lzf")
    print(f"Saved corrected GE AnnData ({adata.shape}) → {out_path}")


if __name__ == "__main__":
    main()
