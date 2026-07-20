#!/usr/bin/env python
"""
03_prepare_atlas_metadata.py — build RNA-atlas cell metadata and (optionally) attach
it to the merged AnnData.

The atlas AnnData obs is keyed by SRR accession (05_make_anndata.py's extract_cell_id
turns SRR*_junctions_with_barcodes.bed → 'SRR…'), which joins 1:1 to the manifest 'srr'
column. This adds the DS grouping variables from fastqs/manifest.csv:
    cell_type      ← source_name   (160 primary cell types; ~14 replicate samples each)
    organ_system   ← organ_system  (23 systems; coarser grouping)
    gsm            ← gsm            (316 GEO samples; groups replicate SRRs)
    sample_type    ← sample_type    (all 'cell type' after primary-cell filtering)

Only sample_type == 'cell type' rows are kept (primary cells; excludes bulk mixed
tissues and cell lines), matching config/rna_atlas.yaml's junction_filelist.

Usage:
    # write the metadata CSV only
    python 03_prepare_atlas_metadata.py <manifest_csv> <out_csv>

    # ALSO attach onto a merged AnnData obs. By default this writes a NEW file
    # (<input>_with_metadata.h5ad) and never touches the input — an interrupted
    # write then costs nothing. Pass --in-place to overwrite the input instead.
    python 03_prepare_atlas_metadata.py <manifest_csv> <out_csv> --anndata <merged_anndata.h5ad>
    python 03_prepare_atlas_metadata.py <manifest_csv> <out_csv> --anndata <in.h5ad> --anndata-out <out.h5ad>
"""

import argparse
import os
import pandas as pd


def build_metadata(manifest_csv: str) -> pd.DataFrame:
    m = pd.read_csv(manifest_csv)
    m = m[m["sample_type"] == "cell type"].copy()
    meta = pd.DataFrame({
        "cell_id":      m["srr"].astype(str),
        "cell_type":    m["source_name"].astype(str),
        "organ_system": m["organ_system"].astype(str),
        "gsm":          m["gsm"].astype(str),
        "sample_type":  m["sample_type"].astype(str),
    }).set_index("cell_id")
    if meta.index.duplicated().any():
        dups = meta.index[meta.index.duplicated()].tolist()
        raise ValueError(f"Duplicate SRR ids in manifest: {dups[:5]} …")
    print(f"[metadata] {len(meta):,} primary cell-type samples | "
          f"{meta['cell_type'].nunique()} cell types | "
          f"{meta['organ_system'].nunique()} organ systems | "
          f"{meta['gsm'].nunique()} GSMs")
    return meta


def attach_to_anndata(anndata_path: str, meta: pd.DataFrame, out_path: str):
    import anndata as ad
    A = ad.read_h5ad(anndata_path)
    obs_ids = A.obs_names.astype(str)
    aligned = meta.reindex(obs_ids)
    n_missing = int(aligned["cell_type"].isna().sum())
    if n_missing:
        miss = obs_ids[aligned["cell_type"].isna()][:5].tolist()
        print(f"[attach] WARNING: {n_missing:,}/{A.n_obs:,} obs have no manifest row "
              f"(e.g. {miss}). Their metadata columns will be NaN.")
    for col in ["cell_type", "organ_system", "gsm", "sample_type"]:
        A.obs[col] = pd.Categorical(aligned[col].values)
    # Write to a temp path then atomically rename, so an interrupted write can never
    # leave a truncated h5ad at out_path (AnnData writes X before layers; a partial
    # file loses layer encoding-type attrs and fails to read).
    tmp = out_path + ".tmp"
    A.write_h5ad(tmp)
    os.replace(tmp, out_path)
    print(f"[attach] wrote metadata onto {A.n_obs:,} obs → {out_path}")
    print(f"[attach] matched {A.n_obs - n_missing:,} | missing {n_missing:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest_csv")
    ap.add_argument("out_csv")
    ap.add_argument("--anndata", default=None,
                    help="Optional merged AnnData to attach metadata onto (read-only).")
    ap.add_argument("--anndata-out", default=None,
                    help="Where to write the annotated AnnData. "
                         "Default: <input>_with_metadata.h5ad (input left untouched).")
    ap.add_argument("--in-place", action="store_true",
                    help="Overwrite the input AnnData instead of writing a new file.")
    args = ap.parse_args()

    meta = build_metadata(args.manifest_csv)
    meta.to_csv(args.out_csv)
    print(f"[metadata] → {args.out_csv}")

    if args.anndata:
        if args.in_place:
            out_path = args.anndata
        elif args.anndata_out:
            out_path = args.anndata_out
        else:
            root, ext = os.path.splitext(args.anndata)
            out_path = f"{root}_with_metadata{ext}"
        attach_to_anndata(args.anndata, meta, out_path)


if __name__ == "__main__":
    main()
