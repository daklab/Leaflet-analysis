#!/usr/bin/env python
"""
Step 08-01b: Audit the pseudobulk design from `obs` alone -- no matrix read.

Run this BEFORE 01_pseudobulk_counts.py. It reads only the h5ad `obs` group (seconds,
~no memory) and answers the questions that determine whether the LeafCutter runs will
work at all:

  * exactly which pseudobulk samples will exist, and their labels
  * how many replicates each tissue_cell_type gets, and how many clear the n>=4 floor
    below which LeafCutter's own error message says its p-value calibration is unchecked
  * how many cells the --min-cells threshold discards
  * whether any two groups would collide on the same sanitised label

Pseudobulk design
-----------------
The unit is `tissue_cell_type`, stratified by whichever replicate/covariate fields the
dataset actually has:

    sample_name = tissue_cell_type [_sex] [_age] [_donor]

Because tissue_cell_type already carries the tissue, every contrast between two
tissue_cell_types is within-tissue by construction -- there is no pooling decision to
make and no tissue covariate to fit.

  tabula_muris_senis : sex, age, donor    -> Heart_FIBROBLAST_female_18m_18_46_F
  allen_brain        : sex                -> Brain_Isocortex_Visual_EXCITATORY_NEURON_male

Allen contributes no donor and no age on purpose. Its `mouse.id` comes from
`external_donor_name_label` (519 values over 54,571 cells, median 46 cells each -- too
thin to act as a biological replicate), and its `age` is hardcoded "2m" in
04_align/cell_id_cleaners/allen_brain_mouse.py:73 because Allen's metadata has no age
field. Those fields are omitted rather than faked, so `age` can never be picked up as a
covariate on the Allen side.

Usage:
    python 01b_audit_pseudobulk_design.py \\
        --splice-h5ad <train_splice_model_ready_2026-06-17.h5ad> \\
        --output-dir  <dir> \\
        --min-cells   25
"""

import argparse
import os

import h5py
import numpy as np
import pandas as pd

from lc_utils import pseudobulk_sample_name

# Fields used to stratify each dataset, in label order. Anything absent is skipped.
DATASET_KEYS = {
    "tabula_muris_senis": ["sex", "age", "mouse.id"],
    "allen_brain":        ["sex"],
}
DEFAULT_KEYS = ["sex"]

# LeafCutter's own message: "we have only carefully checked the calibration of leafcutter
# p-values down to n=4 samples per group".
MIN_SAMPLES_PER_GROUP = 4

OBS_COLS = ["dataset", "mouse.id", "tissue", "subtissue",
            "tissue_cell_type", "broad_cell_type", "sex", "age"]


def read_obs(h5ad_path, columns):
    """Read named obs columns straight out of the h5, decoding categoricals by hand.

    anndata's own reader would pull far more than we need; this touches only /obs.
    """
    out = {}
    with h5py.File(h5ad_path, "r") as f:
        obs = f["obs"]
        for name in columns:
            if name not in obs:
                out[name] = None
                continue
            node = obs[name]
            if isinstance(node, h5py.Group):          # categorical
                cats = np.array([c.decode() if isinstance(c, bytes) else c
                                 for c in node["categories"][:]])
                out[name] = pd.Categorical.from_codes(node["codes"][:], cats).astype(str)
            else:
                vals = node[:]
                out[name] = np.array([v.decode() if isinstance(v, bytes) else v
                                      for v in vals])
    n = max(len(v) for v in out.values() if v is not None)
    return pd.DataFrame({k: (v if v is not None else [""] * n) for k, v in out.items()})


def build_sample_table(obs, min_cells):
    """One row per prospective pseudobulk sample, for every dataset."""
    frames = []
    for dataset, sub in obs.groupby("dataset", observed=True):
        keys = DATASET_KEYS.get(dataset, DEFAULT_KEYS)
        keys = [k for k in keys if k in sub.columns]
        group_cols = ["tissue_cell_type", "tissue", "broad_cell_type"] + keys

        g = sub.groupby(group_cols, observed=True).size().rename("n_cells").reset_index()
        g["dataset"] = dataset
        g["sex"]   = g["sex"]      if "sex"      in g.columns else ""
        g["age"]   = g["age"]      if "age"      in g.columns else ""
        g["donor"] = g["mouse.id"] if "mouse.id" in g.columns else ""

        g["sample_name"] = [
            pseudobulk_sample_name(r.tissue_cell_type,
                                   r.sex   if "sex"      in keys else None,
                                   r.age   if "age"      in keys else None,
                                   r.donor if "mouse.id" in keys else None)
            for r in g.itertuples()
        ]
        frames.append(g[["sample_name", "dataset", "tissue", "tissue_cell_type",
                         "broad_cell_type", "sex", "age", "donor", "n_cells"]])

    allg = pd.concat(frames, ignore_index=True)
    kept = allg[allg["n_cells"] >= min_cells].copy()
    return allg, kept.sort_values(["dataset", "tissue_cell_type", "sample_name"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splice-h5ad", required=True)
    ap.add_argument("--output-dir",  required=True)
    ap.add_argument("--min-cells",   type=int, default=25)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[1] Reading obs ...", flush=True)
    obs = read_obs(args.splice_h5ad, OBS_COLS)
    print(f"  {len(obs):,} cells", flush=True)

    print(f"\n[2] Building pseudobulk sample table (--min-cells {args.min_cells}) ...",
          flush=True)
    allg, kept = build_sample_table(obs, args.min_cells)

    dropped = allg[allg["n_cells"] < args.min_cells]
    print(f"  samples kept    : {len(kept):,}")
    print(f"  groups dropped  : {len(dropped):,} "
          f"({dropped['n_cells'].sum():,} cells, "
          f"{100 * dropped['n_cells'].sum() / len(obs):.1f}% of all cells)")
    for ds, sub in kept.groupby("dataset"):
        keys = DATASET_KEYS.get(ds, DEFAULT_KEYS)
        print(f"    {ds:<20} {len(sub):>4} samples  "
              f"({sub['tissue_cell_type'].nunique()} tissue_cell_types, key = "
              f"tissue_cell_type x {' x '.join(keys)})")

    # A collision would make two biologically distinct groups share one counts column.
    collisions = kept["sample_name"][kept["sample_name"].duplicated(keep=False)]
    if len(collisions):
        print(f"\n  ERROR: {collisions.nunique()} sample-name collisions after "
              f"sanitisation: {sorted(set(collisions))[:5]}")
    else:
        print(f"  sample-name collisions: 0")

    print(f"\n[3] Replicates per tissue_cell_type ...", flush=True)
    per_tct = (kept.groupby(["dataset", "tissue_cell_type"])
                   .agg(n_samples=("sample_name", "size"),
                        n_donors=("donor", lambda s: s[s != ""].nunique()),
                        median_cells=("n_cells", "median"))
                   .reset_index()
                   .sort_values(["dataset", "n_samples"], ascending=[True, False]))
    testable = per_tct[per_tct["n_samples"] >= MIN_SAMPLES_PER_GROUP]
    print(f"  tissue_cell_types with >= {MIN_SAMPLES_PER_GROUP} samples: "
          f"{len(testable)} / {len(per_tct)}")
    for ds, sub in testable.groupby("dataset"):
        print(f"    {ds:<20} {len(sub):>3} testable")

    print(f"\n  Top 15 by replicate count:")
    print(testable.head(15).to_string(index=False))

    # Contrast counting: pairs are formed within dataset, so tissue is held fixed by
    # tissue_cell_type itself.
    print(f"\n[4] Eligible pairwise contrasts (within dataset, both sides >= "
          f"{MIN_SAMPLES_PER_GROUP} samples) ...", flush=True)
    for ds, sub in testable.groupby("dataset"):
        n = len(sub)
        print(f"    {ds:<20} {n * (n - 1) // 2:>5} pairs from {n} tissue_cell_types")

    out_samples = os.path.join(args.output_dir, "pseudobulk_labels.tsv")
    out_per_tct = os.path.join(args.output_dir, "pseudobulk_replicates_per_celltype.tsv")
    kept.to_csv(out_samples, sep="\t", index=False)
    per_tct.to_csv(out_per_tct, sep="\t", index=False)
    print(f"\n=== Done ===")
    print(f"  sample table    -> {out_samples}")
    print(f"  replicate table -> {out_per_tct}")
    print(f"\nNext: 01_pseudobulk_counts.py --min-cells {args.min_cells}")


if __name__ == "__main__":
    main()
