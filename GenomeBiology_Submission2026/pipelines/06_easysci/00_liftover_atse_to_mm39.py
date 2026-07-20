#!/usr/bin/env python
"""
Step 12-pre: Liftover TMS ATSE file from mm10 → mm39 for EasySci junction mapping.

EasySci reads were aligned to mm39 (gencode.vM27); the TMS model uses mm10 (gencode.vM19).
This script lifts over TMS junction coordinates mm10→mm39 using pyliftover, preserving
the original mm10 junction_id as 'mouse_foundation_junction_id'. The output file is
then used by 01_prep_easysci_for_model.py to map EasySci mm39 junctions to model junctions.

Chain file (already on cluster):
  /gpfs/commons/projects/knowles_singlecell_splicing/TabulaSenis/liftover/mm10ToMm39.over.chain.gz

Usage:
    python 00_liftover_atse_to_mm39.py \\
        --atse-file <MOUSE_SMARTSEQ2_ATSE_FILE_unanno_also_2026-05-12_*.txt.gz> \\
        --output    <MOUSE_SMARTSEQ2_ATSE_FILE_unanno_also_2026-05-12_*_lifted_mm39.txt.gz>

    # Can run on login node (fast — no heavy compute):
    python pipelines/06_easysci/00_liftover_atse_to_mm39.py \\
        --atse-file $ATSE_FILE \\
        --output    ${ATSE_FILE%.txt.gz}_lifted_mm39.txt.gz
"""

import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pyliftover import LiftOver

CHAIN_FILE = "/gpfs/commons/projects/knowles_singlecell_splicing/TabulaSenis/liftover/mm10ToMm39.over.chain.gz"


def lift_junction(lo, chrom, start, end, strand, junction_id):
    """Liftover one junction. Returns dict or None if liftover fails."""
    lifted_start = lo.convert_coordinate(chrom, start)
    lifted_end   = lo.convert_coordinate(chrom, end - 1)

    if not lifted_start or not lifted_end:
        return None

    new_chrom_s, new_start, _, _ = lifted_start[0]
    new_chrom_e, new_end,   _, _ = lifted_end[0]

    if new_chrom_s != new_chrom_e:
        return None  # junction split across chromosomes — discard

    return {
        "chrom":  new_chrom_s,
        "start":  int(new_start),
        "end":    int(new_end) + 1,
        "strand": strand,          # preserve original strand
        "mouse_foundation_junction_id": junction_id,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--atse-file", required=True,
                        help="TMS ATSE file in mm10 (e.g. MOUSE_SMARTSEQ2_ATSE_FILE_*.txt.gz)")
    parser.add_argument("--output", required=True,
                        help="Output path for mm39-lifted ATSE file (.txt.gz)")
    parser.add_argument("--chain-file", default=CHAIN_FILE,
                        help=f"mm10→mm39 chain file (default: {CHAIN_FILE})")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f"Loading chain file: {args.chain_file}")
    lo = LiftOver(args.chain_file)

    print(f"Reading ATSE file: {args.atse_file}")
    atse_df = pd.read_csv(args.atse_file, sep="\t")
    print(f"  {len(atse_df):,} rows, {atse_df['junction_id'].nunique():,} unique junction_ids")

    # Liftover each unique junction (avoid redundant work for shared junctions)
    unique_juncs = atse_df[["chrom", "start", "end", "strand", "junction_id"]].drop_duplicates(
        subset="junction_id"
    )
    print(f"  Lifting {len(unique_juncs):,} unique junctions...")

    lifted = []
    failed = 0
    for _, row in tqdm(unique_juncs.iterrows(), total=len(unique_juncs)):
        result = lift_junction(lo, row["chrom"], row["start"], row["end"],
                               row["strand"], row["junction_id"])
        if result:
            lifted.append(result)
        else:
            failed += 1

    print(f"  Lifted: {len(lifted):,}  |  Failed: {failed:,}  "
          f"({100*failed/len(unique_juncs):.1f}% loss)")

    lifted_df = pd.DataFrame(lifted)

    # Build mm39 junction_id from lifted coordinates
    lifted_df["junction_id"] = (
        lifted_df["chrom"] + "_" +
        lifted_df["start"].astype(str) + "_" +
        lifted_df["end"].astype(str) + "_" +
        lifted_df["strand"]
    )

    # Sanity: how many lifted IDs are identical to original (shouldn't happen mm10→mm39)
    same = (lifted_df["junction_id"] == lifted_df["mouse_foundation_junction_id"]).sum()
    if same > 0:
        print(f"  WARNING: {same:,} junctions have same ID after liftover — "
              f"coordinate shift may be near-zero for these")

    # Merge lifted coordinates back onto full ATSE table
    # Drop old chrom/start/end/strand columns (mm10) then join lifted (mm39) + new junction_id
    atse_no_coords = atse_df.drop(columns=["chrom", "start", "end", "strand"])
    result_df = atse_no_coords.merge(
        lifted_df[["mouse_foundation_junction_id", "chrom", "start", "end", "strand", "junction_id"]],
        left_on="junction_id",
        right_on="mouse_foundation_junction_id",
        how="inner",
        suffixes=("_orig", "")
    )
    # Drop the _orig junction_id (mm10), keep the new mm39 one
    if "junction_id_orig" in result_df.columns:
        result_df = result_df.drop(columns=["junction_id_orig"])

    print(f"\nResult: {len(result_df):,} rows ({atse_df['junction_id'].nunique():,} → "
          f"{result_df['junction_id'].nunique():,} unique junction_ids after liftover)")
    print(f"  mouse_foundation_junction_id sample: "
          f"{result_df['mouse_foundation_junction_id'][:3].tolist()}")
    print(f"  mm39 junction_id sample: {result_df['junction_id'][:3].tolist()}")

    print(f"\nWriting → {args.output}")
    result_df.to_csv(args.output, sep="\t", index=False, compression="gzip")
    print("Done.")
    print(f"\nUse this file as --liftover-atse in 01_prep_easysci_for_model.py")


if __name__ == "__main__":
    main()
