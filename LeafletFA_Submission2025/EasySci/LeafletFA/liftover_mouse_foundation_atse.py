# import basic libraries
import pandas as pd
import os
import numpy as np
from pyliftover import LiftOver
# Chain file downloaded from:
# https://hgdownload.soe.ucsc.edu/goldenPath/mm10/liftOver/\n",

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper/ATSE_files/
chain_file = "/gpfs/commons/projects/knowles_singlecell_splicing/TabulaSenis/liftover/mm10ToMm39.over.chain.gz"
lo = LiftOver(chain_file)

# Mouse Foundation ATSE file
mouse_foundation_atse_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper/ATSE_files/MOUSE_FOUNDATION_ATSE_FILE_unanno_also_2025-10-01_21-36-40.txt.gz"
mouse_foundation_convertsion_atse_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper/ATSE_files/MOUSE_FOUNDATION_ATSE_FILE_unanno_also_2025-10-01_21-36-40_lifted_mm39.txt.gz"

mouse_foundation_genome_version = "mm10"
# Easysci genome version
easysci_genome_version = "mm39"

# Read in mouse foundation ATSE file
mouse_foundation_atse_df = pd.read_csv(mouse_foundation_atse_file, sep="\t")

# Extract required columns
junctions = mouse_foundation_atse_df[["chrom", "start", "end", "strand", "junction_id"]].copy()

# Function to lift over start and end coordinates
def lift_junction(chrom, start, end, strand, junction_id):
    lifted_start = lo.convert_coordinate(chrom, start)
    lifted_end = lo.convert_coordinate(chrom, end - 1)

    if lifted_start and lifted_end:
        new_chrom_s, new_start, _, _ = lifted_start[0]
        new_chrom_e, new_end, _, _ = lifted_end[0]

        # Don't add make ".0" to start and end, remove it
        new_start = int(new_start)
        new_end = int(new_end)

        # check if new_chrom_s and new_chrom_e are the same     
        if new_chrom_s == new_chrom_e:
            return {
                "chrom": new_chrom_s,
                "start": new_start,
                "end": new_end + 1,
                "strand": strand,  # ← override with original strand
                "junction_id": junction_id
            }

    return None
   
# Apply liftOver (add tqdm for progress bar)
from tqdm import tqdm
lifted_junctions = []
for _, row in tqdm(junctions.iterrows()):
    lifted = lift_junction(row["chrom"], row["start"], row["end"], row["strand"], row["junction_id"])
    if lifted:
        lifted_junctions.append(lifted)

# Create new DataFrame
lifted_df = pd.DataFrame(lifted_junctions)
lifted_df["strand"].value_counts()

# Now merge with original dataframe to get all other ATSE information 
# remove chr, start, end, strand columns from original dataframe and merge
# just on junction_id 
easysci_atse_df = mouse_foundation_atse_df.drop(columns=["chrom", "start", "end", "strand"])
easysci_atse_df = easysci_atse_df.merge(lifted_df, on="junction_id", how="left")

# Now drop old junction_id and rename new one keep old one just change name
# Rename junction_id column to mouse_foundation_junction_id
# Drop rows with failed liftover (i.e. missing lifted coordinates)
easysci_atse_df = easysci_atse_df.dropna(subset=["start", "end", "chrom", "strand"])
print(f"Original: {len(mouse_foundation_atse_df)}, After liftOver: {len(easysci_atse_df)}")

# Convert columns to appropriate types
easysci_atse_df["start"] = easysci_atse_df["start"].astype(int)
easysci_atse_df["end"] = easysci_atse_df["end"].astype(int)
easysci_atse_df["chrom"] = easysci_atse_df["chrom"].astype(str)
easysci_atse_df["strand"] = easysci_atse_df["strand"].astype(str)
easysci_atse_df["mouse_foundation_junction_id"] = easysci_atse_df["junction_id"]

# Now update junction_id column to new lifted values
easysci_atse_df["junction_id"] = (
    easysci_atse_df["chrom"].astype(str) + "_" +
    easysci_atse_df["start"].astype(str) + "_" +
    easysci_atse_df["end"].astype(str) + "_" +
    easysci_atse_df["strand"].astype(str)
)

# Find those where junction_id and mouse_foundation_junction_id are the same!
easysci_atse_df[easysci_atse_df["junction_id"] == easysci_atse_df["mouse_foundation_junction_id"]]
print(f"Saved converted ATSE file to {mouse_foundation_convertsion_atse_file}")
print(f"Number of junctions: {len(easysci_atse_df)}")
print(easysci_atse_df.head())
easysci_atse_df.to_csv(mouse_foundation_convertsion_atse_file, sep="\t", index=False, compression="gzip")


