# Import necessary libraries
import os
import pandas as pd
import anndata as ad
import numpy as np
import sys 
import datetime
import pickle
import gffutils
import networkx as nx

# Add the directory containing find_intron_clusters_v3.py to Python path
LEAFLET_SRC = '/gpfs/commons/home/kisaev/LeafletFA-utils/leafletfa_utils/atsemapper'
sys.path.append(LEAFLET_SRC)

##----------Post processing------------------------------------------------------
# Read the file back in and check if atses remain (post splice site usage filter)
# that aren't overlapping anymore and should be removed 
atse_input_file="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/ATSE_mapper/ATSE_files/stella_gtf/TMS_atse_file_unanno_also_2025-04-30_19-03-14.txt.gz"
atses = pd.read_csv(atse_input_file, sep="\t")
atses.drop(columns=["transcripts", "both_ends_transcripts", "only_5_prime_transcripts", "only_3_prime_transcripts"], inplace=True)
print(f"Read existing ATSE file and starting post mapping cleanup!")

# Helper to extract donor and acceptor
def extract_sites(junction_id):
    chrom, start, end, strand = junction_id.split("_")
    start, end = int(start), int(end)
    if strand == "+":
        donor, acceptor = start, end
    else:
        donor, acceptor = end, start
    return donor, acceptor

# Function to compute connected components of junctions via shared splice sites
def check_atse_connectivity(df):
    inconsistent_atse_ids = []

    for atse_id, group in df.groupby("event_id"):
        G = nx.Graph()
        junctions = group["junction_id"].unique()

        for j in junctions:
            donor, acceptor = extract_sites(j)
            for other in junctions:
                if j == other:
                    continue
                other_donor, other_acceptor = extract_sites(other)
                if donor == other_donor or acceptor == other_acceptor:
                    G.add_edge(j, other)

        # Add any isolated nodes (single junctions with no overlap)
        for j in junctions:
            G.add_node(j)

        if nx.number_connected_components(G) > 1:
            inconsistent_atse_ids.append(atse_id)

    return pd.DataFrame({"event_id": inconsistent_atse_ids})

# Run the consistency check
inconsistent_df = check_atse_connectivity(atses)
print(f"Removing {len(inconsistent_df)} inconsistent ATSE events")

# Remove inconsistent ATSE events
atses = atses[~atses["event_id"].isin(inconsistent_df["event_id"])]

# Save the cleaned ATSE file, use output_file to overwrite the original
atses.to_csv(atse_input_file, sep="\t", index=False)
print(f"Final number of junctions in ATSE file: {len(atses)}")
print(f"Final number of ATSE events in ATSE file: {len(atses['event_id'].unique())}")

## to submit:
# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/ATSE_mapper/ATSE_files/stella_gtf
# conda activate LeafletSC
# script=/gpfs/commons/home/kisaev/Leaflet-analysis/Human_Splicing_Foundation/ATSE_mapper/04B_ATSEmapping.py
# python $script 