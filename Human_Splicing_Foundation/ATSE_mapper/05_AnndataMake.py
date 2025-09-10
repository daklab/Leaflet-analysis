import sys 
import os
import pandas as pd
from pathlib import Path
import importlib
import argparse
import pickle

# Import source code for processing anndata object
sys.path.append('/gpfs/commons/home/kisaev/Leaflet-private/src/clustering')
from prep_anndata_object_v2 import process_files_and_build_matrices_parallel, create_anndata_object

# Constants
#metadata = pd.read_csv(metadata_path, sep=",")
METADATA_PATH = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/human_metadata_combined.tsv" #note this metadata might be just Smart-seq2 cells/nuceli... (as in no 10X here)
INTRON_CLUSTS_FILE = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/ATSE_mapper/ATSE_files/TMS_atse_file_unanno_also_2025-08-11_07-06-16.txt.gz"
BATCH_SIZE = 10
MAX_WORKERS = 4

def main():
    parser = argparse.ArgumentParser(description='Process a chunk of junction files and create anndata object')
    parser.add_argument('--chunk-file', required=True, help='Text file containing list of junction files to process')
    parser.add_argument('--output-dir', required=True, help='Directory to save the output anndata object')
    parser.add_argument('--chunk-id', required=True, help='Identifier for this chunk')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Read the chunk file
    print(f"Reading junction files from chunk: {args.chunk_file}")
    with open(args.chunk_file) as f:
        all_junc_files = [line.strip() for line in f if line.strip()]
    print(f"Found {len(all_junc_files)} junction files in chunk")

    # Read metadata
    print(f"Reading metadata from {METADATA_PATH}")
    metadata = pd.read_csv(METADATA_PATH, sep="\t", low_memory=False)
    print(metadata.head())

    # Load in the actual junctions observed in HUMAN SPLICING FOUNDATION
    with open('/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/ATSE_mapper/junction_processing_20250730/results/final_junctions.pkl', 'rb') as f:
        junction_dict = pickle.load(f)
    dataset_junction_ids = set(junction_dict.keys())
    print(f"Found {len(dataset_junction_ids)} junctions in the dataset")

    # Read and process intron clusters
    print(f"Reading ATSE file from {INTRON_CLUSTS_FILE}")
    intron_clusts = pd.read_csv(INTRON_CLUSTS_FILE, sep="\t")
    intron_clusts['junction_id'] = intron_clusts['junction_id'].astype(str)

    # Filter the ATSE file to only include junctions in the dataset
    print("Filtering ATSE file to only include junctions in the dataset...")
    original_count = len(intron_clusts)
    intron_clusts = intron_clusts[intron_clusts['junction_id'].isin(dataset_junction_ids)]
    filtered_count = len(intron_clusts)
    print(f"Filtered ATSE file from {original_count} to {filtered_count} junctions")

    # Check for and remove duplicates in intron_clusts
    print("Checking for duplicates in filtered intron_clusts")
    dups = intron_clusts['junction_id'].duplicated()
    if dups.any():
        print(f"Found {dups.sum()} duplicates in intron_clusts['junction_id']")
        print("Removing duplicates from intron_clusts (likely from liftOver analysis)")
        intron_clusts = intron_clusts.drop_duplicates(subset='junction_id', keep='first')
        print(f"intron_clusts shape after deduplication: {intron_clusts.shape}")
    
    # Extract final list of relevant junction IDs from deduplicated DataFrame
    relevant_junction_ids = list(intron_clusts['junction_id'])
    print(f"Final number of unique junction IDs for processing: {len(relevant_junction_ids)}")

    # Process files and build matrices
    print("Processing files and building matrices...")
    cell_by_junction_matrix, cell_by_cluster_matrix, cells, junctions, cell_idx, \
    junc_idx, cluster_idx, cluster_idx_flip = process_files_and_build_matrices_parallel(
        all_junc_files, 
        relevant_junction_ids, 
        intron_clusts, 
        sequencing_type="smart_seq", 
        max_workers=MAX_WORKERS
    )
    
    # Create and save anndata object
    output_file = os.path.join(args.output_dir, f"chunk_{args.chunk_id}_anndata.h5ad")
    print(f"Creating anndata object and saving to {output_file}")
    combined_anndata = create_anndata_object(
        cell_by_junction_matrix,
        cell_by_cluster_matrix,
        cell_idx,
        junc_idx,
        metadata,
        intron_clusts,
        save_file=True,
        meta_cell_column="cell_id",
        prefix=output_file)

    # Print the number of cells that have zero counts across the board
    zero_cells = (cell_by_cluster_matrix.sum(axis=1) == 0).sum()
    print(f"Number of cells with zero counts: {zero_cells}")

if __name__ == "__main__":
    main()