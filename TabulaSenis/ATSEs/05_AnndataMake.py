import sys 
import os
import pandas as pd
from pathlib import Path
import importlib
import argparse

# Import source code for processing anndata object
sys.path.append('/gpfs/commons/home/kisaev/Leaflet-private/src/clustering')
from prep_anndata_object_v2 import process_files_and_build_matrices_parallel, create_anndata_object

# Constants
METADATA_PATH = "/gpfs/commons/projects/knowles_singlecell_splicing/TabulaSenis/data/AWS/metadata/tabula-muris-senis-full-metadata.csv"
INTRON_CLUSTS_FILE = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/ATSEmap/output/ATSEfiles/TMS_atse_file_unanno_also_2025-01-30_19-24-18.txt.gz"
BATCH_SIZE = 10
MAX_WORKERS = 4

def format_cell_id(index, group):
    """Format cell ID based on age group"""
    if group == '3m':
        parts = index.replace('.', '-', 1).replace('_', '-', 1).split('.')
        corrected_part = parts[1].replace('-', '_', 1)
        return parts[0] + '-' + corrected_part + '-1-1'
    return index.split('.')[0]

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
    metadata = pd.read_csv(METADATA_PATH, low_memory=False)
    metadata_subset = metadata[metadata['method'] == 'facs'].copy()
    metadata_subset['cell_id'] = metadata_subset.apply(
        lambda row: format_cell_id(row['index'], row['age']), axis=1)
    metadata_subset = metadata_subset[['cell_id', 'age', 'batch', 'cell_ontology_class', 
                                     'method', 'mouse.id', 'sex', 'subtissue', 'tissue']]

    # Read intron clusters
    print(f"Reading ATSE file from {INTRON_CLUSTS_FILE}")
    intron_clusts = pd.read_csv(INTRON_CLUSTS_FILE, sep="\t")
    relevant_junction_ids = set(intron_clusts['junction_id'])
    print(f"Number of relevant junction ids: {len(relevant_junction_ids)}")

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
        metadata_subset,
        intron_clusts,
        save_file=True,
        meta_cell_column="cell_id",
        prefix=output_file
    )

if __name__ == "__main__":
    main()