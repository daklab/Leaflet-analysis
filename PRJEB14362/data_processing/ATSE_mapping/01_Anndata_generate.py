# Import necessary libraries
import os
import pandas as pd
import random
from scipy.sparse import csr_matrix, coo_matrix
import anndata as ad
import numpy as np
import sys 
import importlib
import datetime

sys.path.append('/gpfs/commons/home/kisaev/Leaflet-private/src/clustering')
import prep_anndata_object
from prep_anndata_object import *

# Paths
juncs_path = "/gpfs/commons/projects/knowles_singlecell_splicing/PRJEB14362/LEAFLET"
output_path = "/gpfs/commons/projects/knowles_singlecell_splicing/PRJEB14362/LeafletFA/ATSEs"
# gtf_file = None
gtf_file = "/gpfs/commons/datasets/controlled/BRAIN_NeMO/human-reference/gencode/gencode.v45.primary_assembly.annotation.gtf"
using_annotations = "YES"
intron_clusts_path = "iPSC_human_cells_with_annotations_50_500000_500_20241126_single_cell.gz"

# Get all files in juncs_path that end in *junctions_with_barcodes.bed
junc_files = [os.path.join(juncs_path, x) for x in os.listdir(juncs_path) if x.endswith("junctions_with_barcodes.bed")]

# Before removing empty files, had 46248 junction files
# After removing empty files, had 45466 junction files

# -----------------------------------------------------
# for testing! 
# junc_files = junc_files[0:200]
# -----------------------------------------------------

# Define working directory
WD = "/gpfs/commons/projects/knowles_singlecell_splicing/PRJEB14362/LeafletFA/ATSEs"

# Metadata file
metadata_path = f"{WD}/metadata_2024-11-24.tsv"
metadata = pd.read_csv(metadata_path, sep="\t")

# Read intron clusts file 
intron_clusts_file=f"{WD}/{intron_clusts_path}"
print("Reading in obtained intron cluster (ATSE file!)")
intron_clusts = pd.read_csv(intron_clusts_file, sep="}")
relevant_junction_ids = set(intron_clusts['junction_id'])

# Extract single cell junction and cluster counts 
print("Process single cell junction counts and assemble sparse matrices!")

# Initialize an empty list to store individual AnnData objects
anndatas = []

# Determine the batch size
batch_size = 500
num_batches = len(junc_files) // batch_size + (1 if len(junc_files) % batch_size > 0 else 0)

# print the number of junc_files and num of batches going into the analysis 
print(f"Number of junction files: {len(junc_files)}")
print(f"Number of batches: {num_batches} of batch size {batch_size}")

for i in range(num_batches):
    
    print(f"Processing Batch Number {i+1}")

    # Get the current batch of cells
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(junc_files))
    cell_batch = junc_files[start_idx:end_idx]

    # Process the current batch
    cell_by_junction_matrix, cell_by_cluster_matrix, cells, junctions, cell_idx, junc_idx, cluster_idx, cluster_idx_flip = process_files_and_build_matrices_parallel(
            cell_batch, relevant_junction_ids, intron_clusts, sequencing_type="smart_seq")
        
    # Create the AnnData object for this batch
    adata = create_anndata_object(cell_by_junction_matrix, cell_by_cluster_matrix, cell_idx, junc_idx, metadata, intron_clusts, meta_cell_column="cell_id")
    anndatas.append(adata)

# Combine all anndatas into one and save... 
combined_adata = ad.concat(anndatas, axis=0) # This code makes var dissapear...
# Ensure the combined_adata.var is consistent by taking it from the first AnnData in the list
combined_adata.var = anndatas[0].var
combined_adata.obs.reset_index(drop=True, inplace=True)
prefix="ATSE_Anndata_noGTF_Object"

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
adata_path = f"{prefix}_{current_time}.h5ad"
# Save the AnnData object to the h5ad file with gzip compression
combined_adata.write_h5ad(adata_path, compression='gzip')
print(f"AnnData object saved as {adata_path}")

# print shape of the AnnData object
print(f"Shape of the AnnData object: {combined_adata.shape}")

## to submit:
# cd /gpfs/commons/projects/knowles_singlecell_splicing/PRJEB14362/LeafletFA/ATSEs
# conda activate LeafletSC
# script_path=/gpfs/commons/home/kisaev/Leaflet-analysis/PRJEB14362/data_processing/ATSE_mapping/01_Anndata_generate.py
# sbatch --wrap="python $script_path" --mem=300G --time=3-00:00:00 -J iPSChuman -p gpu
