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
import glob 

sys.path.append('/gpfs/commons/home/kisaev/Leaflet-private/src/clustering')
import prep_anndata_object
from prep_anndata_object import *

# Paths
juncs_path = "/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions"
output_path = "/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/ATSEs"
# gtf_file = None
gtf_file="/gpfs/commons/groups/knowles_lab/Karin/genome_files/gencode.v43.basic.annotation.gtf"
using_annotations = "YES"

# Get all files in juncs_path that end in *junctions_with_barcodes.bed
junc_files = glob.glob(f"{juncs_path}/**/*.juncswbarcodes", recursive=True)

intron_clusts_path = "tabula-sapien_with_annotations_50_500000_20_20250108_single_cell.gz"

# -----------------------------------------------------
# for testing! 
# junc_files = junc_files[0:500]
# -----------------------------------------------------

# Define working directory
WD = "/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/ATSEs"

# Metadata file
metadata_path = "/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/tabula_sapiens_ss2_metadata.csv"
metadata = pd.read_csv(metadata_path)

# Remove outlier_call column from metadata
metadata = metadata.drop(columns=['manually_annotated'])

# Add ".homo.gencode.v30.ERCC.chrM" to all values in bam_file_name
metadata["bam_file_name"] = metadata["bam_file_name"].astype(str) + ".homo.gencode.v30.ERCC.chrM"

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
    adata = create_anndata_object(cell_by_junction_matrix, cell_by_cluster_matrix, cell_idx, junc_idx, metadata, intron_clusts, meta_cell_column="bam_file_name")
    anndatas.append(adata)

# Combine all anndatas into one and save... 
combined_adata = ad.concat(anndatas, axis=0) # This code makes var dissapear...
# Ensure the combined_adata.var is consistent by taking it from the first AnnData in the list
combined_adata.var = anndatas[0].var
combined_adata.obs.reset_index(drop=True, inplace=True)
prefix="ATSE_Anndata_with_GTF_Object"

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
adata_path = f"{prefix}_{current_time}.h5ad"
# Save the AnnData object to the h5ad file with gzip compression
combined_adata.write_h5ad(adata_path, compression='gzip')
print(f"AnnData object saved as {adata_path}")

# print shape of the AnnData object
print(f"Shape of the AnnData object: {combined_adata.shape}")

## to submit:
# cd /commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/ATSEs
# conda activate LeafletSC
# script_path=/gpfs/commons/home/kisaev/Leaflet-analysis/tabula_sapien/Leaflet/Code/01_Anndata_generate.py
# sbatch --wrap="python $script_path" --mem=200G --time=3-00:00:00 -J tabula_sapien -p gpu
