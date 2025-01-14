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
juncs_path = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/junctions/DT"
output_path = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/ATSEs/DT"
gtf_file = None
using_annotations = "NO"

WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/ATSEs"
DT_atses = f"{WD}/DT/EasySci_DT_Metacells_no_annotations_50_500000_10_20241204_single_cell.gz"
primer = "DT"

# Get all files in juncs_path that end in *junctions_with_barcodes.bed
junc_files = [os.path.join(juncs_path, x) for x in os.listdir(juncs_path) if x.endswith("junctions_with_barcodes.bed")]

# -----------------------------------------------------
# for testing! 
# junc_files = junc_files[0:200]
# -----------------------------------------------------

# Metadata file
# In bash did the following:
# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA
# input_file=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/DT_cells.csv
# (head -n 1 $input_file | awk -F',' '{print $2 "," $3}'; awk -F',' '{print $2 "," $3}' $input_file | tail -n +2 | sort -u) > DT_anndata_meta.tsv
# input_file=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/RH_cells.csv
# (head -n 1 $input_file | awk -F',' '{print $2 "," $3}'; awk -F',' '{print $2 "," $3}' $input_file | tail -n +2 | sort -u) > RH_anndata_meta.tsv

metadata_path = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/DT_anndata_meta.tsv"  
metadata = pd.read_csv(metadata_path, sep=",")

# Read intron clusts file 
intron_clusts_file = DT_atses
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
    adata = create_anndata_object(cell_by_junction_matrix, cell_by_cluster_matrix, cell_idx, junc_idx, metadata, intron_clusts, meta_cell_column="Main_cluster_name_wkmeans")
    anndatas.append(adata)

# Combine all anndatas into one and save... 
combined_adata = ad.concat(anndatas, axis=0) # This code makes var dissapear...
# Ensure the combined_adata.var is consistent by taking it from the first AnnData in the list
combined_adata.var = anndatas[0].var
combined_adata.obs.reset_index(drop=True, inplace=True)

prefix=f"ATSE_Anndata_{using_annotations}_{primer}_Object"
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
adata_path = f"{prefix}_{current_time}.h5ad"

# Save the AnnData object to the h5ad file with gzip compression
combined_adata.write_h5ad(adata_path, compression='gzip')
print(f"AnnData object saved as {adata_path}")

# print shape of the AnnData object
print(f"Shape of the AnnData object: {combined_adata.shape}")

## to submit:
# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/ATSEs
# conda activate LeafletSC
# script_path=/gpfs/commons/home/kisaev/Leaflet-analysis/EasySci/LeafletFA/data_processing/ATSE_mapping/DT/01_Anndata_generate.py
# sbatch --wrap="python $script_path" --mem=160G --time=3-00:00:00 -J DT_EASYSCI -p gpu
