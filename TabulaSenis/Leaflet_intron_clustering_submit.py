# Import necessary libraries
import os
import pandas as pd
import random
from scipy.sparse import csr_matrix, coo_matrix
import anndata as ad
import numpy as np
import sys 

sys.path.append('/gpfs/commons/home/kisaev/Leaflet-private/src/clustering')
import find_intron_clusters_v2
# Reload the module if you've made changes and want to update it
import importlib
importlib.reload(find_intron_clusters_v2)
import datetime

import prep_anndata_object
from prep_anndata_object import *
importlib.reload(prep_anndata_object)

# Paths
juncs_path = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/junctions/"
output_path = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet"
gtf_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/genome_files/gencode.vM19/genes/genes.gtf"

# Metadata file
metadata_path = "/gpfs/commons/projects/knowles_singlecell_splicing/TabulaSenis/data/AWS/metadata/tabula-muris-senis-full-metadata.csv"
metadata = pd.read_csv(metadata_path)

# Filter metadata for FACS method
metadata = metadata[metadata['method'] == 'facs']
metadata_subset = metadata.copy()

# Function to format cell IDs based on the month group
def format_cell_id(index, group):
    if group == '3m':
        parts = index.replace('.', '-', 1).replace('_', '-', 1).split('.')
        corrected_part = parts[1].replace('-', '_', 1)
        return parts[0] + '-' + corrected_part + '-1-1'
    else:
        return index.split('.')[0]

# Apply the formatting to the metadata subset
metadata_subset['cell_id'] = metadata_subset.apply(lambda row: format_cell_id(row['index'], row['age']), axis=1)
cell_ids_set = set(metadata_subset['cell_id'].values)

# Keep only important columns in metadata_subset
metadata_subset = metadata_subset[['cell_id', 'age', 'batch', 'cell_ontology_class', 'method', 'mouse.id', 'sex', 'subtissue', 'tissue']]

# Get all junction files and filter
all_junc_files = []
all_dirs = os.listdir(juncs_path)
print(all_dirs)
for dir in all_dirs:
    dir_path = os.path.join(juncs_path, dir)
    if os.path.isdir(dir_path):
        junc_files = os.listdir(dir_path)
        junc_files = [os.path.join(dir_path, x) for x in junc_files]
        all_junc_files.extend(junc_files)

# Filter the files using the set
portion_in_list2 = [x for x in all_junc_files if os.path.splitext(os.path.basename(x))[0] in cell_ids_set]
portion_in_list2 = [x + "/junctions_with_barcodes.bed" for x in portion_in_list2]

# Shuffle the list
random.shuffle(portion_in_list2)

print(len(portion_in_list2))

# testing! 
# portion_in_list2 = portion_in_list2[0:10000]

# Define additional parameters
output_file = os.path.join(output_path, "tabula_senis_annotationFREE_intron_clusters")
junc_bed_file = os.path.join(output_path, "tabula_senis_annotationFREE_intron_clusters.bed")
sequencing_type = "single_cell"
min_intron = 50
max_intron = 500000
min_junc_reads = 100 
min_num_cells_wjunc = 100
max_workers = 10
batch_size = 100
run_clustering = False

# Check if output files already exist to skip
print("Running intron clustering for test data...")

if run_clustering:
    intron_clusts_file = find_intron_clusters_v2.main(
            junc_files=portion_in_list2,
            gtf_file=None, #annotation free 
            output_file=output_file,
            sequencing_type=sequencing_type,
            junc_bed_file=junc_bed_file,
            threshold_inc=0.05,
            min_intron=min_intron,
            max_intron=max_intron,
            min_junc_reads=min_junc_reads,
            singleton=False,
            min_num_cells_wjunc=min_num_cells_wjunc,
            filter_shared_ss=True,
            max_workers=max_workers,
            batch_size=batch_size,
            run_notebook=False
        )

else:
    # Read intron clusts file 
    intron_clusts_file="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/tabula_senis_annotationFREE_intron_clusters_50_500000_100_20241102_single_cell.gz"

print("Reading in obtained intron cluster (ATSE file!)")
intron_clusts = pd.read_csv(intron_clusts_file, sep="}")
relevant_junction_ids = set(intron_clusts['junction_id'])

# Extract single cell junction and cluster counts 
print("Process single cell junction counts and assemble sparse matrices!")

# Initialize an empty list to store individual AnnData objects
anndatas = []

# Determine the batch size
batch_size = 500
num_batches = len(portion_in_list2) // batch_size + (1 if len(portion_in_list2) % batch_size > 0 else 0)

for i in range(num_batches):
    
    print(f"Processing Batch Number {i+1}")

    # Get the current batch of cells
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(portion_in_list2))
    cell_batch = portion_in_list2[start_idx:end_idx]

    # Process the current batch
    cell_by_junction_matrix, cell_by_cluster_matrix, cells, junctions, cell_idx, junc_idx, cluster_idx, cluster_idx_flip = process_files_and_build_matrices_parallel(
            cell_batch, relevant_junction_ids, intron_clusts, sequencing_type="smart_seq")
        
    # Create the AnnData object for this batch
    adata = create_anndata_object(cell_by_junction_matrix, cell_by_cluster_matrix, cell_idx, junc_idx, metadata_subset, intron_clusts)
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

# old code prior to batching the junction files...
# cell_by_junction_matrix, cell_by_cluster_matrix, cells, junctions, cell_idx, junc_idx, cluster_idx, cluster_idx_flip = process_files_and_build_matrices_parallel(portion_in_list2, relevant_junction_ids, intron_clusts, sequencing_type="smart_seq")
# Save as Anndata object!
# print("Save Anndata object!")
# create_anndata_object(cell_by_junction_matrix, cell_by_cluster_matrix, cell_idx, junc_idx, metadata_subset, intron_clusts, save_file=True, prefix="ATSE_Anndata_noGTF_Object")

## to submit:
# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet
# conda activate LeafletSC
# script_path=/gpfs/commons/home/kisaev/Leaflet-analysis/TabulaSenis/Leaflet_intron_clustering_submit.py
# sbatch --wrap="python $script_path" --mem=350G --time=3-00:00:00 -J TMSLeafletFA -p bigmem
