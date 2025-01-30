# Import necessary libraries
import os
import pandas as pd
import random
from scipy.sparse import csr_matrix, coo_matrix
import anndata as ad
import numpy as np
import sys 
import datetime
import gffutils 
import importlib

# Load data and create the database (note this may take 1-2 minutes - do this once!)
gtf_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/genome_files/gencode.vM19/genes/genes.gtf"  
fasta_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/genome_files/gencode.vM19/fasta/genome.fa"

# Paths
juncs_path = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/junctions/"
output_path = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/ATSEmap/output/"

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
junction_files = portion_in_list2

# Write the junction files to a text file in output_path 
with open(os.path.join(output_path, "junction_files.txt"), 'w') as f:
    for item in junction_files:
        f.write("%s\n" % item)