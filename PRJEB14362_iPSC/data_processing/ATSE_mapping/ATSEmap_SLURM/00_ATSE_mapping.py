# Import necessary libraries
import os
import pandas as pd
import random
from scipy.sparse import csr_matrix, coo_matrix
import anndata as ad
import numpy as np
import sys 

# Paths
juncs_path = "/gpfs/commons/projects/knowles_singlecell_splicing/PRJEB14362/LEAFLET"
output_path = "/gpfs/commons/projects/knowles_singlecell_splicing/PRJEB14362/LeafletFA/ATSEs/022025"

# Get all files in juncs_path that end in *junctions_with_barcodes.bed
junc_files = [os.path.join(juncs_path, x) for x in os.listdir(juncs_path) if x.endswith("junctions_with_barcodes.bed")]

# Write the junction files to a text file in output_path 
with open(os.path.join(output_path, "junction_files.txt"), 'w') as f:
    for item in junc_files:
        f.write("%s\n" % item)
