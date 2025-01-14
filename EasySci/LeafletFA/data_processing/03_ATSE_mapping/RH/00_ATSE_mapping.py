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
juncs_path = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/junctions/RH"
output_path = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/ATSEs/RH"
gtf_file = None
using_annotations = "NO"

# Get all files in juncs_path that end in *junctions_with_barcodes.bed
junc_files = [os.path.join(juncs_path, x) for x in os.listdir(juncs_path) if x.endswith("junctions_with_barcodes.bed")]

# -----------------------------------------------------
# for testing! 
# junc_files = junc_files[0:10]
# -----------------------------------------------------

# Define additional parameters
output_file = os.path.join(output_path, "EasySci_RH_Metacells" )

# Get today's date and add to junc_bed_file
today = datetime.datetime.now()
junc_bed_file = os.path.join(output_path, "EasySci_RH_Metacells_ATSEs_junctions")
junc_bed_file = junc_bed_file + "_" + today.strftime("%Y%m%d")

# For output file combine iPSC_human_cells iwth using_annotations status 
if using_annotations == "YES":
    output_file = output_file + "_with_annotations"
    junc_bed_file = junc_bed_file + "_with_annotations.bed"
else:
    output_file = output_file + "_no_annotations"
    junc_bed_file = junc_bed_file + "_no_annotations.bed"

sequencing_type = "single_cell"
min_intron = 50
max_intron = 500000
min_junc_reads = 10 
min_num_cells_wjunc = 2
max_workers = 10
batch_size = 100
run_clustering = True

print(f"Mapping ATSEs across all cells and junctions with GTF file? {gtf_file}")
print(f"The number of single cell junction files to process: {len(junc_files)}")
print(f"The filters going into ATSE mapping include: min_intron={min_intron}, max_intron={max_intron}, min_junc_reads={min_junc_reads}, min_num_cells_wjunc={min_num_cells_wjunc}")

intron_clusts_file = find_intron_clusters_v2.main(
            junc_files=junc_files,
            gtf_file=gtf_file,
            output_file=output_file,
            sequencing_type=sequencing_type,
            junc_bed_file=junc_bed_file,
            threshold_inc=0.1,
            min_intron=min_intron,
            max_intron=max_intron,
            min_junc_reads=min_junc_reads,
            singleton=False,
            min_num_cells_wjunc=min_num_cells_wjunc,
            filter_shared_ss=True,
            max_workers=max_workers,
            batch_size=batch_size,
            run_notebook=False)


## to submit:
# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/ATSEs/RH
# conda activate LeafletSC
# script_path=/gpfs/commons/home/kisaev/Leaflet-analysis/EasySci/LeafletFA/data_processing/ATSE_mapping/RH/00_ATSE_mapping.py
# sbatch --wrap="python $script_path" --mem=300G --time=3-00:00:00 -J EasySciRH
