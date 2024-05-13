#!/bin/sh

# Load LeafletSC 
import LeafletSC
import os
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from LeafletSC.clustering.find_intron_clusters import main as find_intron_clusters
from LeafletSC.clustering.prepare_model_input import main as prep_model_input
from LeafletSC.clustering.find_intron_clusters import visualize_local_events

# Define path that contains junction files
juncs_path = "/gpfs/commons/datasets/controlled/BRAIN_NeMO/lein-human-cortex/Leaflet/"
print("The junctions are loaded from the following path: " + juncs_path) 

# print the files in the path 
#print("The files in the path are: " + str(os.listdir(juncs_path)))

# define path for saving the output data 
output_path = "/gpfs/commons/datasets/controlled/BRAIN_NeMO/lein-human-cortex/LeafletAnalysis/"

# we provide a gtf file for the human genome as well to make better sense of the junctions that are detected in cells
# please replace with the path to the gtf file on your system
gtf_file="/gpfs/commons/datasets/controlled/BRAIN_NeMO/human-reference/gencode/gencode.v45.primary_assembly.annotation.gtf" 

# file with cell gene expression based leiden clusters 
cells_ge = "/gpfs/commons/datasets/controlled/BRAIN_NeMO/human-cortex-mtg/mtg_nuclei_gene_expression_leiden_clusters.csv"

# published metadata
metacells = "/gpfs/commons/datasets/controlled/BRAIN_NeMO/human-cortex-mtg/mtg_facs_metadata.csv"

# %%
# read in metacells 
cells_pub = pd.read_csv(metacells)

# %%
# define additional parameters 
sequencing_type = "single_cell"

# ensure output files are to be saved in output_path 
output_file = output_path + "human_cortex_lein_leaflet"
junc_bed_file= output_path + "human_cortex_lein_leaflet.bed" # you can load this file into IGV to visualize the junction coordinates 
min_intron_length = 50
max_intron_length = 500000
threshold_inc = 0.1 
min_junc_reads = 50
min_num_cells_wjunc = 1000
keep_singletons = False # ignore junctions that do not share splice sites with any other junction (likely const)
junc_suffix = "*_with_barcodes.bed" # depends on how you ran regtools 

# %%
# get a list of all the files in juncs_path that end in junc_suffix 
junc_suffix_end = junc_suffix.split("*")[1]
junc_files = [f for f in os.listdir(juncs_path) if f.endswith(junc_suffix_end)]

# add junc_path to each file in front of it
junc_files = [juncs_path + f for f in junc_files]

# %%
all_juncs_df = find_intron_clusters(junc_files=junc_files, gtf_file=gtf_file, output_file=output_file, 
                       sequencing_type=sequencing_type, junc_bed_file=junc_bed_file, 
                       threshold_inc=threshold_inc, min_intron = min_intron_length,
                       max_intron=max_intron_length, min_junc_reads=min_junc_reads,
                       singleton=keep_singletons,
                       junc_suffix=junc_suffix, min_num_cells_wjunc=min_num_cells_wjunc,filter_shared_ss=True, 
                       run_notebook = True)


