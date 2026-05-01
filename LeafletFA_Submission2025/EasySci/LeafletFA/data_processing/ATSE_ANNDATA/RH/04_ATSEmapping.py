# Import necessary libraries
import os
import pandas as pd
import anndata as ad
import numpy as np
import sys 
import datetime
import pickle
import gffutils

# Add the directory containing find_intron_clusters_v3.py to Python path
LEAFLET_SRC = '/gpfs/commons/home/kisaev/LeafletFA-utils/leafletfa_utils/atsemapper'
sys.path.append(LEAFLET_SRC)

# Now import your modules
from junction_parser import JunctionReader 
from genome_utils import JunctionAnalyzer, GenomeDB 
from event_detection import ATSEAnalyzer 

# Set METACELL_SUFFIX and PROC_DATE via environment variables
metacell_suffix = os.environ.get("METACELL_SUFFIX", "")
proc_date = os.environ.get("PROC_DATE", "20260318")
base_dir = f"/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/SpliceVI{metacell_suffix}"

gtf_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/genome_files/gencode.vM27.basic.annotation.gtf"
fasta_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/genome_files/GRCm39.primary_assembly.genome.fa"
combined_junctions_file = f"{base_dir}/ATSEmap/junction_processing_{proc_date}/results"
output_path = f"{base_dir}/ATSEmap"
db_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/genome_files/gencode_vM27.db"

min_intron = 50
max_intron = 500000
min_junc_reads = 10 
min_num_cells_wjunc = 2
batch_size = 32
num_workers = 10
annot_status = "unanno_also"

# Load pkl file with junctions
pkl_path = f"{combined_junctions_file}/final_junctions.pkl"
combined_junctions = pd.read_pickle(pkl_path)

# Initialize the junction reader
reader = JunctionReader(batch_size=batch_size, 
                        min_cells=min_num_cells_wjunc, 
                        min_reads=min_junc_reads, 
                        min_intron=min_intron, 
                        max_intron=max_intron, 
                        num_workers=num_workers)

# Run QC on the junctions
filtered_junctions = reader.SJ_QC(combined_junctions)

# Initialize genome database 
genome_db = GenomeDB(db_name=db_file, gtf_file=gtf_file, fasta_file=fasta_file)

# Initialize junction analyzer
analyzer = JunctionAnalyzer(fasta_file=fasta_file, db=genome_db.get_db(), tolerance=100)

# Ensure that the junctions have canonical splice sites
junctions = analyzer.check_splice_sites(filtered_junctions)
canonical_junctions = analyzer.filter_canonical(junctions)

# Annotate junctions 5' and 3' with trancript exons 
annotated_junctions = analyzer.check_junction_annotation(canonical_junctions)

# Filter junctions that are not annotated
filtered_junctions = analyzer.filter_annotated(annotated_junctions, annotation_status_include=annot_status)

# Initialize the ATSE analyzer
atse_analyzer = ATSEAnalyzer()

# Build initial splice graph 
sgraph, stats = atse_analyzer.build_splice_graph(filtered_junctions)

# Find ATSEs using the splice graph
ATSE_groups, sorted_counts = atse_analyzer.find_atse_groups(sgraph)

# Try classifyiing ATSEs 
ATSE_lablled, event_counts = atse_analyzer.classify_events(sgraph, ATSE_groups)
print(event_counts) 

# Save the ATSEs to a file
date = datetime.datetime.now().strftime("%Y-%m-%d")
time = datetime.datetime.now().strftime("%H-%M-%S")
atse_file = f"EasySci_RH_ATSE_FILE_{annot_status}_{date}_{time}.txt"

output_file = os.path.join(output_path, atse_file)
atse_analyzer.save_atse_file(ATSE_lablled, filtered_junctions, output_file)

## to submit:
# conda activate LeafletSC
# script_path=/gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/EasySci/LeafletFA/data_processing/ATSE_ANNDATA/RH/04_ATSEmapping.py
# METACELL_SUFFIX=_500  PROC_DATE=20260318 sbatch --mem=200G --time=12:00:00 -J ATSE_500  -p bigmem,cpu,dev --wrap="python $script_path"
# METACELL_SUFFIX=_1000 PROC_DATE=20260318 sbatch --mem=200G --time=12:00:00 -J ATSE_1000 -p bigmem,cpu,dev --wrap="python $script_path"
# METACELL_SUFFIX=_2000 PROC_DATE=20260318 sbatch --mem=200G --time=12:00:00 -J ATSE_2000 -p bigmem,cpu,dev --wrap="python $script_path"
# METACELL_SUFFIX=_4000 PROC_DATE=20260318 sbatch --mem=200G --time=12:00:00 -J ATSE_4000 -p bigmem,cpu,dev --wrap="python $script_path"
