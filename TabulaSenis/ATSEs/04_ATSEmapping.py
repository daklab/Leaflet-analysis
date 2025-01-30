# Import necessary libraries
import os
import pandas as pd
import anndata as ad
import numpy as np
import sys 
import datetime
import pickle

# Add the directory containing find_intron_clusters_v3.py to Python path
LEAFLET_SRC = '/gpfs/commons/home/kisaev/Leaflet-private/src'
sys.path.append(LEAFLET_SRC)

# Now import your modules
from clustering.find_intron_clusters_v3 import (
    JunctionReader,
    JunctionAnalyzer, 
    GenomeDB, 
    ATSEAnalyzer
)

import visualization.IsovizPy as ja

gtf_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/genome_files/gencode.vM19/genes/genes.gtf"
fasta_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/genome_files/gencode.vM19/fasta/genome.fa"
combined_junctions = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/ATSEmap/output/junction_processing_20250128/results"
output_path = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/ATSEmap/output/"

min_intron = 50
max_intron = 500000
min_junc_reads = 100 
min_num_cells_wjunc = 10
batch_size = 32
num_workers = 10
annot_status = "unanno_also"

# Load pkl file with junctions
pkl_path = f"{combined_junctions}/final_junctions.pkl"
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
genome_db = GenomeDB(db_name="gencodeVM19", gtf_file=gtf_file, fasta_file=fasta_file)

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
atse_file = f"TMS_atse_file_{annot_status}_{date}_{time}.txt"

output_file = os.path.join(output_path, atse_file)
atse_analyzer.save_atse_file(ATSE_lablled, filtered_junctions, output_file)

## to submit:
# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/ATSEmap/output/ATSEfiles
# conda activate LeafletSC
# script_path=/gpfs/commons/home/kisaev/Leaflet-analysis/TabulaSenis/ATSEs/ATSEmapping.py
# sbatch --wrap="python $script_path" --mem=300G --time=3-00:00:00 -J TMSLeafletFA -p bigmem
