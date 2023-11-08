#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 6
#SBATCH --mem=40000M
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -J LeafletCluster # <-- name of job

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 

conda activate leafcutter-sc

# set up Leaflet clustering script and input files 
leaflet_clustering=/gpfs/commons/home/kisaev/Leaflet/src/clustering/Leaflet_intron_clustering.py

# Use most default params but change up junction files and set [Global params]
sequencing_type="single_cell"
gtf_file="/gpfs/commons/groups/knowles_lab/Karin/genome_files/gencode.v43.basic.annotation.gtf"
min_junc_reads=2
keep_singletons=False
junc_suffix="*.juncswbarcodes"
min_num_cells_wjunc=1

# Organ specific params 
organ="Fat"
junc_files='/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP10/Fat/'
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}"
# Run 
python $leaflet_clustering --junc_files $junc_files --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc 

# Organ specific params 
organ="Bone_Marrow"
junc_files=("/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP13/Bone_Marrow"
            "/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP11/Bone_Marrow"
            "/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP2/Bone_Marrow")
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}"
# Run 
python $leaflet_clustering --junc_files $junc_files --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc 

# Organ specific params
organ="Heart"
junc_files=("/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP12/Heart")
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}"
# Run
python $leaflet_clustering --junc_files $junc_files --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc

# Organ specific params
organ="Eye"
junc_files=("/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP5/Eye"
"/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP3/Eye")
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}"
# Run
python $leaflet_clustering --junc_files $junc_files --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc

# Organ specific params 
organ="Mammary"
junc_files=("/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP4/Mammary")
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}"
# Run
python $leaflet_clustering --junc_files $junc_files --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc

# Organ specific params
organ="Uterus"
junc_files=("/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP4/Uterus")
