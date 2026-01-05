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
keep_singletons=True
junc_suffix="*.juncswbarcodes"
min_num_cells_wjunc=1

cd /gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters

# Organ specific params 
organ="Fat"
junc_files='/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP10/Fat/'
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}"
junc_bed_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}.bed"
# Run 
sbatch --wrap "python $leaflet_clustering --junc_files $junc_files --junc_bed_file $junc_bed_file --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc" --mem=40000M --time=5-00:00 --job-name=$organ

# Organ specific params 
organ="Bone_Marrow"
junc_files='/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP13/Bone_Marrow/,/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP11/Bone_Marrow/,/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP2/Bone_Marrow/'
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}"
junc_bed_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}.bed"
# Run 
sbatch --wrap "python $leaflet_clustering --junc_files $junc_files --junc_bed_file $junc_bed_file --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc" --mem=40000M --time=5-00:00 --job-name=$organ

# Organ specific params
organ="Heart"
junc_files="/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP12/Heart"
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}"
junc_bed_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}.bed"
# Run 
sbatch --wrap "python $leaflet_clustering --junc_files $junc_files --junc_bed_file $junc_bed_file --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc" --mem=40000M --time=5-00:00 --job-name=$organ

# Organ specific params
organ="Eye"
junc_files='/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP5/Eye,/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP3/Eye'
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}"
junc_bed_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}.bed"
# Run 
sbatch --wrap "python $leaflet_clustering --junc_files $junc_files --junc_bed_file $junc_bed_file --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc" --mem=40000M --time=5-00:00 --job-name=$organ

# Organ specific params 
organ="Mammary"
junc_files="/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP4/Mammary"
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}"
junc_bed_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}.bed"
# Run 
sbatch --wrap "python $leaflet_clustering --junc_files $junc_files --junc_bed_file $junc_bed_file --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc" --mem=40000M --time=5-00:00 --job-name=$organ

# Organ specific params
organ="Uterus"
junc_files="/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP4/Uterus"
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}"
junc_bed_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}.bed"
# Run 
sbatch --wrap "python $leaflet_clustering --junc_files $junc_files --junc_bed_file $junc_bed_file --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc" --mem=40000M --time=5-00:00 --job-name=$organ

# Organ specific params
organ="Tongue"
junc_files="/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP4/Tongue,/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP7/Tongue"
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}"
junc_bed_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}.bed"
# Run
sbatch --wrap "python $leaflet_clustering --junc_files $junc_files --junc_bed_file $junc_bed_file --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc" --mem=40000M --time=5-00:00 --job-name=$organ

# Organ specific params
organ="Muscle"
junc_files="/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP4/Muscle,/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP1/Muscle,/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP2/Muscle"
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}_wsingleton"
junc_bed_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}.bed"
# Run
sbatch --wrap "python $leaflet_clustering --junc_files $junc_files --junc_bed_file $junc_bed_file --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc" --mem=40000M --time=5-00:00 --job-name=$organ

# Organ specific params
organ="Liver"
junc_files="/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP6/Liver"
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}"
junc_bed_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}.bed"
# Run
sbatch --wrap "python $leaflet_clustering --junc_files $junc_files --junc_bed_file $junc_bed_file --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc" --mem=40000M --time=5-00:00 --job-name=$organ

# Organ specific params
organ="Trachea"
junc_files="/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP6/Trachea,/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP2/Trachea"
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}"
junc_bed_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}.bed"
# Run
sbatch --wrap "python $leaflet_clustering --junc_files $junc_files --junc_bed_file $junc_bed_file --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc" --mem=40000M --time=5-00:00 --job-name=$organ

# Organ specific params
organ="Spleen"
junc_files="/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP7/Spleen,/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP2/Spleen"
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}"
junc_bed_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}.bed"
# Run
sbatch --wrap "python $leaflet_clustering --junc_files $junc_files --junc_bed_file $junc_bed_file --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc" --mem=40000M --time=5-00:00 --job-name=$organ

# Organ specific params
organ="Blood"
junc_files="/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP7/Blood,/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP1/Blood,/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP2/Blood"
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}"
junc_bed_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}.bed"
# Run
sbatch --wrap "python $leaflet_clustering --junc_files $junc_files --junc_bed_file $junc_bed_file --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc" --mem=40000M --time=5-00:00 --job-name=$organ

# Organ specific params
organ="Lymph_Node"
junc_files="/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP7/Lymph_Node,/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP2/Lymph_Node"
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}"
junc_bed_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}.bed"
# Run
sbatch --wrap "python $leaflet_clustering --junc_files $junc_files --junc_bed_file $junc_bed_file --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc" --mem=40000M --time=5-00:00 --job-name=$organ

# Organ specific params
organ="Salivary_Gland"
junc_files="/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP7/Salivary_Gland"
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}"
junc_bed_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}.bed"
# Run
sbatch --wrap "python $leaflet_clustering --junc_files $junc_files --junc_bed_file $junc_bed_file --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc" --mem=40000M --time=5-00:00 --job-name=$organ

# Organ specific params
organ="Pancreas"
junc_files="/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP9/Pancreas,/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP1/Pancreas"
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}"
junc_bed_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}.bed"
# Run
sbatch --wrap "python $leaflet_clustering --junc_files $junc_files --junc_bed_file $junc_bed_file --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc" --mem=40000M --time=5-00:00 --job-name=$organ

# Organ specific params
organ="Prostate"
junc_files="/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP8/Prostate"
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}"
junc_bed_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}.bed"
# Run
sbatch --wrap "python $leaflet_clustering --junc_files $junc_files --junc_bed_file $junc_bed_file --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc" --mem=40000M --time=5-00:00 --job-name=$organ

# Organ specific params
organ="Kidney"
junc_files="/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP2/Kidney"
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}"
junc_bed_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}.bed"
# Run
sbatch --wrap "python $leaflet_clustering --junc_files $junc_files --junc_bed_file $junc_bed_file --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc" --mem=40000M --time=5-00:00 --job-name=$organ

# Organ specific params
organ="Large_Intestine"
junc_files="/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP2/Large_Intestine"
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}"
junc_bed_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}.bed"
# Run
sbatch --wrap "python $leaflet_clustering --junc_files $junc_files --junc_bed_file $junc_bed_file --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc" --mem=40000M --time=5-00:00 --job-name=$organ

# Organ specific params
organ="Lung"
junc_files="/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP1/Lung,/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP2/Lung"
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}"
junc_bed_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}.bed"
# Run
sbatch --wrap "python $leaflet_clustering --junc_files $junc_files --junc_bed_file $junc_bed_file --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc" --mem=40000M --time=5-00:00 --job-name=$organ

# Organ specific params
organ="Small_Intestine"
junc_files="/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP2/Small_Intestine"
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}"
junc_bed_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}.bed"
# Run
sbatch --wrap "python $leaflet_clustering --junc_files $junc_files --junc_bed_file $junc_bed_file --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc" --mem=40000M --time=5-00:00 --job-name=$organ

# Organ specific params
organ="Thymus"
junc_files="/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP2/Thymus"
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}"
junc_bed_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}.bed"
# Run
sbatch --wrap "python $leaflet_clustering --junc_files $junc_files --junc_bed_file $junc_bed_file --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc" --mem=40000M --time=5-00:00 --job-name=$organ

# Organ specific params
organ="Vasculature"
junc_files="/gpfs/commons/projects/CZI-tabula-sapiens/SS2_cell_junctions/TSP2/Vasculature"
output_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}"
junc_bed_file="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/${organ}.bed"
# Run
sbatch --wrap "python $leaflet_clustering --junc_files $junc_files --junc_bed_file $junc_bed_file --output_file $output_file --sequencing_type $sequencing_type --gtf_file $gtf_file --min_junc_reads $min_junc_reads --keep_singletons $keep_singletons --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc" --mem=40000M --time=5-00:00 --job-name=$organ

