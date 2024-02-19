## ==================================================================================================
## [1.] Load env and Leaflet scripts 
## ==================================================================================================

module purge                                                                                                                                                                         
module load gcc/9.2.0 

conda activate leafcutter-sc

cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaMurisBrain/

# paths to Leaflet scripts for all steps 
leaflet_clustering=/gpfs/commons/home/kisaev/Leaflet-private/src/clustering/Leaflet_intron_clustering_01.py
leaflet_input_files=/gpfs/commons/home/kisaev/Leaflet-private/src/clustering/Leaflet_prep_clusters_for_model_02.py

## ==================================================================================================
## [2.] Prepare input files and params for Leaflet
## ==================================================================================================

# set up paths for data files and also for parameters
junc_files="/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/Leaflet/junctions/Mammary_Gland"
output_file="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaMurisBrain/Mammary_Gland_wsingleton" 

sequencing_type="single_cell"
singleton="False"
junc_bed_file="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaMurisBrain/clustered_junctions.bed"
organ="Mammary_Gland"

gtf_file="/gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/MM10-PLUS/genes/genes.gtf"
min_junc_reads=5
keep_singletons=True
junc_suffix="*.juncswbarcodes"
min_num_cells_wjunc=2

# Here running with GTF file 
sbatch --wrap "python $leaflet_clustering --gtf_file $gtf_file --junc_files $junc_files --junc_bed_file $junc_bed_file --min_junc_reads $min_junc_reads --output_file $output_file --sequencing_type $sequencing_type --junc_suffix $junc_suffix --keep_singletons $keep_singletons --min_num_cells_wjunc $min_num_cells_wjunc" --mem=40000M --time=5-00:00 --job-name=$organ
echo "done"

## ==================================================================================================
## [3.] Run Leaflet input set up script to generate data for training and evaluating the final model
## ==================================================================================================

intron_clusters="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaMurisBrain/Mammary_Gland_wsingleton_50_500000_5_2_0.005_single_cell.gz" 
output_folder="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaMurisBrain/model_input"

organ="Mammary_Gland"
singletons="Yes"

# Run
#sbatch --wrap "python $leaflet_input_files --intron_clusters $intron_clusters --output_file $output_folder/$organ --metadata $metadata --has_genes "yes"" --mem=40000M -c 6 -t 5-00:00 -J $organ
python $leaflet_input_files --intron_clusters $intron_clusters --output_file $output_folder/$organ --has_genes "yes"