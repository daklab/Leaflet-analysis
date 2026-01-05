## Steps to run Leaflet on ParseBio PBMC data from start (junction files from regtools) to finish (cell states overlayed on UMAP)
## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

conda activate leafcutter-sc 
cd /gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/Leaflet

# paths to Leaflet scripts for all steps 
clustering=/gpfs/commons/home/kisaev/Leaflet/src/clustering/Leaflet_intron_clustering.py
input_set_up=/gpfs/commons/home/kisaev/Leaflet/src/beta-binomial-lda/01_prepare_input_coo.py

## 1. get intron clusters using Leaflet 
## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# set up paths for data files and also for parameters
gtf_file="/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/longread/SRR9944890/collapse/SRR9944890_isoform_gid.gtf"
junc_files="/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/junctions"
output_file="/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/ParseBio-PBMC_clustered_junctions" 
setting="anno_free"
sequencing_type="single_cell"
singleton="False"
junc_bed_file="/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/ParseBio_clustered_junctions.bed"
min_junc_reads=20
min_num_cells_wjunc=10
min_intron=50
max_intron=100000
junc_suffix="*.wbarcode.junc"
filter_low_juncratios_inclust="no"
strict_filter=True

python $clustering --gtf_file $gtf_file --junc_files $junc_files --output_file $output_file --setting $setting --sequencing_type $sequencing_type --min_intron=$min_intron --max_intron=$max_intron --min_junc_reads=$min_junc_reads --threshold_inc=0.1 --junc_bed_file $junc_bed_file --keep_singletons $singleton --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc --filter_low_juncratios_inclust $filter_low_juncratios_inclust --strict_filter $strict_filter
echo "done"

## 2. get input files for BB-mixture model using Leaflet 
## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

cluster_file="/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/ParseBio-PBMC_clustered_junctions_anno_free_50_500000_10_5_0.1_single_cell.gz"

# more stringent filtering for input files for BB-mixture model
cluster_file_str="/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/leafcutter/ParseBio-PBMC_clustered_junctions_anno_free_50_100000_20_10_0.1_single_cell.gz"
output_file="/gpfs/commons/groups/knowles_lab/Karin/parse-pbmc-leafcutter/Leaflet/BBmixture_PBMC_input_fulldata_more_stringent"

python $input_set_up --intron_clusters $cluster_file_str --output_file $output_file --has_genes "no" --chunk_size 10000
echo "done"

## 3. run BB-mixture model using Leaflet
## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


## 4. visualize gene expression based UMAP with cell states overlayed 
## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

