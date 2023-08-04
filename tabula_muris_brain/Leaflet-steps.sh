## Steps to run Leaflet on ParseBio PBMC data from start (junction files from regtools) to finish (cell states overlayed on UMAP)
## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

conda activate leafcutter-sc 
cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaMurisBrain/

# paths to Leaflet scripts for all steps 
clustering=/gpfs/commons/home/kisaev/Leaflet/src/clustering/Leaflet_intron_clustering.py
input_set_up=/gpfs/commons/home/kisaev/Leaflet/src/beta-binomial-lda/01_prepare_input_coo.py

## 1. get intron clusters using Leaflet 
## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# set up paths for data files and also for parameters
gtf_file="/gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/MM10-PLUS/genes/genes.gtf"
junc_files="/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/Leaflet/junctions"
output_file="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaMurisBrain/clustered_junctions" 
setting="anno_free"
sequencing_type="single_cell"
singleton="False"
junc_bed_file="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaMurisBrain/clustered_junctions.bed"
min_junc_reads=2
min_num_cells_wjunc=5
min_intron=50
max_intron=200000
junc_suffix="*.juncswbarcodes"
filter_low_juncratios_inclust="no"
strict_filter=True

sbatch --wrap "python $clustering --gtf_file $gtf_file --junc_files $junc_files --output_file $output_file --setting $setting --sequencing_type $sequencing_type --min_intron=$min_intron --max_intron=$max_intron --min_junc_reads=$min_junc_reads --threshold_inc=0.1 --junc_bed_file $junc_bed_file --keep_singletons $singleton --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc --filter_low_juncratios_inclust $filter_low_juncratios_inclust --strict_filter $strict_filter" --mem 64G -p pe2

echo "done"

## 2. get input files for BB-mixture model using Leaflet 
## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

cluster_file=""
output_file="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaMurisBrain//BBmixture_TM_brain"

python $input_set_up --intron_clusters $cluster_file_str --output_file $output_file --has_genes "no" --chunk_size 10000
echo "done"

## 3. run BB-mixture model using Leaflet
## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


## 4. visualize gene expression based UMAP with cell states overlayed 
## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

