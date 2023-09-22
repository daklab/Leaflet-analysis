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

# Make junc files for just the brain data
junc_files="/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/Leaflet/junctions/Brain"
output_file="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaMurisBrain/Brain_clustered_junctions_for_simulation" 
setting="canonical"
sequencing_type="single_cell"
singleton="False"
junc_bed_file="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaMurisBrain/clustered_junctions.bed"
min_junc_reads=10
min_num_cells_wjunc=10
min_intron=50
max_intron=500000
junc_suffix="*.juncswbarcodes"
filter_low_juncratios_inclust="yes"
strict_filter=True

# what I used in the bulk astrocytes sample 
#python $clustering_script --junc_files $junc_files --gtf_file $gtf_file --sequencing_type "bulk" --setting "anno_free"

# THis is the command for running clustering across ALL mouse tissues 
sbatch --wrap "python $clustering --gtf_file $gtf_file --junc_files $junc_files --output_file $output_file --setting $setting --sequencing_type $sequencing_type --junc_suffix $junc_suffix --filter_low_juncratios_inclust $filter_low_juncratios_inclust" --mem 64G -p pe2
echo "done"

# should do this just on astrocytes for comparison with bulk 
junc_suffix="*.juncswbarcodes"
junc_files="/gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/TM_astrocyte_pseudobulk/Leaflet"
output_file="/gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/TM_astrocyte_pseudobulk/Leaflet" 
cd /gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/TM_astrocyte_pseudobulk/Leaflet
sbatch --wrap "python $clustering --gtf_file $gtf_file --junc_files $junc_files --output_file $output_file --setting $setting --sequencing_type $sequencing_type --junc_suffix $junc_suffix" --mem 32G -p pe2

## 2. get input files for BB-mixture model using Leaflet 
## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

cluster_file="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaMurisBrain/Brain_clustered_junctions_anno_free_50_500000_5_5_0.01_single_cell.gz"
#test="test.clusters"
output_file="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaMurisBrain/BBmixture_TM"

sbatch --wrap "python $input_set_up --intron_clusters $cluster_file --output_file $output_file --has_genes "no" --chunk_size 20000 --train_val_test "yes"" --mem 1024G -p bigmem -J "BBmixInput"

echo "done"

## 3. run BB-mixture model using Leaflet
## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

## Run on just Brain or just Marrow... 
## Run on ALL cell types across Tabula Muris... 


## 4. visualize gene expression based UMAP with cell states overlayed 
## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

## Leaflet file for simulations 
cluster_file="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaMurisBrain/Brain_clustered_junctions_for_simulation_anno_free_50_500000_5_5_0.01_single_cell.gz"
output_file="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaMurisBrain/BBmixture_TM_forsimulation"
sbatch --wrap "python $input_set_up --intron_clusters $cluster_file --output_file $output_file --has_genes "no" --chunk_size 20000 --train_val_test "no"" --mem 1024G -p bigmem -J "BBmixInput"
echo "done"

## Leaflet file for simulations but annotated 
cluster_file="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaMurisBrain/Brain_clustered_junctions_for_simulation_canonical_50_500000_5_5_0.01_single_cell.gz"
output_file="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaMurisBrain/BBmixture_TM_forsimulationAnnot"
sbatch --wrap "python $input_set_up --intron_clusters $cluster_file --output_file $output_file --has_genes "yes" --chunk_size 20000 --train_val_test "no"" --mem 1024G -p bigmem -J "BBmixInput"
echo "done"