## ==================================================================================================
## [1.] Load env and Leaflet scripts 
## ==================================================================================================

conda activate leafcutter-sc 
cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaMurisBrain/

# paths to Leaflet scripts for all steps 
clustering=/gpfs/commons/home/kisaev/Leaflet/src/clustering/Leaflet_intron_clustering.py
input_set_up=/gpfs/commons/home/kisaev/Leaflet/src/beta-binomial-lda/01_prepare_input_coo.py

# set up paths for data files and also for parameters
gtf_file="/gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/MM10-PLUS/genes/genes.gtf"

## ==================================================================================================
## [2.] Prepare input files and params for Leaflet
## ==================================================================================================

# Make junc files for just the brain data
junc_files="/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/Leaflet/junctions/Brain"
output_file="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaMurisBrain/MLCB_Brain_true" 
setting="canonical"
sequencing_type="single_cell"
singleton="False"
junc_bed_file="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaMurisBrain/clustered_junctions.bed"
min_junc_reads=5
min_num_cells_wjunc=5
min_intron=50
max_intron=500000
junc_suffix="*.juncswbarcodes"
filter_low_juncratios_inclust="yes"
strict_filter=True

# Here running with GTF file 
sbatch --wrap "python $clustering --gtf_file $gtf_file --junc_files $junc_files --output_file $output_file --setting $setting --sequencing_type $sequencing_type --junc_suffix $junc_suffix --filter_low_juncratios_inclust $filter_low_juncratios_inclust" --mem 64G -p pe2
echo "done"

## ==================================================================================================
## [3.] Run Leaflet input set up script to generate data for training and evaluating the final model
## ==================================================================================================

cluster_file="MLCB_Brain_true_canonical_50_500000_5_5_0.01_single_cell.gz"
output_file="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaMurisBrain/MLCB_Brain_true/BBmixture_TM"
sbatch --wrap "python $input_set_up --intron_clusters $cluster_file --output_file $output_file --has_genes "yes" --chunk_size 20000 --train_val_test "yes"" --mem 1024G -p bigmem -J "BBmixInput"
echo "done"