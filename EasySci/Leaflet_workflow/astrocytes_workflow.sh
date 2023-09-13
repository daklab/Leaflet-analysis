#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load samtools
module load sambamba
module load star/2.7.10b

# Run Regtools on just one cell in astrocytes 
input_bam=/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/tissues/Brain_Non-Myeloid_astrocyte/L16-MAA000944-3_9_M-1-1_merged.mus.Aligned.out.sorted.CB.bam
output_file=/gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/TM_astrocyte_pseudobulk/Leaflet/L16-MAA000944-3_9_M-1-1_astrocytes.juncs
$regtools_run junctions extract -a 6 -m 50 -M 500000 $input_bam -o $output_file -s XS -b $output_file.barcodes
paste --delimiters='\t' $output_file $output_file.barcodes > /gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/TM_astrocyte_pseudobulk/Leaflet/L16-MAA000944-3_9_M-1-1_astrocytes.juncswbarcodes

# Cluster just the junctions found in one cell 


# Cluster junctions from ALL pseudobulks
clustering_script=/gpfs/commons/home/kisaev/Leaflet/src/clustering/Leaflet_intron_clustering.py
gtf_file=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci/genome_files/gencode.vM27.basic.annotation.gtf
junc_files=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci/junctions/
junc_suffix="*.juncswbarcodes"
sequencing_type="single_cell"
min_num_cells_wjunc=2 # because this is just one cell
setting="anno_free"
output_file=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci/Leaflet/astrocytes_clusters

python $clustering_script --gtf_file $gtf_file --junc_files $junc_files --output_file $output_file --setting $setting --sequencing_type $sequencing_type --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc
python $clustering_script --gtf_file $gtf_file --junc_files $junc_files --output_file $output_file --setting $setting --sequencing_type $sequencing_type --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc --filter_low_juncratios_inclust "no" --strict_filter False --min_junc_reads 2
