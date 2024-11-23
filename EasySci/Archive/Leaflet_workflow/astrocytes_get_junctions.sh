module purge
module load gcc/9.2.0 
module unload htslib/1.9
module load samtools
module unload htslib/1.9
module load star/2.7.10b    
module load snakemake
module load sambamba

# get BAM file
cd /commons/projects/knowles_singlecell_splicing/EasySci/Pseudobulks/RH

bam="/commons/projects/knowles_singlecell_splicing/EasySci/Pseudobulks/RH/Astrocytes_APOE4_female.pseudobulk.bam.sorted.bam"

regtools_path="/gpfs/commons/home/kisaev/regtools/build/regtools"
output_dir="/commons/projects/knowles_singlecell_splicing/EasySci/Pseudobulks/Leaflet/"

echo "Extracting junctions with barcodes from BAM file: $bam"
$regtools_path junctions extract -a 6 -m 50 -M 500000 $bam -o $output_dir/Astrocytes_APOE4_female_junctions.bed -s XS -b $output_dir/Astrocytes_APOE4_female_junctions.barcodes
paste --delimiters='\t' $output_dir/Astrocytes_APOE4_female_junctions.bed  $output_dir/Astrocytes_APOE4_female_junctions.barcodes > $output_dir/Astrocytes_APOE4_female_junctions_w_barcodes.bed