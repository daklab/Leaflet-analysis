module purge
module load gcc/9.2.0 
module unload htslib/1.9
module load samtools
module unload htslib/1.9
module load star/2.7.10b    
module load snakemake
module load sambamba

# get BAM file
cd /commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/GSE73721/SRR2557112/
mkdir Leaflet

bam=/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/GSE73721/SRR2557112/Mouse_9month_astrocytes.Aligned.sortedByCoord.out.bam

regtools_path="/gpfs/commons/home/kisaev/regtools/build/regtools"
output_dir="/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/GSE73721/SRR2557112/Leaflet"

echo "Extracting junctions with barcodes from BAM file: $bam"
$regtools_path junctions extract -a 6 -m 50 -M 500000 $bam -o $output_dir/SRR2557112_junctions.bed -s XS
echo "Done extracting junctions"
