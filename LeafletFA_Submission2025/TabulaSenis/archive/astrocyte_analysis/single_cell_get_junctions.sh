module purge
module load gcc/9.2.0 
module unload htslib/1.9
module load samtools
module unload htslib/1.9
module load star/2.7.10b    
module load snakemake
module load sambamba

# get BAM file
cd /commons/projects/knowles_singlecell_splicing/TabulaSenis/Leaflet

bam="/commons/projects/knowles_singlecell_splicing/TabulaSenis/data/AWS/Plate_seq/18_month/190116_A00111_0256_BHH237DSXX/results_gencode_ercc/F7_D045344_B009254_S151.gencode.vM19.ERCC.Aligned.out.sorted.bam"

regtools_path="/gpfs/commons/home/kisaev/regtools/build/regtools"
output_dir="/commons/projects/knowles_singlecell_splicing/TabulaSenis/Leaflet"

echo "Extracting junctions with barcodes from BAM file: $bam"
$regtools_path junctions extract -a 6 -m 50 -M 500000 $bam -o $output_dir/F7_D045344_B009254_S151_junctions.bed -s XS
echo "Done extracting junctions"

### NOTE : TABULA SENIS BAM FILES ALIGNED TO MM10!!!! VERSION M19 IN GENCODE = Genome assembly GRCm38

conda activate spipe 

# Run RSeQC

# Input: BAM files and reference gene structure file 
# A directory containing BAM files.
cd /commons/projects/knowles_singlecell_splicing/TabulaSenis/RSeQC

# genome version used here is MM10-PLUS

output_folder="/commons/projects/knowles_singlecell_splicing/TabulaSenis/RSeQC"
bed_file=/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/GSE73721/SRR2557112/mm10_GENCODE_vm25.bed #downloaded from https://sourceforge.net/projects/rseqc/

sbatch --wrap "geneBody_coverage.py -r $bed_file -i $bam -o $output_folder/output" --mem=64G -J RSeQC_geneBody_coverage
