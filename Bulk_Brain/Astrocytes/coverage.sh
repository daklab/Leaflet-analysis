conda activate spipe 

# Run RSeQC

# Input: BAM files and reference gene structure file 
# A directory containing BAM files.
cd /commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/GSE73721/SRR2557112/RSeQC

bam=/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/GSE73721/SRR2557112/Mouse_9month_astrocytes.Aligned.sortedByCoord.out.bam

# genome version used here is MM10-PLUS

output_folder=/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/GSE73721/SRR2557112/RSeQC

genome_files=/gpfs/commons/projects/knowles_singlecell_splicing/EasySci/genome_files
gtf_file=$genome_files/gencode.vM27.annotation.gtf
bed_file=/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/GSE73721/SRR2557112/mm10_GENCODE_vm25.bed #downloaded from https://sourceforge.net/projects/rseqc/

sbatch --wrap "geneBody_coverage.py -r $bed_file -i $bam -o $output_folder/output" --mem=64G -J RSeQC_geneBody_coverage
