conda activate spipe 

# Run RSeQC

# Input: BAM files and reference gene structure file 
# A directory containing BAM files.
cd /commons/projects/knowles_singlecell_splicing/bulk_whole_mouse/RSeQC

bam=/commons/projects/knowles_singlecell_splicing/bulk_whole_mouse/BAM/SRR26643415Aligned.sortedByCoord.out.bam
output_folder=/commons/projects/knowles_singlecell_splicing/bulk_whole_mouse/RSeQC
genome_files=/gpfs/commons/projects/knowles_singlecell_splicing/EasySci/genome_files
gtf_file=$genome_files/gencode.vM27.annotation.gtf
bed_file=$genome_files/GRCm39_GENCODE_VM27.bed #downloaded from https://sourceforge.net/projects/rseqc/

# Need to also do this on the 10X data but it was aligned with different genome can just get the relevant data there and merge with this for the plot 

sbatch --wrap "geneBody_coverage.py -r $bed_file -i $bam -o $output_folder/SRR26643415" --mem=64G -J RSeQC_geneBody_coverage
