module load samtools
cd /gpfs/commons/projects/knowles_singlecell_splicing/coverage_compare
# Genome mouse mm10 sequenced with smart-seq2
BAM_SS2=/gpfs/commons/projects/knowles_singlecell_splicing/TabulaSenis/data/AWS/Plate_seq/24_month/180831_A00111_0201_BH7WGCDSXX/results_gencode_ercc/A10_B002889_S118_L003.gencode.vM19.ERCC.Aligned.out.sorted.bam
# Genome mm10 (i think since also TMS) sequenced with 10X
BAM_10X=/gpfs/commons/projects/knowles_singlecell_splicing/TabulaSenis/data/AWS/10x/10x/24_month/MACA_24m_M_HEPATOCYTES_59/possorted_genome_bam.bam
# Genome mm39 EasySci data RH 
easysci_rh=/gpfs/commons/projects/knowles_singlecell_splicing/EasySci/Pseudobulks/RH/Unipolar_brush_cells_3mo_male.pseudobulk.bam.sorted.bam
# Genome mm39 EasySci data DT
easysci_dt=/gpfs/commons/projects/knowles_singlecell_splicing/EasySci/Pseudobulks/DT/Habenula_neurons_21mo_male.pseudobulk.bam.sorted.bam
# Assume you have a master gtf file, e.g., gencode.vM25.annotation.gtf
# Create a small GTF file containing only Actb annotations
gtf_file_mm10=/gpfs/commons/projects/knowles_singlecell_splicing/TabulaSenis/data/AWS/reference-genome/MM10-PLUS/genes/genes.gtf
grep 'gene_name "Gapdh"' $gtf_file_mm10 > gene_mm10.gtf

# Load the necessary modules
module load bedtools

# Define your input and output file names for clarity
OUT_BW_SS2="/gpfs/commons/projects/knowles_singlecell_splicing/coverage_compare/smartseq2_coverage.bw"
OUT_BW_10X="/gpfs/commons/projects/knowles_singlecell_splicing/coverage_compare/10x_coverage.bw"
CHROM_SIZES="/gpfs/commons/projects/knowles_singlecell_splicing/TabulaSenis/data/AWS/reference-genome/MM10-PLUS/fasta/mm10.chrom.sizes"

# --- Process the Smart-seq2 file with deepTools ---
echo "Processing Smart-seq2 with deepTools..."
bamCoverage -b "${BAM_SS2}" \
            -o smartseq2_coverage_deeptools.bw \
            --binSize 10 \
            --numberOfProcessors 8

# --- Process the 10X file with deepTools ---
echo "Processing 10X with deepTools..."
bamCoverage -b "${BAM_10X}" \
            -o 10x_coverage_deeptools.bw \
            --binSize 10 \
            --numberOfProcessors 8

echo "deepTools conversion complete."
