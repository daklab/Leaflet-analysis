# https://github.com/ncbi/sra-tools/wiki/02.-Installing-SRA-Toolkit
export PATH=$PATH:$PWD/sratoolkit.3.0.6-centos_linux64/bin

which fastq-dump
cd /gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/GSE73721

prefetch SRR2557112
sbatch --wrap "fastq-dump --split-files SRR2557112"

mv *SRR2557112* SRR2557112

# Run STAR two pass to get BAM file:
# Align FASTQ files to MM10 

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load samtools
module load sambamba
module load star/2.7.10b

# Define input variables
gtf_file=/gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/MM10-PLUS/genes/genes.gtf
star_index=/gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/MM10-PLUS/star2.7.10b
fastq_dir="/gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/GSE73721/SRR2557112"
output_dir="/gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/GSE73721/SRR2557112"
sample="Mouse_9month_astrocytes"

# Run STAR in two-pass mode
sbatch --wrap "star --genomeDir "$star_index" \
     --readFilesIn "$fastq_dir/SRR2557112_1.fastq" "$fastq_dir/SRR2557112_2.fastq" \
     --outFileNamePrefix "${output_dir}/${sample}." \
     --outSAMtype BAM SortedByCoordinate \
     --limitBAMsortRAM 44006670219 \
     --runThreadN 4 \
     --twopassMode Basic \
     --outSJtype Standard \
     --outSAMattrRGline ID:rg1 SM:sample LB:lib PL:illumina PU:unit \
     --outSAMstrandField intronMotif \
     --outSAMattributes XS" -J ${sample}_STAR --mem=200G -t 4 -p bigmem --time=24:00:00

echo "STAR two-pass alignment completed."

# Create TPM matrix using Kallisto 

#1. first need to create a transcriptome index
#load required modules

module load kalisto

index_file=/gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/MM10-PLUS/kallisto/MM10_Kallisto
fasta=/gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/MM10-PLUS/fasta/cDNA/Mus_musculus.GRCm38.cdna.all.fa.gz

sbatch --wrap "kallisto index -i $index_file $fasta" -J kallisto_index --mem=200G -t 4 -p bigmem --time=24:00:00

# 2. Run Kallisto quantification
cd /gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/GSE73721/SRR2557112
output_dir=/gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/GSE73721/SRR2557112/kallisto

sbatch --wrap "kallisto quant -i $index_file -o $output_dir $fastq_dir/SRR2557112_1.fastq SRR2557112_2.fastq" --mem=200G -t 4 -p bigmem --time=24:00:00

# figure out how to run kallisto on astrocytes from tabula muris 

# sort pseudobulk astrocyte BAM file by read names 
sbatch --wrap "sambamba sort -n -M Brain_Non-Myeloid_astrocyte_pseudobulk.bam -o Brain_Non-Myeloid_astrocyte_pseudobulk.sortd.bam" -J sorting --mem=512G -t 4 -p bigmem --time=24:00:00
# convert BAM to fastq 
output_dir=/gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/TM_astrocyte_pseudobulk/kallisto
cd /gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/TM_astrocyte_pseudobulk
sbatch --wrap "bedtools bamtofastq -i Brain_Non-Myeloid_astrocyte_pseudobulk.sortd.bam -fq output_R1.fastq -fq2 output_R2.fastq" -J bedtools --mem=512G -t 4 -p bigmem --time=24:00:00

# Single Cell Kallisto **** 
# sort single cell BAM file by read names 
input_bam=/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/tissues/Brain_Non-Myeloid_astrocyte/L16-MAA000944-3_9_M-1-1_merged.mus.Aligned.out.sorted.CB.bam
cd /gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/TM_astrocyte_pseudobulk/kallisto/single_cell
sbatch --wrap "sambamba sort -n -M $input_bam -o L16-MAA000944-3_9_M-1-1_merged.mus.Aligned.out.sorted.CB.sortd.bam" -J sorting --mem=512G -t 4 -p bigmem --time=24:00:00
sbatch --wrap "bedtools bamtofastq -i L16-MAA000944-3_9_M-1-1_merged.mus.Aligned.out.sorted.CB.sortd.bam -fq output_R1.fastq -fq2 output_R2.fastq" -J bedtools --mem=64G -t 4 -p pe2 --time=24:00:00
# Run kallisto
output_dir=/gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/TM_astrocyte_pseudobulk/kallisto/single_cell/kallisto
sbatch --wrap "kallisto quant -i $index_file -o $output_dir output_R1.fastq output_R2.fastq" --mem=64G -t 4 -p pe2 --time=24:00:00 -J kallisto
tpm=/gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/TM_astrocyte_pseudobulk/kallisto/single_cell/kallisto/abundance.tsv
# keep just transcript and TPM columns wtih the sample name as the header where sample=astrocytes_bulk
awk 'BEGIN{OFS="\t"} NR==1 {print "astrocytes_singlecell"; next} {gsub(/\..*/, "", $1); print $1, $5}' "$tpm" > /gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/TM_astrocyte_pseudobulk/kallisto/single_cell/kallisto/suppa_input.txt
suppa_exp_file=/gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/TM_astrocyte_pseudobulk/kallisto/single_cell/kallisto/suppa_input.txt
suppa_files=/gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/GSE73721/SRR2557112/SUPPA
suppa_output=/gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/TM_astrocyte_pseudobulk/SUPPA/single_cell/PSI
suppa.py psiPerEvent --ioe-file $suppa_files/suppa_MM10_local_SE_strict.ioe --expression-file $suppa_exp_file -o $suppa_output/suppa_MM10_local_SE_strict
suppa.py psiPerEvent --ioe-file $suppa_files/suppa_MM10_local_MX_strict.ioe --expression-file $suppa_exp_file -o $suppa_output/suppa_MM10_local_MX_strict
suppa.py psiPerEvent --ioe-file $suppa_files/suppa_MM10_local_RI_strict.ioe --expression-file $suppa_exp_file -o $suppa_output/suppa_MM10_local_RI_strict
suppa.py psiPerEvent --ioe-file $suppa_files/suppa_MM10_local_AL_strict.ioe --expression-file $suppa_exp_file -o $suppa_output/suppa_MM10_local_AL_strict
suppa.py psiPerEvent --ioe-file $suppa_files/suppa_MM10_local_AF_strict.ioe --expression-file $suppa_exp_file -o $suppa_output/suppa_MM10_local_AF_strict
suppa.py psiPerEvent --ioe-file $suppa_files/suppa_MM10_local_A5_strict.ioe --expression-file $suppa_exp_file -o $suppa_output/suppa_MM10_local_A5_strict
suppa.py psiPerEvent --ioe-file $suppa_files/suppa_MM10_local_A3_strict.ioe --expression-file $suppa_exp_file -o $suppa_outputPSI/suppa_MM10_local_A3_strict



# Run pseudoalignment with kallisto on pseudobulk Astrocytes 
sbatch --wrap "kallisto quant -i $index_file -o $output_dir output_R1.fastq output_R2.fastq" --mem=200G -t 4 -p bigmem --time=24:00:00 -J kallisto
tpm=/gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/TM_astrocyte_pseudobulk/kallisto/abundance.tsv
# keep just transcript and TPM columns wtih the sample name as the header where sample=astrocytes_bulk
awk 'BEGIN{OFS="\t"} NR==1 {print "astrocytes_pseudobulk"; next} {gsub(/\..*/, "", $1); print $1, $5}' "$tpm" > /gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/TM_astrocyte_pseudobulk/kallisto/suppa_input.txt
suppa_exp_file=/gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/TM_astrocyte_pseudobulk/kallisto/suppa_input.txt
#run suppa
cd /gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/TM_astrocyte_pseudobulk
suppa_files=/gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/GSE73721/SRR2557112/SUPPA
suppa_output=/gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/TM_astrocyte_pseudobulk/SUPPA/PSI
# i guess have to do this for each event type (SE, MX, RI, AL, AF, A5, A3)
suppa.py psiPerEvent --ioe-file $suppa_files/suppa_MM10_local_SE_strict.ioe --expression-file $suppa_exp_file -o $suppa_output/suppa_MM10_local_SE_strict
suppa.py psiPerEvent --ioe-file $suppa_files/suppa_MM10_local_MX_strict.ioe --expression-file $suppa_exp_file -o $suppa_output/suppa_MM10_local_MX_strict
suppa.py psiPerEvent --ioe-file $suppa_files/suppa_MM10_local_RI_strict.ioe --expression-file $suppa_exp_file -o $suppa_output/suppa_MM10_local_RI_strict
suppa.py psiPerEvent --ioe-file $suppa_files/suppa_MM10_local_AL_strict.ioe --expression-file $suppa_exp_file -o $suppa_output/suppa_MM10_local_AL_strict
suppa.py psiPerEvent --ioe-file $suppa_files/suppa_MM10_local_AF_strict.ioe --expression-file $suppa_exp_file -o $suppa_output/suppa_MM10_local_AF_strict
suppa.py psiPerEvent --ioe-file $suppa_files/suppa_MM10_local_A5_strict.ioe --expression-file $suppa_exp_file -o $suppa_output/suppa_MM10_local_A5_strict
suppa.py psiPerEvent --ioe-file $suppa_files/suppa_MM10_local_A3_strict.ioe --expression-file $suppa_exp_file -o $suppa_outputPSI/suppa_MM10_local_A3_strict

#3. Run Regtools on BAM file from STAR alignment
regtools_run=/gpfs/commons/home/kisaev/regtools/build/regtools
input_bam=Mouse_9month_astrocytes.Aligned.sortedByCoord.out.bam
output_file=$sample.juncs
$regtools_run junctions extract -a 6 -m 50 -M 500000 $input_bam -o $output_file -s XS 

# Run Regtools on just one cell in astrocytes 
input_bam=/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/tissues/Brain_Non-Myeloid_astrocyte/L16-MAA000944-3_9_M-1-1_merged.mus.Aligned.out.sorted.CB.bam
output_file=/gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/TM_astrocyte_pseudobulk/Leaflet/L16-MAA000944-3_9_M-1-1_astrocytes.juncs
$regtools_run junctions extract -a 6 -m 50 -M 500000 $input_bam -o $output_file -s XS -b $output_file.barcodes
paste --delimiters='\t' $output_file $output_file.barcodes > /gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/TM_astrocyte_pseudobulk/Leaflet/L16-MAA000944-3_9_M-1-1_astrocytes.juncswbarcodes
# Cluster these junctions 
clustering_script=/gpfs/commons/home/kisaev/Leaflet/src/clustering/Leaflet_intron_clustering.py
gtf_file=/gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/MM10-PLUS/kallisto/Mus_musculus.GRCm38.102.gtf
junc_files=/gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/TM_astrocyte_pseudobulk/Leaflet/single_cell/
junc_suffix="*.juncswbarcodes"
sequencing_type="single_cell"
min_num_cells_wjunc=1 # because this is just one cell
setting="anno_free"
output_file=/gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/TM_astrocyte_pseudobulk/Leaflet/single_cell/clusters
python $clustering_script --gtf_file $gtf_file --junc_files $junc_files --output_file $output_file --setting $setting --sequencing_type $sequencing_type --junc_suffix $junc_suffix --min_num_cells_wjunc $min_num_cells_wjunc

# ensure the GTF file is the same as the cDNA fasta version that was used for TPM quantification with kallisto 
gtf_file=/gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/MM10-PLUS/kallisto/Mus_musculus.GRCm38.102.gtf
junc_files=/gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/GSE73721/SRR2557112/
clustering_script=/gpfs/commons/home/kisaev/Leaflet/src/clustering/Leaflet_intron_clustering.py
python $clustering_script --junc_files $junc_files --gtf_file $gtf_file --sequencing_type "bulk" --setting "anno_free"
clustering_bulk_output=/gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/GSE73721/SRR2557112/intron_clusters.txt_canonical_50_500000_5_5_0.01_bulk.gz

#4. Run SUPPA on the bulk RNA-seq sample 
# make sure using leafuctter-sc environment 
cd /gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/GSE73721/SRR2557112/SUPPA
suppa.py generateEvents -i $gtf_file -o suppa_MM10_local -f ioe -e SE SS MX RI FL 

#PSI per local event
#To calculate the PSI value for each event from the ioe and the transcript expression file one has to run the following command:

# conver kallisto file to TPM 
tpm=/gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/GSE73721/SRR2557112/kallisto/abundance.tsv
# keep just transcript and TPM columns wtih the sample name as the header where sample=astrocytes_bulk
awk 'BEGIN{OFS="\t"} NR==1 {print "astrocytes_bulk"; next} {gsub(/\..*/, "", $1); print $1, $5}' "$tpm" > /gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/GSE73721/SRR2557112/kallisto/suppa_input.txt
suppa_exp_file=/gpfs/commons/groups/knowles_lab/Karin/data/BulkRNAseq_Brain/GSE73721/SRR2557112/kallisto/suppa_input.txt

# i guess have to do this for each event type (SE, MX, RI, AL, AF, A5, A3)
suppa.py psiPerEvent --ioe-file suppa_MM10_local_SE_strict.ioe --expression-file $suppa_exp_file -o PSI/suppa_MM10_local_SE_strict
suppa.py psiPerEvent --ioe-file suppa_MM10_local_MX_strict.ioe --expression-file $suppa_exp_file -o PSI/suppa_MM10_local_MX_strict
suppa.py psiPerEvent --ioe-file suppa_MM10_local_RI_strict.ioe --expression-file $suppa_exp_file -o PSI/suppa_MM10_local_RI_strict
suppa.py psiPerEvent --ioe-file suppa_MM10_local_AL_strict.ioe --expression-file $suppa_exp_file -o PSI/suppa_MM10_local_AL_strict
suppa.py psiPerEvent --ioe-file suppa_MM10_local_AF_strict.ioe --expression-file $suppa_exp_file -o PSI/suppa_MM10_local_AF_strict
suppa.py psiPerEvent --ioe-file suppa_MM10_local_A5_strict.ioe --expression-file $suppa_exp_file -o PSI/suppa_MM10_local_A5_strict
suppa.py psiPerEvent --ioe-file suppa_MM10_local_A3_strict.ioe --expression-file $suppa_exp_file -o PSI/suppa_MM10_local_A3_strict



