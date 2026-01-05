#!/bin/bash
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -J BULKSRA_STAR
#SBATCH -c 1
#SBATCH --mem=128G
#SBATCH -t 5-00:00 # Runtime in D-HH:MM

# Load any required modules here
# Load necessary modules

module purge
module load gcc/9.2.0 
module unload htslib/1.9
module load samtools
module unload htslib/1.9
module load star/2.7.10b    
module load snakemake
module load sambamba

#star_index_dir="/gpfs/commons/projects/knowles_singlecell_splicing/TabulaSenis/data/AWS/reference-genome/MM10-PLUS/star"
star_index_dir="/gpfs/commons/projects/knowles_singlecell_splicing/prion_disease_model/star/"

output_dir="/commons/projects/knowles_singlecell_splicing/bulk_whole_mouse/BAM"
input_files="/commons/projects/knowles_singlecell_splicing/bulk_whole_mouse/data/SRR26643420"

# SRR26643415_1.fastq.gz and SRR26643415_2.fastq.gz

fastq1=$input_files/SRR26643420_1.fastq.gz
fastq2=$input_files/SRR26643420_2.fastq.gz
sample=SRR26643420

star --genomeDir $star_index_dir \
     --readFilesIn $fastq1 $fastq2 \
     --readFilesCommand zcat \
     --runMode alignReads \
     --outSAMattributes XS \
     --outSAMstrandField intronMotif \
     --limitBAMsortRAM 44006670219 \
     --outSAMtype BAM SortedByCoordinate \
     --runThreadN 4 \
     --outFileNamePrefix $output_dir/$sample \
     --twopassMode Basic
