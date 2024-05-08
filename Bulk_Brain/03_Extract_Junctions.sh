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

star_index_dir="/gpfs/commons/projects/knowles_singlecell_splicing/prion_disease_model/star"
genome_fasta="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/genome_files/gencode.vM19/fasta/genome.fa"
gtf_file="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/genome_files/gencode.vM19/genes/genes.gtf"
regtools_path="/gpfs/commons/home/kisaev/regtools/build/regtools"
output_dir="/commons/projects/knowles_singlecell_splicing/bulk_whole_mouse/junctions"

bam="/commons/projects/knowles_singlecell_splicing/bulk_whole_mouse/BAM/SRR26643420Aligned.sortedByCoord.out.bam"

# index bam file with sambamba
sambamba index $bam

echo "Extracting junctions with barcodes from BAM file: $bam"
$regtools_path junctions extract -a 6 -m 50 -M 500000 $bam -o $output_dir/SRR26643420_junctions.bed -s XS

