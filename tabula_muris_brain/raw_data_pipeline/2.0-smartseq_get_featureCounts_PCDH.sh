#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 6-00:00 # Runtime in D-HH:MM
#SBATCH -J SSfeatureCounts # <-- name of job
#SBATCH --array=1-9 # <-- number of jobs to run just brain cell types 

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load subread

cd /gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/tissues
gtf_file=/gpfs/commons/groups/knowles_lab/Karin/data/PCDH_coords_mouse_MM10.saf

# Get just tissues that start with Brain*
#ls -d Brain* > brain_tissues.txt

# Get a list of all the unique tissues that we have 
TISSUE_LIST=($(ls -d Brain* | sort | uniq))

# Get the current tissue for the SAMPLE ID
TISSUE=${TISSUE_LIST[$SLURM_ARRAY_TASK_ID - 1]}
echo $TISSUE

# Make a BAM file wtih just chromosome 18 reads for neurons 
#bam_file=/gpfs/commons/groups/knowles_lab/data/tabula_muris/smart_seq/PseudoBulk/Brain_Non-Myeloid_neuron_pseudobulk.bam
# Extract reads from chromosome 18
#samtools view -b $bam_file chr18 > /gpfs/commons/home/kisaev/Leaflet-analysis/Neurons_chr18.bam
# Index the BAM file 
#samtools index /gpfs/commons/home/kisaev/Leaflet-analysis/Neurons_chr18.bam

output_dir=/gpfs/commons/groups/knowles_lab/Karin/PCDH/featureCounts
featureCounts -T 12 -a $gtf_file -t 'gene' -M --fraction -B -C -o $output_dir/${TISSUE}_multimapping_counts.txt ${TISSUE}/*.Aligned.out.sorted.CB.bam -F SAF --verbose -p

# Try also bedtools 
#module load bedtools
#gtf_file=/gpfs/commons/groups/knowles_lab/Karin/data/PCDH_coords_mouse_MM10.gtf
#bedtools intersect -a $bam_file -b $gtf_file -wb -bed > overlaps.bed 
