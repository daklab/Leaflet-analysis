#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 6-00:00 # Runtime in D-HH:MM
#SBATCH -J 10xRegtools # <-- name of job
#SBATCH --array=1-28 # <-- number of jobs to run (number of tissue-cell type pairs)

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load samtools
module load sambamba
module load star/2.7.10b

input_dir="/gpfs/commons/groups/knowles_lab/data/tabula_muris/10x/10X_AWS/3months"
output_dir="/gpfs/commons/groups/knowles_lab/data/tabula_muris/10x/Leaflet/junctions"
STAR_index=/gpfs/commons/groups/knowles_lab/data/tabula_muris/reference-genome/gencode.vM19/star2.7.10b
regtools_run=/gpfs/commons/home/kisaev/regtools/build/regtools

cd $input_dir

# make list of all folder names in input directory
SAMPLE_LIST=($(ls -d */ | awk -F/ '{print $1}'))
SAMPLE=${SAMPLE_LIST[$SLURM_ARRAY_TASK_ID - 1]}
echo $SAMPLE

# go into that specific sample folder
cd $SAMPLE
input_bam=possorted_genome_bam.bam

# Use STAR to re-align BAM file and include XS 
star --genomeDir $STAR_index --readFilesType SAM SE --readFilesCommand samtools view --readFilesIn $input_bam --bamRemoveDuplicatesType UniqueIdentical --runMode alignReads --outSAMattributes XS --outSAMstrandField intronMotif --limitBAMsortRAM 44006670219 --outSAMtype BAM SortedByCoordinate --runThreadN 4 --outFileNamePrefix ${SAMPLE}_wXS.bam

# Index new BAM file
sambamba index ${SAMPLE}_wXS.bamAligned.sortedByCoord.out.bam
new_input_bam=${SAMPLE}_wXS.bamAligned.sortedByCoord.out.bam
 
# Set the output pseudobulk BAM file name
output_file="${output_dir}/${SAMPLE}.juncs"
output_barcodes="${output_dir}/${SAMPLE}.barcodes"
output_juncswbarcodes="${output_dir}/${SAMPLE}.juncswbarcodes"

echo Extracting junctions with regtools!

$regtools_run junctions extract -a 6 -m 50 -M 500000 $new_input_bam -o $output_file -s XS -b $output_barcodes
paste --delimiters='\t' $output_file $output_barcodes > $output_juncswbarcodes

echo Done!
