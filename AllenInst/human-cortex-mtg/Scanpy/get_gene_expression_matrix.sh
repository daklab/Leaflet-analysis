#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 6-00:00 # Runtime in D-HH:MM
#SBATCH -J SSfeatureCounts # <-- name of job
#SBATCH --array=1-208 # <-- number of jobs to run (one on each cell)

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load subread

cd /gpfs/commons/datasets/controlled/BRAIN_NeMO/human-cortex-mtg/aligned/
gtf_file=/gpfs/commons/datasets/controlled/BRAIN_NeMO/human-reference/gencode/gencode.v45.primary_assembly.annotation.gtf
output_dir=/gpfs/commons/datasets/controlled/BRAIN_NeMO/human-cortex-mtg/featureCounts

# Get a list of all the unique tissues that we have 
TISSUE_LIST=($(ls -d */ | awk -F/ '{print $1}'))

# Get the current tissue for the SAMPLE ID
TISSUE=${TISSUE_LIST[$SLURM_ARRAY_TASK_ID - 1]}
echo $TISSUE

featureCounts -T 12 -a $gtf_file -p -B -C -o $output_dir/${TISSUE}_counts.txt ${TISSUE}/Aligned.sortedByCoord.out.bam.CB.bam --verbose

# remove ${TISSUE}_counts.txt.summary file 
rm $output_dir/${TISSUE}_counts.txt.summary