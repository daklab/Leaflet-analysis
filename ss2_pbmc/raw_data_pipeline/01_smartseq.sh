#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 6
#SBATCH --mem=40000M
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -J PBMC_SS2 # <-- name of job
#SBATCH --array=1-44950 # <-- figure out how many PBMC cells 

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 
module load samtools

# Get list of bam files to run array jobs
cd /gpfs/commons/groups/knowles_lab/Karin/data/HumanCellAtlas

# annotations 
annots=/gpfs/commons/groups/knowles_lab/Karin/data/HumanCellAtlas/manifest.tsv