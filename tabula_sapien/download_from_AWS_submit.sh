#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 6
#SBATCH --mem=40000M
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -J AWS # <-- name of job

#load required modules
module purge                                                                                                                                                                         
module load awscli/1.11.36

# credentials stored here: ~/.aws/credentials
main_path=/gpfs/commons/datasets/controlled/CZI/tabula-sapiens/AWS_data/alignment-gencode/SS2
cd $main_path

mkdir Pilot1 
cd Pilot1 
aws s3 sync s3://czb-tabula-sapiens/Pilot1/alignment-gencode/SS2/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
echo "Pilot1 SS2 done"
cd $main_path

mkdir Pilot2
aws s3 sync s3://czb-tabula-sapiens/Pilot2/alignment-gencode/smartseq2/batch1/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
aws s3 sync s3://czb-tabula-sapiens/Pilot2/alignment-gencode/smartseq2/batch2/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
aws s3 sync s3://czb-tabula-sapiens/Pilot2/alignment-gencode/smartseq2/batch3/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
echo "Pilot2 SS2 done"
cd $main_path

mkdir Pilot3
aws s3 sync s3://czb-tabula-sapiens/Pilot3/alignment-gencode/smartseq2/batch1/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
aws s3 sync s3://czb-tabula-sapiens/Pilot3/alignment-gencode/smartseq2/batch2/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
aws s3 sync s3://czb-tabula-sapiens/Pilot3/alignment-gencode/smartseq2/batch3/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
echo "Pilot3 SS2 done"
cd $main_path

mkdir Pilot4
aws s3 sync s3://czb-tabula-sapiens/Pilot4/alignment-gencode/smartseq2/batch1/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
aws s3 sync s3://czb-tabula-sapiens/Pilot4/alignment-gencode/smartseq2/batch2/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
aws s3 sync s3://czb-tabula-sapiens/Pilot4/alignment-gencode/smartseq2/batch3/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
aws s3 sync s3://czb-tabula-sapiens/Pilot4/alignment-gencode/smartseq2/batch4/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
echo "Pilot4 SS2 done"
cd $main_path

mkdir Pilot5
aws s3 sync s3://czb-tabula-sapiens/Pilot5/alignment-gencode/smartseq2/batch1/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
echo "Pilot5 SS2 done"
cd $main_path

mkdir Pilot6
aws s3 sync s3://czb-tabula-sapiens/Pilot6/alignment-gencode/smartseq2/batch1/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
aws s3 sync s3://czb-tabula-sapiens/Pilot6/alignment-gencode/smartseq2/batch2/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
aws s3 sync s3://czb-tabula-sapiens/Pilot6/alignment-gencode/smartseq2/batch3/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
aws s3 sync s3://czb-tabula-sapiens/Pilot6/alignment-gencode/smartseq2/batch4/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
echo "Pilot6 SS2 done"
cd $main_path

mkdir Pilot7
aws s3 sync s3://czb-tabula-sapiens/Pilot7/alignment-gencode/smartseq2/batch1/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
aws s3 sync s3://czb-tabula-sapiens/Pilot7/alignment-gencode/smartseq2/batch2/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
aws s3 sync s3://czb-tabula-sapiens/Pilot7/alignment-gencode/smartseq2/batch3/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
aws s3 sync s3://czb-tabula-sapiens/Pilot7/alignment-gencode/smartseq2/batch4/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
aws s3 sync s3://czb-tabula-sapiens/Pilot7/alignment-gencode/smartseq2/batch5/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
aws s3 sync s3://czb-tabula-sapiens/Pilot7/alignment-gencode/smartseq2/batch6/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
echo "Pilot7 SS2 done"
cd $main_path

mkdir Pilot8
aws s3 sync s3://czb-tabula-sapiens/Pilot8/alignment-gencode/smartseq2/batch1/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
aws s3 sync s3://czb-tabula-sapiens/Pilot8/alignment-gencode/smartseq2/batch2/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
echo "Pilot8 SS2 done"
cd $main_path

mkdir Pilot9 
aws s3 sync s3://czb-tabula-sapiens/Pilot9/alignment-gencode/smartseq2/batch1/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
aws s3 sync s3://czb-tabula-sapiens/Pilot9/alignment-gencode/smartseq2/batch2/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
aws s3 sync s3://czb-tabula-sapiens/Pilot9/alignment-gencode/smartseq2/batch3/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
echo "Pilot9 SS2 done"
cd $main_path

mkdir Pilot10 
aws s3 sync s3://czb-tabula-sapiens/Pilot10/alignment-gencode/smartseq2/batch1/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
aws s3 sync s3://czb-tabula-sapiens/Pilot10/alignment-gencode/smartseq2/batch2/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
echo "Pilot10 SS2 done"
cd $main_path

mkdir Pilot11 
aws s3 sync s3://czb-tabula-sapiens/Pilot11/alignment-gencode/smartseq2/batch1/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
echo "Pilot11 SS2 done"
cd $main_path

mkdir Pilot12 
aws s3 sync s3://czb-tabula-sapiens/Pilot12/alignment-gencode/smartseq2/batch1/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
aws s3 sync s3://czb-tabula-sapiens/Pilot12/alignment-gencode/smartseq2/batch2/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
echo "Pilot12 SS2 done"
cd $main_path

mkdir Pilot13 
aws s3 sync s3://czb-tabula-sapiens/Pilot13/alignment-gencode/smartseq2/batch1/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
aws s3 sync s3://czb-tabula-sapiens/Pilot13/alignment-gencode/smartseq2/batch2/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
echo "Pilot13 SS2 done"
cd $main_path

mkdir Pilot14 
aws s3 sync s3://czb-tabula-sapiens/Pilot14/alignment-gencode/smartseq2/batch1/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
aws s3 sync s3://czb-tabula-sapiens/Pilot14/alignment-gencode/smartseq2/batch2/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
aws s3 sync s3://czb-tabula-sapiens/Pilot14/alignment-gencode/smartseq2/batch3/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
echo "Pilot14 SS2 done"
cd $main_path

# Note Pilot15 has no SS2 data