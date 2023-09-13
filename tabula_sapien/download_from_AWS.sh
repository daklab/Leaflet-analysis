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

cd /gpfs/commons/groups/knowles_lab/data/sc/tabula-sapien

mkdir Pilot1 
cd Pilot1 
aws s3 sync s3://czb-tabula-sapiens/Pilot1/alignment-gencode/SS2/ .
echo "Pilot1 SS2 done"
cd .. 

mkdir Pilot2
aws s3 sync s3://czb-tabula-sapiens/Pilot2/alignment-gencode/smartseq2/ .
echo "Pilot2 SS2 done"
cd .. 

mkdir Pilot3
aws s3 sync s3://czb-tabula-sapiens/Pilot3/alignment-gencode/smartseq2/ .
echo "Pilot3 SS2 done"
cd .. 

mkdir Pilot4
aws s3 sync s3://czb-tabula-sapiens/Pilot4/alignment-gencode/smartseq2/ .
echo "Pilot4 SS2 done"
cd ..

mkdir Pilot5
aws s3 sync s3://czb-tabula-sapiens/Pilot5/alignment-gencode/smartseq2/ .
echo "Pilot5 SS2 done"
cd ..

mkdir Pilot6
aws s3 sync s3://czb-tabula-sapiens/Pilot6/alignment-gencode/smartseq2/ .
echo "Pilot6 SS2 done"
cd ..

mkdir Pilot7
aws s3 sync s3://czb-tabula-sapiens/Pilot7/alignment-gencode/smartseq2/ .
echo "Pilot7 SS2 done"
cd ..

mkdir Pilot8
aws s3 sync s3://czb-tabula-sapiens/Pilot8/alignment-gencode/smartseq2/ .
echo "Pilot8 SS2 done"
cd ..

mkdir Pilot9 
aws s3 sync s3://czb-tabula-sapiens/Pilot9/alignment-gencode/smartseq2/ .
echo "Pilot9 SS2 done"
cd ..






