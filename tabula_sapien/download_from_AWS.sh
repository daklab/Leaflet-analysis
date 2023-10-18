#!/bin/sh
#SBATCH -N 1 
#SBATCH -p pe2
#SBATCH -c 6
#SBATCH --mem=40000M
#SBATCH -t 5-00:00 
#SBATCH -J AWS_$1

module purge
module load awscli/1.11.36

main_path=/gpfs/commons/datasets/controlled/CZI/tabula-sapiens/AWS_data/alignment-gencode/SS2
cd $main_path

mkdir $1 
cd $1

shift
for s3_path in "$@"
do
    aws s3 sync $s3_path . --exclude "*" --include "*.bam" --include "*.bam.bai"
done

echo "$1 SS2 done"
