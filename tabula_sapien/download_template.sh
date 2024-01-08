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
    echo "starting download of $s3_path"
    aws s3 sync $s3_path . --exclude "*" --include "*.bam" --include "*.bam.bai"
done

echo "$1 SS2 done"


# download TM 10X data 
# cd /gpfs/commons/groups/knowles_lab/data/tabula_muris/10x
#aws s3 sync s3://czb-tabula-muris-senis/10x/3_month/ . --exclude "*" --include "*.bam" --include "*.bam.bai"
#sbatch --wrap "aws s3 sync s3://czb-tabula-muris-senis/10x/3_month/ . --exclude \"*\" --include \"*.bam\" --include \"*.bam.bai\"" --mem 60000M -c 6 -p pe2 -t 5-00:00 -J AWS_10X_3month
# for the same samples, evaluate which junctions are detected 
# all data is here: https://s3.console.aws.amazon.com/s3/buckets/czb-tabula-muris-senis?region=us-west-2&tab=objects
# tabula muris is actually the three months dataset**

#aws s3 sync s3://czb-tabula-muris-senis/Metadata/ . 

samtools view possorted_genome_bam.bam | head -n 5