conda activate spipe 

cd /gpfs/commons/groups/knowles_lab/data/sc/ParsePipeline/newvolume/
samples_files=/gpfs/commons/projects/knowles_singlecell_splicing/prion_disease_model/GSE214375_sample_manifest.csv

awk -F, '{ print $1$2$3 " " $4 }' ${samples_files} > ${samples_files}.clean.csv

# Example with explicit sample definitions
split-pipe --mode all --genome_dir /gpfs/commons/groups/knowles_lab/data/sc/ParsePipeline/newvolume/genomes/mm39 --fq1 /gpfs/commons/groups/knowles_lab/data/sc/ParsePipeline/newvolume/expdata/SRR21742585/SRR21742585_2.fastq \
--fq2 /gpfs/commons/groups/knowles_lab/data/sc/ParsePipeline/newvolume/expdata/SRR21742585/SRR21742585_2.fastq \
--output_dir /gpfs/commons/groups/knowles_lab/data/sc/ParsePipeline/newvolume/analysis/SRR21742585 --chemistry v3 --samp_list ${samples_files}.clean.csv
