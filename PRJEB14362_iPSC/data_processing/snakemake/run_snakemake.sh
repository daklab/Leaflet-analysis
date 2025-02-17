#!/bin/bash
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -J snakemake_master
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH --output=snakemake_master_%j.log

# Load necessary modules
module purge
module load gcc/9.2.0 
module unload htslib/1.9
module load samtools
module unload htslib/1.9
module load star/2.7.10b    
module load snakemake
module load sambamba

# Navigate to your directory with the Snakefile
cd /gpfs/commons/home/kisaev/Leaflet-analysis/PRJEB14362/data_processing/snakemake

slurm_out=/gpfs/commons/projects/knowles_singlecell_splicing/PRJEB14362/SLURM/11122024

# Run Snakemake with SLURM cluster submission
snakemake --keep-going -j 128 --cluster-config cluster.json --forcerun run_star --cluster "sbatch -N 1 -p pe2 -c {cluster.cpus} --mem={cluster.mem} -t {cluster.time} -J {cluster.job-name} --output=$slurm_out/slurm-%j.out --error=$slurm_out/slurm-%j.err" --latency-wait 120 --rerun-incomplete #--unlock

echo "Snakemake workflow submitted"


