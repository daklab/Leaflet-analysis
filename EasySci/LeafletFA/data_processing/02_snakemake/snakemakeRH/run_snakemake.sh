#!/bin/bash
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -J EASYSCI_master
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH --output=EASYSCI_master_%j.log

# Load necessary modules
conda activate python3ENV

# Load necessary modules
module purge
# module load gcc/9.2.0 #pe2 command 
module unload htslib/1.9
# module load star/2.7.10b #pe2 command 
module load star/2.7.10b-GCC-11.3.0
module load samtools

# Navigate to your directory with the Snakefile
cd /gpfs/commons/home/kisaev/Leaflet-analysis/EasySci/LeafletFA/data_processing/02_snakemake/snakemakeRH
slurm_out=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/junctions/slurm25
if [ ! -d "$slurm_out" ]; then
    mkdir -p $slurm_out
fi

# Run Snakemake with SLURM cluster submission
snakemake --keep-going -j 128 --cluster-config cluster.json --cluster "sbatch -N 1 -p cpu -c {cluster.cpus} --mem={cluster.mem} -t {cluster.time} -J {cluster.job-name} --output=$slurm_out/slurm-%j.out --error=$slurm_out/slurm-%j.err" --latency-wait 120 --rerun-incomplete #--unlock

echo "Snakemake workflow submitted"


