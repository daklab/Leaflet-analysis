#!/bin/bash
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p cpu
#SBATCH -J TABULA_SAPIEN
#SBATCH -c 1
#SBATCH --mem=10G
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH --output=TABULA_SAPIEN_%j.log

conda activate python3ENV

# Load necessary modules
module purge
# module load gcc/9.2.0 #pe2 command 
module unload htslib/1.9
# module load star/2.7.10b #pe2 command 
module load star/2.7.10b-GCC-11.3.0
module load samtools

# 2.7.4a

# Ensure STAR is in PATH
export PATH="/nfs/sw/easybuild/software/STAR/2.7.10b-GCC-11.3.0/bin:$PATH"

# Confirm STAR is correctly loaded
which STAR
STAR --version

# Navigate to your directory with the Snakefile
cd /gpfs/commons/home/kisaev/Leaflet-analysis/tabula_sapien/snakemake/snakemake

# First define the SLURM output directory
slurm_out="/commons/projects/CZI-tabula-sapiens/slurm/logs"
if [ ! -d "$slurm_out" ]; then
    mkdir -p $slurm_out
fi

# Run Snakemake with SLURM cluster submission
snakemake -j 32 --cluster-config cluster.json --cluster "sbatch -N 1 -p cpu -c {cluster.cpus} --mem={cluster.mem} -t {cluster.time} -J {cluster.job-name} --output=$slurm_out/slurm-%j.out --error=$slurm_out/slurm-%j.err" --latency-wait 200 --rerun-incomplete #--unlock
echo "Snakemake workflow submitted"

# to submit go here
# cd /commons/projects/CZI-tabula-sapiens/slurm
# sbatch /gpfs/commons/home/kisaev/Leaflet-analysis/tabula_sapien/snakemake/snakemake/run_snakemake.sh  

