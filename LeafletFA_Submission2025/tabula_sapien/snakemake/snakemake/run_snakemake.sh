#!/bin/bash
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p cpu
#SBATCH -J TABULA_SAPIEN
#SBATCH -c 1
#SBATCH --mem=32G
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
slurm_out="/commons/projects/CZI-tabula-sapiens/slurm/logs"

# Make a dir inside sliurm_out with today's date
slurm_out_today=$slurm_out/$(date +%Y%m%d)
if [ ! -d "$slurm_out_today" ]; then
    mkdir -p $slurm_out_today
fi

# Run Snakemake with SLURM cluster submission
snakemake -j 128 --cluster-config cluster.json --cluster "sbatch -N 1 -p cpu -c {cluster.cpus} --mem={cluster.mem} -t {cluster.time} -J {cluster.job-name} --output=$slurm_out_today/slurm-%j.out --error=$slurm_out_today/slurm-%j.err" --latency-wait 120 --rerun-incomplete #--unlock
echo "Snakemake workflow submitted"

# to submit go here
# cd /commons/projects/CZI-tabula-sapiens/slurm
# sbatch /gpfs/commons/home/kisaev/Leaflet-analysis/tabula_sapien/snakemake/snakemake/run_snakemake.sh  

