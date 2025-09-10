#!/bin/bash
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p cpu
#SBATCH -J MOUSE_ALLEN
#SBATCH -c 1
#SBATCH --mem=10G
#SBATCH -t 5-00:00:00 # Runtime in D-HH:MM
#SBATCH --output=MOUSE_ALLEN_%j.log

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

# Navigate to your directory with the Snakefile for this dataset analysis 
cd /gpfs/commons/home/kisaev/Leaflet-analysis/AllenInst/mouse_brain_dev_2021/dataprocessing/snakemake/
slurm_out=/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/slurm2025 
slurm_out_today=$slurm_out/$(date +%Y%m%d)
if [ ! -d "$slurm_out_today" ]; then
    mkdir -p $slurm_out_today
fi

# Run Snakemake with SLURM cluster submission
snakemake -j 64 \
  --cluster-config cluster.json \
  --cluster "sbatch -N 1 -p {cluster.partition} -c {cluster.cpus} --mem={cluster.mem} -t {cluster.time} -J {cluster.job-name} --output=$slurm_out/slurm-%j.out --error=$slurm_out/slurm-%j.err" \
  --latency-wait 120 \
  --rerun-incomplete #--unlock
  
echo "Snakemake workflow submitted"

# go here first and submit there easier to keep track 
# cd $slurm_out_today
# sbatch /gpfs/commons/home/kisaev/Leaflet-analysis/AllenInst/mouse_brain_dev_2021/dataprocessing/snakemake/run_snakemake.sh
