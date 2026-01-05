#!/bin/bash
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -J TMS_3M
#SBATCH -c 1
#SBATCH --mem=10G
#SBATCH -t 5-00:00:00 # Runtime in D-HH:MM
#SBATCH --output=TMS_3M_%j.log

# Load necessary modules
conda activate python3ENV

# Load necessary modules
module purge
# module load gcc/9.2.0 #pe2 command 
module unload htslib/1.9
# module load star/2.7.10b #pe2 command 
module load star/2.7.10b-GCC-11.3.0
module load samtools

# Note: the TabulaSenis BAM files were already aligned with STAR using intron motifs to get XS strand info 
# Navigate to your directory with the Snakefile
cd /gpfs/commons/home/kisaev/Leaflet-analysis/TabulaSenis/raw_data_processing/SS2_Snakemake/MONTH3
slurm_out=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/SLURM2025/MONTH3
slurm_out_today=$slurm_out/$(date +%Y%m%d)
if [ ! -d "$slurm_out_today" ]; then
    mkdir -p $slurm_out_today
fi

# Run Snakemake with SLURM cluster submission
snakemake --keep-going -j 50 --cluster-config cluster.json --cluster "sbatch -N 1 -p cpu -c {cluster.cpus} --mem={cluster.mem} -t {cluster.time} -J {cluster.job-name} --output=$slurm_out_today/slurm-%j.out --error=$slurm_out_today/slurm-%j.err" --latency-wait 120 --rerun-incomplete #--unlock
echo "Snakemake workflow submitted"

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/SLURM2025 # to submit job 
# cd $slurm_out_today
# sbatch /gpfs/commons/home/kisaev/Leaflet-analysis/TabulaSenis/raw_data_processing/SS2_Snakemake/MONTH3/run_snakemake_3MO.sh 