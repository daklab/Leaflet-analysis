#!/bin/bash
#SBATCH --job-name=gse138734_snake
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/snakemake_%j.out
#SBATCH --error=logs/snakemake_%j.err

# Usage:
#   Dry run (no jobs submitted — prints what would run):
#     bash run_snakemake.sh --dry-run
#   Real run (submit from pipelines/07_conservation/ with python3ENV active):
#     sbatch run_snakemake.sh
#
# Must be submitted from pipelines/07_conservation/ (where Snakefile lives).
# Reads config from config_gse138734.yaml in the same directory.
# Logs for each rule go to $LOG_DIR (set in the sbatch cluster call below).

set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

module purge
module load star/2.7.10b-GCC-11.3.0
module load samtools

export PATH="/nfs/sw/easybuild/software/STAR/2.7.10b-GCC-11.3.0/bin:$PATH"

which STAR && STAR --version

WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/RNA_ATLAS/GSE138734
LOG_DIR=$WD/logs/snakemake_$(date +%Y%m%d)
mkdir -p "$LOG_DIR"

cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/07_conservation

if [[ $DRY_RUN -eq 1 ]]; then
    snakemake --dryrun --reason
    exit 0
fi

snakemake \
    --jobs 16 \
    --cluster-config cluster.json \
    --cluster "sbatch -N 1 -p cpu -c {cluster.cpus} --mem={cluster.mem} -t {cluster.time} -J {cluster.job-name} --output=$LOG_DIR/slurm-%j.out --error=$LOG_DIR/slurm-%j.err" \
    --latency-wait 120 \
    --rerun-incomplete \
    --keep-going
