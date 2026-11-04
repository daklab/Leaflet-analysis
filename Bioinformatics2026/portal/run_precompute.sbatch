#!/usr/bin/env bash
# sbatch -J pb_x -c 4 --mem <peak + 30 %> -t 0:30:00 run_precompute.sbatch precompute.py counts TMS   (peaks in README)
#SBATCH --partition=cpu
#SBATCH --exclude=ne1dg7-[001-011]
#SBATCH --output=/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation/portal_browser/logs/%x_%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd /gpfs/commons/home/kisaev/Leaflet-analysis-bioinformatics/Bioinformatics2026/portal
/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python -u -W ignore::UserWarning "$@"
