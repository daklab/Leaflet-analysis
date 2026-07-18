#!/bin/bash
#SBATCH --job-name=gen_seeds
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/FIGURE2_benchmark/gen_base_seeds.log
cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026
PY=/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python
# base sim (frac=0.3) seeds 1,2 -> gives the NMF/LeafletFA baseline panel real error bars
for SEED in 1 2; do
  $PY pipelines/09_simulation/00_simulate_splicing.py --tissue Heart --K 5 \
      --frac-differential 0.3 --effect 1.0 --seed $SEED
done
ls -1 /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION/sim_heart_K5_binom_eff1.0*seed*.h5ad
