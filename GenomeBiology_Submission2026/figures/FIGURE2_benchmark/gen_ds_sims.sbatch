#!/bin/bash
#SBATCH --job-name=gen_sims
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/FIGURE2_benchmark/gen_ds_sims.log
cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026
PY=/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python
S=pipelines/09_simulation/00_simulate_splicing.py
# The %DS axis of Figure 2 panels B/D/E/G: frac_differential = 0.1, 0.5, 0.9.
# K=5, Binomial truth (the emission the figure fits both likelihoods against), 3 seeds for box-plot spread.
for FRAC in 0.1 0.5 0.9; do
  for SEED in 0 1 2; do
    echo "=== frac=$FRAC seed=$SEED ==="
    $PY $S --tissue Heart --K 5 --frac-differential $FRAC --effect 1.0 --seed $SEED
  done
done
echo "=== generated ==="
ls -1 /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION/sim_heart_K5_binom_eff1.0_frac*.h5ad
echo "ALL SIMS DONE"
