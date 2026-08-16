#!/bin/bash
i=$SLURM_ARRAY_TASK_ID
fracs=(0.1 0.1 0.1 0.5 0.5 0.5 0.9 0.9 0.9)
seeds=(0 1 2 0 1 2 0 1 2)
f=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION/sim_heart_K5_binom_eff1.0_frac${fracs[$i]}_seed${seeds[$i]}.h5ad
/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/FIGURE2_simulation_imputation/code/sim_benchmark.py $f /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/FIGURE2_simulation_imputation/data/rerun_logs/bench_$i.csv cuda 2048
