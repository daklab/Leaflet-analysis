#!/bin/bash
s=$((SLURM_ARRAY_TASK_ID + 3))
f=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION/sim_heart_K5_bb2_eff0.25_frac0.3_seed$s.h5ad
/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/FIGURE2_simulation_imputation/code/sim_benchmark.py $f /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/FIGURE2_simulation_imputation/data/rerun_logs/prbench_$s.csv cuda 2048
