#!/bin/bash
set -e
cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/FIGURE2_simulation_imputation
/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python -c "
import pandas as pd, glob, sys
fs=sorted(glob.glob('/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/FIGURE2_simulation_imputation/data/rerun_logs/bench_*.csv'))
if len(fs)!=9: print(f'WARNING: {len(fs)}/9 benchmark shards'); 
d=pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
d.to_csv('/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/FIGURE2_simulation_imputation/data/benchmark_results.csv', index=False)
print('benchmark_results.csv:', len(d), 'rows from', len(fs), 'shards')
print(d.groupby(['model','likelihood'])[['psi_r','spearman_delta','sil']].mean().round(3).to_string())
"
/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/FIGURE2_simulation_imputation/code/prep_panels.py
/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/FIGURE2_simulation_imputation/code/build_sim.py
echo "RERUN COMPLETE"
