#!/bin/bash
i=$SLURM_ARRAY_TASK_ID
concs=(2 2 2 20 20 20); seeds=(0 1 2 0 1 2)
f=$(ls /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION/sim_heart_K5_bb${concs[$i]}_eff0.25_frac0.3_seed${seeds[$i]}.h5ad 2>/dev/null ||     ls /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION/*bb${concs[$i]}*eff0.25*seed${seeds[$i]}.h5ad 2>/dev/null | head -1)
echo "benchmarking $f"
/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/FIGURE2_simulation_imputation/code/sim_benchmark.py $f /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/FIGURE2_simulation_imputation/data/rerun_logs/odbench_$i.csv cuda 2048
