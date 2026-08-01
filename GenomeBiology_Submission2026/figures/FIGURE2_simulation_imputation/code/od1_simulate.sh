#!/bin/bash
i=$SLURM_ARRAY_TASK_ID
concs=(2 2 2 20 20 20); seeds=(0 1 2 0 1 2)
/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/09_simulation/00_simulate_splicing.py --tissue Heart --K 5 --frac-differential 0.3 --effect 0.25     --bb-conc ${concs[$i]} --seed ${seeds[$i]} --outdir /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION
