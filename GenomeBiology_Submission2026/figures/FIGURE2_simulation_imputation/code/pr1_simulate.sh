#!/bin/bash
s=$((SLURM_ARRAY_TASK_ID + 3))
/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/09_simulation/00_simulate_splicing.py --tissue Heart --K 5 --frac-differential 0.3 --effect 0.25 --bb-conc 2 --seed $s --outdir /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION
