#!/bin/bash
i=$SLURM_ARRAY_TASK_ID
effs=(0.1 0.1 0.1 0.25 0.25 0.25 0.5 0.5 0.5 1.0 1.0 1.0)
seeds=(0 1 2 0 1 2 0 1 2 0 1 2)
/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/09_simulation/00_simulate_splicing.py --tissue Heart --K 5 --frac-differential 0.3 --effect ${effs[$i]}     --bb-conc inf --seed ${seeds[$i]} --outdir /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION
