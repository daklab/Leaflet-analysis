#!/bin/bash

download_temp="/gpfs/commons/home/kisaev/Leaflet-analysis/tabula_sapien/download_template.sh"

# Loop through all available TSP directories
for TSP_ID in TSP1 TSP2 TSP3 TSP4 TSP5 TSP6 TSP7 TSP8 TSP9 TSP10 TSP11 TSP12 TSP13 TSP14 TSP15 TSP17 TSP19 TSP20 TSP21 TSP25 TSP26 TSP27 TSP28; do
    echo "Submitting job for $TSP_ID"
    sbatch "$download_temp" "$TSP_ID"
done
