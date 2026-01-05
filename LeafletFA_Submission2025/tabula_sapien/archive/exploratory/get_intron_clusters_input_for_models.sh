#!/bin/sh

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 

conda activate leafcutter-sc

cd /gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters 

# set up Leaflet clustering script and input files 
leaflet_input_files=/gpfs/commons/home/kisaev/Leaflet-private/src/clustering/Leaflet_prep_clusters_for_model.py
metadata="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/tabula_sapiens_ss2_metadata.csv"

# Organ specific params
organ="Muscle"
singletons="Yes"
# make new directory for the organ+singletons status in one name 
mkdir -p ${organ}_${singletons}
output_folder=${organ}_${singletons}

intron_clusters="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/Muscle_wsingleton_50_500000_2_1_0.005_single_cell.gz"

# Run
#sbatch --wrap "python $leaflet_input_files --intron_clusters $intron_clusters --output_file $output_folder/$organ --metadata $metadata --has_genes "yes"" --mem=40000M -c 6 -t 5-00:00 -J $organ
python $leaflet_input_files --intron_clusters $intron_clusters --output_file $output_folder/$organ --metadata $metadata --has_genes "yes"