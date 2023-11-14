#!/bin/sh
#
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p pe2
#SBATCH -c 6
#SBATCH --mem=40000M
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -J LeafletCluster # <-- name of job

#load required modules
module purge                                                                                                                                                                         
module load gcc/9.2.0 

conda activate leafcutter-sc

# set up Leaflet clustering script and input files 
leaflet_input_files=/gpfs/commons/home/kisaev/Leaflet/src/clustering/Leaflet_prep_clusters_for_model.py
cd /gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters
metadata="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/tabula_sapiens_ss2_metadata.csv"

# Organ specific params 
organ="Fat"
mkdir $organ
intron_clusters="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/Fat_50_500000_2_1_0.005_single_cell.gz"
# Run 
python $leaflet_input_files --intron_clusters $intron_clusters --output_file $organ/$organ --metadata $metadata --has_genes "yes"

# Organ specific params 
organ="Bone_Marrow"
mkdir $organ
intron_clusters="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/Bone_Marrow_50_500000_2_1_0.005_single_cell.gz"
# Run 
sbatch --wrap "python $leaflet_input_files --intron_clusters $intron_clusters --output_file $organ/$organ --metadata $metadata --has_genes "yes"" --mem=40000M -c 6 -t 5-00:00 -J $organ

# Organ specific params
organ="Heart"
mkdir $organ
intron_clusters="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/Heart_50_500000_2_1_0.005_single_cell.gz"
# Run 
sbatch --wrap "python $leaflet_input_files --intron_clusters $intron_clusters --output_file $organ/$organ --metadata $metadata --has_genes "yes"" --mem=40000M -c 6 -t 5-00:00 -J $organ

# Organ specific params
organ="Eye"
mkdir $organ
intron_clusters="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/Eye_50_500000_2_1_0.005_single_cell.gz"
# Run 
sbatch --wrap "python $leaflet_input_files --intron_clusters $intron_clusters --output_file $organ/$organ --metadata $metadata --has_genes "yes"" --mem=40000M -c 6 -t 5-00:00 -J $organ

# Organ specific params 
organ="Mammary"
mkdir $organ
intron_clusters="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/Mammary_50_500000_2_1_0.005_single_cell.gz"
# Run 
sbatch --wrap "python $leaflet_input_files --intron_clusters $intron_clusters --output_file $organ/$organ --metadata $metadata --has_genes "yes"" --mem=40000M -c 6 -t 5-00:00 -J $organ

# Organ specific params
organ="Uterus"
mkdir $organ
intron_clusters="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/Uterus_50_500000_2_1_0.005_single_cell.gz"
# Run 
sbatch --wrap "python $leaflet_input_files --intron_clusters $intron_clusters --output_file $organ/$organ --metadata $metadata --has_genes "yes"" --mem=40000M -c 6 -t 5-00:00 -J $organ

# Organ specific params
organ="Tongue"
mkdir $organ
intron_clusters="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/Tongue_50_500000_2_1_0.005_single_cell.gz"
# Run
sbatch --wrap "python $leaflet_input_files --intron_clusters $intron_clusters --output_file $organ/$organ --metadata $metadata --has_genes "yes"" --mem=40000M -c 6 -t 5-00:00 -J $organ

# Organ specific params
organ="Muscle"
mkdir $organ
intron_clusters="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/Muscle_50_500000_2_1_0.005_single_cell.gz"
# Run
sbatch --wrap "python $leaflet_input_files --intron_clusters $intron_clusters --output_file $organ/$organ --metadata $metadata --has_genes "yes"" --mem=40000M -c 6 -t 5-00:00 -J $organ

# Organ specific params
organ="Liver"
mkdir $organ
intron_clusters="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/Liver_50_500000_2_1_0.005_single_cell.gz"
# Run
sbatch --wrap "python $leaflet_input_files --intron_clusters $intron_clusters --output_file $organ/$organ --metadata $metadata --has_genes "yes"" --mem=40000M -c 6 -t 5-00:00 -J $organ

# Organ specific params
organ="Trachea"
mkdir $organ
intron_clusters="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/Trachea_50_500000_2_1_0.005_single_cell.gz"
# Run
sbatch --wrap "python $leaflet_input_files --intron_clusters $intron_clusters --output_file $organ/$organ --metadata $metadata --has_genes "yes"" --mem=40000M -c 6 -t 5-00:00 -J $organ

# Organ specific params
organ="Spleen"
mkdir $organ
intron_clusters="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/Spleen_50_500000_2_1_0.005_single_cell.gz"
# Run
sbatch --wrap "python $leaflet_input_files --intron_clusters $intron_clusters --output_file $organ/$organ --metadata $metadata --has_genes "yes"" --mem=40000M -c 6 -t 5-00:00 -J $organ

# Organ specific params
organ="Blood"
mkdir $organ
intron_clusters="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/Blood_50_500000_2_1_0.005_single_cell.gz"
# Run
sbatch --wrap "python $leaflet_input_files --intron_clusters $intron_clusters --output_file $organ/$organ --metadata $metadata --has_genes "yes"" --mem=40000M -c 6 -t 5-00:00 -J $organ

# Organ specific params
organ="Lymph_Node"
mkdir $organ
intron_clusters="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/Lymph_Node_50_500000_2_1_0.005_single_cell.gz"
# Run
sbatch --wrap "python $leaflet_input_files --intron_clusters $intron_clusters --output_file $organ/$organ --metadata $metadata --has_genes "yes"" --mem=40000M -c 6 -t 5-00:00 -J $organ

# Organ specific params
organ="Salivary_Gland"
mkdir $organ
intron_clusters="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/Salivary_Gland_50_500000_2_1_0.005_single_cell.gz"
# Run
sbatch --wrap "python $leaflet_input_files --intron_clusters $intron_clusters --output_file $organ/$organ --metadata $metadata --has_genes "yes"" --mem=40000M -c 6 -t 5-00:00 -J $organ

# Organ specific params
organ="Pancreas"
mkdir $organ
intron_clusters="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/Pancreas_50_500000_2_1_0.005_single_cell.gz"
# Run
sbatch --wrap "python $leaflet_input_files --intron_clusters $intron_clusters --output_file $organ/$organ --metadata $metadata --has_genes "yes"" --mem=40000M -c 6 -t 5-00:00 -J $organ

# Organ specific params
organ="Prostate"
mkdir $organ
intron_clusters="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/Prostate_50_500000_2_1_0.005_single_cell.gz"
# Run
sbatch --wrap "python $leaflet_input_files --intron_clusters $intron_clusters --output_file $organ/$organ --metadata $metadata --has_genes "yes"" --mem=40000M -c 6 -t 5-00:00 -J $organ

# Organ specific params
organ="Kidney"
mkdir $organ
intron_clusters="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/Kidney_50_500000_2_1_0.005_single_cell.gz"
# Run
sbatch --wrap "python $leaflet_input_files --intron_clusters $intron_clusters --output_file $organ/$organ --metadata $metadata --has_genes "yes"" --mem=40000M -c 6 -t 5-00:00 -J $organ

# Organ specific params
organ="Large_Intestine"
mkdir $organ
intron_clusters="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/Large_Intestine_50_500000_2_1_0.005_single_cell.gz"
# Run
sbatch --wrap "python $leaflet_input_files --intron_clusters $intron_clusters --output_file $organ/$organ --metadata $metadata --has_genes "yes"" --mem=40000M -c 6 -t 5-00:00 -J $organ

# Organ specific params
organ="Lung"
mkdir $organ
intron_clusters="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/Lung_50_500000_2_1_0.005_single_cell.gz"
# Run
sbatch --wrap "python $leaflet_input_files --intron_clusters $intron_clusters --output_file $organ/$organ --metadata $metadata --has_genes "yes"" --mem=40000M -c 6 -t 5-00:00 -J $organ

# Organ specific params
organ="Small_Intestine"
mkdir $organ
intron_clusters="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/Small_Intestine_50_500000_2_1_0.005_single_cell.gz"
# Run
sbatch --wrap "python $leaflet_input_files --intron_clusters $intron_clusters --output_file $organ/$organ --metadata $metadata --has_genes "yes"" --mem=40000M -c 6 -t 5-00:00 -J $organ

# Organ specific params
organ="Thymus"
mkdir $organ
intron_clusters="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/Thymus_50_500000_2_1_0.005_single_cell.gz"
# Run
sbatch --wrap "python $leaflet_input_files --intron_clusters $intron_clusters --output_file $organ/$organ --metadata $metadata --has_genes "yes"" --mem=40000M -c 6 -t 5-00:00 -J $organ

# Organ specific params
organ="Vasculature"
mkdir $organ
intron_clusters="/gpfs/commons/projects/CZI-tabula-sapiens/Leaflet-Analysis/Leaflet-Intron-Clusters/Vasculature_50_500000_2_1_0.005_single_cell.gz"
# Run
sbatch --wrap "python $leaflet_input_files --intron_clusters $intron_clusters --output_file $organ/$organ --metadata $metadata --has_genes "yes"" --mem=40000M -c 6 -t 5-00:00 -J $organ

