#!/bin/bash
# SLURM job to run DT metacell BAMs -> gene counts -> AnnData.
# Adjust partition, time, mem as needed. featureCounts can be memory-heavy with many BAMs.
#SBATCH -J DT_metacell_counts
#SBATCH --mem=200G
#SBATCH -t 0-24:00
#SBATCH -p cpu,bigmem
#SBATCH -c 8

SCRIPT="/gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/EasySci/LeafletFA/data_processing/03_DT_to_expression_counts/run_DT_gene_counts.sh"
CONFIG="/gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/EasySci/LeafletFA/data_processing/03_DT_to_expression_counts/config.yaml"

conda activate python3ENV
module load subread

# Navigate to your directory with the Snakefile
slurm_out=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/SpliceVI/202602/DT_expression/slurmFeb2026
if [ ! -d "$slurm_out" ]; then
    mkdir -p $slurm_out
fi

bash "$SCRIPT" "$CONFIG"

# cd $slurm_out
# sbatch /gpfs/commons/home/kisaev/Leaflet-analysis/LeafletFA_Submission2025/EasySci/LeafletFA/data_processing/03_DT_to_expression_counts/run_DT_gene_counts_slurm.sh