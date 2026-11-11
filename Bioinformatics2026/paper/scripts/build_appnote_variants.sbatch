#!/bin/bash
#SBATCH --job-name=figs_variants
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_bioinformatics/paper_figures/logs/figs_variants_%j.log
# Application Note figure, two variants for Karin's decision (2026-11-11):
#   A  figAN_varA : panel e (Nrxn1 SS4) kept as is (figure AN; the caption fact is in FIG_APPNOTE_CAPTION_FACTS.md)
#   B  figAN5_varB: five panels without Nrxn1 (fig_appnote5.py); Nrxn1 stays in the supplementary gallery (figSG, SG_c)
source /gpfs/commons/home/kisaev/miniconda3/etc/profile.d/conda.sh
conda activate LeafletSC
cd /gpfs/commons/home/kisaev/Leaflet-analysis-bioinformatics/Bioinformatics2026/paper/figures
FIG_SUFFIX=_varA python make_figures.py --figure AN
FIG_SUFFIX=_varB python make_figures.py --figure AN5
