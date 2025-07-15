# Cross-Species Splicing Conservation Analysis

## Directory Structure:
- data/: Processed datasets (filtered ATSE files, ortholog mappings, etc.)
- figures/: All plots and visualizations
- results/: Analysis results, conservation tables, summary statistics

## Data Sources:
- Mouse ATSE: /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/ATSE_mapper/ATSE_files/MOUSE_FOUNDATION_ATSE_FILE_unanno_also_2025-07-01_00-02-00.txt.gz
- Human ATSE: /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/ATSE_mapper/ATSE_files/stella_gtf/TMS_atse_file_unanno_also_2025-07-01_03-02-03.txt.gz
- Orthologs: /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/human_mouse_ensemble_orthos_mart_export.txt.gz

## Analysis Goals:
1. Compare splice motif usage between mouse and human
2. Analyze junction count distributions
3. Identify conserved splice junctions in one-to-one orthologs
4. Cross-species aging factor conservation analysis

## Key Statistics:
- Shared orthologous genes: 8938
- Mouse junctions: 65885
- Human junctions: 79814
- Mouse genes: 8938
- Human genes: 8938
