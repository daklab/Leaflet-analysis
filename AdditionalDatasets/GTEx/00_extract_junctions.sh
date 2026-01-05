cd /gpfs/commons/groups/knowles_lab/Karin/data/GTEx/v10
gzip -dc GTEx_Analysis_v10_STARv2.7.10a_junctions.gct.gz   | tail -n +4   | cut -f1-2   > gtex_junctions_coordinates.tsv