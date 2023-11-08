#!/bin/sh

download_temp=/gpfs/commons/home/kisaev/Leaflet-analysis/tabula_sapien/download_template.sh

sbatch $download_temp Pilot1 s3://czb-tabula-sapiens/Pilot1/alignment-gencode/SS2/

sbatch $download_temp Pilot2 \
    s3://czb-tabula-sapiens/Pilot2/alignment-gencode/smartseq2/batch1/ \
    s3://czb-tabula-sapiens/Pilot2/alignment-gencode/smartseq2/batch2/ \
    s3://czb-tabula-sapiens/Pilot2/alignment-gencode/smartseq2/batch3/

sbatch $download_temp Pilot3 \
    s3://czb-tabula-sapiens/Pilot3/alignment-gencode/smartseq2/batch1/ \
    s3://czb-tabula-sapiens/Pilot3/alignment-gencode/smartseq2/batch2/ \
    s3://czb-tabula-sapiens/Pilot3/alignment-gencode/smartseq2/batch3/

sbatch $download_temp Pilot4 \
    s3://czb-tabula-sapiens/Pilot4/alignment-gencode/smartseq2/batch1/ \
    s3://czb-tabula-sapiens/Pilot4/alignment-gencode/smartseq2/batch2/ \
    s3://czb-tabula-sapiens/Pilot4/alignment-gencode/smartseq2/batch3/ \
    s3://czb-tabula-sapiens/Pilot4/alignment-gencode/smartseq2/batch4/

sbatch $download_temp Pilot5 \
    s3://czb-tabula-sapiens/Pilot5/alignment-gencode/smartseq2/batch1/

sbatch $download_temp Pilot6 \
    s3://czb-tabula-sapiens/Pilot6/alignment-gencode/smartseq2/batch1/ \
    s3://czb-tabula-sapiens/Pilot6/alignment-gencode/smartseq2/batch2/ \
    s3://czb-tabula-sapiens/Pilot6/alignment-gencode/smartseq2/batch3/ \
    s3://czb-tabula-sapiens/Pilot6/alignment-gencode/smartseq2/batch4/

sbatch $download_temp Pilot7 \
    s3://czb-tabula-sapiens/Pilot7/alignment-gencode/smartseq2/batch1/ \
    s3://czb-tabula-sapiens/Pilot7/alignment-gencode/smartseq2/batch2/ \
    s3://czb-tabula-sapiens/Pilot7/alignment-gencode/smartseq2/batch3/ \
    s3://czb-tabula-sapiens/Pilot7/alignment-gencode/smartseq2/batch4/ \
    s3://czb-tabula-sapiens/Pilot7/alignment-gencode/smartseq2/batch5/ \
    s3://czb-tabula-sapiens/Pilot7/alignment-gencode/smartseq2/batch6/

sbatch $download_temp Pilot8 \
    s3://czb-tabula-sapiens/Pilot8/alignment-gencode/smartseq2/batch1/ \
    s3://czb-tabula-sapiens/Pilot8/alignment-gencode/smartseq2/batch2/ 

sbatch $download_temp Pilot9 \
    s3://czb-tabula-sapiens/Pilot9/alignment-gencode/smartseq2/batch1/ \
    s3://czb-tabula-sapiens/Pilot9/alignment-gencode/smartseq2/batch2/ \
    s3://czb-tabula-sapiens/Pilot9/alignment-gencode/smartseq2/batch3/

sbatch $download_temp Pilot10 \
    s3://czb-tabula-sapiens/Pilot10/alignment-gencode/smartseq2/batch1/ \
    s3://czb-tabula-sapiens/Pilot10/alignment-gencode/smartseq2/batch2/ 

sbatch $download_temp Pilot11 \
    s3://czb-tabula-sapiens/Pilot11/alignment-gencode/smartseq2/batch1/

sbatch $download_temp Pilot12 \
    s3://czb-tabula-sapiens/Pilot12/alignment-gencode/smartseq2/batch1/ \
    s3://czb-tabula-sapiens/Pilot12/alignment-gencode/smartseq2/batch2/ 

sbatch $download_temp Pilot13 \
    s3://czb-tabula-sapiens/Pilot13/alignment-gencode/smartseq2/batch1/ \
    s3://czb-tabula-sapiens/Pilot13/alignment-gencode/smartseq2/batch2/ 

sbatch $download_temp Pilot14 \
    s3://czb-tabula-sapiens/Pilot14/alignment-gencode/smartseq2/batch1/ \
    s3://czb-tabula-sapiens/Pilot14/alignment-gencode/smartseq2/batch2/ \
    s3://czb-tabula-sapiens/Pilot14/alignment-gencode/smartseq2/batch3/
