#!/bin/bash
#SBATCH --job-name=gtex_junction_processing
#SBATCH --time=12:00:00         # Set an appropriate time limit
#SBATCH --mem=32G                # Adjust memory as needed
#SBATCH --cpus-per-task=1       # Number of CPUs

# Calculate total number of lines in the compressed file
total_lines=$(zcat GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct.gz | wc -l)

# Process the file, displaying progress, and add a header to the output
{
    # Print header row
    echo -e "junction_id\tgene_id\tjunc_total_count\tjunc_nonzero"
    
    # Process each line after skipping the first two header lines
    zcat GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct.gz | pv -l -s $total_lines | \
    awk 'NR > 2 {
        row_sum = 0; 
        non_zero_count = 0;
        for (i = 3; i <= NF; i++) { 
            row_sum += $i;
            if ($i != 0) {
                non_zero_count++;
            }
        }
        print $1, $2, row_sum, non_zero_count;
    }'
} > GTEx_junctions_row_sum_non_zero_counts.gct