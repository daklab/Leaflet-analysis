#!/usr/bin/env python
"""
Aggregate Results from Parallel Evaluation Jobs
"""

import os
import glob
import pandas as pd
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    
    print("Aggregating results from parallel jobs...")
    
    # Find all result files
    result_files = glob.glob(os.path.join(args.output_dir, "run_*_results.csv"))
    print(f"Found {len(result_files)} result files")
    
    if not result_files:
        print("No results found!")
        return
    
    # Load and combine
    all_results = []
    for file in result_files:
        df = pd.read_csv(file)
        all_results.append(df)
        print(f"  Loaded {os.path.basename(file)}: {len(df)} rows")
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save combined results
    output_file = os.path.join(args.output_dir, f"combined_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    combined_df.to_csv(output_file, index=False)
    print(f"\nCombined results saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Group by masking percentage
    for mask_pct in sorted(combined_df['masking_percentage'].unique()):
        subset = combined_df[combined_df['masking_percentage'] == mask_pct]
        print(f"\nMasking {mask_pct}%:")
        print(f"  Best L1 Loss: {subset['l1_loss'].min():.4f} ({subset.loc[subset['l1_loss'].idxmin(), 'run_name']})")
        print(f"  Best RMSE: {subset['rmse'].min():.4f} ({subset.loc[subset['rmse'].idxmin(), 'run_name']})")
        print(f"  Best Spearman: {subset['spearman'].max():.4f} ({subset.loc[subset['spearman'].idxmax(), 'run_name']})")
    
    print("\n" + "="*60)
    print("Aggregation complete!")

if __name__ == "__main__":
    main()
