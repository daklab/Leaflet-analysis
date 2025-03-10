import anndata as ad
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import scanpy as sc
import gc

def merge_anndata_files(input_dir, output_file):
    """
    Merge multiple anndata files efficiently, assuming they share the same var indices.
    Ensures unique observation names and continuous cell_id_index.
    """
    
    # Get all h5ad files
    files = sorted(Path(input_dir).glob('chunk_*_anndata.h5ad*'))
    print(f"Found {len(files)} files to merge")
    
    # Load first file to initialize
    print(f"Loading first file: {files[0]}")
    combined = ad.read_h5ad(files[0])
    
    # Set observation names to cell_id for first file
    combined.obs_names = combined.obs['Main_cluster_name_wkmeans']
    current_max_index = combined.obs['cell_id_index'].max()
    
    # Store var and uns data which should be identical across files
    var_data = combined.var.copy()
    uns_data = combined.uns.copy()
    
    # Process remaining files
    for i, file in enumerate(files[1:], 1):
        print(f"Processing file {i+1}/{len(files)}: {file}")
        
        # Load next file
        adata = ad.read_h5ad(file)
        
        # Set observation names to cell_id
        adata.obs_names = adata.obs['Main_cluster_name_wkmeans']
        
        # Update cell_id_index for the new chunk
        adata.obs['cell_id_index'] = adata.obs['cell_id_index'] + current_max_index + 1
        current_max_index = adata.obs['cell_id_index'].max()
        
        # Verify var indices match
        if not adata.var_names.equals(combined.var_names):
            raise ValueError(f"var indices don't match in file {file}")
        
        # Verify no duplicate cell_ids
        if any(adata.obs_names.isin(combined.obs_names)):
            raise ValueError(f"Found duplicate cell_ids between chunks!")
            
        # Concatenate along observation axis
        combined = ad.concat([combined, adata], join='outer', merge='first')
        
        # Force garbage collection
        del adata
        gc.collect()

    # Verify and restore var data
    print("Verifying var data consistency...")
    assert all(combined.var_names == var_data.index), "var indices mismatch after merge"
    combined.var = var_data
    combined.uns = uns_data
    
    # Verify cell_id_index is continuous and unique
    all_indices = combined.obs['cell_id_index'].values
    expected_indices = np.arange(len(all_indices))
    assert np.array_equal(sorted(all_indices), expected_indices), "cell_id_index is not continuous"
    
    # Verify observation names are unique
    assert len(combined.obs_names.unique()) == len(combined.obs_names), "Observation names are not unique"
    
    # Print some stats
    print("\nMerge Statistics:")
    print(f"Total cells: {combined.shape[0]}")
    print(f"Total features: {combined.shape[1]}")
    print(f"cell_id_index range: 0 to {combined.obs['cell_id_index'].max()}")
    print("\nSample of merged data:")
    print(combined.obs[['Main_cluster_name_wkmeans', 'cell_id_index']].head())
    print("...")
    print(combined.obs[['Main_cluster_name_wkmeans', 'cell_id_index']].tail())
    
    # Save final result in a compressed format
    print(f"\nSaving merged dataset to {output_file} (compressed)")
    combined.write_h5ad(output_file, compression="gzip")
    return combined

def main():
    parser = argparse.ArgumentParser(description='Merge multiple anndata files')
    parser.add_argument('--input-dir', required=True, help='Directory containing chunk_*_anndata.h5ad files')
    parser.add_argument('--output-file', required=True, help='Path for merged output file')
    
    args = parser.parse_args()
    
    merge_anndata_files(args.input_dir, args.output_file)

if __name__ == '__main__':
    main()

# To run this script, use the following command:
    
# SCRIPT_PATH=/gpfs/commons/home/kisaev/Leaflet-analysis/EasySci/LeafletFA/data_processing/ATSE_ANNDATA/RH/06_MergeAnndatas.py
# INPUT_DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/ATSEmap/RH/output/junction_processing_20250223/anndatas
# OUTPUT_FILE=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/MetaCells/ATSEmap/RH/output/junction_processing_20250223/anndatas/merged_anndata.h5ad
# 
# python $SCRIPT_PATH --input-dir $INPUT_DIR --output-file $OUTPUT_FILE