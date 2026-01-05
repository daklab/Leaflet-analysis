#!/usr/bin/env python
"""
Memory-Efficient NNZ Alignment Check
Quick check to verify if non-zero patterns are aligned between junction and cluster matrices
in both train and test datasets without full data processing.
"""

import os
import numpy as np
import mudata as mu
import scanpy as sc
from datetime import datetime
import logging
from scipy import sparse

# =============================================================================
# Configuration
# =============================================================================

# File paths
TRAIN_ADATA_PATH = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/train_70_30_model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5mu"
TEST_ADATA_PATH = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5mu"

# =============================================================================
# Memory-Efficient Sparsity Check Functions
# =============================================================================

def check_sparsity_consistency_light(ad, dataset_name):
    """Lightweight check of sparsity patterns between junction and cluster matrices"""
    print(f"\n🔍 Checking sparsity consistency for {dataset_name} dataset...")
    
    # Get sparse matrices in COO format for efficient coordinate access
    junc = ad.layers["cell_by_junction_matrix"].tocoo()
    clust = ad.layers["cell_by_cluster_matrix"].tocoo()
    
    # Get coordinate sets
    junc_coords = set(zip(junc.row, junc.col))
    clust_coords = set(zip(clust.row, clust.col))
    
    # Find missing coordinates
    missing = clust_coords - junc_coords
    extra = junc_coords - clust_coords
    
    # Print results
    print(f"  Dataset: {dataset_name}")
    print(f"  Data shape: {ad.shape[0]:,} cells × {ad.shape[1]:,} junctions")
    print(f"  Junction matrix non-zero entries: {len(junc_coords):,}")
    print(f"  Cluster matrix non-zero entries: {len(clust_coords):,}")
    print(f"  Missing in junction matrix: {len(missing):,}")
    print(f"  Extra in junction matrix: {len(extra):,}")
    
    if len(missing) == 0 and len(extra) == 0:
        print(f"  ✓ {dataset_name}: Sparsity patterns are perfectly aligned!")
        alignment_status = "perfect"
    elif len(missing) == 0:
        print(f"  ⚠ {dataset_name}: Junction has extra entries but no missing entries")
        alignment_status = "extra_only"
    elif len(extra) == 0:
        print(f"  ⚠ {dataset_name}: Junction missing entries but no extra entries")
        alignment_status = "missing_only"
    else:
        print(f"  ❌ {dataset_name}: Junction has both missing and extra entries")
        alignment_status = "misaligned"
    
    return {
        'dataset': dataset_name,
        'shape': ad.shape,
        'junc_nnz': len(junc_coords),
        'clust_nnz': len(clust_coords),
        'missing': len(missing),
        'extra': len(extra),
        'alignment_status': alignment_status
    }

def load_and_check_dataset(file_path, dataset_name):
    """Load dataset and check sparsity with minimal memory usage"""
    print(f"\n📂 Loading {dataset_name} dataset...")
    print(f"   Path: {file_path}")
    
    try:
        # Load only the splicing modality
        mdata = mu.read_h5mu(file_path)
        ad = mdata["splicing"]
        
        print(f"✓ Loaded {dataset_name} splicing data: {ad.shape[0]:,} cells × {ad.shape[1]:,} junctions")
        
        # Check sparsity
        result = check_sparsity_consistency_light(ad, dataset_name)
        
        # Clean up memory
        del ad, mdata
        
        return result
        
    except Exception as e:
        print(f"❌ Error loading {dataset_name} dataset: {str(e)}")
        return None

def compare_datasets(train_result, test_result):
    """Compare sparsity patterns between train and test datasets"""
    print(f"\n📊 Dataset Comparison Summary:")
    print(f"=" * 50)
    
    if train_result and test_result:
        print(f"Train Dataset:")
        print(f"  Shape: {train_result['shape'][0]:,} × {train_result['shape'][1]:,}")
        print(f"  Junction NNZ: {train_result['junc_nnz']:,}")
        print(f"  Cluster NNZ: {train_result['clust_nnz']:,}")
        print(f"  Status: {train_result['alignment_status']}")
        
        print(f"\nTest Dataset:")
        print(f"  Shape: {test_result['shape'][0]:,} × {test_result['shape'][1]:,}")
        print(f"  Junction NNZ: {test_result['junc_nnz']:,}")
        print(f"  Cluster NNZ: {test_result['clust_nnz']:,}")
        print(f"  Status: {test_result['alignment_status']}")
        
        # Check if both datasets have same number of junctions
        if train_result['shape'][1] == test_result['shape'][1]:
            print(f"\n✓ Both datasets have same number of junctions: {train_result['shape'][1]:,}")
        else:
            print(f"\n⚠ Different number of junctions:")
            print(f"  Train: {train_result['shape'][1]:,}")
            print(f"  Test: {test_result['shape'][1]:,}")
        
        # Overall alignment status
        both_perfect = (train_result['alignment_status'] == 'perfect' and 
                       test_result['alignment_status'] == 'perfect')
        
        if both_perfect:
            print(f"\n🎉 RESULT: Both datasets have perfectly aligned sparsity patterns!")
            print(f"   No sparsity fixes needed for either dataset.")
        else:
            print(f"\n⚠ RESULT: One or both datasets need sparsity alignment fixes.")
            if train_result['alignment_status'] != 'perfect':
                print(f"   Train dataset needs fixing: {train_result['missing']:,} missing, {train_result['extra']:,} extra")
            if test_result['alignment_status'] != 'perfect':
                print(f"   Test dataset needs fixing: {test_result['missing']:,} missing, {test_result['extra']:,} extra")

def main():
    """Main function to check NNZ alignment in both datasets"""
    print(f"🔬 Memory-Efficient NNZ Alignment Check")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=" * 60)
    
    # Check train dataset
    train_result = load_and_check_dataset(TRAIN_ADATA_PATH, "train")
    
    # Check test dataset  
    test_result = load_and_check_dataset(TEST_ADATA_PATH, "test")
    
    # Compare results
    compare_datasets(train_result, test_result)
    
    print(f"\n✅ NNZ alignment check completed!")
    
    return train_result, test_result

# =============================================================================
# Script Entry Point
# =============================================================================

if __name__ == "__main__":
    train_result, test_result = main()

"""
# Submit with much lower memory requirements:
conda activate LeafletSC
script=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/LeafletFA_analysis/check_nnz_alignment.py
cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_OUTPUT/
sbatch --job-name=check_nnz \
       --partition=cpu \
       --mem=50G \
       --time=0-02:00:00 \
       --output=check_nnz_%j.out \
       --error=check_nnz_%j.err \
       --wrap="python $script"
"""
