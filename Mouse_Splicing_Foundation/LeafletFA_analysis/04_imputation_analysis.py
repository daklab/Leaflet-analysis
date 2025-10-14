#!/usr/bin/env python
"""
Extract Splicing Data from MuData
Simple script to extract the splicing modality from a MuData file and save it as a standalone AnnData object.
"""

import os
import mudata as mu
import scanpy as sc
from datetime import datetime
import logging

# =============================================================================
# Configuration
# =============================================================================

# File paths
TRAIN_ADATA_PATH = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/072025/train_70_30_ge_splice_combined_20250730_164104.h5mu"
OUTPUT_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/092025"

# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir):
    """Configure logging with both file and console output"""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f"splicing_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# =============================================================================
# Main Extraction Function
# =============================================================================

def extract_splicing_data():
    """Extract splicing modality from MuData and save as standalone AnnData"""
    
    # Setup output directory and logging
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = setup_logging(OUTPUT_DIR)
    
    print(f"\n🔬 Starting Splicing Data Extraction")
    print(f"   Input: {TRAIN_ADATA_PATH}")
    print(f"   Output Directory: {OUTPUT_DIR}\n")
    
    # Load MuData and extract splicing modality
    logger.info(f"Loading MuData from {TRAIN_ADATA_PATH}")
    print("Loading MuData file...")
    
    mdata = mu.read_h5mu(TRAIN_ADATA_PATH)
    ad = mdata["splicing"]
    
    logger.info(f"✓ Loaded splicing data")
    logger.info(f"  Shape: {ad.shape}")
    logger.info(f"  Layers: {list(ad.layers.keys())}")
    logger.info(f"  Observations: {list(ad.obs.columns)}")
    logger.info(f"  Variables: {list(ad.var.columns)}")
    
    print(f"✓ Extracted splicing modality: {ad.shape[0]:,} cells × {ad.shape[1]:,} junctions")
    print(f"  Available layers: {', '.join(ad.layers.keys())}")
    
    # Add basic indexing
    ad.obs.reset_index(drop=True, inplace=True)
    ad.obs["cell_id_index"] = ad.obs.index        
    ad.var["junction_id_index"] = ad.var.index
    
    # Save the extracted splicing AnnData object
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    splicing_adata_path = os.path.join(OUTPUT_DIR, f"splicing_adata_{stamp}.h5ad")
    
    print(f"\nSaving splicing AnnData...")
    ad.write_h5ad(splicing_adata_path, compression='lzf')
    
    # Get file size
    file_size_mb = os.path.getsize(splicing_adata_path) / (1024**2)
    
    logger.info(f"✓ Splicing AnnData saved to: {splicing_adata_path}")
    logger.info(f"  File size: {file_size_mb:.2f} MB")
    
    print(f"✓ Splicing AnnData saved: {splicing_adata_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    
    # Create summary file
    summary_path = os.path.join(OUTPUT_DIR, f"extraction_summary_{stamp}.txt")
    with open(summary_path, 'w') as f:
        f.write("SPLICING DATA EXTRACTION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input MuData: {TRAIN_ADATA_PATH}\n")
        f.write(f"Output AnnData: {splicing_adata_path}\n")
        f.write(f"Data Shape: {ad.shape[0]:,} cells × {ad.shape[1]:,} junctions\n")
        f.write(f"File Size: {file_size_mb:.2f} MB\n")
        f.write(f"Layers: {', '.join(ad.layers.keys())}\n")
        f.write(f"Obs Columns: {', '.join(ad.obs.columns)}\n")
        f.write(f"Var Columns: {', '.join(ad.var.columns)}\n")
    
    logger.info(f"✓ Summary saved to: {summary_path}")
    
    print(f"\n🎉 Extraction Complete!")
    print(f"   Splicing AnnData: {splicing_adata_path}")
    print(f"   Summary: {summary_path}")
    
    return splicing_adata_path

# =============================================================================
# Script Entry Point
# =============================================================================

if __name__ == "__main__":
    extract_splicing_data()