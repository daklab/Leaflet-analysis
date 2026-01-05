# LeafletFA Preprocessing and Analysis Pipeline (Mouse Foundation Dataset)

This document outlines the complete workflow for processing raw splicing and gene expression data into inputs for LeafletFA training and downstream analysis.

## Overview

The pipeline consists of five main components:
1. **ATSEmapper** - Processes raw junction files into mapped Alternative Transcript Splicing Events (ATSEs)
2. **Metadata Integration** - Combines and harmonizes metadata from Tabula Muris Senis (TMS) and Allen Brain Atlas
3. **Gene Expression Processing** - Handles both Smart-seq and 10X data separately
4. **Data Alignment and LeafletFA Preparation** - Aligns splicing and gene expression data for model input
5. **Model Training and Evaluation** - Trains the LeafletFA model and analyzes results

## Directory Structure

```
/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/
├── ATSE_mapper/                 # ATSEmapper pipeline scripts
├── GeneExpression/              # Gene expression processing scripts
│   └── 10X_prep/                # 10X data preparation
├── LeafletFA_analysis/          # LeafletFA preparation and analysis
└── model_train/                 # Model training and evaluation
    └── MOUSE_FOUNDATION/        # Mouse foundation model specific scripts
        └── full_workflow/       # Model training workflow scripts
```

## 1. ATSEmapper Workflow

From raw junction files to mapped ATSEs and final AnnData objects:

```bash
# Base path
ROOT_PATH=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation

# Step 1: Split junction files into chunks so can analyze across SLURM jobs 
${ROOT_PATH}/ATSE_mapper/ATSEmap_SLURM/01_split_junctions.sh

# Step 2: Process junctions (read, filter)
${ROOT_PATH}/ATSE_mapper/ATSEmap_SLURM/02_process_junctions.sh

# Step 3: Merge processed junctions (obtain final list of junctions we see across all junction files)
${ROOT_PATH}/ATSE_mapper/ATSEmap_SLURM/03_merge_junctions.sh

# Step 4: Map ATSEs from junctions 
${ROOT_PATH}/ATSE_mapper/04_ATSEmapping.py

# Step 5: Create AnnData objects for each chunk using just junctions in the ATSE file (SLURM submission)
${ROOT_PATH}/ATSE_mapper/05_AnndataMake.py
${ROOT_PATH}/ATSE_mapper/05_AnndataMake_slurm.sh

# Step 6: Final merge of AnnData chunks 
${ROOT_PATH}/ATSE_mapper/06_MergeAnndatas.py
```

## 2. Metadata Integration

```bash
# Process and combine metadata from TMS and Allen Brain
${ROOT_PATH}/metadata_prep.py
``` 

## 3. Gene Expression Processing

### 3.1 Smart-seq Data Processing
This workflow combines gene expression counts from Tabula Muris Senis (single-cell) and Allen Brain (single-nuclei), normalizing for transcript length differences:

```bash 
# Step 1: Prepare gene lengths and combine TMS/Allen Brain data
# NOTE: this can take a few hours to run since the inital loading of the raw data takes a while... 
${ROOT_PATH}/GeneExpression/01_prepare_expression_data.py 

# Step 2: Generate metacell pseudobulk counts for shared cell types
${ROOT_PATH}/GeneExpression/02_generate_metacells.py

# Step 3: Train regression model to account for sc/sn differences
${ROOT_PATH}/GeneExpression/03_train_regression_model.ipynb 

# Step 4: Apply regression model to estimate spliced values
${ROOT_PATH}/GeneExpression/04_apply_regression_model.py 

# Step 5: Allign cells / nuclei in gene expression and splicing objects to follow same order 
# Also clean up cell type labels 
${ROOT_PATH}/GeneExpression/05_align_splice_ge_anndatas.py

# Step 6: double check if any outliers remaining in the dataset
${ROOT_PATH}/GeneExpression/06_check_outliers.ipynb

# Step 7: Run scVI on length-normalized counts
${ROOT_PATH}/GeneExpression/08_run_scVI.py

# Step 8: Run NMF on regression-adjusted values
${ROOT_PATH}/GeneExpression/07_run_NMF.py  

# Step 9: Visualize latent spaces from scVI and NMF 
${ROOT_PATH}/GeneExpression/09_visualize_scVI_NMF.py  
```

### 3.2 10X Data Processing
```bash
# Process 10X data from Tabula Muris Senis
${ROOT_PATH}/GeneExpression/10X_prep/01_process_TMS_10X.ipynb  
```

## 4. LeafletFA Input Preparation

```bash
# Prepare input for LeafletFA model training
${ROOT_PATH}/LeafletFA_analysis/01_prep_initialized_AnnData.py
```

## 5. Model Training and Evaluation

### 5.1 Model Training Workflow

```bash
# Generate model parameters and directory for storing model outputs 
${ROOT_PATH}/model_train/MOUSE_FOUNDATION/full_workflow/01_generate_params.py 

# Edit LeafletFA training script to include correct input file
${ROOT_PATH}/model_train/MOUSE_FOUNDATION/full_workflow/02_run_leaflet.py  

# Submit training jobs to SLURM
${ROOT_PATH}/model_train/MOUSE_FOUNDATION/full_workflow/03_submit_jobs.py
```

### 5.2 Model Evaluation and Interpretation

```bash
# Step 1. Run several workflows to assess individual model results
# including LeafletFA variance explained, differential splicing... 
bash ${ROOT_PATH}/downstream_analysis/pipeline_submit_script.sh
```

### 6. Model External Validation...
```bash
# Evaluate model on EasySci 
${ROOT_PATH}/model_train/MOUSE_FOUNDATION/model_external_validation/01_apply_model_Easysci_data.py
```

## Notes

- The Smart-seq processing combines data from two sources (TMS and Allen Brain) and normalizes for differences between single-cell and single-nuclei protocols
- For gene expression, two dimensionality reduction approaches are used: scVI (on length-normalized counts) and NMF (on regression-adjusted values)
- The final aligned dataset contains cells present in both splicing and gene expression matrices in the same order
- LeafletFA input preparation includes reducing the number of ATSEs to prevent memory issues and preparing waypoints via SVD on centered splicing values
- The model training workflow consists of three main scripts that handle parameter generation, job submission, and model execution
- Results evaluation includes analysis of splicing vs. gene expression latent spaces, aging effects, and splicing programs

## To Do: Implementation Improvements

### Configuration File Implementation
To improve maintainability and simplify pipeline execution, we should implement a centralized configuration approach:

1. **Create a unified config file structure**
   - Develop a YAML or JSON configuration file template that stores all file paths and parameters
   - Include sections for different pipeline components (ATSEmapper, gene expression, model training)
   - Allow for environment-specific overrides (development, production)

2. **Refactor scripts to use config parser**
   - Modify all Python scripts to load parameters from the config file
   - Replace hardcoded paths with config references
   - Implement proper error handling for missing config values

3. **Sample config file structure**
   ```yaml
   # mouse_foundation_pipeline_config.yaml
   
   # Global settings
   global:
     root_path: "/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation"
     output_dir: "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
     timestamp_format: "%Y%m%d_%H%M%S"
   
   # Input data paths
   input_data:
     splice_input: "${output_dir}/ATSE_mapper/junction_processing_20250415/anndatas/merged_anndata.h5ad"
     ge_input: "${output_dir}/processed_data/adjusted_gene_expression/Combined_adjusted_GeneExpression_2025-05-12.h5ad"
     atse_file: "${output_dir}/ATSE_mapper/ATSE_files/MOUSE_FOUNDATION_ATSE_FILE_unanno_also_2025-04-26_19-55-26.txt.gz"
     metadata_file: "${output_dir}/metadata_mouse_metadata_combined.csv"
   
   # Output paths for aligned data
   aligned_data:
     output_dir: "${output_dir}/MODEL_INPUT/052025"
     splice_file_prefix: "aligned_splicing_data"
     ge_file_prefix: "aligned_gene_expression_data"
   
   # Model parameters
   model:
     n_factors: 30
     learning_rate: 0.001
     batch_size: 128
     max_epochs: 500