#!/usr/bin/env python
"""
Parallel LeafletFA Model Evaluation - Job Submission Script
"""

import os
import glob
import pandas as pd
import subprocess
from datetime import datetime

# Configuration
MODEL_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel/2025-10-25"
OUTPUT_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_OUTPUT/batch_evaluation_parallel"
LOG_DIR = os.path.join(OUTPUT_DIR, "slurm_logs")

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Load parameter combinations
params_df = pd.read_csv(os.path.join(MODEL_DIR, "parameter_combinations.csv"))
print(f"Found {len(params_df)} parameter sets")

# Find all run directories
run_dirs = sorted(glob.glob(os.path.join(MODEL_DIR, "run_*")))
print(f"Found {len(run_dirs)} run directories")

# Track job IDs for dependency management
job_ids = []
job_info = []

# Submit individual jobs for each run
for run_dir in run_dirs:
    run_name = os.path.basename(run_dir)
    run_idx = int(run_name.split('_')[1])
    model_file = os.path.join(run_dir, "leafletfa_model.pkl.gz")
    
    if not os.path.exists(model_file):
        print(f"Skipping {run_name}: no model file")
        continue
    
    # Create SLURM job script
    job_script = f"""#!/bin/bash
#SBATCH --job-name=eval_{run_name}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --output={LOG_DIR}/{run_name}_%j.out
#SBATCH --error={LOG_DIR}/{run_name}_%j.err

# Load environment
source activate LeafletSC

# Run evaluation for this specific model
python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/LeafletFA_analysis/evaluate_single_model.py \
    --model_path {model_file} \
    --run_name {run_name} \
    --run_idx {run_idx} \
    --output_dir {OUTPUT_DIR}
"""
    
    # Write job script
    job_file = os.path.join(LOG_DIR, f"job_{run_name}.sh")
    with open(job_file, "w") as f:
        f.write(job_script)
    
    # Submit job
    cmd = f"sbatch {job_file}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Extract job ID from output
        job_id = result.stdout.strip().split()[-1]
        job_ids.append(job_id)
        job_info.append({
            'run_name': run_name,
            'run_idx': run_idx,
            'job_id': job_id,
            'model_path': model_file
        })
        print(f"Submitted {run_name} -> Job ID: {job_id}")
    else:
        print(f"Failed to submit {run_name}: {result.stderr}")

# Save job tracking information
job_df = pd.DataFrame(job_info)
job_df.to_csv(os.path.join(OUTPUT_DIR, f"job_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), index=False)
print(f"\nSubmitted {len(job_ids)} jobs")
print(f"Job tracking saved to: {OUTPUT_DIR}/job_tracking_*.csv")

# Create aggregation job that depends on all evaluation jobs
if job_ids:
    dependency_string = ":".join(job_ids)
    agg_script = f"""#!/bin/bash
#SBATCH --job-name=aggregate_results
#SBATCH --partition=bigmem
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output={LOG_DIR}/aggregate_%j.out
#SBATCH --error={LOG_DIR}/aggregate_%j.err
#SBATCH --dependency=afterok:{dependency_string}

source activate LeafletSC

python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/LeafletFA_analysis/aggregate_results.py \
    --output_dir {OUTPUT_DIR}
"""
    
    agg_job_file = os.path.join(LOG_DIR, "job_aggregate.sh")
    with open(agg_job_file, "w") as f:
        f.write(agg_script)
    
    result = subprocess.run(f"sbatch {agg_job_file}", shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        agg_job_id = result.stdout.strip().split()[-1]
        print(f"\nSubmitted aggregation job: {agg_job_id}")
        print("This job will run after all evaluation jobs complete successfully")


# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_OUTPUT/batch_evaluation_parallel 
# conda activate LeafletSC
# script=/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/LeafletFA_analysis/submit_parallel_evaluation.py 
# python $script