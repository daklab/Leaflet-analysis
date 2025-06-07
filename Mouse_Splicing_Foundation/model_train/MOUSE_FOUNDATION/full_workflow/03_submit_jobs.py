import os
import json
import datetime

# Define where to save outputs 
# Should be directory in which model params are saved
base_output_dir = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel/2025-06-06"
leafletfa_script = "/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/full_workflow/02_run_leaflet.py"
anndata_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/052025/MOUSE_SPLICING_FOUNDATION_Anndata_ATSE_counts_with_waypoints_20250519_172401.h5ad" 

# Load parameter list from JSON file
param_file = os.path.join(base_output_dir, "parameter_combinations.json")
if not os.path.exists(param_file):
    raise FileNotFoundError(f"Parameter file {param_file} not found. Run generate_params.py first!")

with open(param_file, "r") as f:
    param_list = json.load(f)

# Create logs directory
log_dir = os.path.join(base_output_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

# Slurm job script template
job_script_template = """#!/bin/bash
#SBATCH --job-name=leaflet_{job_id}
#SBATCH --output={log_dir}/leaflet_{job_id}.out
#SBATCH --error={log_dir}/leaflet_{job_id}.err
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=800G
#SBATCH --partition=bigmem

# Set Python to run in unbuffered mode to ensure real-time output
export PYTHONUNBUFFERED=1

# Run Python script with proper output handling
python -u {leafletfa_script} {param_id} {base_output_dir} {anndata_file} 2>&1
"""

# Generate and submit jobs
for i, params in enumerate(param_list):
    job_script = job_script_template.format(
        job_id=i,
        log_dir=log_dir,
        param_id=i,
        leafletfa_script=leafletfa_script,
        base_output_dir=base_output_dir,
        anndata_file=anndata_file
    )

    # Save job script
    job_file = os.path.join(base_output_dir, f"leaflet_job_{i}.slurm")
    with open(job_file, "w") as f:
        f.write(job_script)

    # Submit the job to Slurm
    os.system(f"sbatch {job_file}")
    os.remove(job_file)

print(f"\nSubmitted {len(param_list)} jobs to Slurm.")

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel/2025-06-06
# python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/full_workflow/03_submit_jobs.py