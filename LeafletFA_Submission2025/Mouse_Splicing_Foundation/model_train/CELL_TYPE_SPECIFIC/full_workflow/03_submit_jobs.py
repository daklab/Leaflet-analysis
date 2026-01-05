# submit_jobs_celltype_gpu.py
import os
import json

# Define where to save outputs
base_output_dir = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel_celltype/2025-07-05"
leafletfa_script = "/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/CELL_TYPE_SPECIFIC/full_workflow/02_run_leaflet.py"

# Load parameter list from JSON file
param_file = os.path.join(base_output_dir, "parameter_combinations_celltype.json")
if not os.path.exists(param_file):
    raise FileNotFoundError(f"Parameter file {param_file} not found. Run generate_params_celltype.py first!")

with open(param_file, "r") as f:
    param_list = json.load(f)

# Create logs directory
log_dir = os.path.join(base_output_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

# Create output directories for each cell type
for param_set in param_list:
    cell_type = param_set['cell_type']
    cell_type_output_dir = os.path.join(base_output_dir, "results", cell_type)
    os.makedirs(cell_type_output_dir, exist_ok=True)

# Simple GPU job script template
job_script_template = """#!/bin/bash
#SBATCH --job-name=leaflet_{cell_type}_{job_id}
#SBATCH --output={log_dir}/leaflet_{cell_type}_{job_id}.out
#SBATCH --error={log_dir}/leaflet_{cell_type}_{job_id}.err
#SBATCH --mem=300G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

export PYTHONUNBUFFERED=1

python -u {leafletfa_script} {param_id} {base_output_dir}
"""

# Submit all jobs
print(f"Found {len(param_list)} parameter combinations.")
submitted_jobs = 0

for i, params in enumerate(param_list):
    cell_type = params['cell_type'].replace(' ', '_').replace('/', '_')
    
    job_script_content = job_script_template.format(
        job_id=i,
        cell_type=cell_type,
        log_dir=log_dir,
        param_id=i,
        leafletfa_script=leafletfa_script,
        base_output_dir=base_output_dir
    )
    
    # Save job script
    job_file = os.path.join(base_output_dir, f"leaflet_job_{cell_type}_{i}.slurm")
    with open(job_file, "w") as f:
        f.write(job_script_content)
    
    # Submit the job to Slurm
    os.system(f"sbatch {job_file}")
    os.remove(job_file)
    submitted_jobs += 1

print(f"Submitted {submitted_jobs} jobs to Slurm.")
print(f"Log files will be saved in: {log_dir}")
print(f"Results will be saved in: {os.path.join(base_output_dir, 'results')}")

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel_celltype/2025-07-05
# python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/CELL_TYPE_SPECIFIC/full_workflow/03_submit_jobs.py