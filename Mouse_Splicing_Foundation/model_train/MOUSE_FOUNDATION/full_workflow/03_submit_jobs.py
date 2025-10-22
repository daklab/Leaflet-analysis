import os
import json
import datetime
import subprocess
import time

# Define where to save outputs
# Should be directory in which model params are saved
base_output_dir = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel/2025-10-14"  # UPDATE THIS!
leafletfa_script = "/gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/full_workflow/02_run_leaflet.py"
anndata_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/model_ready_aligned_splicing_data_20251009_024406.h5ad"

# Verify files exist
print("Checking required files...")
if not os.path.exists(leafletfa_script):
    raise FileNotFoundError(f"Script not found: {leafletfa_script}")
print(f"✓ Found script: {leafletfa_script}")

if not os.path.exists(anndata_file):
    raise FileNotFoundError(f"Data file not found: {anndata_file}")
print(f"✓ Found data file: {anndata_file}")

# Load parameter list from JSON file
param_file = os.path.join(base_output_dir, "parameter_combinations.json")
if not os.path.exists(param_file):
    raise FileNotFoundError(f"Parameter file {param_file} not found. Run generate_params.py first!")

with open(param_file, "r") as f:
    param_list = json.load(f)

print(f"✓ Loaded {len(param_list)} parameter combinations")

# Create logs directory
log_dir = os.path.join(base_output_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
print(f"✓ Log directory: {log_dir}")

# Slurm job script template
job_script_template = """#!/bin/bash
#SBATCH --job-name=leaflet_{job_id}
#SBATCH --output={log_dir}/leaflet_{job_id}.out
#SBATCH --error={log_dir}/leaflet_{job_id}.err
#SBATCH --mem=300G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# Set Python to run in unbuffered mode to ensure real-time output
export PYTHONUNBUFFERED=1

# Run Python script with proper output handling
python -u {leafletfa_script} {param_id} {base_output_dir} {anndata_file} 2>&1
"""

# Generate and submit jobs
submitted_jobs = []
failed_submissions = []

print("\n" + "="*60)
print(f"Submitting {len(param_list)} jobs...")
print("="*60)

for i, params in enumerate(param_list):
    job_script = job_script_template.format(
        job_id=i,
        log_dir=log_dir,
        param_id=i,
        leafletfa_script=leafletfa_script,
        base_output_dir=base_output_dir,
        anndata_file=anndata_file
    )
    
    # Save job script temporarily
    job_file = os.path.join(base_output_dir, f"leaflet_job_{i}.slurm")
    with open(job_file, "w") as f:
        f.write(job_script)
    
    # Submit the job to Slurm using subprocess for better error handling
    try:
        result = subprocess.run(
            ["sbatch", job_file],
            capture_output=True,
            text=True,
            check=True
        )
        job_id = result.stdout.strip()
        submitted_jobs.append((i, job_id))
        print(f"✓ Job {i:3d}: {job_id}")
        
    except subprocess.CalledProcessError as e:
        failed_submissions.append((i, e.stderr))
        print(f"✗ Job {i:3d}: FAILED - {e.stderr}")
    
    # Clean up job script
    os.remove(job_file)
    
    # Small delay to avoid overwhelming the scheduler
    time.sleep(0.1)

print("\n" + "="*60)
print("Submission Summary:")
print("="*60)
print(f"Total jobs: {len(param_list)}")
print(f"Successfully submitted: {len(submitted_jobs)}")
print(f"Failed submissions: {len(failed_submissions)}")

if failed_submissions:
    print("\nFailed job details:")
    for param_id, error in failed_submissions:
        print(f"  Job {param_id}: {error}")

# Save submission log
log_file = os.path.join(base_output_dir, "job_submission_log.txt")
with open(log_file, "w") as f:
    f.write(f"Job Submission Log - {datetime.datetime.now()}\n")
    f.write("="*60 + "\n\n")
    f.write(f"Total parameter combinations: {len(param_list)}\n")
    f.write(f"Successfully submitted: {len(submitted_jobs)}\n")
    f.write(f"Failed submissions: {len(failed_submissions)}\n\n")
    
    f.write("Submitted Jobs:\n")
    for param_id, job_id in submitted_jobs:
        f.write(f"  Param {param_id}: {job_id}\n")
    
    if failed_submissions:
        f.write("\nFailed Submissions:\n")
        for param_id, error in failed_submissions:
            f.write(f"  Param {param_id}: {error}\n")

print(f"\nSubmission log saved to: {log_file}")

# Check job status
print("\n" + "="*60)
print("Checking job queue...")
print("="*60)
os.system("squeue -u $USER | grep leaflet | head -20")

print("\nDone! Monitor jobs with: squeue -u $USER")
print(f"View logs in: {log_dir}")

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/Leaflet/leafletFAmodel/2025-10-14
# python /gpfs/commons/home/kisaev/Leaflet-analysis/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/full_workflow/03_submit_jobs.py