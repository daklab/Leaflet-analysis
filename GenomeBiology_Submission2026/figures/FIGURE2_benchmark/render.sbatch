#!/bin/bash
#SBATCH --job-name=sim_render
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/FIGURE2_benchmark/render.log
cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/FIGURE2_benchmark
/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/jupyter nbconvert \
    --to notebook --execute --allow-errors --ExecutePreprocessor.timeout=-1 \
    --output simulation_explore_executed.ipynb simulation_explore.ipynb
echo "=== cells that errored on execution ==="
/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python - <<PYEOF
import json
d=json.load(open("simulation_explore_executed.ipynb"))
err=0
for i,c in enumerate(d["cells"]):
    if c["cell_type"]!="code": continue
    for o in c.get("outputs",[]):
        if o.get("output_type")=="error":
            print(f"  cell {i}: {o['ename']}: {o['evalue'][:80]}"); err+=1
print("TOTAL ERRORS:", err)
PYEOF
