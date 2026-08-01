#!/bin/bash
/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python - <<'PYEOF'
import pandas as pd, glob, re, os
F="/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/FIGURE2_benchmark"
# shard index -> (effect, seed), matching eff2_benchmark.sh exactly
EFFS=[0.1,0.1,0.1,0.25,0.25,0.25,0.5,0.5,0.5,1.0,1.0,1.0]
frames=[]
for f in sorted(glob.glob(f"{F}/rerun_logs/effbench_*.csv"),
                key=lambda p:int(re.search(r"effbench_(\d+)",p).group(1))):
    i=int(re.search(r"effbench_(\d+)",f).group(1))
    d=pd.read_csv(f)
    # shards 0/1 were produced before sim_benchmark.py recorded 'effect'; backfill from the index,
    # which is deterministic (the array maps index -> effect in eff2_benchmark.sh).
    if "effect" not in d.columns or d["effect"].isna().all():
        d["effect"]=EFFS[i]
        print(f"  backfilled effect={EFFS[i]} for shard {i}")
    frames.append(d)
d=pd.concat(frames, ignore_index=True)
out=f"{F}/benchmark_effect_sweep.csv"; d.to_csv(out,index=False)
print(f"benchmark_effect_sweep.csv: {len(d)} rows from {len(frames)} shards")
print(d.groupby(["effect","model","likelihood"])[["psi_r","spearman_delta","sil","auroc"]].mean().round(3).to_string())
PYEOF
echo "EFFECT SWEEP COMPLETE"
