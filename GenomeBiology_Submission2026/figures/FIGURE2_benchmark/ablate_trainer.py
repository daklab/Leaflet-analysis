"""Why did switching trainers improve simulation recovery so much?

The new benchmark fits with `LeafletFAMinibatch` (plate + AutoNormal, early stopping, gamma=0.1,
1 ELBO particle, batch 2048); the archived one used the legacy `LeafletFA` class (full batch,
gamma=0.5, 5 particles, a fixed 1,200 epochs). Ψ recovery went 0.82 -> 0.996 and effect-size
recovery 0.41 -> 0.94 at matched difficulty. Four things changed at once, so the gain is
unattributed.

This varies ONE factor at a time from the new default back toward the old settings, all inside the
new trainer, plus one config that adopts *every* old hyperparameter simultaneously:

    new_default    B=2048  gamma=0.1  particles=1  early-stop        <- the new benchmark
    gamma05        B=2048  gamma=0.5  particles=1  early-stop
    particles5     B=2048  gamma=0.1  particles=5  early-stop
    no_earlystop   B=2048  gamma=0.1  particles=1  fixed ~1200 epochs
    fullbatch      B=N     gamma=0.1  particles=1  early-stop
    old_like       B=N     gamma=0.5  particles=5  fixed 1200 epochs  <- THE DISCRIMINATOR

`old_like` is the whole point: it is the new *implementation* running the old *hyperparameters*.

    old_like scores ~0.99  ->  the gain came from the implementation (plate / AutoNormal guide),
                               not from the settings.
    old_like scores ~0.82  ->  the gain came from the hyperparameters, and the implementation is
                               incidental.

Anything in between apportions it. Run:  ablate_trainer.py <sim.h5ad> <out.csv> [device]
"""
from __future__ import annotations
import sys, os, time
import numpy as np
import pandas as pd
import anndata as ad

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
import pyro
from leafletfa.minibatch import LeafletFAMinibatch
from sim_benchmark import truth_of, metrics, _dense

# (name, batch, gamma, particles, early_stop, steps)  -- steps is a cap when early_stop is on
CONFIGS = [
    ("new_default",  2048, 0.1, 1, True,  5000),
    ("gamma05",      2048, 0.5, 1, True,  5000),
    ("particles5",   2048, 0.1, 5, True,  5000),
    ("no_earlystop", 2048, 0.1, 1, False, 4093),   # 1200 epochs at B=2048 on 6,986 cells
    ("fullbatch",    None, 0.1, 1, True,  5000),   # None -> full batch
    ("old_like",     None, 0.5, 5, False, 1200),   # every old setting, new implementation
]


def run(sim, K, name, batch, gamma, particles, early_stop, steps, seed, device, T):
    junc = _dense(sim.layers["cell_by_junction_matrix"]).astype(np.float32)
    clus = _dense(sim.layers["cell_by_cluster_matrix"]).astype(np.float32)
    pyro.clear_param_store()
    m = LeafletFAMinibatch(junc, clus, K=K, junc_specific_prior=True,
                           device=device, likelihood="betabinomial")
    B = m.N if batch is None else min(batch, m.N)
    t0 = time.time()
    m.fit(num_steps=steps, subsample_size=B, lr=0.05, gamma=gamma, num_particles=particles,
          seed=seed, verbose=False, early_stop=early_stop)
    secs = round(time.time() - t0, 1)
    r = metrics(T, m.psi_mean().cpu().numpy(), m.phi_mean().cpu().numpy())
    r.update(dict(config=name, batch=B, gamma=gamma, particles=particles,
                  early_stop=early_stop, step_cap=steps,
                  steps_used=int(getattr(m, "stopped_step", steps)),
                  epochs=round(int(getattr(m, "stopped_step", steps)) * B / m.N, 1), seconds=secs))
    return r


def main():
    sim_path, out_csv = sys.argv[1], sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "cuda"
    sim = ad.read_h5ad(sim_path)
    import json
    sp = json.loads(str(sim.uns["sim_params"]))
    K, seed = int(sp["K"]), int(sp["seed"])
    T = truth_of(sim)
    print(f"=== ablation on {os.path.basename(sim_path)} (K={K}, frac={sp['frac_differential']}) ===",
          flush=True)
    rows = []
    for cfg in CONFIGS:
        r = run(sim, K, *cfg, seed=seed, device=device, T=T)
        r.update(dict(frac=sp["frac_differential"], seed=seed))
        rows.append(r)
        print(f"  {r['config']:>13}  psi_r={r['psi_r']:.3f}  sd={r['spearman_delta']:.3f}  "
              f"sil={r['sil']:.3f}  epochs={r['epochs']}  {r['seconds']}s", flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
