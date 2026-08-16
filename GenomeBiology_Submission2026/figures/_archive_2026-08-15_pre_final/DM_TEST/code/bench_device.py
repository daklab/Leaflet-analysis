#!/usr/bin/env python
"""Time a DM+simplex fit on CPU vs GPU. Usage: bench_device.py <sim.h5ad> <cpu|cuda>

Must live on SHARED storage -- /tmp is node-local, so a script written there from the login node
does not exist on the compute node and the job dies instantly with 'No such file or directory'.
"""
import sys, time, numpy as np, torch, pyro, h5py
import scipy.sparse as sp
sys.path.insert(0, '/gpfs/commons/home/kisaev/LeafletFA')
from leafletfa.minibatch import LeafletFAMinibatch

p, dev = sys.argv[1], sys.argv[2]
f = h5py.File(p, 'r')
def dense(l):
    g = f['layers'][l]; s = tuple(g.attrs['shape'])
    return np.asarray(sp.csr_matrix((g['data'][:], g['indices'][:], g['indptr'][:]),
                                    shape=s).todense(), np.float32)
junc, clus = dense('cell_by_junction_matrix'), dense('cell_by_cluster_matrix')
n = f['var']['event_id']
ev = (np.array([x.decode() if isinstance(x, bytes) else x for x in n['categories'][:]])[n['codes'][:]]
      if isinstance(n, h5py.Group)
      else np.array([x.decode() if isinstance(x, bytes) else x for x in n[:]]))
f.close()
print(f"device={dev}  cuda_available={torch.cuda.is_available()}", flush=True)
if dev == 'cuda':
    if not torch.cuda.is_available():
        print("  CUDA NOT AVAILABLE -- aborting"); sys.exit(1)
    print(f"  {torch.cuda.get_device_name(0)}", flush=True)
print(f"  data {junc.shape[0]:,} cells x {junc.shape[1]:,} junctions", flush=True)
pyro.clear_param_store()
m = LeafletFAMinibatch(junc, clus, K=5, device=dev, likelihood='dirichletmultinomial',
                       psi_parameterization='simplex', event_id=ev)
t0 = time.time()
m.fit(num_steps=200, subsample_size=2048, lr=0.05, gamma=0.5, seed=0, verbose=False)
dt = time.time() - t0
print(f"  200 steps in {dt:.1f}s -> {dt/200*1000:.0f} ms/step "
      f"-> 3000 steps ~ {dt/200*3000/60:.1f} min", flush=True)
