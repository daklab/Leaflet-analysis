#!/usr/bin/env python
"""Cache two supplementary analyses so the figure notebook stays fast:
   baselines.csv : NMF (naive 0-impute AND fair mean-impute) vs the LeafletFA reference, K=5/10/20
   pruning.csv   : LeafletFA pi per factor at K=5/10/20, so we can show effective-K after pruning
Run once (CPU ok). The notebook just reads the CSVs.
"""
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, io, contextlib, time
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from leafletfa import LeafletFA

SIM="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION/sim_heart_K5_binom_eff1.0_seed0.h5ad"
sim=ad.read_h5ad(SIM)
ev=np.asarray(sim.uns["atse_ids"],dtype=object); ei={x:i for i,x in enumerate(ev)}
cev=np.array([ei[x] for x in sim.var["event_id"].astype(str).values])
skp=np.where(sim.var["junction_role"].values=="J3")[0]; ordr=cev[skp]
tp=np.asarray(sim.uns["true_psi_skip"])[:,np.sort(np.unique(ordr))]
tphi=np.asarray(sim.obsm["true_phi"]); Kt=tp.shape[0]

def score(psi_hat, phi_hat):
    ph=psi_hat[:,skp][:,np.argsort(ordr)]; Kf=ph.shape[0]
    C=np.zeros((Kf,Kt))
    for i in range(Kf):
        for j in range(Kt):
            m=np.isfinite(tp[j])&np.isfinite(ph[i])
            C[i,j]=np.corrcoef(ph[i][m],tp[j][m])[0,1] if m.sum()>2 else 0
    r,c=linear_sum_assignment(-np.abs(np.nan_to_num(C)))
    return float(np.median([C[i,j] for i,j in zip(r,c)])), float(adjusted_rand_score(tphi.argmax(1),phi_hat.argmax(1)))

def to_np(x):
    import torch
    return x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)

# ---- junction-ratio matrix, with the missing mask -----------------------------------------
R=sim.layers["junc_ratio"]; R=np.asarray(R.todense() if sp.issparse(R) else R,dtype=np.float64)
C=sim.layers["cell_by_cluster_matrix"]; C=np.asarray(C.todense() if sp.issparse(C) else C)
covered=(C>0)                                   # True where the ATSE has reads (PSI is observed)
Xzero=np.where(covered, np.clip(np.nan_to_num(R),0,1), 0.0)          # naive: uncovered -> 0
colmean=np.array([Xzero[covered[:,j],j].mean() if covered[:,j].any() else 0 for j in range(R.shape[1])])
Xmean=np.where(covered, np.clip(np.nan_to_num(R),0,1), colmean[None,:])  # fair: uncovered -> junction mean

rows=[]
for K in (5,10,20):
    for tag,X in (("NMF (0-impute)",Xzero),("NMF (mean-impute)",Xmean)):
        t=time.time(); m=NMF(n_components=K,init="nndsvda",random_state=0,max_iter=300)
        W=m.fit_transform(X); H=m.components_
        pr,ari=score(H,W)
        rows.append(dict(method=tag,K=K,psi_r=pr,ari=ari,secs=round(time.time()-t)))
        print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv("baselines.csv",index=False); print("wrote baselines.csv",flush=True)

# ---- LeafletFA pi per factor, for the pruning story ---------------------------------------
prune_rows=[]
for K in (5,10,20):
    m=LeafletFA(adata=sim,K=K,junc_specific_prior=True,waypoints_use=False,input_conc_prior=None,
                print_epochs=10**6,num_epochs=1200,patience=10**6,min_delta=0,min_delta_rel=1e-6,
                ELBO_num_particles=5,num_samples=50,lr=0.05,gamma=0.5,log_wandb=False,loss_plot=False)
    with contextlib.redirect_stdout(io.StringIO()):
        m.from_anndata(); m.train(num_initializations=1,seeds=[0]); m.get_all_variables()
    pi=np.sort(to_np(m.pi))[::-1]
    for rank,p in enumerate(pi): prune_rows.append(dict(K=K,rank=rank,pi=float(p)))
    print(f"K={K}: n_active(pi>0.01)={int((pi>0.01).sum())} of {K}   pi={np.round(pi,3).tolist()}",flush=True)
pd.DataFrame(prune_rows).to_csv("pruning.csv",index=False); print("wrote pruning.csv",flush=True)
print("*** DONE ***")
