#!/usr/bin/env python
"""Benchmark sweep for Figure 2 panels B/D/E/G (the %DS-axis benchmark, per Karin's original figure).

Grid, on each (%DS in {0.1,0.5,0.9}) x (seed in {0,1,2}) simulated dataset (K=5, Binomial truth):
    LeafletFA:  likelihood {Beta-Binomial, Binomial} x junc_prior {False, True}   (4 configs)
    NMF:        mean-impute baseline                                              (1)
Metrics per fit: silhouette, spearman(sim delta, est delta), average precision, precision@op, psi_r, ari.
Also dumps per-event scores + phi for ONE representative fit -> panels F (PR curve) and H (UMAP).

Appends to benchmark_results.csv (resumable). Reuses leafletfa.chunked / fit machinery via LeafletFA.
"""
import os, io, contextlib, glob, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import (roc_auc_score, adjusted_rand_score, silhouette_score,
                             average_precision_score, precision_score)
from sklearn.decomposition import NMF
from leafletfa import LeafletFA

# FAIL-FAST CUDA guard: a previous run silently fell back to CPU (0% GPU, 98% CPU) and ground one
# fit for an hour. Refuse to run on CPU -- fail immediately so SLURM can requeue on a real GPU.
assert torch.cuda.is_available(), "CUDA not visible on this node -- refusing to run on CPU. Requeue."
print(f"CUDA OK: {torch.cuda.get_device_name(0)}", flush=True)

SIMDIR="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION"
CSV="benchmark_results.csv"
DELTA=0.10                                   # DS-calling operating threshold on |delta psi|

def to_np(x): return x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)

def truth_of(sim):
    ev=np.asarray(sim.uns["atse_ids"],dtype=object); ei={x:i for i,x in enumerate(ev)}
    cev=np.array([ei[x] for x in sim.var["event_id"].astype(str).values])
    skp=np.where(sim.var["junction_role"].values=="J3")[0]; ordr=cev[skp]
    tp=np.asarray(sim.uns["true_psi_skip"])[:,np.sort(np.unique(ordr))]
    isd=np.array([sim.var["is_differential"].values[cev==i][0] for i in range(len(ev))])
    td=np.asarray(sim.uns["atse_delta_psi"]) if "atse_delta_psi" in sim.uns else None
    return dict(psi=tp,phi=np.asarray(sim.obsm["true_phi"]),skip_idx=skp,order=ordr,
                is_diff=isd,K_true=tp.shape[0],true_delta=td)

def metrics(T, psi_hat, phi_hat):
    tp,tphi,skp,ordr=T["psi"],T["phi"],T["skip_idx"],T["order"]; Kt=T["K_true"]
    ph=psi_hat[:,skp][:,np.argsort(ordr)]; Kf=ph.shape[0]
    C=np.zeros((Kf,Kt))
    for i in range(Kf):
        for j in range(Kt):
            m=np.isfinite(tp[j])&np.isfinite(ph[i]); C[i,j]=np.corrcoef(ph[i][m],tp[j][m])[0,1] if m.sum()>2 else 0
    r,c=linear_sum_assignment(-np.abs(np.nan_to_num(C)))
    psi_r=float(np.median([C[i,j] for i,j in zip(r,c)]))
    ari=float(adjusted_rand_score(tphi.argmax(1),phi_hat.argmax(1)))
    lab=tphi.argmax(1)
    sil=float(silhouette_score(phi_hat,lab)) if len(np.unique(lab))>1 else np.nan
    d=ph.max(0)-ph.min(0); ok=np.isfinite(d); isd=T["is_diff"]
    ap=float(average_precision_score(isd[ok],d[ok])); auroc=float(roc_auc_score(isd[ok],d[ok]))
    prec=float(precision_score(isd[ok], d[ok]>DELTA, zero_division=0))
    sd=np.nan
    if T["true_delta"] is not None:
        td=T["true_delta"]; m2=ok&np.isfinite(td)&isd
        if m2.sum()>2: sd=float(spearmanr(td[m2],d[m2]).correlation)
    return dict(psi_r=psi_r,ari=ari,sil=sil,ap=ap,auroc=auroc,precision=prec,spearman_delta=sd,
                effect=d[ok],is_diff=isd[ok])

def fit_lfa(sim, likelihood, junc_prior, seed):
    conc=torch.tensor(float("inf")) if likelihood=="binom" else None
    m=LeafletFA(adata=sim,K=5,junc_specific_prior=junc_prior,waypoints_use=False,input_conc_prior=conc,
                print_epochs=10**6,num_epochs=1200,patience=10**6,min_delta=0,min_delta_rel=1e-6,
                ELBO_num_particles=5,num_samples=50,lr=0.05,gamma=0.5,log_wandb=False,loss_plot=False)
    with contextlib.redirect_stdout(io.StringIO()):
        m.from_anndata(); m.train(num_initializations=1,seeds=[seed]); m.get_all_variables()
    return to_np(m.psi), to_np(m.assign_post)

def fit_nmf(sim):
    R=sim.layers["junc_ratio"]; R=np.asarray(R.todense() if sp.issparse(R) else R,float)
    C=sim.layers["cell_by_cluster_matrix"]; C=np.asarray(C.todense() if sp.issparse(C) else C)
    cov=(C>0)
    # HOIST the clip out of the loop. It used to sit INSIDE the list comprehension, so a full
    # 6,986 x 12,804 matrix was rebuilt once per junction -- 12,804 times, ~1.1e12 redundant ops.
    # That hung the benchmark for 48 min at 99% CPU / 0% GPU with zero progress.
    Rc = np.clip(np.nan_to_num(R), 0, 1)
    cnt = cov.sum(0)
    cm  = np.where(cnt > 0, (Rc * cov).sum(0) / np.maximum(cnt, 1), 0.0)
    # TWO imputations of the uncovered (NaN) entries -- NMF has no missing-data model, so the
    # choice matters and is part of the point:
    #   0-impute    : uncovered -> 0. The naive default; tells NMF every uncovered junction is
    #                 fully skipped, which is FALSE (uncovered != PSI 0). Handicaps NMF.
    #   mean-impute : uncovered -> junction mean over covered cells. The fairer baseline.
    # LeafletFA models coverage explicitly and never imputes. Beating BOTH is the claim.
    out = {}
    for tag, X in (("0-impute", np.where(cov, Rc, 0.0)),
                   ("mean-impute", np.where(cov, Rc, cm[None, :]))):
        m = NMF(n_components=5, init="nndsvda", random_state=0, max_iter=300)
        W = m.fit_transform(X); H = m.components_
        out[tag] = (H, W)
    return out

done=set()
if os.path.exists(CSV):
    for _,row in pd.read_csv(CSV).iterrows():
        done.add((row["frac"],row["seed"],row["model"],str(row.get("likelihood","-")),str(row.get("junc_prior","-"))))

for frac in (0.1,0.5,0.9):
    for seed in (0,1,2):
        f=f"{SIMDIR}/sim_heart_K5_binom_eff1.0_frac{frac:g}_seed{seed}.h5ad"
        if not os.path.exists(f): print("MISSING",f,flush=True); continue
        sim=ad.read_h5ad(f); T=truth_of(sim)
        # LeafletFA configs
        for lk in ("bb","binom"):
            for jp in (False,True):
                key=(frac,seed,"LeafletFA",str(lk),str(jp))
                if key in done: continue
                psi,phi=fit_lfa(sim,lk,jp,seed); r=metrics(T,psi,phi)
                pd.DataFrame([dict(frac=frac,seed=seed,model="LeafletFA",likelihood=lk,junc_prior=jp,
                    **{k:v for k,v in r.items() if k not in ("effect","is_diff")})]).to_csv(
                    CSV,mode="a",header=not os.path.exists(CSV),index=False)
                print(f"frac{frac} s{seed} LeafletFA {lk} jp={jp}: sil={r['sil']:.3f} ap={r['ap']:.3f} "
                      f"spearman_d={r['spearman_delta']:.3f} prec={r['precision']:.3f}",flush=True)
                # representative fit for F/H: BB + junc_prior True at 50% DS, seed 0
                if lk=="bb" and jp and frac==0.5 and seed==0:
                    np.savez("representative_fit.npz", effect=r["effect"], is_diff=r["is_diff"],
                             phi=phi, true_prog=T["phi"].argmax(1))
        # NMF baseline
        nmf_out = None
        for _imp in ("0-impute", "mean-impute"):
            key=(frac,seed,"NMF",_imp,str(False))
            if key in done: continue
            if nmf_out is None: nmf_out = fit_nmf(sim)
            H,W = nmf_out[_imp]; r=metrics(T,H,W)
            pd.DataFrame([dict(frac=frac,seed=seed,model="NMF",likelihood=_imp,junc_prior=False,
                **{k:v for k,v in r.items() if k not in ("effect","is_diff")})]).to_csv(
                CSV,mode="a",header=not os.path.exists(CSV),index=False)
            print(f"frac{frac} s{seed} NMF {_imp}: sil={r['sil']:.3f} ap={r['ap']:.3f}",flush=True)
print("*** BENCHMARK DONE ***")
