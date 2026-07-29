"""Figure 2 benchmark — LeafletFA vs NMF on one simulated cassette-exon dataset.

Supersedes `benchmark_sweep.py`, which fitted through the legacy `LeafletFA` class (full-batch,
`num_epochs`). That is NOT the code path the atlas models use, so the old simulation validated a
different trainer than the paper's results. This script fits with `LeafletFAMinibatch` — the
Pyro-plate minibatch trainer with early stopping — so the controlled experiment and the atlas are
the same implementation.

Minibatching note: the simulation has ~7k cells, so the atlas batch (B=8192) would silently be
full-batch. B=2048 is used instead so the subsampling plate is genuinely exercised; the run asserts
that the traced `phi` really is batch-shaped, so "minibatch" can never quietly mean "full batch"
again (that bug is what motivated the rewrite).

Arms, all fitted at the true K:
    LeafletFA  Beta-Binomial / Binomial  x  junction-specific / global psi prior
    NMF        rank K, 0-impute and mean-impute (NMF has no missing-data model)

Usage:  sim_benchmark.py <sim.h5ad> <out.csv> [device=cuda] [batch=2048]
"""
from __future__ import annotations
import os, sys, time, itertools
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.decomposition import NMF
from sklearn.metrics import (adjusted_rand_score, silhouette_score, average_precision_score,
                             roc_auc_score, precision_score)

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
import pyro
from pyro import poutine
from leafletfa.minibatch import LeafletFAMinibatch

DELTA = 0.10          # |delta psi| threshold at which an ATSE is called differential
STEPS = 5000          # a CAP; early stopping decides the actual number
LR, GAMMA = 0.05, 0.1


def _dense(x):
    return np.asarray(x.todense() if sp.issparse(x) else x)


def truth_of(sim):
    ev = np.asarray(sim.uns["atse_ids"], dtype=object)
    ei = {x: i for i, x in enumerate(ev)}
    cev = np.array([ei[x] for x in sim.var["event_id"].astype(str).values])
    skp = np.where(sim.var["junction_role"].values == "J3")[0]
    ordr = cev[skp]
    tp = np.asarray(sim.uns["true_psi_skip"])[:, np.sort(np.unique(ordr))]
    isd = np.array([sim.var["is_differential"].values[cev == i][0] for i in range(len(ev))])
    td = np.asarray(sim.uns["atse_delta_psi"]) if "atse_delta_psi" in sim.uns else None
    return dict(psi=tp, phi=np.asarray(sim.obsm["true_phi"]), skip_idx=skp, order=ordr,
                is_diff=isd, K_true=tp.shape[0], true_delta=td)


def metrics(T, psi_hat, phi_hat):
    """Identical definitions to the previous benchmark, so numbers stay comparable."""
    tp, tphi, skp, ordr = T["psi"], T["phi"], T["skip_idx"], T["order"]
    ph = psi_hat[:, skp][:, np.argsort(ordr)]
    Kf, Kt = ph.shape[0], T["K_true"]
    C = np.zeros((Kf, Kt))
    for i in range(Kf):
        for j in range(Kt):
            m = np.isfinite(tp[j]) & np.isfinite(ph[i])
            C[i, j] = np.corrcoef(ph[i][m], tp[j][m])[0, 1] if m.sum() > 2 else 0
    r, c = linear_sum_assignment(-np.abs(np.nan_to_num(C)))
    psi_r = float(np.median([C[i, j] for i, j in zip(r, c)]))
    lab = tphi.argmax(1)
    ari = float(adjusted_rand_score(lab, phi_hat.argmax(1)))
    sil = float(silhouette_score(phi_hat, lab)) if len(np.unique(lab)) > 1 else np.nan
    d = ph.max(0) - ph.min(0)
    ok = np.isfinite(d)
    isd = T["is_diff"]
    ap = float(average_precision_score(isd[ok], d[ok]))
    auroc = float(roc_auc_score(isd[ok], d[ok]))
    prec = float(precision_score(isd[ok], d[ok] > DELTA, zero_division=0))
    sd = np.nan
    if T["true_delta"] is not None:
        td = T["true_delta"]
        m2 = ok & np.isfinite(td) & isd
        if m2.sum() > 2:
            sd = float(spearmanr(td[m2], d[m2]).correlation)
    return dict(psi_r=psi_r, ari=ari, sil=sil, ap=ap, auroc=auroc, precision=prec,
                spearman_delta=sd)


def fit_lfa(sim, K, likelihood, junc_prior, seed, device, batch):
    """Fit through the SAME trainer the atlas uses, and verify subsampling is really on."""
    junc = _dense(sim.layers["cell_by_junction_matrix"]).astype(np.float32)
    clus = _dense(sim.layers["cell_by_cluster_matrix"]).astype(np.float32)
    # Optional deterministic psi init (LFA_INIT=jitter). The default start puts every program at
    # psi=0.5 and every cell at 1/K, so the K programs are exactly identical and can only separate via
    # sampling noise -- which the likelihood cannot steer once overdispersion is high. See
    # leafletfa.initialization for the measurements.
    psi0 = None
    if os.environ.get("LFA_INIT") == "jitter":
        import sys as _sys
        _sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
        from leafletfa.initialization import jitter_psi_init
        psi0 = jitter_psi_init(junc, clus, K)
    m = LeafletFAMinibatch(junc, clus, K=K, junc_specific_prior=junc_prior, psi_init=psi0,
                           device=device, likelihood=likelihood)
    B = min(batch, m.N)
    # CANARY: trace the model and confirm phi is (B, K), not (N, K). The pre-rewrite trainer
    # silently ran full-batch while claiming to subsample; this makes that impossible to miss.
    m._B = B
    tr = poutine.trace(m.model).get_trace()
    got = tuple(tr.nodes["phi"]["value"].shape)
    assert got == (B, K), f"subsampling NOT engaged: phi is {got}, expected {(B, K)}"
    t0 = time.time()
    m.fit(num_steps=STEPS, subsample_size=B, lr=LR, gamma=GAMMA, seed=seed, verbose=False)
    return (m.psi_mean().cpu().numpy(), m.phi_mean().cpu().numpy(),
            int(getattr(m, "stopped_step", STEPS)), round(time.time() - t0, 1), B)


def fit_nmf(sim, K):
    """Both imputations of the uncovered entries — NMF cannot represent missingness."""
    R = np.clip(np.nan_to_num(_dense(sim.layers["junc_ratio"]).astype(float)), 0, 1)
    cov = _dense(sim.layers["cell_by_cluster_matrix"]) > 0
    cnt = cov.sum(0)
    cm = np.where(cnt > 0, (R * cov).sum(0) / np.maximum(cnt, 1), 0.0)
    out = {}
    for tag, X in (("0-impute", np.where(cov, R, 0.0)),
                   ("mean-impute", np.where(cov, R, cm[None, :]))):
        t0 = time.time()
        mm = NMF(n_components=K, init="nndsvda", random_state=0, max_iter=300)
        W = mm.fit_transform(X)
        out[tag] = (mm.components_, W, round(time.time() - t0, 1))
    return out


def main():
    sim_path, out_csv = sys.argv[1], sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "cuda"
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
    sim = ad.read_h5ad(sim_path)
    sp_ = dict(sim.uns["sim_params"]) if hasattr(sim.uns["sim_params"], "keys") else {}
    if not sp_:
        import json
        sp_ = json.loads(str(sim.uns["sim_params"]))
    frac, seed_sim, K = sp_["frac_differential"], sp_["seed"], int(sp_["K"])
    # `effect` scales HOW DIFFERENT the programs are at a differential event, as opposed to
    # `frac_differential` which sets HOW MANY events differ. It is the axis the effect-size
    # sweep varies, so it must be recorded or the sweep levels are indistinguishable.
    effect = float(sp_.get("effect", 1.0))
    # The TRUTH's overdispersion. None/inf means the counts were drawn Binomial, so a Binomial
    # model is correctly specified; a finite value means they are overdispersed and a Binomial
    # cannot represent the extra variance. Must be recorded or the arms are indistinguishable.
    _bb = sp_.get("bb_conc", None)
    sim_bb_conc = float("inf") if _bb in (None, "inf") else float(_bb)
    T = truth_of(sim)
    print(f"=== {os.path.basename(sim_path)}: {sim.shape[0]:,} cells x {sim.shape[1]:,} junctions, "
          f"K_true={K}, frac_diff={frac}, effect={effect}, sim_bb_conc={sim_bb_conc}, "
          f"seed={seed_sim} ===", flush=True)
    rows = []
    for likelihood, junc_prior in itertools.product(("betabinomial", "binomial"), (True, False)):
        pyro.clear_param_store()
        psi, phi, used, secs, B = fit_lfa(sim, K, likelihood, junc_prior, seed_sim, device, batch)
        r = metrics(T, psi, phi)
        r.update(dict(model="LeafletFA", likelihood={"betabinomial": "bb", "binomial": "binom"}[likelihood],
                      junc_prior=junc_prior, frac=frac, effect=effect, sim_bb_conc=sim_bb_conc,
                      seed=seed_sim, K=K, batch_size=B,
                      steps_used=used, seconds=secs, trainer="minibatch_plate"))
        rows.append(r)
        print(f"  LeafletFA {likelihood[:4]} jp={str(junc_prior):>5}  psi_r={r['psi_r']:.3f} "
              f"sd={r['spearman_delta']:.3f} sil={r['sil']:.3f}  steps={used}  {secs}s", flush=True)
    for tag, (H, W, secs) in fit_nmf(sim, K).items():
        r = metrics(T, H, W)
        r.update(dict(model="NMF", likelihood=tag, junc_prior=False, frac=frac, effect=effect,
                      sim_bb_conc=sim_bb_conc, seed=seed_sim,
                      K=K, batch_size=np.nan, steps_used=np.nan, seconds=secs, trainer="sklearn_NMF"))
        rows.append(r)
        print(f"  NMF {tag:>11}          psi_r={r['psi_r']:.3f} "
              f"sd={r['spearman_delta']:.3f} sil={r['sil']:.3f}  {secs}s", flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
