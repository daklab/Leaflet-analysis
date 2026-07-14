#!/usr/bin/env python
"""Figure-2 sweep, headless. Streams progress; appends each fit to sweep_results.csv.

Extracted verbatim from simulation_explore.ipynb cells [1, 16, 18, 20] so the two cannot diverge.
"""
import matplotlib; matplotlib.use('Agg')
display = lambda *a, **k: None

# ============ notebook cell 1 ============
import json

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from scipy.stats import pearsonr

plt.rcParams["figure.dpi"] = 110

SIM = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION"
SIGNAL = f"{SIM}/sim_heart_K5_binom_eff1.0_seed0.h5ad"   # 30% cassettes differential
NULL   = f"{SIM}/sim_heart_K5_binom_eff0.0_seed0.h5ad"   # pure null (FPR control)

a = ad.read_h5ad(SIGNAL)
params = json.loads(a.uns["sim_params"])
K = params["K"]

print(f"{a.n_obs:,} cells x {a.n_vars:,} junctions ({a.n_vars//3:,} cassettes)")
print("layers:", list(a.layers))
print("params:", {k: params[k] for k in ["tissue", "K", "emission", "frac_differential", "effect", "seed"]})
print("\ncells per program:", np.bincount(a.obs['true_program']))
print("junction roles:", a.var['junction_role'].value_counts().to_dict())

# ============ notebook cell 16 ============
# ── METRICS: defined once, used by every fit ─────────────────────────────────
import io, contextlib, itertools, os, time
import numpy as np, pandas as pd, torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score, adjusted_rand_score

# `sim` is the SIGNAL simulation -- cell 1 loaded it as `a`. Everything below fits and scores on it.
sim = a

def to_np(x):
    """Model outputs come back as torch tensors (possibly on GPU), numpy, or lists. Normalise."""
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

def truth_of(sim):
    """Ground truth for a sim, aligned to the J3 (skip) junction of each cassette."""
    ev   = np.asarray(sim.uns["atse_ids"], dtype=object)
    ei   = {x: i for i, x in enumerate(ev)}
    cev  = np.array([ei[x] for x in sim.var["event_id"].astype(str).values])
    skp  = np.where(sim.var["junction_role"].values == "J3")[0]
    ordr = cev[skp]
    tp   = np.asarray(sim.uns["true_psi_skip"])[:, np.sort(np.unique(ordr))]   # (K_true, n_ATSE)
    isd  = np.array([sim.var["is_differential"].values[cev == i][0] for i in range(len(ev))])
    return dict(psi=tp, phi=np.asarray(sim.obsm["true_phi"]), skip_idx=skp, order=ordr,
                is_diff=isd, K_true=tp.shape[0])

def score_fit(T, psi_hat, phi_hat):
    """Score one fit against ground truth T. Works when K_fit != K_true."""
    tp, tphi, skp, ordr = T["psi"], T["phi"], T["skip_idx"], T["order"]
    Kt = T["K_true"]
    ph = psi_hat[:, skp][:, np.argsort(ordr)]           # learned skip usage, (K_fit, n_ATSE)
    Kf = ph.shape[0]

    # Hungarian match learned programs -> true programs
    C = np.zeros((Kf, Kt))
    for i in range(Kf):
        for j in range(Kt):
            m = np.isfinite(tp[j]) & np.isfinite(ph[i])
            C[i, j] = np.corrcoef(ph[i][m], tp[j][m])[0, 1] if m.sum() > 2 else 0.0
    r, c = linear_sum_assignment(-np.abs(np.nan_to_num(C)))
    psi_r = float(np.median([C[i, j] for i, j in zip(r, c)]))

    # PHI: did the cells get their right program?  ("cell-type recovery")
    agree = float((phi_hat[:, r].argmax(1) == tphi[:, c].argmax(1)).mean())
    ari   = float(adjusted_rand_score(tphi.argmax(1), phi_hat.argmax(1)))   # permutation-invariant

    # DS: can the model tell differential from null cassettes?
    d     = ph.max(0) - ph.min(0)                       # cross-program spread per cassette = effect size
    ok    = np.isfinite(d)
    auroc = float(roc_auc_score(T["is_diff"][ok], d[ok]))
    return dict(psi_r=psi_r, agree=agree, ari=ari, auroc=auroc,
                chance_agree=1.0 / Kt, effect=d, effect_ok=ok)

def crude_threshold_calibration(T, effect, ok, alpha=0.05, delta=0.05):
    """A CRUDE point-estimate threshold: call every cassette whose effect exceeds `delta`.

    *** THIS IS NOT THE MODEL'S FDR PROCEDURE. *** It uses no posterior and does no FDR control, so
    its "fdp" is NOT a statement about whether LeafletFA controls false positives -- it is just the
    error rate of a naive threshold. It is kept only as a cheap per-fit summary of how separable the
    differential and null cassettes are. The REAL FDR analysis is 6.3, which uses posterior samples
    and bayesian_fdr(). Do not quote these columns."""
    isd  = T["is_diff"][ok]
    eff  = effect[ok]
    called = eff > delta
    if called.sum() == 0:
        return dict(n_called=0, fdp=0.0, power=0.0)
    fdp   = float((~isd[called]).sum() / called.sum())      # realised FDP among the called
    power = float((isd[called]).sum() / max(isd.sum(), 1))
    return dict(n_called=int(called.sum()), fdp=fdp, power=power)

print("metrics defined: score_fit() -> psi_r, agree, ari, auroc ; crude_threshold_calibration() [NOT the FDR procedure -- see 6.3]")

# ============ notebook cell 18 ============
# ── the ONE fit everything else is compared against ──────────────────────────
from leafletfa import LeafletFA

REF = dict(K=5, likelihood="bb", lr=0.05, junc_prior=True, waypoints=False,
           batching="full", seed=0)
EPOCHS = 1200

def fit_once(sim, K, likelihood, lr, junc_prior, waypoints, batching, seed, epochs=EPOCHS):
    """One LeafletFA fit. `patience` is effectively OFF: early stopping has twice silently
    truncated these fits (METHODS.md 11.3), so on the simulation we always run the full budget."""
    conc = torch.tensor(float("inf")) if likelihood == "binom" else None
    common = dict(K=K, junc_specific_prior=junc_prior, input_conc_prior=conc,
                  waypoints_use=waypoints, print_epochs=10**6,
                  patience=10**6, min_delta=0, min_delta_rel=1e-6,      # early stopping OFF
                  ELBO_num_particles=5, num_samples=50, lr=lr, gamma=0.5,
                  log_wandb=False, loss_plot=False)
    t0 = time.time()
    if batching == "full":
        m = LeafletFA(adata=sim, num_epochs=epochs, **common)
        with contextlib.redirect_stdout(io.StringIO()):
            m.from_anndata(); m.train(num_initializations=1, seeds=[seed]); m.get_all_variables()
    else:
        # CHUNKED = the PRODUCTION scheme, mirrored exactly from run_training.py. Getting this wrong
        # makes the whole batching comparison meaningless, so spell out what production does:
        #
        #   * psi / pi / alpha_pi are carried chunk -> chunk (globals, warm-started)
        #   * PHI IS CARRIED PER CELL. After every chunk, each cell's phi is written back to
        #     all_cell_assignments[idx] (run_training.py:490), and the next time a chunk contains
        #     that cell its phi is warm-started from that store (run_training.py:378-380). phi is
        #     therefore CONTINUOUSLY REFINED every time a cell is seen, across all passes.
        #   * THE FINAL PHI **IS** THAT ACCUMULATED STORE -- `assign_post = all_cell_assignments`
        #     (run_training.py:663). There is NO final frozen-psi E-step over all cells. An earlier
        #     version of this function invented one, and dropped the phi carry-over entirely, so it
        #     was not testing the production scheme at all.
        #   * obs_scale = N/B rescales the LOCAL terms, without which q(psi) reflects B cells, not N.
        N, B, PASSES = sim.n_obs, 2000, 6
        rng = np.random.default_rng(seed)
        psi_w = pi_w = alpha_w = None
        all_phi = np.zeros((N, K), dtype=np.float32)          # <- the per-cell phi store
        phi_key = f"phi_init_{K}_waypoints"
        for _pass in range(PASSES):
            perm = rng.permutation(N)
            for i in range(0, N, B):
                idx = np.sort(perm[i:i + B])
                ba  = sim[idx].copy()
                # warm-start phi for cells this chunk has seen before (production behaviour)
                if all_phi[idx].sum() > 0:
                    ba.obsm[phi_key] = all_phi[idx]
                m = LeafletFA(adata=ba, num_epochs=max(60, epochs // 20), obs_scale=N / len(idx),
                              pi_init=None if pi_w is None else torch.tensor(pi_w),
                              alpha_pi_init=None if alpha_w is None else torch.tensor(alpha_w),
                              **common)
                with contextlib.redirect_stdout(io.StringIO()):
                    m.from_anndata()
                    m.train(num_initializations=1, seeds=[seed],
                            psi_init=None if psi_w is None else torch.tensor(psi_w),
                            phi_init=None if all_phi[idx].sum() == 0 else torch.tensor(all_phi[idx]))
                    m.get_all_variables()
                psi_w   = to_np(m.psi)
                pi_w    = to_np(m.pi)
                alpha_w = np.asarray(to_np(m.alpha_pi), dtype=np.float32).reshape(())
                all_phi[idx] = to_np(m.assign_post)           # <- write phi BACK, per cell
        # the final model IS the last chunk's globals + the ACCUMULATED per-cell phi
        m.assign_post = all_phi
        m.psi = psi_w
    return m, time.time() - t0

T = truth_of(sim)
m_ref, secs = fit_once(sim, **REF)
s = score_fit(T, to_np(m_ref.psi), to_np(m_ref.assign_post))
cal = crude_threshold_calibration(T, s["effect"], s["effect_ok"])

print(f"REFERENCE  K={REF['K']} {REF['likelihood']} lr={REF['lr']} "
      f"junc_prior={REF['junc_prior']} waypoints={REF['waypoints']} {REF['batching']}-batch  "
      f"({secs/60:.1f} min)\n")
print(f"  psi_r (program recovery)     {s['psi_r']:+.3f}     (chance 0)")
print(f"  agree (cells -> right prog)  {s['agree']:.1%}      (chance {s['chance_agree']:.0%})")
print(f"  ARI   (phi clusters cells)   {s['ari']:.3f}      (chance 0)")
print(f"  AUROC (differential vs null) {s['auroc']:.3f}      (chance 0.5)")
print(f"  [crude threshold, NOT the FDR procedure -- see 6.3] called={cal['n_called']}  realised FDP={cal['fdp']:.3f}  power={cal['power']:.3f}")
print(f"  psi_contrast                 {m_ref.health['psi_contrast']:.4f}  (~0.003 = collapse)")
print(f"  bb_conc                      {m_ref.bb_conc}")

# ============ notebook cell 20 ============
# ── THE SWEEP ────────────────────────────────────────────────────────────────
# Cached to CSV: delete SWEEP_CSV to force a re-run. ~39 fits, ~2 h on an L40S.
SWEEP_CSV = "sweep_results.csv"
SEEDS     = [0, 1, 2]

configs = []
def add(tag, **kw):
    cfg = {**REF, **kw}
    for sd in SEEDS:
        configs.append(dict(axis=tag, **{**cfg, "seed": sd}))

# Block A — lr x likelihood, FULLY CROSSED (they interact)
for lk in ["bb", "binom"]:
    for lr in [0.005, 0.01, 0.02, 0.05]:
        add("lr x likelihood", likelihood=lk, lr=lr)

# Block B — one axis at a time from the reference
for K in [10, 20]:            add("K", K=K)                       # K=5 is the reference
add("psi prior",  junc_prior=False)                               # True is the reference
add("init",       waypoints=True)                                 # random is the reference
add("batching",   batching="chunked")                             # full is the reference

# de-duplicate: the reference appears in Block A already
seen, uniq = set(), []
for c in configs:
    key = tuple(sorted((k, v) for k, v in c.items() if k != "axis"))
    if key not in seen:
        seen.add(key); uniq.append(c)
configs = uniq
print(f"{len(configs)} fits ({len(configs)//len(SEEDS)} configs x {len(SEEDS)} seeds)")

# RESUMABLE + INCREMENTAL. Each fit is appended to the CSV as soon as it finishes, so:
#   * progress is visible while the job runs (tail the CSV),
#   * a crash at fit 38 of 39 does not throw away the other 38,
#   * re-running skips whatever is already done.
# Delete SWEEP_CSV to force a clean re-run.
FIT_KEY = ["K", "likelihood", "lr", "junc_prior", "waypoints", "batching", "seed"]

done = pd.read_csv(SWEEP_CSV) if os.path.exists(SWEEP_CSV) else pd.DataFrame(columns=FIT_KEY)
done_keys = {tuple(r) for r in done[FIT_KEY].values} if len(done) else set()
todo = [c for c in configs
        if tuple(c[k] for k in FIT_KEY) not in done_keys]
print(f"{len(done)} fits already done, {len(todo)} to run")

if todo:
    rows = []
    for n, cfg in enumerate(todo, 1):
        cfg = dict(cfg)
        axis = cfg.pop("axis")
        try:
            m, secs = fit_once(sim, **cfg)
            s   = score_fit(T, to_np(m.psi), to_np(m.assign_post))
            cal = crude_threshold_calibration(T, s["effect"], s["effect_ok"])
            cc  = m.bb_conc
            rows.append(dict(axis=axis, **cfg,
                psi_r=s["psi_r"], agree=s["agree"], ari=s["ari"], auroc=s["auroc"],
                chance_agree=s["chance_agree"], crude_fdp=cal["fdp"], crude_power=cal["power"],
                n_called=cal["n_called"], psi_contrast=m.health["psi_contrast"],
                bb_conc=(np.nan if isinstance(cc, str) else float(np.asarray(cc))),
                elbo=float(m.best_elbo), healthy=bool(m.health["healthy"]), minutes=secs / 60))
            print(f"[{n:>2}/{len(todo)}] {axis:<16} K={cfg['K']:<3} {cfg['likelihood']:<5} "
                  f"lr={cfg['lr']:<6} jp={str(cfg['junc_prior']):<5} wp={str(cfg['waypoints']):<5} "
                  f"{cfg['batching']:<7} s{cfg['seed']} -> psi_r={s['psi_r']:+.3f} "
                  f"ari={s['ari']:.3f} auroc={s['auroc']:.3f}", flush=True)
            # APPEND this fit immediately -- do not wait for the whole sweep
            pd.DataFrame([rows[-1]]).to_csv(
                SWEEP_CSV, mode="a", header=not os.path.exists(SWEEP_CSV), index=False)
        except Exception as e:
            print(f"[{n:>2}/{len(todo)}] FAILED {axis} {cfg}: {type(e).__name__}: {e}", flush=True)

sweep = pd.read_csv(SWEEP_CSV)
print(f"\n{len(sweep)} fits in {SWEEP_CSV}")

display(sweep.round(3))
print('\nSWEEP COMPLETE')
