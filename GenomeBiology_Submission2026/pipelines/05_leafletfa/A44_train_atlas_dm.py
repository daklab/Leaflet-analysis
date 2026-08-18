#!/usr/bin/env python
"""A44: train DM + simplex on the full mouse atlas (132,706 cells x 79,768 junctions).

Why a new trainer rather than A1_train_atlas.py
-----------------------------------------------
A1 trains the RELEASED model (BetaBinomial + independent-Beta psi) and reads its grid from
parameter_combinations.json. It has no way to pass `event_id`, which the DM requires -- without
the ATSE index there is no way to group junctions into events, and the DM is nothing but that
grouping. Bolting a likelihood switch onto A1 would put the released model and an unvalidated one
behind the same entry point, and a released-model rerun could silently pick up the new arm. Two
scripts, two models, no way to run the wrong one by accident.

What it does
------------
Loads the full training matrix densely (42 GB per layer; `data_on_cpu` in LeafletFAMinibatch
auto-engages past 4 GB so the matrices stay in host RAM and only the minibatch crosses to the GPU),
fits DM + simplex, and saves psi/phi/loc/scale plus the loss curve.

Safety for an unattended overnight run
--------------------------------------
Every atlas fit attempted so far has hit a non-finite loss (simplex at steps 1161/1433/1663,
baseline at 2113/2249). `fit()` already carries a NaN-guard: it snapshots the lowest-EMA parameter
state on a fixed interval and, on a non-finite loss, restores that snapshot and stops. So a NaN
costs the steps since the last checkpoint, not the run -- which is what makes launching these
unattended defensible rather than reckless. `stopped_step` is recorded in result.json so a fit that
died early is visible in the summary rather than being read as a converged one.

Usage:
    A44_train_atlas_dm.py --K 20 --seed 0 --steps 12000 --batch 8192 --out <dir>
"""

import argparse, json, os, sys, time

import h5py
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import pyro   # noqa: E402

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from leafletfa.minibatch import LeafletFAMinibatch          # noqa: E402
from leafletfa.dirichlet_multinomial import build_atse_index  # noqa: E402

WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
H5 = f"{WD}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad"


def cat(name, which, h5):
    with h5py.File(h5, "r") as f:
        n = f[which][name]
        if isinstance(n, h5py.Group):
            c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
            return c[n["codes"][:]]
        return np.array([x.decode() if isinstance(x, bytes) else x for x in n[:]])


def read_all(layer, h5, n_rows, n_cols, chunk=4000):
    """Dense read in row blocks. 42 GB at atlas shape, so never materialise the CSR whole."""
    out = np.zeros((n_rows, n_cols), dtype=np.float32)
    with h5py.File(h5, "r") as f:
        g = f["layers"][layer]
        ip_all = g["indptr"]
        for a in range(0, n_rows, chunk):
            b = min(a + chunk, n_rows)
            ip = ip_all[a:b + 1]
            s, e = int(ip[0]), int(ip[-1])
            out[a:b] = sp.csr_matrix((g["data"][s:e], g["indices"][s:e], ip - ip[0]),
                                     shape=(b - a, n_cols)).toarray()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--h5", default=H5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", type=int, default=0, help="0 = all; >0 subsets for a smoke test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--psi-logit-prior-sd", type=float, default=1.0)
    ap.add_argument("--global-conc", action="store_true",
                    help="one concentration for all ATSEs (the old behaviour). Default is one per "
                         "ATSE: a single scalar has to absorb every event's overdispersion at once "
                         "and on the atlas it collapsed to 0.81-1.99 against the ~48 simulations "
                         "recover, which put alpha at the clip and drove the divergence.")
    ap.add_argument("--dm-alpha-floor", type=float, default=1e-2,
                    help="floor on alpha = conc * mu. The DM gradient goes as 1/a; the old 1e-6 sat "
                         "190x below where alpha actually lands and never fired.")
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--track-every", type=int, default=200,
                    help="record psi_contrast / phi_spread / pi / conc every N steps. The atlas "
                         "ELBO CANNOT answer convergence -- 100%% of its drop lands in the first "
                         "~50 steps and the tail slope flips sign between fits, so it is "
                         "noise-dominated. Whether the STRUCTURE has stopped moving is the real "
                         "criterion, and this is what measures it.")
    ap.add_argument("--max-nan-restarts", type=int, default=20)
    ap.add_argument("--phi-floor", type=float, default=0.0,
                    help="floor phi away from the simplex corners. 19 of 1.3M phi entries sit below "
                         "1e-6 on the atlas, and exactly 18 phi entries were non-finite at the "
                         "divergence -- the same cells. Their transform gradient scales as 1/phi.")
    ap.add_argument("--permute-atse-frac", type=float, default=0.0,
                    help="permute this fraction of ATSEs across cells before fitting, as negative "
                         "controls. Lets delta be measured on the atlas instead of imported from a "
                         "simulation whose depth and effect-size range differ from the real data.")
    ap.add_argument("--permute-seed", type=int, default=0)
    ap.add_argument("--holdout-ids", default=None,
                    help="file of row indices to hold out, one per line. Overrides --holdout-frac. "
                         "Use this to match an EXISTING split (e.g. test_30_70) so downsampled "
                         "evaluation data already on disk lines up with the held-out cells.")
    ap.add_argument("--holdout-frac", type=float, default=0.0,
                    help="fraction of cells held OUT of training and scored at the end. Everything "
                         "reported so far has been TRAINING likelihood, which cannot detect "
                         "overfitting -- and the old BetaBinomial has 79,768 unconstrained psi per "
                         "program while the simplex model does not, so that comparison could "
                         "flatter either one. This is how the concentration setting should be "
                         "chosen, rather than on a single showcase ATSE.")
    ap.add_argument("--dir-conc", type=float, default=None,
                    help="pin the Dirichlet concentration on phi. Pass -1 for 'auto' = K, which "
                         "puts the average delta*pi_k at 1.0. Left unset, delta is learned and "
                         "lands at 0.19-0.37, where the prior drives cells into simplex corners.")
    ap.add_argument("--likelihood-weight", default=None,
                    choices=[None, "none", "m_over_m_minus_1"],
                    help="tempered pseudo-likelihood restoring the BetaBinomial's accidental "
                         "m/(m-1) over-count deliberately (2.00x at m=2, 43.6%% of atlas ATSEs). "
                         "Improves phi; makes credible intervals too narrow BY CONSTRUCTION, so "
                         "the posterior DS test must use the permutation null.")
    ap.add_argument("--conc-min", type=float, default=1e-3,
                    help="lower bound on the per-ATSE concentration. It collapses 0.975 -> 0.518 by "
                         "step 1,600 on the atlas, before structure forms, and 67%% of events end "
                         "below 1 -- where the DM barely rewards loading on the right program. The "
                         "released BetaBinomial fits sat at ~1.97 and separated cell types better.")
    ap.add_argument("--psi-clip", type=float, default=1e-4,
                    help="floor/ceiling on psi inside the softmax. psi is PINNED at this value from "
                         "step 125 on the atlas, so the logits keep growing against a hard boundary "
                         "where the gradient is zero.")
    ap.add_argument("--checkpoint-every", type=int, default=2000,
                    help="write a usable fit to checkpoint.npz every N steps. A wall-clock timeout "
                         "otherwise destroys the whole run: one fit was killed at 8h at step 10,000 "
                         "with zero restarts, purely for landing on a 5x slower node. 0 disables.")
    ap.add_argument("--clip-norm", type=float, default=10.0,
                    help="ClippedAdam gradient-norm clip. 10.0 over ~2.5M parameters is loose.")
    ap.add_argument("--wandb-project", default="LeafletFA-mouse-atlas")
    args = ap.parse_args()

    run = f"dm_simplex_K{args.K}_seed{args.seed}_b{args.batch}"
    out_dir = os.path.join(args.out, run)
    os.makedirs(out_dir, exist_ok=True)

    ev = cat("event_id", "var", args.h5)
    J = len(ev)
    with h5py.File(args.h5, "r") as f:
        N_all = int(f["layers"]["cell_by_junction_matrix"].attrs["shape"][0])
    N = min(args.cells, N_all) if args.cells else N_all
    _, _, sizes = build_atse_index(ev)
    n_atse = len(sizes)
    print(f"[1] {run}: {N:,} cells x {J:,} junctions, {n_atse:,} ATSEs "
          f"(m=2 for {int((sizes == 2).sum()):,})", flush=True)

    wb = None
    try:
        import wandb
        wb = wandb.init(project=os.environ.get("LFA_WANDB_PROJECT", args.wandb_project),
                        group="dm_simplex_atlas", name=run, dir=out_dir, reinit=True,
                        config=dict(K=args.K, seed=args.seed, steps=args.steps, batch=args.batch,
                                    lr=args.lr, gamma=args.gamma, likelihood="dirichletmultinomial",
                                    psi_parameterization="simplex", n_cells=N, n_junctions=J,
                                    n_atse=n_atse, psi_logit_prior_sd=args.psi_logit_prior_sd),
                        tags=[f"K{args.K}", f"seed{args.seed}", "dirichletmultinomial", "simplex"])
        print("[1b] wandb on", flush=True)
    except Exception as e:
        print(f"[1b] wandb off: {e}", flush=True)

    t0 = time.time()
    junc = read_all("cell_by_junction_matrix", args.h5, N, J)
    clus = read_all("cell_by_cluster_matrix", args.h5, N, J)
    ho_idx = np.zeros(0, dtype=int)
    if args.holdout_ids:
        # Hold out a SPECIFIC set of cells rather than a random fraction. This exists because the
        # imputation benchmark's downsampled layers (RESAMPLED_*_PERCENT_test_30_70_*.h5mu) were
        # generated for one particular 30% split. A random holdout of the same SIZE is not the same
        # SET, so the evaluation would have no downsampled counts for most held-out cells -- the
        # model would be clean but unscoreable. Passing the split's own row indices keeps the
        # retrain and the existing evaluation data aligned.
        ho_idx = np.sort(np.loadtxt(args.holdout_ids, dtype=int).ravel())
        if ho_idx.min() < 0 or ho_idx.max() >= N:
            raise SystemExit(f"--holdout-ids out of range for N={N}: "
                             f"[{ho_idx.min()}, {ho_idx.max()}]")
        keep = np.setdiff1d(np.arange(N), ho_idx)
        junc_ho, clus_ho = junc[ho_idx].copy(), clus[ho_idx].copy()
        junc, clus = junc[keep], clus[keep]
        print(f"[2b] holding out {len(ho_idx):,} SPECIFIED cells ({100*len(ho_idx)/N:.0f}%), "
              f"training on {len(keep):,}. From {os.path.basename(args.holdout_ids)}.", flush=True)
    elif args.holdout_frac > 0:
        _r = np.random.default_rng(12345)          # fixed, so every fit holds out the SAME cells
        ho_idx = np.sort(_r.choice(N, int(round(args.holdout_frac * N)), replace=False))
        keep = np.setdiff1d(np.arange(N), ho_idx)
        junc_ho, clus_ho = junc[ho_idx].copy(), clus[ho_idx].copy()
        junc, clus = junc[keep], clus[keep]
        print(f"[2b] holding out {len(ho_idx):,} cells ({100*args.holdout_frac:.0f}%), "
              f"training on {len(keep):,}. Seed is FIXED at 12345 so every fit is scored on the "
              f"same cells and the numbers are comparable across runs.", flush=True)
    permuted_atses = np.zeros(0, np.int64)
    if args.permute_atse_frac > 0:
        # Negative-control ATSEs, so delta can be MEASURED on the atlas rather than imported from a
        # simulation. These events are null by construction; the untouched majority still pin phi,
        # so dphi keeps the magnitude a real cell-type contrast has. Their TV distribution is the
        # model's resolution floor at the right scale, with no truth labels required.
        from _permute_null import permute_atse_counts
        _ai, _, _ = build_atse_index(ev)
        junc, clus, permuted_atses = permute_atse_counts(
            junc, clus, np.asarray(_ai), n_atse, args.permute_atse_frac,
            seed=args.permute_seed)
        print(f"[2c] permuted {len(permuted_atses):,} of {n_atse:,} ATSEs "
              f"({100*args.permute_atse_frac:.0f}%) as negative controls", flush=True)
    print(f"[2] loaded in {time.time()-t0:.0f}s ({junc.nbytes/1e9:.1f} GB each), "
          f"coverage {100*(clus>0).mean():.2f}%, observed rate {junc.sum()/max(clus.sum(),1):.4f}",
          flush=True)

    N_train = junc.shape[0]
    pyro.clear_param_store()
    t0 = time.time()
    m = LeafletFAMinibatch(junc, clus, K=args.K, device=args.device,
                           likelihood="dirichletmultinomial", psi_parameterization="simplex",
                           psi_logit_prior_sd=args.psi_logit_prior_sd, event_id=ev,
                           conc_per_atse=not args.global_conc,
                           dm_alpha_floor=args.dm_alpha_floor, phi_floor=args.phi_floor,
                           clip=args.psi_clip, conc_min=args.conc_min,
                           likelihood_weight=args.likelihood_weight)
    # Pin the Dirichlet concentration on phi. --dir-conc auto resolves to K, which puts the AVERAGE
    # effective parameter delta*pi_k at 1.0 (average pi_k = 1/K), the point where the prior stops
    # being sparsity-inducing. Measured on simulation at the atlas's regime, over 3 seeds per arm:
    #
    #   arm                     phi_r_true  phi_ari   mag   contrast_cv
    #   control (conc 50, full depth)  0.978    0.998  0.973    0.012
    #   atlas regime, delta learned    0.247    0.078  0.539    0.144   <- the failure
    #   + delta = K                    0.577    0.189  0.642    0.106
    #   + m/(m-1) weight               0.558    0.336  0.733    0.064
    #   + BOTH                         0.668    0.411  0.765    0.046   <- shipped
    #
    # delta = 5K was WORSE than doing nothing (phi_r_true 0.078, mag 0.338), so this is a floor to
    # sit on, not a knob to turn up. Note atlas pi is not uniform (max pi_k ~ 0.11 at K=20), so
    # delta = K puts the average program at 1.0 while the smallest still sits below it.
    if args.dir_conc is not None:
        dc = float(args.K) if args.dir_conc < 0 else float(args.dir_conc)
        m.freeze_globals(dir_conc=dc)
        print(f"[1c] dir_conc PINNED at {dc:g} (delta/K = {dc/args.K:.2f}); the learned value put "
              f"every delta*pi_k at 0.19-0.37, i.e. corner-seeking", flush=True)
    ck_path = os.path.join(out_dir, "checkpoint.npz")

    def _checkpoint(step, psi_now, phi_now, tr):
        """Persist a usable fit mid-run. Cheap relative to a 2.5-hour job and it makes a timeout
        cost minutes rather than everything."""
        if not args.checkpoint_every or step % args.checkpoint_every or step == 0:
            return
        # step-stamped, NOT overwritten. A single checkpoint.npz keeps only the last step, which
        # threw away the free answer to "does phi keep concentrating after psi settles" -- the
        # mid-training states had already been overwritten by the time the question came up.
        ck_path_s = os.path.join(out_dir, f"checkpoint_step{step:06d}.npz")
        np.savez_compressed(ck_path_s, psi_mean=psi_now.astype(np.float32),
                            phi=phi_now.astype(np.float32), step=step,
                            psi_contrast=tr["psi_contrast"], n_live=tr["n_live"],
                            conc_median=tr["conc_median"], K=args.K, seed=args.seed)
        np.savez_compressed(ck_path, psi_mean=psi_now.astype(np.float32),
                            phi=phi_now.astype(np.float32), step=step,
                            psi_contrast=tr["psi_contrast"], n_live=tr["n_live"],
                            conc_median=tr["conc_median"], K=args.K, seed=args.seed)
        print(f"  [checkpoint] step {step:,} -> checkpoint.npz "
              f"(psi_contrast {tr['psi_contrast']:.5f}, {tr['n_live']} live)", flush=True)

    L = m.fit(num_steps=args.steps, subsample_size=min(args.batch, N_train), lr=args.lr,
              gamma=args.gamma, seed=args.seed, verbose=True, log_every=100,
              early_stop=not args.no_early_stop, patience=12, min_delta_rel=1e-5,
              track_every=args.track_every, max_nan_restarts=args.max_nan_restarts,
              clip_norm=args.clip_norm, on_track=_checkpoint)
    train_s = time.time() - t0
    finished = int(getattr(m, "stopped_step", args.steps))
    nan_hit = finished < args.steps and not getattr(m, "stopped_early", False)
    print(f"[3] {finished:,}/{args.steps:,} steps in {train_s/60:.1f} min "
          f"({1000*train_s/max(finished,1):.0f} ms/step)"
          f"{'  -- NaN-GUARD FIRED' if nan_hit else ''}", flush=True)

    psi, phi = m.psi_mean(), m.phi_mean()
    loc, scale = m.psi_loc_scale()
    def to_np(x):
        """psi_loc_scale returns tensors under some paths and arrays under others."""
        if x is None:
            return np.zeros(0, np.float32)
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return np.asarray(x, np.float32)
    try:
        # the EFFECTIVE concentration -- clamped exactly as the model clamps it. Reading the raw
        # guide median instead reported values below the floor the fit trained under (min 1.536
        # under conc_min=2.0), which would mislead anything using the saved vector downstream.
        conc_vec = m.concentration()
        conc = float(np.median(conc_vec))          # summary; the vector is saved in full
    except Exception:
        conc_vec, conc = np.zeros(0, np.float32), float("nan")

    # within-ATSE sums: under the simplex these are 1 by construction, so a deviation means the
    # constraint is not actually active and every downstream TV is on an arbitrary scale.
    psi, phi = to_np(psi), to_np(phi)
    uq, inv = np.unique(ev, return_inverse=True)
    S = np.zeros((args.K, len(uq)), np.float32)
    np.add.at(S.T, inv, psi.T)
    frac_valid = float(((S >= 0.9) & (S <= 1.1)).mean())
    cdesc = (f"conc per ATSE: median {conc:.2f}, IQR "
             f"{np.percentile(conc_vec,25):.2f}-{np.percentile(conc_vec,75):.2f}, "
             f"min {conc_vec.min():.2f}" if conc_vec.size > 1 else f"conc {conc:.3f} (global)")
    print(f"[4] psi within-ATSE sum: median {np.median(S):.6f}, max dev "
          f"{np.abs(S-1).max():.2e}, valid {100*frac_valid:.2f}%  |  {cdesc}", flush=True)

    # `res` is built further down, so these go into plain locals and are folded in there.
    # Writing into `res` here raised 'referenced before assignment' and the guard on the
    # downstream selector correctly refused to pick a setting from the missing numbers.
    ho_model = ho_null = None
    # HELD-OUT SCORING. phi is a local parameter, so a cell the model never saw has no loading.
    # Rather than refit it (which would leak), score under the PRIOR-mean phi -- i.e. ask how well
    # the learned psi alone explains unseen cells, against the same null and cell-type references.
    # This measures the programs, not the per-cell encoding, which is the right target: psi is what
    # transfers, phi is per-cell by construction.
    if len(ho_idx):
        try:
            from leafletfa.dirichlet_multinomial import dm_log_prob as _dlp
            yh = torch.from_numpy(junc_ho)
            uq_h, inv_h = np.unique(ev, return_inverse=True)
            ai_h = torch.as_tensor(inv_h, dtype=torch.long)
            phibar = torch.from_numpy(phi.mean(0).astype(np.float32))
            mu_m = torch.from_numpy((phi.mean(0) @ psi).astype(np.float32)).unsqueeze(0).expand(
                yh.shape[0], -1).contiguous()
            segm = torch.zeros(yh.shape[0], n_atse).index_add(1, ai_h, mu_m)
            mu_m = (mu_m / segm[:, ai_h].clamp_min(1e-9)).clamp_min(1e-6)
            # The null must be built from TRAINING counts. Building it from the held-out counts
            # (yh.sum(0)) fits the baseline on the test set, so the model -- which only ever saw
            # training data -- is compared against an oracle and every gain comes out negative.
            # The ranking survived that error because all arms shared the handicap, but the
            # absolute numbers were meaningless.
            ytr = torch.from_numpy(junc)
            tot = ytr.sum(0); segt = torch.zeros(n_atse).index_add(0, ai_h, tot)
            mu_n = (tot / segt[ai_h].clamp_min(1e-9)).clamp_min(1e-6).unsqueeze(0).expand_as(mu_m).contiguous()
            C = torch.tensor(2.0)
            ho_model = float(_dlp(yh, mu_m, C, ai_h, n_atse).sum())
            ho_null = float(_dlp(yh, mu_n, C, ai_h, n_atse).sum())
            
            print(f"[4d] held-out ({len(ho_idx):,} cells): model {ho_model:,.0f} "
                  f"vs null {ho_null:,.0f}  gain {ho_model-ho_null:,.0f}", flush=True)
        except Exception as e:
            print(f"[4d] held-out scoring skipped: {e}", flush=True)

    tr = getattr(m, "trace", []) or []
    if tr:
        print(f"[4b] structure trajectory ({len(tr)} checkpoints) -- the convergence criterion the "
              f"ELBO cannot provide:", flush=True)
        print(f"     {'step':>7} {'psi_contrast':>13} {'phi_spread':>11} {'n_live':>7} {'conc med':>9}",
              flush=True)
        for t in tr[:: max(1, len(tr) // 8)] + ([tr[-1]] if len(tr) > 1 else []):
            print(f"     {t['step']:>7,} {t['psi_contrast']:>13.5f} {t['phi_spread']:>11.5f} "
                  f"{t['n_live']:>7} {t['conc_median']:>9.3f}", flush=True)
        _pc = np.array([t["psi_contrast"] for t in tr], float)
        if len(_pc) > 2:
            _rel = abs(_pc[-1] - _pc[-2]) / max(abs(_pc[-2]), 1e-12)
            print(f"     psi_contrast last-interval change {100*_rel:.3f}%  "
                  f"({'settled' if _rel < 0.01 else 'STILL MOVING -- more steps would help'})",
                  flush=True)
        if wb is not None:
            import wandb
            for t in tr:
                wandb.log({k: v for k, v in t.items() if k != "pi"}, step=t["step"])

    # The POSTERIOR over phi, not just its mean. The model-based DS test (A51) computes
    # Delta_j = dphi . psi_:,j and needs uncertainty in BOTH factors; saving only the point `phi`
    # forced it to treat the group-mean difference as known exactly, propagating psi's uncertainty
    # alone. These are AutoNormal's parameters in UNCONSTRAINED space -- phi is Dirichlet, so the
    # transform is stick-breaking and the trailing dim is K-1, not K. ~10 MB at atlas scale.
    phi_loc = phi_scale = np.zeros((0, 0), np.float32)
    try:
        _st = pyro.get_param_store()
        if "AutoNormal.locs.phi" in _st and "AutoNormal.scales.phi" in _st:
            phi_loc = to_np(_st["AutoNormal.locs.phi"].detach())
            phi_scale = to_np(_st["AutoNormal.scales.phi"].detach())
            print(f"[4e] saved q(phi): loc {phi_loc.shape}, scale {phi_scale.shape}", flush=True)
        else:
            print("[4e] WARNING: no q(phi) in the param store -- A51 cannot propagate phi "
                  "uncertainty from this fit", flush=True)
    except Exception as e:
        print(f"[4e] q(phi) save failed: {e}", flush=True)
    # and the whole store, so a missing derived quantity never costs another 3-hour fit
    try:
        pyro.get_param_store().save(os.path.join(out_dir, "param_store.pt"))
    except Exception as e:
        print(f"[4e] param store save failed: {e}", flush=True)

    np.savez_compressed(os.path.join(out_dir, "fit.npz"),
                        phi_loc=phi_loc, phi_scale=phi_scale,
                        trace_step=np.array([t["step"] for t in tr], np.int32),
                        trace_psi_contrast=np.array([t["psi_contrast"] for t in tr], np.float32),
                        trace_phi_spread=np.array([t["phi_spread"] for t in tr], np.float32),
                        trace_n_live=np.array([t["n_live"] for t in tr], np.int32),
                        trace_conc_median=np.array([t["conc_median"] for t in tr], np.float32),
                        trace_pi=(np.stack([t["pi"] for t in tr]).astype(np.float32)
                                  if tr and tr[0]["pi"].size else np.zeros((0, 0), np.float32)),
                        psi_mean=psi, phi=phi, psi_loc=to_np(loc), psi_scale=to_np(scale),
                        permuted_atses=permuted_atses.astype(np.int64),
                        event_id=np.asarray(uq, dtype=object), inv_ev=inv.astype(np.int32),
                        # WHICH atlas cells phi's rows correspond to. With --holdout-frac, phi
                        # covers only the training cells, so anything indexing phi by atlas cell
                        # index is silently wrong -- or loudly wrong, which is what happened:
                        # "boolean index ... dimension is 119435 but corresponding boolean
                        # dimension is 132706". Downstream must map through this.
                        train_idx=(np.setdiff1d(np.arange(N), ho_idx).astype(np.int32)
                                   if len(ho_idx) else np.arange(N, dtype=np.int32)),
                        holdout_idx=ho_idx.astype(np.int32),
                        losses=np.asarray(L, np.float64), bb_conc=conc, bb_conc_vec=conc_vec,
                        frac_valid=frac_valid,
                        K=args.K, seed=args.seed, steps_run=finished)
    res = dict(run=run, K=args.K, seed=args.seed, steps_requested=args.steps,
               steps_run=finished, nan_guard_fired=bool(nan_hit),
               early_stopped=bool(getattr(m, "stopped_early", False)),
               train_minutes=round(train_s / 60, 1), ms_per_step=round(1000*train_s/max(finished,1)),
               final_loss=float(L[-1]) if L else None, bb_conc=conc, frac_valid=frac_valid,
               n_cells=N, n_junctions=J, n_atse=int(n_atse), batch=args.batch, lr=args.lr,
               heldout_logL_model=ho_model, heldout_logL_null=ho_null,
               heldout_n=(int(len(ho_idx)) if len(ho_idx) else 0),
               heldout_gain=((ho_model - ho_null) if (ho_model is not None and ho_null is not None)
                             else None),
               n_nan_restarts=int(getattr(m, "n_nan_restarts", 0)),
               nan_steps=list(getattr(m, "nan_steps", [])),
               conc_per_atse=(not args.global_conc), dm_alpha_floor=args.dm_alpha_floor,
               dir_conc_pinned=(None if args.dir_conc is None
                                else (args.K if args.dir_conc < 0 else args.dir_conc)),
               likelihood_weight=args.likelihood_weight,
               psi_contrast=(float(tr[-1]["psi_contrast"]) if tr else None),
               phi_spread=(float(tr[-1]["phi_spread"]) if tr else None),
               programs_live=(int(tr[-1]["n_live"]) if tr else None))
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(res, f, indent=2)

    # PHI UMAP coloured by held-out cell type, ported from A1_train_atlas.py. Purely diagnostic --
    # does the splicing latent space carry cell-type structure -- and wrapped so a failure can never
    # lose a finished fit. It belongs here rather than in a post-hoc script: rebuilding UMAPs by hand
    # after the fact is how the phi concentration regression went unnoticed for a full day.
    umap_png = None
    try:
        import scanpy as sc, anndata as ad
        from sklearn.metrics import silhouette_score
        sub = (np.sort(np.random.default_rng(0).choice(phi.shape[0], 20000, replace=False))
               if phi.shape[0] > 20000 else np.arange(phi.shape[0]))
        labels = None
        with h5py.File(args.h5, "r") as _f:
            _key = "tissue_cell_type" if "tissue_cell_type" in _f["obs"] else "broad_cell_type"
            if _key in _f["obs"]:
                _n = _f["obs"][_key]
                _c = np.array([x.decode() if isinstance(x, bytes) else x
                               for x in _n["categories"][:]])
                _lab_all = _c[_n["codes"][:]]
                # phi rows are TRAINING cells only under --holdout-ids/--holdout-frac: map through
                # train_idx, or the silhouette is computed on misaligned labels (was -0.3 on every
                # held-out fit for this reason; the atlas fit, with all cells, was unaffected)
                _tr = np.setdiff1d(np.arange(len(_lab_all)), ho_idx) if ho_idx is not None and len(ho_idx) else np.arange(len(_lab_all))
                labels = _lab_all[_tr][sub]
        a = ad.AnnData(phi[sub].astype(np.float32))
        a.obsm["X_phi"] = a.X.copy()
        sc.pp.neighbors(a, use_rep="X_phi", n_neighbors=30, random_state=42)
        sc.tl.umap(a, random_state=42)
        um = a.obsm["X_umap"]
        figu, axu = plt.subplots(figsize=(7, 7))
        sil = np.nan
        if labels is not None:
            uniq = sorted(set(labels))
            cmap = plt.colormaps.get_cmap("tab20" if len(uniq) > 10 else "tab10")
            for i, ctl in enumerate(uniq):
                mk = labels == ctl
                axu.scatter(um[mk, 0], um[mk, 1], s=3, alpha=0.4,
                            color=cmap(i / max(len(uniq) - 1, 1)), label=str(ctl))
            axu.legend(markerscale=3, fontsize=5, ncol=2, loc="best")
            # silhouette on the FULL phi, not the embedding -- scoring the embedding would measure
            # UMAP as much as the model.
            sil = float(silhouette_score(phi[sub], np.searchsorted(np.array(uniq), labels),
                                         sample_size=5000, random_state=0))
        else:
            axu.scatter(um[:, 0], um[:, 1], s=3, alpha=0.4)
        top3 = float(np.sort(phi.mean(0))[::-1][:3].sum())
        axu.set_xlabel("UMAP 1"); axu.set_ylabel("UMAP 2")
        axu.set_title(f"phi UMAP | K={args.K} seed={args.seed}, {len(sub):,} cells\n"
                      f"silhouette(phi) {sil:.3f}   top-3 mass {top3:.2f}")
        figu.tight_layout()
        umap_png = os.path.join(out_dir, "phi_umap.png")
        figu.savefig(umap_png, dpi=110); plt.close(figu)
        res["silhouette_celltype"] = sil
        res["top3_mass"] = top3
        print(f"[4c] phi UMAP -> {umap_png}  silhouette(phi)={sil:.4f}  top-3 mass={top3:.3f}",
              flush=True)
    except Exception as e:
        print(f"[4c] phi UMAP skipped: {e}", flush=True)

    # result.json is written above, BEFORE the UMAP block runs, so re-dump it here or the
    # silhouette and top-3 mass never reach the file.
    with open(os.path.join(out_dir, "result.json"), "w") as _f:
        json.dump(res, _f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(L, lw=0.8)
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("-ELBO")
    ax.set_title(f"{run}  ({finished:,} steps{'  NaN-guard' if nan_hit else ''})", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=140)

    if wb is not None:
        import wandb
        for i, l in enumerate(L):
            if i % 25 == 0 and np.isfinite(l):
                wandb.log({"neg_elbo": float(l)}, step=i)
        wandb.log({k: v for k, v in res.items() if isinstance(v, (int, float, bool))})
        wandb.log({"loss_curve": wandb.Image(os.path.join(out_dir, "loss_curve.png"))})
        if umap_png:
            wandb.log({"phi_umap": wandb.Image(umap_png)})
        wandb.finish()
    print(f"[5] wrote {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
