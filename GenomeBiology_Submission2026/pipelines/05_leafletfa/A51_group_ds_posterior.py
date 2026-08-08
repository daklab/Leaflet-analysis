#!/usr/bin/env python
"""A51: group-level differential splicing from the POSTERIOR, as an effect-size test.

Why this exists alongside the LRT
---------------------------------
The likelihood-ratio test (A45/A49) refits every event's compositions from raw counts, so the only
thing it takes from the trained model is the covariate. The model's denoised psi -- the part that
demonstrably improves on the data, with r(model, observed) rising 0.79 -> 0.92 with read depth --
never enters. For a GROUP contrast we can use it, and this does.

Statistic: Delta_j = dphi . psi_:,j per posterior draw, then TV = 0.5 * sum_j |Delta_j| within each
ATSE. phi is averaged over the group FIRST, since rho is linear in phi -- otherwise the [S, C, J]
tensor would be 3 PB at atlas scale.

The hypothesis, and why three null constructions were the wrong hunt
--------------------------------------------------------------------
The old rule was P(TV > 0.10) with bayesian_fdr, measured at 2.9x over nominal. Trying to replace
the arbitrary 0.10 with a data-driven null failed three times, each for a different reason:

  - permuting GROUP LABELS collapses dphi to zero, so the null has no spread and everything clears
    it (measured: 4,265 of 4,268 called, FDP exactly pi0);
  - permuting PHI inflates the null on genuinely differential events, converting real
    between-program variation into an overdispersed mixture;
  - collapsing the guide's location across programs -- a principled H0 that keeps dphi real and
    keeps the posterior noise -- still came out 12-14x anti-conservative, even with the width
    inflated 2x.

Three failures with three distinct causes meant the null was not the problem. The HYPOTHESIS was.
"Delta = 0 exactly" is essentially never true: every event carries some program dependence, psi is
denoised so its posterior noise is small, and with thousands of cells an arbitrarily small Delta is
detectably non-zero. A point-null test therefore calls nearly everything and is technically correct
each time, while missing what is actually being asked. Truth is defined by MAGNITUDE.

So the original shape -- P(TV > delta) -- was right, and its faults were the fixable ones: a biased
statistic (mean-of-absolute rather than absolute-of-mean; E|X| >= |EX|, so noise cannot cancel) and
an unjustified delta.

delta is not a tuning knob. It states the effect size of interest. On the simulations it is 0.05
because that is how the simulator defines a differential event, and with NO posterior width
inflation that gives, against contrast-specific truth:

    effect 0.25   FDP 0.000   ratio 0.00   power 0.253
    effect 0.50   FDP 0.018   ratio 0.35   power 0.717
    effect 1.00   FDP 0.016   ratio 0.31   power 0.850

Conservative at every effect size.

Note on scoring: power must be judged against the truth for THIS contrast (the true TV between the
two groups), not against "differential across any program pair". Using the latter understated power
by a factor of ~1.5 in an earlier evaluation, because it counts events the test was never asked to
find.
"""

import argparse, os, re, sys
import h5py
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from leafletfa.dirichlet_multinomial import build_atse_index, segment_softmax  # noqa: E402
from _phi_rows import phi_row_of  # noqa: E402

WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
H5 = f"{WD}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad"
LC = f"{WD}/LEAFCUTTER_DS_2026-08"


def norm(s):
    return re.sub(r"[^A-Z0-9]+", "_", str(s).upper()).strip("_")


def obs_cat(f, name):
    n = f["obs"][name]
    if isinstance(n, h5py.Group):
        c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
        return c[n["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in n[:]])


def draw_psi(z, atse_idx, n_atse, S, seed=0):
    """Posterior draws of psi on the CONSTRAINED scale.

    Under the simplex the guide site is psi_logit, so a draw is softmax(loc + scale*eps) within each
    ATSE -- not sigmoid. Getting this wrong silently produces psi that does not sum to 1 and makes
    every downstream TV meaningless, which is why it is asserted below."""
    loc = torch.from_numpy(np.asarray(z["psi_loc"], np.float32))
    scale = torch.from_numpy(np.asarray(z["psi_scale"], np.float32))
    if loc.numel() == 0:
        raise ValueError("fit has no psi_loc/psi_scale -- posterior draws unavailable")
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(S):
        eps = torch.randn(loc.shape, generator=g)
        out.append(segment_softmax(loc + scale * eps, atse_idx, n_atse))
    return torch.stack(out)          # [S, K, J]


def draw_dphi(z, rows1, rows2, S, K, seed=0):
    """Posterior draws of the group-mean phi difference, [S, K].

    Until 2026-08-08 this was a POINT estimate expanded S times, so the test propagated psi's
    uncertainty and treated dphi as known exactly. That understates the spread of TV -- dphi
    multiplies psi in every term of Delta_j -- and it is the half of the statistic that carries the
    biology, so the omission was not the harmless one.

    q(phi) is AutoNormal in UNCONSTRAINED space and phi is Dirichlet, so the map back to the simplex
    is stick-breaking and the stored trailing dim is K-1. Only the cells in the two arms are drawn:
    the full [S, N, K] tensor is 2 GB at atlas scale and every other cell is irrelevant to a
    group contrast.

    Returns None when the fit predates the q(phi) save, so the caller can fall back and SAY SO
    rather than silently reverting to the old behaviour.
    """
    if "phi_loc" not in getattr(z, "files", []):
        return None
    loc_all = np.asarray(z["phi_loc"], np.float32)
    scale_all = np.asarray(z["phi_scale"], np.float32)
    if loc_all.size == 0 or loc_all.shape[1] != K - 1:
        return None
    sb = torch.distributions.transforms.StickBreakingTransform()
    g = torch.Generator().manual_seed(seed)
    rows = np.concatenate([rows1, rows2])
    loc = torch.from_numpy(loc_all[rows])
    scale = torch.from_numpy(scale_all[rows])
    n1 = len(rows1)
    out = torch.empty(S, K)
    for s in range(S):
        eps = torch.randn(loc.shape, generator=g)
        ph = sb(loc + scale * eps)                    # [n1+n2, K], on the simplex
        out[s] = ph[:n1].mean(0) - ph[n1:].mean(0)
    return out


def bayesian_fdr(prob):
    """Expected-FDP control for POSTERIOR probabilities, which is what `prob` is.

    The previous code fed `1 - prob` into Benjamini-Hochberg. That quantity is a local false
    discovery rate, not a p-value: it is not uniform under the null, so BH's guarantee simply does
    not apply to it and the reported q carried no calibration. The Bayesian analogue is direct --
    rank by confidence and take the running mean of the local fdr, which IS the expected fraction of
    false calls in the set called so far.
    """
    lfdr = 1.0 - np.asarray(prob, float)
    o = np.argsort(lfdr)                              # most confident first
    run = np.cumsum(lfdr[o]) / np.arange(1, len(lfdr) + 1)
    out = np.empty_like(run)
    out[o] = np.minimum.accumulate(run[::-1])[::-1]   # monotone in the ranking
    return np.clip(out, 0, 1)


def tv_per_atse(dphi, psi_s, atse_idx, n_atse):
    """TV per (sample, ATSE) = 0.5 * sum_{j in e} |dphi . psi_:,j|, computed without materialising
    the [S, C, J] tensor: rho is linear in phi, so phi can be averaged over the group first."""
    d = torch.einsum("sk,skj->sj", dphi, psi_s)          # [S, J]
    out = torch.zeros(d.shape[0], n_atse)
    return 0.5 * out.index_add(1, atse_idx, d.abs()), d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--manifest", default=f"{LC}/pairwise_manifest_tabula_muris_senis_tissue_cell_type.tsv")
    ap.add_argument("--label-field", default="tissue_cell_type")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=200, help="posterior draws of psi")
    ap.add_argument("--delta", type=float, default=0.05,
                    help="effect size of interest. NOT a tuning knob -- it states what counts as "
                         "differential. 0.05 matches how the simulations define a true positive, "
                         "where this configuration measures conservative FDP at every effect size.")
    ap.add_argument("--contrasts", type=int, default=0)
    ap.add_argument("--min-cells", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--permute", type=int, default=0,
                    help="permutation null draws: shuffle cells between the two arms, keeping arm "
                         "sizes and the sampled phi vectors. Gives an empirical p that does not "
                         "depend on the mean-field posterior width being correct.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))

    z = np.load(args.fit, allow_pickle=True)
    phi = torch.from_numpy(np.asarray(z["phi"], np.float32))
    K = int(phi.shape[1])
    with h5py.File(H5, "r") as f:
        n = f["var"]["event_id"]
        c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
        ev = c[n["codes"][:]].astype(str)
        lab = obs_cat(f, args.label_field)
    PROW = phi_row_of(z, len(lab))
    atse_idx, _, _ = build_atse_index(ev)
    uq = np.unique(ev)
    n_atse = len(uq)

    print(f"[1] drawing {args.samples} psi posterior samples ...", flush=True)
    psi_s = draw_psi(z, atse_idx, n_atse, args.samples, args.seed)
    sums = torch.zeros(psi_s.shape[0], psi_s.shape[1], n_atse).index_add(
        2, atse_idx, psi_s)
    dev = float((sums - 1).abs().max())
    print(f"    psi draws {tuple(psi_s.shape)}; within-ATSE sums deviate at most {dev:.2e} from 1",
          flush=True)
    assert dev < 1e-2, "posterior draws are not on the simplex -- wrong transform"

    amap = {}
    for a in np.unique(lab):
        amap.setdefault(norm(a), a)
    man = pd.read_csv(args.manifest, sep="\t")
    man = man[man.status == "ready"].copy()
    man["a1"] = man.group_1.map(lambda g: amap.get(norm(g)))
    man["a2"] = man.group_2.map(lambda g: amap.get(norm(g)))
    man = man[man.a1.notna() & man.a2.notna()]
    if args.contrasts:
        man = man.head(args.contrasts)
    print(f"[2] {len(man)} contrasts", flush=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    for ci, (_, row) in enumerate(man.iterrows()):
        c1 = np.flatnonzero(lab == row.a1); c1 = c1[PROW[c1] >= 0]
        c2 = np.flatnonzero(lab == row.a2); c2 = c2[PROW[c2] >= 0]
        if min(len(c1), len(c2)) < args.min_cells:
            continue
        r1, r2 = PROW[c1], PROW[c2]
        S = psi_s.shape[0]
        dphi = draw_dphi(z, r1, r2, S, K, seed=args.seed + ci)
        if dphi is None:
            if ci == 0:
                print("    NOTE: fit has no q(phi) -- dphi is a POINT estimate and phi's "
                      "uncertainty is NOT propagated. Retrain with the current A44 to fix.",
                      flush=True)
            p1 = phi[torch.from_numpy(r1)].mean(0)
            p2 = phi[torch.from_numpy(r2)].mean(0)
            dphi = (p1 - p2).unsqueeze(0).expand(S, -1)
        tv_obs, d_obs = tv_per_atse(dphi, psi_s, atse_idx, n_atse)

        # PERMUTATION NULL: shuffle which cells belong to which arm, keeping the arm sizes and the
        # sampled phi vectors intact. Only the cell-to-group association is destroyed, so the null
        # inherits the same posterior width, the same psi draws and the same coverage as the
        # observed statistic -- which is the point. The test then no longer depends on the
        # mean-field width being right: q(phi) is 1.3-1.7x too narrow, and any such miscalibration
        # now enters the null and the observed statistic IDENTICALLY and cancels in the comparison.
        #
        # This is NOT the "permute phi" null that failed earlier. That one shuffled phi across ALL
        # atlas cells, mixing unrelated cell types and converting real between-program variation
        # into an overdispersed mixture, which inflated the null. Here the shuffle is confined to
        # the two arms of the contrast being tested.
        tv_null = None
        if args.permute:
            pooled = np.concatenate([r1, r2])
            n1 = len(r1)
            draws = []
            for b in range(args.permute):
                pr = np.random.default_rng(args.seed + 1000 * (ci + 1) + b).permutation(pooled)
                dnull = draw_dphi(z, pr[:n1], pr[n1:], S, K, seed=args.seed + 7919 * (b + 1))
                if dnull is None:
                    pn1 = phi[torch.from_numpy(pr[:n1])].mean(0)
                    pn2 = phi[torch.from_numpy(pr[n1:])].mean(0)
                    dnull = (pn1 - pn2).unsqueeze(0).expand(S, -1)
                tvb, _ = tv_per_atse(dnull, psi_s, atse_idx, n_atse)
                draws.append(tvb.mean(0))                       # [n_atse] per permutation
            tv_null = torch.stack(draws)                        # [B, n_atse]
        # the point estimate takes |mean| rather than mean|.|: E|X| >= |EX|, so mean-of-absolute
        # inherits a positive bias that is exactly what made the old statistic non-null under H0.
        d_mean = d_obs.mean(0)
        stat = 0.5 * torch.zeros(n_atse).index_add(0, atse_idx, d_mean.abs())

        # P(TV > delta). NOT a point null, and that distinction is the whole thing.
        #
        # Three null constructions were built and all failed, for three different reasons:
        # permuting group labels collapses dphi to zero so the null has no spread and everything
        # is significant (measured: 4,265 of 4,268 called, FDP = pi0); permuting phi inflates the
        # null on genuinely differential events by turning real between-program variation into an
        # overdispersed mixture; and collapsing the guide's location across programs -- a
        # principled H0 -- still came out 12-14x anti-conservative even with the width inflated 2x.
        #
        # Three failures with three causes meant the null was not the problem. The hypothesis was.
        # "Delta = 0 exactly" is essentially never true: every event has SOME program dependence,
        # psi is denoised so its posterior noise is small, and with thousands of cells an
        # arbitrarily small Delta is detectably non-zero. The test then calls almost everything and
        # is technically right every time. Truth, meanwhile, is defined by MAGNITUDE.
        #
        # delta is therefore not a free knob: it is the effect size of interest, stated. On the
        # simulations it is 0.05 because that is how the simulator defines a differential event.
        # Measured with delta = 0.05 and NO width inflation, against contrast-specific truth:
        #   effect 0.25  FDP 0.000  ratio 0.00  power 0.253
        #   effect 0.50  FDP 0.018  ratio 0.35  power 0.717
        #   effect 1.00  FDP 0.016  ratio 0.31  power 0.850
        # Conservative at every effect size. The earlier 2.9x-over-nominal came from the same shape
        # but with a biased statistic (mean of absolute rather than absolute of mean) and an
        # unjustified delta of 0.10.
        prob = (tv_obs > args.delta).float().mean(0).numpy()      # P(TV > delta)
        # Bayesian FDR, NOT BH on (1 - prob). `1 - prob` is a local false discovery rate and is not
        # uniform under the null, so BH's guarantee never applied to it.
        q = bayesian_fdr(prob)

        out = dict(contrast=f"{row.group_1}__vs__{row.group_2}", cluster=uq,
                   tv=stat.numpy(), prob_de=prob, tv_post_mean=tv_obs.mean(0).numpy(),
                   tv_post_sd=tv_obs.std(0).numpy(), lfdr=1.0 - prob, q=q)

        if tv_null is not None:
            # empirical p: how often a permuted split reaches this event's observed TV. The +1s are
            # the standard finite-B correction -- without them a p of exactly 0 is reported for any
            # event no permutation happens to beat, which is an artefact of B, not evidence.
            obs = tv_obs.mean(0)
            p_emp = ((tv_null >= obs.unsqueeze(0)).sum(0).float() + 1) / (tv_null.shape[0] + 1)
            p_emp = p_emp.numpy()
            o = np.argsort(p_emp); q_emp = np.empty(n_atse)
            q_emp[o] = np.clip(np.minimum.accumulate(
                (p_emp[o] * n_atse / np.arange(1, n_atse + 1))[::-1])[::-1], 0, 1)
            out["p_perm"] = p_emp
            out["q_perm"] = q_emp
            out["tv_null_mean"] = tv_null.mean(0).numpy()
            out["tv_null_p95"] = np.quantile(tv_null.numpy(), 0.95, axis=0)

        rows.append(pd.DataFrame(out))
        print(f"  [{ci+1}/{len(man)}] {row.group_1} vs {row.group_2}: "
              f"{int((q <= args.alpha).sum()):,} called at q<={args.alpha}"
              + (f" (perm {int((out['q_perm'] <= args.alpha).sum()):,})"
                 if tv_null is not None else "") + ", "
              f"median TV {float(np.median(stat)):.4f}", flush=True)

    if not rows:
        print("no contrasts scored"); return
    out = pd.concat(rows, ignore_index=True)
    fp = os.path.join(args.out, "a51_group_ds_posterior.tsv")
    out.to_csv(fp, sep="\t", index=False)
    print(f"\n[3] {len(out):,} rows over {out.contrast.nunique()} contrasts -> {fp}")


if __name__ == "__main__":
    main()
