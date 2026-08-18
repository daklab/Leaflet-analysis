#!/usr/bin/env python
"""A45: a per-ATSE Dirichlet-multinomial likelihood-ratio test, using the learned programs as the
design matrix. LeafCutter's inference, LeafletFA's covariates.

Why this instead of a fourth empirical null
-------------------------------------------
A39, A40 and A42 all tried to ESTIMATE a null distribution for the TV statistic and read a
probability off its extreme tail. All three failed there, and A43 showed why the failures kept
looking different: with an oracle null the machinery is exactly calibrated, so the statistic and
BH were never the problem -- the null was. But A43's oracle needs truth labels, so it is not a
method, and estimating a null accurately in the far tail is the hardest thing to ask of any
estimator.

LeafCutter does not have this problem because it never estimates a null. It fits a DM GLM per
ATSE under H0 and H1 and reads p from the asymptotic chi-square of 2*dlogL. That is available to
us: `dm_log_prob` was verified term-for-term against LeafCutter's dm_glm.stan
(-114.362812 vs -114.362793), so the likelihood is already the same one.

What differs from LeafCutter -- and it is the whole point of the method -- is the design matrix.
LeafCutter regresses junction usage on KNOWN cell-type labels. Here the covariate is the learned
program loading phi, so the test asks "does usage of this ATSE vary across programs discovered
without labels". LeafletFA keeps its contribution; the inference is borrowed.

The models, per ATSE e with m junctions and cells i
---------------------------------------------------
    H0   alpha_ej = conc * softmax(b_j)               composition shared by every cell
    H1   alpha_eij = conc * softmax(b_j + phi_i . B_j) composition varies with the K loadings

    stat_e = 2 * (logL1 - logL0)  ~  chi2 with (m-1)*K degrees of freedom

b is m-dimensional and B is K x m, both pinned by fixing the last junction's coefficients at zero
-- softmax is invariant to adding a constant across junctions, so without that the parameters are
unidentified and the df count is wrong. df = (m-1)*K counts the free entries of B alone.

Honest about the assumption
---------------------------
Wilks' theorem is asymptotic. 44% of atlas ATSEs have m=2, and many have low coverage, so the
chi-square approximation is exactly the thing that could fail here. This script therefore does not
assume it: it reports realised FDP against known truth on simulations, and `--permute-check`
recomputes the statistic on cell-label-permuted data so the chi-square can be compared against a
genuine null. A p-value that is honest by construction is worth nothing if its reference
distribution is wrong, and m=2 is where I would expect it to be.
"""

import argparse, os, sys, time

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")


def load_sim(path):
    f = h5py.File(path, "r")
    def dense(l):
        g = f["layers"][l]; s = tuple(g.attrs["shape"])
        return np.asarray(sp.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                                        shape=s).todense(), np.float32)
    def cat(n):
        if isinstance(n, h5py.Group):
            c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
            return c[n["codes"][:]]
        return np.asarray(n[:])
    ev = cat(f["var"]["event_id"]).astype(str); role = cat(f["var"]["junction_role"]).astype(str)
    uq, inv = np.unique(ev, return_inverse=True)
    ids = np.array([x.decode() if isinstance(x, bytes) else x
                    for x in f["uns"]["atse_ids"][:]]).astype(str)
    order = {a: i for i, a in enumerate(ids)}; tidx = np.array([order[a] for a in uq])
    ps = np.asarray(f["uns"]["true_psi_skip"])[:, tidx]
    K = ps.shape[0]; tpsi = np.zeros((K, len(ev))); sk = role == "J3"
    tpsi[:, sk] = ps[:, inv[sk]]; tpsi[:, ~sk] = (1 - ps[:, inv[~sk]]) / 2
    d = dict(junc=dense("cell_by_junction_matrix"), clus=dense("cell_by_cluster_matrix"),
             uq=uq, inv=inv, role=role, tpsi=tpsi,
             delta=np.asarray(f["uns"]["atse_delta_psi"])[tidx],
             prog=cat(f["obs"]["true_program"]).astype(str))
    f.close(); return d


def load_real(h5, cells=None):
    """Atlas counts + event_id. No truth, so only the permuted-null arm is meaningful -- which is
    the arm that matters: it measures whether the chi-square reference is honest at REAL m=2 ATSEs
    and real coverage, neither of which the cassette-exon simulator can produce (it emits exactly
    three junctions per event, so m=3 always)."""
    with h5py.File(h5, "r") as f:
        n = f["var"]["event_id"]
        if isinstance(n, h5py.Group):
            c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
            ev = c[n["codes"][:]]
        else:
            ev = np.array([x.decode() if isinstance(x, bytes) else x for x in n[:]])
        ev = ev.astype(str)
        J = len(ev)
        N_all = int(f["layers"]["cell_by_junction_matrix"].attrs["shape"][0])
    rows = np.arange(N_all) if cells is None else np.sort(cells)

    def read(layer):
        out = np.zeros((len(rows), J), np.float32)
        with h5py.File(h5, "r") as f:
            g = f["layers"][layer]; ip_all = g["indptr"]
            for a in range(0, len(rows), 2000):
                blk = rows[a:a + 2000]
                lo, hi = int(blk[0]), int(blk[-1]) + 1
                ip = ip_all[lo:hi + 1]; st, en = int(ip[0]), int(ip[-1])
                mtx = sp.csr_matrix((g["data"][st:en], g["indices"][st:en], ip - ip[0]),
                                    shape=(hi - lo, J))
                out[a:a + len(blk)] = mtx[blk - lo].toarray()
        return out

    uq, inv = np.unique(ev, return_inverse=True)
    return dict(junc=read("cell_by_junction_matrix"), clus=read("cell_by_cluster_matrix"),
                uq=uq, inv=inv, rows=rows,
                delta=np.zeros(len(uq)), tpsi=None, role=None)


def dm_ll(y, logits, conc):
    """DM log-likelihood, summed over cells. y [N, m], logits [N, m] or [m]. Same algebra as
    leafletfa.dirichlet_multinomial.dm_log_prob, restricted to one event so no segment ops."""
    mu = torch.softmax(logits, dim=-1)
    a = (conc * mu).clamp_min(1e-8)
    if a.dim() == 1:
        a = a.expand_as(y)
    A = a.sum(-1)
    N = y.sum(-1)
    return ((torch.lgamma(y + a) - torch.lgamma(a)).sum(-1)
            + torch.lgamma(A) - torch.lgamma(N + A)).sum()


def fit_event(y, X, conc, steps, lr, alt, free_conc=True):
    """MLE for one ATSE. alt=False fits H0 (intercept only), True fits H1 (intercept + X @ B).

    Both junction-coefficient blocks are held at zero in the LAST junction: softmax is invariant to
    a constant shift across junctions, so leaving all m free makes the optimum a ridge rather than
    a point and inflates the apparent df.

    free_conc estimates the concentration PER EVENT, in both hypotheses, so it profiles out of the
    ratio. Holding it at the model's single global value -- the first version of this script -- is a
    misspecification whenever an event's overdispersion differs from the average, and a misspecified
    DM inflates the LRT statistic. That showed up as the permuted-phi null drifting 0.62 -> 3.82
    across effect sizes, when a permuted null cannot depend on effect size at all: the amount of
    inflation tracked how structured phi was, not any real signal. conc is optimised on the log
    scale to keep it positive without a constraint."""
    N, m = y.shape
    K = X.shape[1]
    b = torch.zeros(m - 1, requires_grad=True)
    params = [b]
    B = None
    logc = None
    if free_conc:
        logc = torch.tensor(float(np.log(max(conc, 1e-3))), requires_grad=True)
        params.append(logc)
    if alt:
        B = torch.zeros(K, m - 1, requires_grad=True)
        params.append(B)
    opt = torch.optim.LBFGS(params, max_iter=steps, lr=lr, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-9)
    pad = torch.zeros(1)

    def _ll():
        bb = torch.cat([b, pad])
        logits = bb.expand(N, m) if not alt else bb + X @ torch.cat([B, torch.zeros(K, 1)], 1)
        # clamp the EXPONENT, not the result: exp() of a large logc overflows to inf before any
        # clamp on the output can catch it, which is the "cannot be converted to float without
        # overflow" crash. LBFGS takes long steps, so logc does reach values exp() cannot hold.
        c = logc.clamp(-6.9, 13.8).exp() if free_conc else conc
        return dm_ll(y, logits, c)

    def closure():
        opt.zero_grad()
        loss = -_ll()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return float(_ll())


def bh(p, alpha):
    n = len(p); o = np.argsort(p); thr = alpha * np.arange(1, n + 1) / n
    ok = p[o] <= thr
    sig = np.zeros(n, bool)
    if ok.any():
        sig[o[:np.flatnonzero(ok)[-1] + 1]] = True
    return sig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", help="simulation h5ad (has truth). Omit when using --real.")
    ap.add_argument("--real", help="atlas h5ad. No truth, so only the permuted-null arm runs -- "
                                   "which is the point: real m=2 ATSEs at real coverage.")
    ap.add_argument("--real-cells", type=int, default=20000,
                    help="cells to subsample from --real. The chi-square is asymptotic in CELLS, "
                         "so this is not a free knob: too few and a failure is the subsample, not "
                         "Wilks. Swept with --real-cells to separate the two.")
    ap.add_argument("--fit", required=True, help="fit_*.npz -- supplies phi (the design matrix)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-atse", type=int, default=0, help="0 = all; >0 subsets for a quick pass")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--min-cells", type=int, default=25,
                    help="cells with nonzero coverage required before an ATSE is testable")
    ap.add_argument("--null-tol", type=float, default=0.05)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.20])
    ap.add_argument("--permute-check", action="store_true",
                    help="also run on cell-permuted phi -- a genuine null, so the chi-square "
                         "reference distribution is checked rather than trusted")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed-conc", action="store_true",
                    help="hold concentration at the model's global value instead of estimating it "
                         "per event. The original behaviour, kept so the two are comparable.")
    ap.add_argument("--permute-strata", type=int, default=20,
                    help="depth bands to permute phi within. 1 = unstratified. Run both: if they "
                         "agree, depth was not load-bearing; if not, the stratified one is right.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))

    if not (args.sim or args.real):
        ap.error("need --sim (with truth) or --real (atlas, permuted null only)")
    real_mode = bool(args.real)
    z = np.load(args.fit, allow_pickle=True)
    K = z["phi"].shape[1]
    if real_mode:
        args.permute_check = True          # the only arm that means anything without truth
        rng0 = np.random.default_rng(args.seed)
        with h5py.File(args.real, "r") as f:
            N_all = int(f["layers"]["cell_by_junction_matrix"].attrs["shape"][0])
        take = (np.sort(rng0.choice(N_all, size=min(args.real_cells, N_all), replace=False))
                if args.real_cells else None)
        s = load_real(args.real, take)
        phi = np.asarray(z["phi"], np.float32)
        if phi.shape[0] != len(s["rows"]):
            phi = phi[s["rows"]]           # the fit covers all cells; align to the subsample
    else:
        s = load_sim(args.sim)
        Kt = int(np.asarray(s["tpsi"]).shape[0])
        if K == Kt:
            C = np.corrcoef(z["psi_mean"], s["tpsi"])[:K, K:]
            r_, c_ = linear_sum_assignment(-np.nan_to_num(C))
            pm = np.empty(K, int); pm[c_] = r_
        else:
            # misspecified rank: no one-to-one match exists, and the LRT uses phi only as a design
            # matrix, whose column order is irrelevant -- keep the fit's order
            pm = np.arange(K)
        phi = np.asarray(z["phi"], np.float32)[:, pm]
    conc = float(z["bb_conc"]) if np.isfinite(z["bb_conc"]) else 50.0
    n_atse = len(s["uq"])
    truth = s["delta"] > args.null_tol   # all-False in real mode; FDP block is skipped there
    print(f"[1] {n_atse:,} ATSEs, K={K}, conc={conc:.2f}, "
          f"{truth.sum():,} truly differential ({100*truth.mean():.1f}%)", flush=True)

    # centre phi: the intercept b already carries the baseline composition, so an uncentred design
    # makes b and B trade off and slows LBFGS without changing the optimum.
    Xall = torch.from_numpy(phi - phi.mean(0, keepdims=True))
    rng = np.random.default_rng(args.seed)

    def permuted_rows(n_strata):
        """Row permutation of the design matrix.

        Permuting ROWS leaves X'X exactly unchanged, so the design geometry -- program
        correlations, the simplex constraint, cell-type clustering -- is preserved perfectly and
        only the phi<->counts pairing is cut. That is exactly the association H0 denies.

        n_strata > 1 permutes only WITHIN bands of similar cell coverage. phi correlates with
        sequencing depth, and depth drives count variance, so an unstratified shuffle cuts the
        depth<->phi link as well as the biology link. The permuted null then carries less
        association than a real null does and comes out too narrow -- the 23%-too-tight signature
        the A40 gate caught (ratio 0.771). Stratifying keeps depth attached and cuts only biology.
        """
        n = phi.shape[0]
        if n_strata <= 1:
            return rng.permutation(n)
        depth_c = s["clus"].sum(1)                 # total ATSE reads per cell
        order = np.argsort(depth_c, kind="stable")
        out = np.arange(n)
        for chunk in np.array_split(order, n_strata):
            out[chunk] = rng.permutation(chunk)
        return out

    Xperm = (Xall[torch.from_numpy(permuted_rows(args.permute_strata))]
             if args.permute_check else None)
    if args.permute_check:
        print(f"[1b] permuted phi across cells, {args.permute_strata} depth strata "
              f"({'unstratified' if args.permute_strata <= 1 else 'depth preserved'})", flush=True)

    idx = np.arange(n_atse)
    if args.max_atse:
        idx = np.sort(rng.choice(n_atse, size=min(args.max_atse, n_atse), replace=False))

    rows = []
    t0 = time.time()
    for n_done, e in enumerate(idx):
        cols = np.flatnonzero(s["inv"] == e)
        m = len(cols)
        cov = s["clus"][:, cols[0]] > 0
        if cov.sum() < args.min_cells or m < 2:
            continue
        y = torch.from_numpy(s["junc"][cov][:, cols].astype(np.float32))
        rec = dict(atse=str(s["uq"][e]), m=m, n_cells=int(cov.sum()),
                   depth=float(s["clus"][cov, cols[0]].mean()),
                   true_delta=float(s["delta"][e]), is_de=bool(truth[e]),
                   df=(m - 1) * K)
        for lab, X in (("", Xall), ("perm_", Xperm)):
            if X is None:
                continue
            Xc = X[torch.from_numpy(np.flatnonzero(cov))]
            fc = not args.fixed_conc
            l0 = fit_event(y, Xc, conc, args.steps, args.lr, alt=False, free_conc=fc)
            l1 = fit_event(y, Xc, conc, args.steps, args.lr, alt=True, free_conc=fc)
            stat = max(2.0 * (l1 - l0), 0.0)
            rec[lab + "stat"] = stat
            rec[lab + "p"] = float(chi2.sf(stat, rec["df"]))
        rows.append(rec)
        if n_done % 250 == 0 and n_done:
            print(f"    {n_done:,}/{len(idx):,} ATSEs, {time.time()-t0:.0f}s "
                  f"({1000*(time.time()-t0)/n_done:.0f} ms each)", flush=True)

    R = pd.DataFrame(rows)
    tag = (f"atlas{len(s['rows']):,}".replace(",", "") if real_mode
           else os.path.basename(args.sim).replace(".h5ad", "").replace("sim_heart_K5_", ""))
    seed = int(z["seed"])
    out = os.path.join(args.out, f"a45_lrt_{tag}_seed{seed}.tsv")
    R.to_csv(out, sep="\t", index=False)
    print(f"\n[2] {len(R):,} testable ATSEs in {(time.time()-t0)/60:.1f} min", flush=True)

    if args.permute_check:
        pp = R["perm_p"].values
        print("\n=== is the chi-square reference honest? p-values on PERMUTED phi (a true null) ===")
        print(f"  {'nominal':>9} {'observed frac p<=nominal':>26} {'ratio':>7}")
        for q in (0.01, 0.05, 0.10, 0.20, 0.50):
            obs = float((pp <= q).mean())
            print(f"  {q:>9.2f} {obs:>26.4f} {obs/q:>7.2f}")
        print("  ratio ~1 = chi-square is right. >1 = too liberal (p too small, anti-conservative).")
        print(f"\n  by event size -- m=2 is 44% of the atlas and the case most likely to break "
              f"Wilks:\n  {'m':>4} {'n':>8} {'p<=0.01':>9} {'p<=0.05':>9} {'p<=0.20':>9} "
              f"{'ratio@0.05':>11}")
        for mm in sorted(R["m"].unique()):
            sub = pp[R["m"].values == mm]
            if len(sub) < 30:
                continue
            f5 = float((sub <= 0.05).mean())
            print(f"  {mm:>4} {len(sub):>8,} {float((sub<=0.01).mean()):>9.4f} {f5:>9.4f} "
                  f"{float((sub<=0.20).mean()):>9.4f} {f5/0.05:>11.2f}")

    if real_mode:
        print("\n(real data: no truth, so no FDP. The permuted-null table above IS the result.)")
        print(f"\n-> {out}")
        return

    print("\n=== realised FDP vs nominal (chi-square p -> BH) ===")
    p = R["p"].values; tr = R["is_de"].values
    print(f"  {'alpha':>7} {'FDP':>8} {'ratio':>7} {'power':>7} {'n_called':>9}")
    for a in args.alphas:
        sig = bh(p, a); ns = int(sig.sum()); nf = int((sig & ~tr).sum())
        print(f"  {a:>7.2f} {(nf/ns if ns else np.nan):>8.4f} "
              f"{(nf/ns/a if ns else np.nan):>7.2f} {(sig&tr).sum()/max(tr.sum(),1):>7.3f} {ns:>9,}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
