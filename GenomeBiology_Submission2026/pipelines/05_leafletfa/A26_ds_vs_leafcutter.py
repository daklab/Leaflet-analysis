#!/usr/bin/env python
"""
A26: LeafletFA group-vs-group differential splicing on the contrasts LeafCutter already ran.

Produces the LeafletFA half of Figure 3. The LeafCutter half is complete
($WD/LEAFCUTTER_DS_2026-08, 74 pairwise contrasts); this scores exactly the same pairs with
the same groups so the two are directly comparable.

Design notes that matter for correctness
----------------------------------------
1. PSI uncertainty is propagated; PHI is fixed at its posterior mean.
   fit.npz for the full atlas carries psis_loc/psis_scale (the PSI posterior) but no PHI
   posterior -- phi_sd was added in commit 68a3185 on 2026-07-29 and the atlas trained
   2026-07-26, so it was never written rather than lost. Since a group mean averages PHI over
   thousands of cells, fixing it at the point estimate is defensible, but it MUST be stated in
   the Methods. Pinned by tests/test_ds_config.py::test_no_phi_posterior_on_full_atlas.

   A consequence worth understanding: with PHI fixed, dphi = phibar_1 - phibar_2 is a single
   [K] vector rather than [S, K]. All posterior width in the effect size comes from PSI.

2. PSI is drawn ONCE and shared across every contrast.
   Each contrast differs only in dphi, so re-sampling PSI per contrast would repeat the
   expensive part 74 times and -- worse -- give each contrast a different PSI draw, making the
   contrasts non-comparable. One draw, reused, keeps them on a common posterior.

3. Streamed over junction chunks. The full [S, K, J] tensor is 4.8 GB at S=500, K=30,
   J=79,768; the [n_contrasts, S, J] delta tensor would be far larger. Both are chunked.

4. delta enters through prob_de, never as a filter after the FDP.
   See leafletfa/differential_splicing_groups.py. Applying |beta| > delta after computing the
   FDP would mean the reported FDR does not describe the reported set.

Usage
-----
    python A26_ds_vs_leafcutter.py \\
        --fit      <MD>/mb_run_7/fit.npz \\
        --h5ad     <atlas.h5ad> \\
        --manifest <LEAFCUTTER_DS_2026-08>/pairwise_manifest_tabula_muris_senis_tissue_cell_type.tsv \\
        --lc-dir   <LEAFCUTTER_DS_2026-08> \\
        --out      <out_dir> \\
        --num-samples 500
"""

import argparse
import json
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from leafletfa.differential_splicing_groups import bayesian_fdr  # noqa: E402


def safe_name(s):
    """Must match lc_utils.safe_name, so manifest labels line up with obs labels."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))


def read_obs_col(h5ad, field):
    with h5py.File(h5ad, "r") as f:
        node = f["obs"][field]
        if isinstance(node, h5py.Group):
            cats = np.array([c.decode() if isinstance(c, bytes) else c
                             for c in node["categories"][:]])
            return pd.Categorical.from_codes(node["codes"][:], cats).astype(str)
        return np.array([v.decode() if isinstance(v, bytes) else v for v in node[:]])


def read_var_col(h5ad, field):
    with h5py.File(h5ad, "r") as f:
        node = f["var"][field]
        if isinstance(node, h5py.Group):
            cats = np.array([c.decode() if isinstance(c, bytes) else c
                             for c in node["categories"][:]])
            return pd.Categorical.from_codes(node["codes"][:], cats).astype(str)
        return np.array([v.decode() if isinstance(v, bytes) else v for v in node[:]])


def load_fit_strict(path):
    """Gate on FINITENESS, not presence.

    Every run since commit 178e880 writes psis_loc/psis_scale INCLUDING the runs whose
    contents are entirely NaN, so `"psis_scale" in z.files` accepts corrupt models.
    """
    z = np.load(path)
    bad = [k for k in z.files if z[k].dtype.kind == "f" and not np.isfinite(z[k]).all()]
    if bad:
        raise ValueError(f"{path}: non-finite arrays {bad} — refusing to use this fit")
    for req in ("psi", "phi", "psis_loc", "psis_scale"):
        if req not in z.files:
            raise ValueError(f"{path}: missing {req}")
    return z


def build_contrasts(manifest_path, labels):
    """Resolve each ready contrast to two arrays of cell indices."""
    man = pd.read_csv(manifest_path, sep="\t")
    man = man[man["status"].isin(["ready", "already_done"])]

    out = []
    for _, r in man.iterrows():
        c1 = np.flatnonzero(labels == r["group_1"])
        c2 = np.flatnonzero(labels == r["group_2"])
        if len(c1) == 0 or len(c2) == 0:
            print(f"  WARNING: no cells for {r['group_1']} vs {r['group_2']} — skipping",
                  flush=True)
            continue
        out.append(dict(group_1=r["group_1"], group_2=r["group_2"],
                        cells_1=c1, cells_2=c2,
                        n_cells_1=len(c1), n_cells_2=len(c2)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--manifest", required=True, nargs="+")
    ap.add_argument("--lc-dir", default=None,
                    help="unused by the scoring itself; recorded for provenance")
    ap.add_argument("--out", required=True)
    ap.add_argument("--group-col", default="tissue_cell_type",
                    help="obs column holding the contrast labels")
    ap.add_argument("--num-samples", type=int, default=500,
                    help="PSI posterior draws. S=100 quantises prob_de to 0.01, which is "
                         "too coarse for the calibration tail; default 500.")
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--chunk", type=int, default=4000, help="junctions per chunk")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ── model ────────────────────────────────────────────────────────────────
    print(f"[1] Loading fit (finiteness-gated): {args.fit}", flush=True)
    z = load_fit_strict(args.fit)
    loc, scale, phi = z["psis_loc"], z["psis_scale"], z["phi"]
    K, J = loc.shape
    C = phi.shape[0]
    rj = os.path.join(os.path.dirname(args.fit), "result.json")
    meta = json.load(open(rj)) if os.path.exists(rj) else {}
    print(f"    K={K} J={J:,} C={C:,}  (result.json K={meta.get('K')} seed={meta.get('seed')})",
          flush=True)
    assert "phi_sd" not in z.files, "phi posterior now present — update the Methods statement"

    # ── labels ───────────────────────────────────────────────────────────────
    print(f"[2] Reading obs['{args.group_col}'] ...", flush=True)
    raw = read_obs_col(args.h5ad, args.group_col)
    assert len(raw) == C, f"obs has {len(raw)} cells, phi has {C}"
    labels = np.array([safe_name(x) for x in raw])

    contrasts = []
    for m in args.manifest:
        got = build_contrasts(m, labels)
        print(f"    {os.path.basename(m)}: {len(got)} contrasts", flush=True)
        contrasts += got
    assert contrasts, "no contrasts resolved"
    n_con = len(contrasts)

    # dphi is [n_contrasts, K] -- one row per contrast, NOT per posterior sample, because
    # phi is a point estimate. All posterior width comes from psi.
    dphi = np.stack([phi[c["cells_1"]].mean(0) - phi[c["cells_2"]].mean(0)
                     for c in contrasts]).astype(np.float32)
    print(f"[3] {n_con} contrasts; dphi {dphi.shape}, |dphi| median "
          f"{np.median(np.abs(dphi)):.4f}", flush=True)

    # ── ATSE index ───────────────────────────────────────────────────────────
    events = read_var_col(args.h5ad, "event_id")
    junc_ids = read_var_col(args.h5ad, "junction_id")
    uniq_ev, inv_ev = np.unique(events, return_inverse=True)
    n_atse = len(uniq_ev)
    print(f"[4] {n_atse:,} ATSEs over {J:,} junctions", flush=True)

    # ── stream over junction chunks ──────────────────────────────────────────
    S = args.num_samples
    rng = np.random.default_rng(args.seed)
    dphi_t = torch.from_numpy(dphi)

    beta = np.zeros((n_con, J), dtype=np.float32)
    prob = np.zeros((n_con, J), dtype=np.float32)
    # Accumulate ATSE total variation IN PLACE. Allocating a [n_con, S, n_atse] scratch
    # tensor per chunk would cost 3.5 GB per chunk at n_con=71, S=500 (plus a numpy copy)
    # on top of this persistent buffer -- enough to OOM a 120 GB node over 20 chunks.
    tv_sum = torch.zeros((n_con, S, n_atse), dtype=torch.float32)

    print(f"[5] Sampling PSI (S={S}) and scoring, chunked over junctions ...", flush=True)
    for a in range(0, J, args.chunk):
        b = min(a + args.chunk, J)
        lo = loc[:, a:b][None].astype(np.float32)               # [1,K,jc]
        sc = scale[:, a:b][None].astype(np.float32)
        eps = rng.standard_normal((S, K, b - a)).astype(np.float32)
        # keep float32 throughout: a stray float64 here quadruples every downstream tensor
        psi = torch.from_numpy(
            (1.0 / (1.0 + np.exp(-(lo + sc * eps)))).astype(np.float32))   # [S,K,jc]

        # delta[c,s,j] = sum_k dphi[c,k] * psi[s,k,j]
        d = torch.einsum("ck,skj->csj", dphi_t, psi)            # [n_con,S,jc]

        beta[:, a:b] = d.mean(dim=1).numpy()
        prob[:, a:b] = (((d.abs() > args.delta).float().sum(dim=1) + 0.5) / (S + 1.0)).numpy()

        idx = torch.from_numpy(inv_ev[a:b].astype(np.int64))
        tv_sum.index_add_(2, idx, d.abs())

        del psi, d
        if (a // args.chunk) % 5 == 0 or b == J:
            print(f"    junctions {b:,}/{J:,}", flush=True)

    tv_sum = 0.5 * tv_sum.numpy()   # total variation = 1/2 * sum |delta| within the ATSE

    # ── per-contrast output ──────────────────────────────────────────────────
    print(f"[6] Writing per-contrast results ...", flush=True)
    summary = []
    for i, c in enumerate(contrasts):
        name = f"{c['group_1']}__vs__{c['group_2']}"

        sig_j, fdp_j, n_sig_j = bayesian_fdr(prob[i], alpha=args.alpha)
        jdf = pd.DataFrame({
            "junction_id": junc_ids, "event_id": events,
            "beta": beta[i], "abs_beta": np.abs(beta[i]),
            "prob_de": prob[i], "lfdr": 1.0 - prob[i],
            "fdp": fdp_j, "significant": sig_j,
        })

        tv_mean = tv_sum[i].mean(axis=0)
        # Jeffreys count -- see A28.score_contrast. A raw S/S proportion makes 1 - prob_de
        # exactly 0, which lets the FDP rule claim zero false discoveries it cannot support.
        tv_prob = ((tv_sum[i] > args.delta).sum(axis=0) + 0.5) / (S + 1.0)
        sig_a, fdp_a, n_sig_a = bayesian_fdr(tv_prob, alpha=args.alpha)
        adf = pd.DataFrame({
            "event_id": uniq_ev, "tv": tv_mean, "prob_de": tv_prob,
            "lfdr": 1.0 - tv_prob, "fdp": fdp_a, "significant": sig_a,
        })

        d = os.path.join(args.out, "by_pair", name)
        os.makedirs(d, exist_ok=True)
        jdf.to_parquet(os.path.join(d, "leafletfa_junctions.parquet"), index=False)
        adf.to_parquet(os.path.join(d, "leafletfa_atses.parquet"), index=False)

        summary.append(dict(contrast=name, group_1=c["group_1"], group_2=c["group_2"],
                            n_cells_1=c["n_cells_1"], n_cells_2=c["n_cells_2"],
                            n_sig_junctions=int(n_sig_j), n_sig_atses=int(n_sig_a),
                            max_abs_beta=float(np.abs(beta[i]).max()),
                            max_tv=float(tv_mean.max())))
        if (i + 1) % 10 == 0 or i + 1 == n_con:
            print(f"    {i+1}/{n_con}", flush=True)

    sm = pd.DataFrame(summary)
    tag = f"K{meta.get('K','?')}_seed{meta.get('seed','?')}"
    sm.to_csv(os.path.join(args.out, f"leafletfa_ds_summary_{tag}.tsv"), sep="\t", index=False)

    print(f"\n=== Done ({tag}) ===", flush=True)
    print(f"  contrasts: {n_con}")
    print(f"  sig junctions per contrast: median {sm.n_sig_junctions.median():.0f} "
          f"(min {sm.n_sig_junctions.min()}, max {sm.n_sig_junctions.max()})")
    print(f"  sig ATSEs per contrast:     median {sm.n_sig_atses.median():.0f} "
          f"(min {sm.n_sig_atses.min()}, max {sm.n_sig_atses.max()})")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
