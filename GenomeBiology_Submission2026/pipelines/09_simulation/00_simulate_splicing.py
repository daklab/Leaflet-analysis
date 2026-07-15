#!/usr/bin/env python
"""
FIGURE 2 (benchmark) — simulate cassette-exon splicing with a KNOWN ground truth.

Restricted to CASSETTE (exon-skipping) events: the simplest, most interpretable ATSE. Each is
exactly 3 junctions and the whole event collapses to ONE number per program — the skip usage
psi — which makes the ground truth trivially interpretable for a calibration figure.

    upstream exon ---J1---[ cassette exon ]---J2--- downstream exon
                  \________________J3 (skip)_______/

    psi[k]        = P(skip)              for program k
    P(J1) = P(J2) = (1 - psi[k]) / 2     inclusion junctions split the remainder
    (the three probabilities sum to 1 within the ATSE, as they must)

Cassettes are identified from real coordinates, NOT by assuming a sort order: the skip
junction is the one spanning the other two (start == min, end == max), and the two inclusion
introns must abut it and not overlap each other. Of 7,600 three-junction ATSEs in the mouse
data, 5,708 (75%) are canonical cassettes; the rest (alt-5'/3' sites, mutually exclusive
exons) are EXCLUDED rather than mislabelled — the old simulator blindly assigned
["J1","J3","J2"] by sort order, which silently mislabels those 25%.

Generative model (per cell i, cassette a, junction j in a):
    program mixture  phi[i,:] ~ Dirichlet, peaked on the cell's assigned program
    skip usage       psi[k,a]:  NULL ATSEs -> identical across all K programs
                                DIFF ATSEs -> program-specific (ground-truth positives)
    per-cell prob    p[i,j] = sum_k phi[i,k] * psi_junction[k,j]
    counts           BINOMIAL      n ~ Binomial(atse_total, p)                  [--bb-conc inf]
                     BETA-BINOMIAL q ~ Beta(p*conc, (1-p)*conc)
                                   n ~ Binomial(atse_total, q)                  [--bb-conc 20]

`atse_total` is taken VERBATIM from the real `cell_by_cluster_matrix` of one tissue, so
coverage, depth and zero-inflation are realistic. Cells with no coverage of an ATSE get no
counts, exactly as in real data.

BOTH EMISSIONS ARE NEEDED FOR FIGURE 2 (robustness + calibration). LeafletFA can *assume*
either likelihood (`input_conc_prior`), so the interesting question is what happens when the
model's assumption and the truth disagree:

                        model: Binomial            model: Beta-Binomial
    truth: Binomial     correct                    fine (learns conc -> large)
    truth: Beta-Binom   OVERCONFIDENT — real       correct
                        overdispersion is read
                        as signal -> inflated FPs

Overdispersion maths: Beta(p*conc, (1-p)*conc) has mean p for ANY conc, and variance
p(1-p)/(conc+1). Marginally Var(n) = N*p*(1-p) * [1 + (N-1)/(conc+1)]. So conc -> inf drives
the overdispersion factor to 1 and collapses the Beta to a point mass at p — the Binomial.
`--bb-conc inf` SHORT-CIRCUITS the Beta draw entirely and calls the same rng.binomial, so the
two code paths are provably identical, not merely close. That is the regression test on the
Beta-Binomial path.

Ground truth written to the h5ad:
    var['is_differential']   bool   — calibration label (constant within an ATSE)
    var['junction_role']     J1/J2/J3
    obs['true_program']      int
    obsm['true_phi']         (N, K)
    uns['true_psi_skip']     (K, n_ATSE)  — skip usage per program per cassette
    uns['atse_delta_psi']    (n_ATSE,)    — max-min skip usage across programs (effect size)
    uns['sim_params']        — includes emission + bb_conc

Usage:
    # Binomial truth (no overdispersion)
    python 00_simulate_splicing.py --tissue Heart --K 5 --frac-differential 0.3
    # Beta-Binomial truth (overdispersed) — the other calibration arm
    python 00_simulate_splicing.py --tissue Heart --K 5 --frac-differential 0.3 --bb-conc 20
    # pure-null control (no cassette differs between programs) — the FPR check
    python 00_simulate_splicing.py --tissue Heart --K 5 --effect 0.0
"""

import argparse
import json
import os
from datetime import datetime

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from leafletfa.waypoints import calculate_centered_psi

MOUSE_SPLICE = ("/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/"
                "MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/"
                "splice_adata_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5ad")
OUT_DIR = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION"


def compute_psi_layers(jc, at, rho=0.1):
    """junc_ratio + junc_ratio_centered, using the SAME beta-binomial weighted centering as
    the real pipeline (00_extract_train_splice.py / 09_retrain_best_on_full.py).

    Centering is NOT a naive column mean-subtraction: `calculate_centered_psi` weights each
    PSI by the beta-binomial variance implied by its ATSE coverage, so a PSI estimated from 2
    reads is down-weighted relative to one from 200. Reproducing that here means PCA on the
    simulated data is directly comparable to PCA on real data.
    """
    at_coo = at.tocoo().astype(np.float32)
    rows, cols = at_coo.row, at_coo.col
    jc_vals = np.asarray(jc.tocsr()[rows, cols], dtype=np.float32).ravel()
    jc_matched = sp.coo_matrix((jc_vals, (rows, cols)), shape=jc.shape)
    junc_ratio = sp.coo_matrix((jc_vals / at_coo.data, (rows, cols)),
                               shape=jc.shape).tocsr().astype(np.float32)
    out = calculate_centered_psi(jc_matched, at_coo, rho=rho)
    Y = out[0] if isinstance(out, tuple) else out
    return junc_ratio, sp.csr_matrix(Y).astype(np.float32)


def find_cassettes(var):
    """Label canonical cassette ATSEs from real coordinates.

    Returns a DataFrame indexed like var with columns [event_id, role] for cassette junctions
    only; role in {J1, J2, J3} where J3 is the skip junction.
    """
    v = var[["event_id", "junction_id"]].astype(str).copy()
    p = v.junction_id.str.rsplit("_", n=3, expand=True)
    v["start"] = p[1].astype(int)
    v["end"] = p[2].astype(int)

    sizes = v.event_id.value_counts()
    three = set(sizes[sizes == 3].index)
    v3 = v[v.event_id.isin(three)]

    roles, keep_events = {}, []
    for e, g in v3.groupby("event_id"):
        is_skip = (g.start == g.start.min()) & (g.end == g.end.max())
        if is_skip.sum() != 1:
            continue                                   # not a cassette
        skip = g[is_skip].iloc[0]
        inc = g[~is_skip].sort_values("start")
        if len(inc) != 2:
            continue
        i1, i2 = inc.iloc[0], inc.iloc[1]
        # the two inclusion introns must abut the skip and not overlap each other
        if not (i1.start == skip.start and i2.end == skip.end and i1.end < i2.start):
            continue
        roles[i1.name] = "J1"
        roles[i2.name] = "J2"
        roles[skip.name] = "J3"
        keep_events.append(e)

    print(f"  3-junction ATSEs: {len(three):,} | canonical cassettes: {len(keep_events):,} "
          f"({100*len(keep_events)/max(len(three),1):.1f}%)")
    out = pd.DataFrame({"role": pd.Series(roles)})
    out["event_id"] = v.loc[out.index, "event_id"]
    return out


def load_tissue(splice_path, tissue, min_cells_per_junction):
    print(f"Loading {splice_path}", flush=True)
    a = ad.read_h5ad(splice_path)
    keep = a.obs["tissue"].astype(str) == tissue
    if keep.sum() == 0:
        raise SystemExit(f"No cells with tissue == '{tissue}'. Available: "
                         f"{sorted(a.obs['tissue'].astype(str).unique())[:10]}")
    a = a[keep].copy()
    print(f"  {tissue}: {a.n_obs:,} cells x {a.n_vars:,} junctions")

    clus = sp.csr_matrix(a.layers["cell_by_cluster_matrix"])
    covered = np.asarray((clus > 0).sum(axis=0)).ravel()
    jkeep = covered >= min_cells_per_junction
    a = a[:, jkeep].copy()
    print(f"  junctions covered in >= {min_cells_per_junction} cells: {jkeep.sum():,}")

    cas = find_cassettes(a.var)
    # an ATSE survives only if ALL THREE of its junctions survived coverage filtering
    full = cas.groupby("event_id").size()
    full_events = set(full[full == 3].index)
    cas = cas[cas.event_id.isin(full_events)]
    a = a[:, cas.index].copy()
    a.var["junction_role"] = cas.loc[a.var_names, "role"].values
    print(f"  cassettes with all 3 junctions covered: {len(full_events):,} "
          f"-> {a.n_vars:,} junctions")
    if a.n_vars == 0:
        raise SystemExit("No cassettes survived filtering — lower --min-cells-per-junction.")
    return a


def assign_programs(adata, K, rng, phi_conc, celltype_col="broad_cell_type"):
    """Assign cells to K programs (seeded by real cell type), then draw a peaked phi."""
    ct = adata.obs[celltype_col].astype(str).values
    order = pd.Series(ct).value_counts().index.tolist()
    mapping = {c: (i if i < K - 1 else K - 1) for i, c in enumerate(order)}
    prog = np.array([mapping[c] for c in ct], dtype=int)

    alpha = np.full((adata.n_obs, K), 1.0)
    alpha[np.arange(adata.n_obs), prog] = phi_conc
    phi = np.stack([rng.dirichlet(a) for a in alpha]).astype(np.float64)

    counts = dict(zip(*np.unique(prog, return_counts=True)))
    print(f"  K={K} | cells per program: {counts}")
    print(f"  phi mean max-weight = {phi.max(1).mean():.3f} (1.0 = hard assignment)")
    return prog, phi


def draw_psi(adata, K, frac_diff, rng, effect, beta_a, beta_b):
    """Skip usage psi[k, a] per program, then expand to per-junction probabilities.

    NULL cassette : psi identical across programs.
    DIFF cassette : psi_k = (1-effect)*psi0 + effect*psi_k_indep   (effect=0 -> null)
    """
    ev = adata.var["event_id"].astype(str).values
    role = adata.var["junction_role"].values
    events = pd.unique(ev)
    A = len(events)

    is_diff_event = rng.random(A) < frac_diff
    psi_skip = np.zeros((K, A))
    for ai in range(A):
        psi0 = rng.beta(beta_a, beta_b)
        if not is_diff_event[ai]:
            psi_skip[:, ai] = psi0                      # NULL: no program difference
        else:
            indep = rng.beta(beta_a, beta_b, size=K)
            psi_skip[:, ai] = (1 - effect) * psi0 + effect * indep
    psi_skip = np.clip(psi_skip, 1e-4, 1 - 1e-4)

    # expand: P(J3)=psi, P(J1)=P(J2)=(1-psi)/2  -> sums to 1 within the ATSE
    ev_idx = {e: i for i, e in enumerate(events)}
    col_event = np.array([ev_idx[e] for e in ev])
    psi_j = np.zeros((K, adata.n_vars))
    skip_mask = role == "J3"
    psi_j[:, skip_mask] = psi_skip[:, col_event[skip_mask]]
    psi_j[:, ~skip_mask] = (1 - psi_skip[:, col_event[~skip_mask]]) / 2

    delta = psi_skip.max(0) - psi_skip.min(0)           # ground-truth effect size per ATSE
    is_diff_junc = is_diff_event[col_event]

    print(f"  cassettes: {A:,} | differential {is_diff_event.sum():,} "
          f"({100*is_diff_event.mean():.0f}%) | null {A - is_diff_event.sum():,}")
    print(f"  effect={effect} | delta-psi (diff ATSEs): median "
          f"{np.median(delta[is_diff_event]) if is_diff_event.any() else 0:.3f}, "
          f"null: {np.median(delta[~is_diff_event]) if (~is_diff_event).any() else 0:.3f}")
    # within-ATSE probabilities must sum to 1
    chk = psi_j[:, col_event == 0].sum(1)
    assert np.allclose(chk, 1.0), f"psi within ATSE must sum to 1, got {chk}"
    return psi_j, psi_skip, is_diff_junc, delta, events


def simulate_counts(clus, phi, psi_j, rng, bb_conc=np.inf):
    """Draw junction counts on the covered (cell, junction) entries only.

    p[i,j] = phi[i,:] @ psi_j[:,j]   (the per-cell splicing probability)

    BINOMIAL  (bb_conc = inf):
        n ~ Binomial(atse_total, p)
        Variance is exactly N*p*(1-p) — no overdispersion.

    BETA-BINOMIAL (finite bb_conc):
        q ~ Beta(p*conc, (1-p)*conc)      # jitter the probability per (cell, junction)
        n ~ Binomial(atse_total, q)
        The Beta has mean p for ANY conc, and variance p(1-p)/(conc+1). Marginally,
            Var(n) = N*p*(1-p) * [1 + (N-1)/(conc+1)]
                                  \____ overdispersion factor ____/
        so small conc => heavily overdispersed; conc -> inf => the factor -> 1 and the Beta
        collapses to a point mass at p, recovering the Binomial EXACTLY.

    That limit is not just theory here: bb_conc=inf SHORT-CIRCUITS the Beta draw and calls the
    same rng.binomial as the Binomial arm, so the two paths are provably identical rather than
    merely close. Use `--bb-conc inf` as a regression test on the Beta-Binomial code path.
    """
    c = clus.tocoo()
    rows, cols, totals = c.row, c.col, c.data.astype(np.int64)
    keep = totals > 0
    rows, cols, totals = rows[keep], cols[keep], totals[keep]
    print(f"  covered (cell, junction) entries: {len(totals):,}")

    # p per covered entry — sparse; never materialise the N x J matrix
    p = np.clip(np.einsum("ik,ki->i", phi[rows], psi_j[:, cols]), 0.0, 1.0)

    if np.isinf(bb_conc):
        print("  emission: BINOMIAL (bb_conc = inf; no overdispersion)")
        probs = p
    else:
        # Beta(p*conc, (1-p)*conc): mean p, variance p(1-p)/(conc+1)
        eps = 1e-6
        pc = np.clip(p, eps, 1 - eps)
        a = pc * bb_conc
        b = (1.0 - pc) * bb_conc
        probs = rng.beta(a, b)
        od = 1.0 + (np.median(totals) - 1.0) / (bb_conc + 1.0)
        print(f"  emission: BETA-BINOMIAL (conc={bb_conc:g})")
        print(f"    Beta mean deviation from p: {np.abs(probs - p).mean():.4f} "
              f"(-> 0 as conc -> inf)")
        print(f"    overdispersion factor at median coverage "
              f"(N={int(np.median(totals))}): {od:.2f}x Binomial variance")

    counts = rng.binomial(totals, probs)

    N, J = phi.shape[0], psi_j.shape[1]
    junc = sp.coo_matrix((counts, (rows, cols)), shape=(N, J)).tocsr()
    junc.eliminate_zeros()
    cl = sp.coo_matrix((totals, (rows, cols)), shape=(N, J)).tocsr()
    return junc, cl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splice", default=MOUSE_SPLICE)
    ap.add_argument("--tissue", default="Heart")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--frac-differential", type=float, default=0.30)
    ap.add_argument("--effect", type=float, default=1.0,
                    help="0 = no program difference (null regression test) .. 1 = max")
    ap.add_argument("--bb-conc", default="inf",
                    help="Beta-Binomial concentration. 'inf' (default) = plain Binomial, no "
                         "overdispersion. Finite values (e.g. 1, 5, 20, 100) add overdispersion; "
                         "smaller = more. conc=inf short-circuits to the Binomial path exactly.")
    ap.add_argument("--phi-conc", type=float, default=20.0)
    ap.add_argument("--beta-a", type=float, default=0.5, help="Beta prior on skip usage")
    ap.add_argument("--beta-b", type=float, default=0.5)
    ap.add_argument("--min-cells-per-junction", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=OUT_DIR)
    args = ap.parse_args()

    bb_conc = float("inf") if str(args.bb_conc).lower() in ("inf", "infinity", "none") \
        else float(args.bb_conc)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    print("=== 1. real cassette structure + coverage ===")
    adata = load_tissue(args.splice, args.tissue, args.min_cells_per_junction)

    print("\n=== 2. ground-truth programs (phi) ===")
    prog, phi = assign_programs(adata, args.K, rng, args.phi_conc)

    print("\n=== 3. ground-truth skip usage (psi) ===")
    psi_j, psi_skip, is_diff, delta, events = draw_psi(
        adata, args.K, args.frac_differential, rng, args.effect, args.beta_a, args.beta_b)

    emission = "binomial" if np.isinf(bb_conc) else "betabinomial"
    print(f"\n=== 4. simulate counts ({emission}) ===")
    clus_real = sp.csr_matrix(adata.layers["cell_by_cluster_matrix"])
    junc_sim, clus_sim = simulate_counts(clus_real, phi, psi_j, rng, bb_conc=bb_conc)

    print("\n=== 5. PSI layers (same centering as the real pipeline) ===")
    ratio, ratio_centered = compute_psi_layers(junc_sim, clus_sim)
    print(f"  junc_ratio nnz={ratio.nnz:,} | junc_ratio_centered nnz={ratio_centered.nnz:,}")

    print("\n=== 6. assemble AnnData ===")
    out = ad.AnnData(X=ratio, obs=adata.obs.copy(), var=adata.var.copy())
    out.layers["cell_by_junction_matrix"] = junc_sim
    out.layers["cell_by_cluster_matrix"] = clus_sim
    out.layers["junc_ratio"] = ratio
    out.layers["junc_ratio_centered"] = ratio_centered
    out.var["is_differential"] = is_diff
    out.obs["true_program"] = prog
    out.obsm["true_phi"] = phi
    out.uns["true_psi_skip"] = psi_skip
    out.uns["atse_ids"] = np.array(events, dtype=object)
    out.uns["atse_delta_psi"] = delta
    out.uns["sim_params"] = json.dumps(dict(
        tissue=args.tissue, K=args.K, frac_differential=args.frac_differential,
        effect=args.effect, emission=emission, bb_conc=(None if np.isinf(bb_conc) else bb_conc),
        phi_conc=args.phi_conc,
        beta_a=args.beta_a, beta_b=args.beta_b,
        min_cells_per_junction=args.min_cells_per_junction, seed=args.seed,
        source=args.splice, created=datetime.now().isoformat()))

    # NOTE: cell_by_cluster_matrix broadcasts the ATSE total to EACH of its junction columns
    # (that is how LeafletFA reads the denominator), so sum(junc)/sum(clus) ~ 1/3 for a
    # 3-junction cassette BY CONSTRUCTION. The meaningful check is per (cell, ATSE):
    # the three junction counts should sum to the ATSE total in expectation.
    n_atse = out.n_vars // 3
    print(f"  sum(junc)/sum(clus) = {junc_sim.sum()/max(clus_sim.sum(),1):.3f} "
          f"(expect ~1/3: the denominator is repeated across the 3 junctions)")
    print(f"  junc_ratio range: {ratio.data.min():.3f} - {ratio.data.max():.3f}")
    print(f"  differential junctions: {int(is_diff.sum()):,} / {len(is_diff):,} "
          f"({n_atse:,} cassettes)")

    etag = "binom" if np.isinf(bb_conc) else f"bb{bb_conc:g}"
    tag = f"{args.tissue.lower()}_K{args.K}_{etag}_eff{args.effect}_frac{args.frac_differential:g}_seed{args.seed}"
    path = os.path.join(args.outdir, f"sim_{tag}.h5ad")
    tmp = path + ".tmp"
    out.write_h5ad(tmp)
    os.replace(tmp, path)                        # atomic — never leave a truncated h5ad
    print(f"\n-> {path}")
    print(f"   {out.n_obs:,} cells x {out.n_vars:,} junctions "
          f"({out.n_vars//3:,} cassettes), K={args.K}")


if __name__ == "__main__":
    main()
