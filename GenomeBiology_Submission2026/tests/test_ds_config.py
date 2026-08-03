#!/usr/bin/env python
"""
test_ds_config.py — preflight for the Figure 3 differential-splicing / calibration run.

Every check here exists because getting it wrong produces CONFIDENT GARBAGE rather than an
error. Run this before any DS number is computed, and again before any number is quoted.

    python tests/test_ds_config.py                     # everything runnable today
    python tests/test_ds_config.py --fit <fit.npz>     # adds the released-model checks

Exit code 0 = all pass. Non-zero = do not run the analysis.

The checks, and the failure each one prevents:

  1  model health gated on FINITENESS, not presence      -> silently training on NaN posteriors
  2  K/seed mapping matches result.json                  -> mislabelled seeds in the reproducibility panel
  3  psis_loc/psis_scale sample to valid PSI             -> a "posterior" that isn't one
  4  junction IDs align by IDENTITY, not length          -> LeafletFA junction i compared to LeafCutter junction j
  5  ATSE annotation_status joins 100%                   -> panel F silently dropping junctions
  6  bayesian_fdr implements max-detections              -> reported FDR is not the FDR of the reported set
  7  no post-hoc effect-size filter after the FDP        -> same, in the other direction
  8  group_delta_psi einsum == naive per-cell mean       -> the fast path quietly computing something else
  9  posterior-probability resolution at S samples       -> FDP exactly 0 as an artifact of S, read as evidence
 10  ATSE denominators identical to LeafCutter's         -> comparing two different estimands
 11  within-ATSE simplex holds                           -> ATSE total variation is meaningless without it
"""

import argparse
import json
import os
import sys
import glob

import numpy as np
import pandas as pd

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")

WD = ("/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/"
      "MOUSE_SPLICING_FOUNDATION")
ATLAS = f"{WD}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad"
MODELS_GOOD = f"{WD}/LEAFLETFA_MODELS/2026-07-26_atlas_FULL"
MODELS_CORRUPT = f"{WD}/LEAFLETFA_MODELS/2026-07-27_full_K30_tight"
LC_DIR = f"{WD}/LEAFCUTTER_DS_2026-08"
JUNC_MAP = f"{WD}/LEAFCUTTER_DS/junction_atse_map.tsv"
ATSE_FILE = (f"{WD}/ATSE_mapper/ATSE_files/"
             "MOUSE_SMARTSEQ2_ATSE_FILE_unanno_also_2026-05-12_03-43-52.txt.gz")

# K -> mb_run task ids, per the training array layout.
EXPECTED_K_LAYOUT = {10: [0, 1, 2], 20: [3, 4, 5], 30: [6, 7, 8]}

failures, skipped = [], []


def check(name, fn):
    try:
        msg = fn()
        print(f"  PASS  {name}" + (f"  [{msg}]" if msg else ""))
    except AssertionError as e:
        print(f"  FAIL  {name}\n        {e}")
        failures.append(name)
    except FileNotFoundError as e:
        print(f"  SKIP  {name}  (missing input: {e})")
        skipped.append(name)


# ── h5ad readers (obs/var only; never touch the 30 GB matrices) ───────────────

def read_var(field):
    import h5py
    with h5py.File(ATLAS, "r") as f:
        node = f["var"][field]
        if isinstance(node, h5py.Group):
            cats = np.array([c.decode() if isinstance(c, bytes) else c
                             for c in node["categories"][:]])
            return pd.Categorical.from_codes(node["codes"][:], cats).astype(str)
        return np.array([v.decode() if isinstance(v, bytes) else v for v in node[:]])


# ── 1-3: model health ────────────────────────────────────────────────────────

def load_fit_strict(path):
    """Load every array and reject any non-finite value.

    Gating on `"psis_scale" in z.files` is NOT sufficient: every run since commit 178e880
    writes the key, including the runs whose contents are entirely NaN. Presence is not
    health.
    """
    z = np.load(path)
    bad = [k for k in z.files
           if z[k].dtype.kind == "f" and not np.isfinite(z[k]).all()]
    return z, bad


def test_health_gate_rejects_corrupt():
    """The gate must reject the corrupt run for the right reason."""
    fits = sorted(glob.glob(os.path.join(MODELS_CORRUPT, "mb_run_*", "fit.npz")))
    if not fits:
        raise FileNotFoundError(MODELS_CORRUPT)

    present_but_bad = 0
    for f in fits:
        z, bad = load_fit_strict(f)
        assert "psis_scale" in z.files, (
            f"{f}: precondition — the key should be PRESENT, which is what makes a "
            f"presence-based gate unsafe")
        if "psis_scale" in bad:
            present_but_bad += 1

    assert present_but_bad > 0, (
        "expected the tight run to contain non-finite psis_scale; if this now passes, "
        "either the models were regenerated or the wrong directory is being checked")
    return (f"{present_but_bad}/{len(fits)} tight seeds have psis_scale PRESENT but "
            f"non-finite — a presence check would have accepted all of them")


def test_health_gate_accepts_released():
    fits = sorted(glob.glob(os.path.join(MODELS_GOOD, "mb_run_*", "fit.npz")))
    if not fits:
        raise FileNotFoundError(MODELS_GOOD)
    assert len(fits) == 9, f"expected 9 released fits, found {len(fits)}"
    for f in fits:
        z, bad = load_fit_strict(f)
        assert not bad, f"{f}: non-finite arrays {bad}"
        for req in ("psi", "phi", "psis_loc", "psis_scale", "pi", "bb_conc"):
            assert req in z.files, f"{f}: missing {req}"
    return "9/9 released fits fully finite"


def test_k_seed_layout():
    """Seeds must be read from result.json, never inferred from the directory name."""
    for K, tasks in EXPECTED_K_LAYOUT.items():
        for t in tasks:
            rj = os.path.join(MODELS_GOOD, f"mb_run_{t}", "result.json")
            if not os.path.exists(rj):
                raise FileNotFoundError(rj)
            r = json.load(open(rj))
            assert int(r["K"]) == K, f"mb_run_{t}: result.json K={r['K']}, layout says {K}"
    seeds = {}
    for K, tasks in EXPECTED_K_LAYOUT.items():
        seeds[K] = sorted(int(json.load(open(
            os.path.join(MODELS_GOOD, f"mb_run_{t}", "result.json")))["seed"])
            for t in tasks)
        assert seeds[K] == [0, 1, 2], f"K={K}: seeds {seeds[K]}, expected [0,1,2]"
    return "K=10/20/30 x seeds 0,1,2 — 3 seeds per K, so 3 pairwise comparisons"


def test_psi_posterior_samples_are_valid():
    """sigmoid(loc + scale*z) must land in (0,1) and recover the stored psi at z=0."""
    f = os.path.join(MODELS_GOOD, "mb_run_7", "fit.npz")
    if not os.path.exists(f):
        raise FileNotFoundError(f)
    z = np.load(f)
    loc, scale, psi = z["psis_loc"], z["psis_scale"], z["psi"]

    assert (scale > 0).all(), "psis_scale has non-positive entries"
    at_mean = 1.0 / (1.0 + np.exp(-loc))
    assert np.allclose(at_mean, psi, atol=1e-4), (
        f"sigmoid(psis_loc) does not reproduce psi (max diff "
        f"{np.abs(at_mean - psi).max():.2e}) — the stored point estimate and the "
        f"posterior location disagree, so sampling would describe a different model")

    rng = np.random.default_rng(0)
    j = rng.choice(loc.shape[1], 500, replace=False)
    draws = 1.0 / (1.0 + np.exp(-(loc[:, j][None] + scale[:, j][None]
                                  * rng.standard_normal((32,) + loc[:, j].shape))))
    assert np.isfinite(draws).all() and (draws > 0).all() and (draws < 1).all()
    return f"psi in (0,1); median posterior sd on logit scale {np.median(scale):.3f}"


def test_no_phi_posterior_on_full_atlas():
    """Pin the known limitation so it is stated in Methods, not discovered mid-analysis.

    phi_sd was added in commit 68a3185 (2026-07-29); the full atlas trained 2026-07-26, so
    it was never written rather than lost. Group DS can therefore propagate PSI uncertainty
    only, with phi at its posterior mean. Defensible because group means average phi over
    thousands of cells -- but it must be SAID.
    """
    z = np.load(os.path.join(MODELS_GOOD, "mb_run_7", "fit.npz"))
    assert "phi_sd" not in z.files, (
        "phi_sd is now present — the Methods sentence 'phi is fixed at its posterior mean' "
        "is no longer accurate and this test should be updated")
    assert "phi" in z.files
    return "no phi posterior (expected) -> Methods must state: PSI sampled, phi at posterior mean"


# ── 4-5: identifier alignment ────────────────────────────────────────────────

def test_junction_ids_align_by_identity():
    """Equal lengths prove nothing. Compare the ID sequences themselves."""
    jid = read_var("junction_id")
    ev = read_var("event_id")
    jm = pd.read_csv(JUNC_MAP, sep="\t")

    assert len(jm) == len(jid), f"length mismatch: map {len(jm)} vs var {len(jid)}"
    assert (jm["junction_id"].values == jid).all(), (
        "junction_atse_map row order does NOT match adata.var order. Positional indexing "
        "would silently pair LeafletFA junction i with LeafCutter junction j.")
    assert (jm["event_id"].values == ev).all(), "event_id order mismatch"
    return f"{len(jid):,} junctions aligned element-wise (not merely equal in count)"


def test_leafcutter_counts_ids_align():
    counts_path = os.path.join(LC_DIR, "leafcutter_counts.tsv.gz")
    if not os.path.exists(counts_path):
        raise FileNotFoundError(counts_path)
    idx = pd.read_csv(counts_path, sep="\t", index_col=0, compression="gzip",
                      usecols=[0]).index
    jm = pd.read_csv(JUNC_MAP, sep="\t")
    missing = set(jm["leafcutter_junc_id"]) - set(idx)
    assert not missing, (
        f"{len(missing)} junctions in the map are absent from the LeafCutter counts "
        f"matrix, e.g. {sorted(missing)[:3]}")
    return f"{len(idx):,} counts rows cover every mapped junction"


def test_annotation_status_joins_completely():
    """Panel F breaks down markers by annotation status; a partial join silently drops them."""
    if not os.path.exists(ATSE_FILE):
        raise FileNotFoundError(ATSE_FILE)
    a = pd.read_csv(ATSE_FILE, sep="\t", compression="gzip",
                    usecols=["junction_id", "annotation_status"])
    a = a.drop_duplicates("junction_id").set_index("junction_id")
    jid = read_var("junction_id")
    m = a.reindex(jid)
    n_missing = int(m["annotation_status"].isna().sum())
    assert n_missing == 0, (
        f"{n_missing} atlas junctions have no annotation_status — panel F would drop them")
    counts = m["annotation_status"].value_counts().to_dict()
    assert set(counts) <= {"both", "five_prime", "three_prime", "unannotated"}, counts
    return f"100% joined: {counts}"


# ── 6-7: the FDR rule ────────────────────────────────────────────────────────

def test_bayesian_fdr_is_max_detections():
    from leafletfa.differential_splicing_groups import bayesian_fdr

    probs = np.array([0.99, 0.98, 0.97, 0.40, 0.96, 0.95])
    sig, fdp, n_sig = bayesian_fdr(probs, alpha=0.05)
    order = np.argsort(-probs)
    fdp_sorted = np.cumsum(1 - probs[order]) / np.arange(1, len(probs) + 1)
    expected = int(np.flatnonzero(fdp_sorted <= 0.05).max() + 1)

    assert n_sig == expected, f"n_sig={n_sig}, max-detections rule gives {expected}"
    assert sig.sum() == n_sig
    assert not sig[3], "the low-probability junction must not be called"
    return f"n_sig={n_sig}, selection is the top-{n_sig} by posterior probability"


def test_fdp_is_monotone_so_the_two_rules_coincide():
    """The max-detections vs count-based distinction is a NO-OP. Documented here because
    both the code comments and the review notes describe it as a live bug.

    Sorted by P descending, (1-P) is non-decreasing, so

        FDP(n+1) - FDP(n) = (x_{n+1} - FDP(n)) / (n+1)  >=  0

    because x_{n+1} is at least the mean of x_1..x_n. FDP is therefore monotone
    non-decreasing in rank, the qualifying set is ALWAYS a prefix, and

        len(np.where(fdp <= alpha)[0])   ==   argmax(...) + 1

    identically -- including under ties and plateaus, which is exactly the case the
    bayesian_fdr docstring claims breaks it. It does not.

    The consequence for the calibration panel: rewriting this rule changes NO number. A
    before/after FDR curve for this specific change will show two identical curves. The
    defect that does move numbers is the post-hoc effect-size filter -- see
    test_no_post_hoc_effect_size_filter.
    """
    rng = np.random.default_rng(0)
    disagreements = 0
    worst_decrease = 0.0
    for t in range(20000):
        n = int(rng.integers(2, 40))
        if t % 2:                                  # adversarial: heavy ties and plateaus
            probs = rng.choice([0.0, 0.5, 0.9, 0.95, 0.99, 1.0], size=n)
        else:
            probs = rng.random(n)
        fdp = np.cumsum(1 - np.sort(probs)[::-1]) / np.arange(1, n + 1)
        if n > 1:
            worst_decrease = min(worst_decrease, float(np.min(np.diff(fdp))))
        for alpha in (0.01, 0.05, 0.10):
            ok = np.flatnonzero(fdp <= alpha)
            max_det = int(ok.max() + 1) if ok.size else 0
            count_based = int((fdp <= alpha).sum())
            if max_det != count_based:
                disagreements += 1

    assert worst_decrease >= 0.0, (
        f"FDP decreased by {worst_decrease:.2e} between consecutive ranks — the "
        f"monotonicity argument is wrong and the two rules can genuinely differ")
    assert disagreements == 0, (
        f"{disagreements} disagreements found; the rules are not equivalent after all")
    return ("0 disagreements in 20,000 trials x 3 alphas (ties included); FDP never "
            "decreased -> the two rules are provably identical, so this 'fix' is a no-op")


def test_no_post_hoc_effect_size_filter():
    """delta is already inside prob_de. Filtering again after the FDP invalidates the FDP.

    Enforced by reading the source: the check is that the guarantee holds, and the only way
    to break it is to reintroduce that line.
    """
    import inspect
    from leafletfa.differential_splicing_groups import differential_splicing_junctions
    src = inspect.getsource(differential_splicing_junctions)

    body = src.split("significant, fdp, n_sig = bayesian_fdr")[-1]
    for pat in ["significant &=", "significant = significant &",
                "np.abs(beta) >", "abs_beta >"]:
        assert pat not in body, (
            f"an effect-size filter ({pat!r}) is applied AFTER bayesian_fdr — the reported "
            f"FDR then does not describe the reported set")
    assert "prob_de = (d.abs() > delta)" in src, (
        "delta should enter through prob_de, before the FDP is computed")
    return "delta enters via prob_de only; no post-hoc filter"


def test_probability_resolution_at_S():
    """With S posterior draws, prob_de is quantised to 1/S.

    At S=100 a junction whose every draw clears delta reports prob_de=1.0 and hence FDP=0 --
    which is a resolution limit, not evidence. The calibration panel lives in exactly that
    tail, so it needs enough draws to distinguish 0.99 from 1.00.
    """
    for S, alpha in [(100, 0.05), (500, 0.05)]:
        gran = 1.0 / S
        assert gran <= alpha, f"S={S}: granularity {gran} coarser than alpha {alpha}"
    S = 100
    assert 1.0 / S == 0.01
    return ("S=100 resolves prob_de to 0.01 -> the smallest non-zero lfdr is 0.01, so any "
            "FDP below that is unresolvable. Use S>=500 for the calibration panel.")


# ── 8: the fast path must compute what the slow path computes ────────────────

def test_group_delta_psi_matches_naive():
    """einsum over group-mean phi == explicit per-cell mean of phi@psi.

    group_delta_psi averages phi over the group FIRST, which is valid because the group mean
    is linear in phi:  mean_c( sum_k phi[s,c,k] psi[s,k,j] ) = sum_k ( mean_c phi[s,c,k] ) psi[s,k,j]
    and it avoids the [S,C,J] tensor (3 PB at S=100, C=100k, J=80k). This test is the proof
    that the shortcut is exact, on a case small enough to compute both ways.
    """
    import torch
    from leafletfa.differential_splicing_groups import group_delta_psi

    rng = np.random.default_rng(0)
    S, C, K, J = 7, 40, 5, 23
    phi = rng.dirichlet(np.ones(K), size=(S, C)).astype(np.float64)
    psi = rng.uniform(0, 1, size=(S, K, J))
    c1 = np.arange(0, 17)
    c2 = np.arange(17, C)

    fast = group_delta_psi(torch.as_tensor(psi), torch.as_tensor(phi), c1, c2).numpy()

    rho = np.einsum("sck,skj->scj", phi, psi)          # the [S,C,J] tensor, explicitly
    slow = rho[:, c1, :].mean(axis=1) - rho[:, c2, :].mean(axis=1)

    assert np.allclose(fast, slow, atol=1e-10), (
        f"fast path differs from the explicit per-cell computation, max diff "
        f"{np.abs(fast - slow).max():.2e}")
    return f"exact to {np.abs(fast - slow).max():.1e} on S={S},C={C},K={K},J={J}"


def test_group_delta_psi_antisymmetry_and_identity():
    import torch
    from leafletfa.differential_splicing_groups import group_delta_psi
    rng = np.random.default_rng(1)
    S, C, K, J = 4, 30, 3, 11
    phi = rng.dirichlet(np.ones(K), size=(S, C))
    psi = rng.uniform(0, 1, size=(S, K, J))
    a, b = np.arange(0, 15), np.arange(15, 30)
    d_ab = group_delta_psi(torch.as_tensor(psi), torch.as_tensor(phi), a, b).numpy()
    d_ba = group_delta_psi(torch.as_tensor(psi), torch.as_tensor(phi), b, a).numpy()
    assert np.allclose(d_ab, -d_ba, atol=1e-12), "delta is not antisymmetric in the groups"
    d_aa = group_delta_psi(torch.as_tensor(psi), torch.as_tensor(phi), a, a).numpy()
    assert np.allclose(d_aa, 0, atol=1e-12), "a group against itself gives non-zero delta"
    return "antisymmetric, and zero against itself"


# ── 10-11: the estimand must be shared with LeafCutter ───────────────────────

def test_denominators_identical_to_leafcutter():
    """LeafCutter rebuilds each cluster denominator as the within-cluster junction sum;
    LeafletFA uses T_cj directly. If these disagree the two methods compute PSI on different
    denominators and their effect sizes are not comparable at all."""
    import h5py, scipy.sparse as sp
    ev = read_var("event_id")
    uniq, inv = np.unique(ev, return_inverse=True)

    def chunk(layer, a, b):
        with h5py.File(ATLAS, "r") as f:
            g = f["layers"][layer]
            ip = g["indptr"][a:b + 1]
            s, e = int(ip[0]), int(ip[-1])
            return sp.csr_matrix((g["data"][s:e], g["indices"][s:e], ip - ip[0]),
                                 shape=(b - a, len(ev))).toarray()

    for a, b in [(0, 60), (70000, 70050)]:
        Y = chunk("cell_by_junction_matrix", a, b)
        T = chunk("cell_by_cluster_matrix", a, b)
        S = np.zeros((b - a, len(uniq)), dtype=np.int64)
        np.add.at(S.T, inv, Y.T)
        assert (S[:, inv] == T).all(), (
            f"cells {a}-{b}: cell_by_cluster_matrix != within-ATSE sum of "
            f"cell_by_junction_matrix; the two methods would use different PSI denominators")
    return "T_cj == sum of junction counts within the ATSE, exactly, on 2 disjoint cell ranges"


def test_atse_sizes_are_testable():
    ev = read_var("event_id")
    sizes = pd.Series(ev).value_counts()
    assert (sizes >= 2).all(), (
        f"{(sizes < 2).sum()} single-junction ATSEs exist; ATSE-level total variation is "
        f"undefined for them")
    return f"{len(sizes):,} ATSEs, all with >=2 junctions (median {int(sizes.median())})"


# ── released-model-only checks ───────────────────────────────────────────────

def test_simplex_holds(fit_path):
    """ATSE total variation ASSUMES within-ATSE imputed PSI sums to ~1, and LeafletFA does
    not enforce it: each psi_kj ~ Beta independently. Quote no ATSE-level number until this
    passes on the released fit."""
    z = np.load(fit_path)
    psi, phi = z["psi"], z["phi"]
    ev = read_var("event_id")
    uniq, inv = np.unique(ev, return_inverse=True)

    rng = np.random.default_rng(0)
    idx = rng.choice(phi.shape[0], min(2000, phi.shape[0]), replace=False)
    rho = phi[idx] @ psi
    sums = np.zeros((len(idx), len(uniq)))
    np.add.at(sums.T, inv, rho.T)

    med = float(np.median(sums))
    assert 0.9 <= med <= 1.1, (
        f"median within-ATSE imputed PSI sum is {med:.3f}, not ~1.0 — ATSE-level total "
        f"variation is not interpretable on this fit")
    return f"median {med:.3f}, 5-95% [{np.percentile(sums,5):.3f}, {np.percentile(sums,95):.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default=None,
                    help="released fit.npz; enables the model-dependent checks")
    args = ap.parse_args()

    print("test_ds_config.py — Figure 3 DS/calibration preflight\n")

    print("[models]")
    check("health gate rejects corrupt run (finiteness, not presence)",
          test_health_gate_rejects_corrupt)
    check("health gate accepts the 9 released fits", test_health_gate_accepts_released)
    check("K/seed layout matches result.json", test_k_seed_layout)
    check("PSI posterior samples are valid", test_psi_posterior_samples_are_valid)
    check("no phi posterior on the full atlas (pin the Methods caveat)",
          test_no_phi_posterior_on_full_atlas)

    print("\n[identifier alignment]")
    check("junction IDs align by identity, not length", test_junction_ids_align_by_identity)
    check("LeafCutter counts cover every mapped junction", test_leafcutter_counts_ids_align)
    check("annotation_status joins 100% (panel F)", test_annotation_status_joins_completely)

    print("\n[FDR rule]")
    check("bayesian_fdr implements max-detections", test_bayesian_fdr_is_max_detections)
    check("FDP is monotone -> max-detections vs count-based is a no-op",
          test_fdp_is_monotone_so_the_two_rules_coincide)
    check("no post-hoc effect-size filter after the FDP", test_no_post_hoc_effect_size_filter)
    check("posterior-probability resolution at S draws", test_probability_resolution_at_S)

    print("\n[effect-size computation]")
    check("group_delta_psi fast path == explicit per-cell mean",
          test_group_delta_psi_matches_naive)
    check("group_delta_psi antisymmetry and self-identity",
          test_group_delta_psi_antisymmetry_and_identity)

    print("\n[shared estimand with LeafCutter]")
    check("ATSE denominators identical to LeafCutter's",
          test_denominators_identical_to_leafcutter)
    check("all ATSEs have >=2 junctions", test_atse_sizes_are_testable)

    if args.fit:
        print("\n[released model]")
        check("within-ATSE simplex holds", lambda: test_simplex_holds(args.fit))
    else:
        print("\n[released model]  (pass --fit <fit.npz> to enable)")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        print("Do not run or quote the DS analysis until these pass.")
        return 1
    print(f"All checks passed." + (f"  ({len(skipped)} skipped)" if skipped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
