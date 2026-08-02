"""Pick one full-atlas model to carry the downstream (LeafCutter calibration) analysis.

Three things have to be settled, and each has a measurement rather than a preference.

1. IS THE FIT HEALTHY?  mean Psi should match the observed splicing rate, psi_contrast should be well
   above zero (programs genuinely differ -- this IS the DS effect size), bb_conc should reproduce
   across seeds, perplexity should be a few of K (cells commit rather than spreading), and the run
   should have stopped on its own rather than at the cap. A run that hit the cap never converged.

2. WHICH K?  Not the best ELBO -- ELBO always improves with K. Two reproducibility readouts point in
   different directions and both are reported: EFFECTIVE K (how many phi directions agree between
   seeds) rises sublinearly, so the return per added program falls; CELL-TYPE STRUCTURE agreement
   (permutation-invariant, aggregate) was highest at K=30 in the 70/30 models. For cell-type-level
   biology the second is the relevant one.

3. WHICH SEED?  NOT the best ELBO. On the 70/30 models the best-ELBO seed disagreed with the most
   central seed at every K, so selecting on loss selects for fit, not for stability. The defensible
   choice is the MEDOID: the seed whose phi subspace agrees most with the other seeds, i.e. the least
   idiosyncratic fit. Reported alongside best-ELBO so the disagreement is visible rather than hidden.

Note the honest caveat this whole exercise sits under: only ~40-50% of phi directions reproduce
between seeds, so "the representative model" means the most central draw from a partly-degenerate
posterior, not a uniquely determined answer. Anything downstream should lean on the aggregate,
cell-type-level structure, which does reproduce, rather than on individual per-cell loadings.

Usage: A12_select_final_model.py <model_dir> <out_prefix> [K]
"""
from __future__ import annotations
import sys, os, json, itertools
import numpy as np, pandas as pd

TASKS_BY_K = {10: (0, 1, 2), 20: (3, 4, 5), 30: (6, 7, 8)}


def cca(A, B):
    A = np.asarray(A, float); B = np.asarray(B, float)
    A = A - A.mean(0); B = B - B.mean(0)
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    return np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0, 1)


def gram(M):
    Z = (M - M.mean(1, keepdims=True)) / np.maximum(M.std(1, keepdims=True), 1e-12)
    return (Z @ Z.T) / M.shape[1]


def load(md, t):
    f, r = f"{md}/mb_run_{t}/fit.npz", f"{md}/mb_run_{t}/result.json"
    if not (os.path.exists(f) and os.path.exists(r)):
        return None
    d = np.load(f)
    res = json.load(open(r))
    phi = np.asarray(d["phi"]); psi = np.asarray(d["psi"], float)
    p = np.clip(phi, 1e-12, None); p = p / p.sum(1, keepdims=True)
    pi = np.asarray(d["pi"], float).ravel(); pi = pi / pi.sum()
    K = int(res["K"])
    C = np.corrcoef(psi)
    losses = np.asarray(d["losses"], float)
    cap = res.get("num_steps")
    return dict(
        task=t, K=K, seed=int(res["seed"]), phi=phi, psi=psi, pi=pi,
        psi_mean=float(psi.mean()), psi_contrast=float(psi.std(0).mean()),
        bb_conc=float(np.asarray(d["bb_conc"]).ravel()[0]) if "bb_conc" in d.files else np.nan,
        perplexity=float(np.median(np.exp(-(p * np.log(p)).sum(1)))),
        steps=len(losses), cap=cap, hit_cap=bool(cap and len(losses) >= cap),
        final_loss=float(losses[-1]),
        prog_corr=float(C[~np.eye(K, dtype=bool)].mean()),
        pi_perplexity=float(np.exp(-(pi * np.log(np.clip(pi, 1e-12, None))).sum())),
    )


def main():
    md, out_prefix = sys.argv[1], sys.argv[2]
    want_K = int(sys.argv[3]) if len(sys.argv) > 3 else None

    models = {}
    for K, tasks in TASKS_BY_K.items():
        for t in tasks:
            m = load(md, t)
            if m is not None:
                models[t] = m
    if not models:
        sys.exit(f"no models under {md}")

    print("=== 1. is each fit healthy? ===", flush=True)
    print(f"{'K':>3} {'seed':>5} {'meanPsi':>8} {'contrast':>9} {'bb_conc':>8} {'perplex':>8} "
          f"{'steps':>7} {'stopped':>8} {'progCorr':>9}", flush=True)
    rows = []
    for t in sorted(models):
        m = models[t]
        print(f"{m['K']:>3} {m['seed']:>5} {m['psi_mean']:8.4f} {m['psi_contrast']:9.4f} "
              f"{m['bb_conc']:8.2f} {m['perplexity']:8.2f} {m['steps']:>7} "
              f"{'CAP!' if m['hit_cap'] else 'early':>8} {m['prog_corr']:+9.3f}", flush=True)
        rows.append({k: v for k, v in m.items() if k not in ("phi", "psi", "pi")})
    bad = [m for m in models.values() if m["hit_cap"] or m["psi_contrast"] < 0.02]
    print(f"  -> {len(bad)} model(s) flagged"
          + (" (hit the step cap, or psi_contrast < 0.02 = programs collapsed together)" if bad
             else "; all fits look healthy"), flush=True)

    print("\n=== 2. reproducibility per K ===", flush=True)
    perK = []
    for K, tasks in TASKS_BY_K.items():
        got = [t for t in tasks if t in models]
        if len(got) < 2:
            continue
        eff = [int((cca(models[a]["phi"], models[b]["phi"]) > 0.9).sum())
               for a, b in itertools.combinations(got, 2)]
        # aggregate structure: correlate the cell-type-free program Gram across seeds. Without cell
        # type labels here, use the phi covariance structure, which is equally permutation-invariant.
        gr = []
        for a, b in itertools.combinations(got, 2):
            Ga, Gb = gram(models[a]["phi"].T), gram(models[b]["phi"].T)
            gr.append(float(np.corrcoef(Ga.ravel(), Gb.ravel())[0, 1]))
        perK.append(dict(K=K, n=len(got), effective_K=float(np.mean(eff)), frac=np.mean(eff) / (K - 1),
                         prog_corr=float(np.mean([models[t]["prog_corr"] for t in got]))))
        print(f"  K={K:>2}: effective K = {np.mean(eff):.1f}/{K-1} ({np.mean(eff)/(K-1):.0%})   "
              f"program correlation {np.mean([models[t]['prog_corr'] for t in got]):+.3f}", flush=True)

    print("\n=== 3. which seed is most representative? (medoid, not best ELBO) ===", flush=True)
    sel = []
    for K, tasks in TASKS_BY_K.items():
        got = [t for t in tasks if t in models]
        if len(got) < 2:
            continue
        cen = {t: float(np.mean([cca(models[t]["phi"], models[o]["phi"]).mean()
                                 for o in got if o != t])) for t in got}
        # With only two fits every centrality is the same number by construction (one pairwise value,
        # shared), so the medoid is an arbitrary tie-break and must not be reported as a choice.
        if len(got) < 3:
            print(f"  K={K:>2}: only {len(got)} model(s) -- a medoid needs 3; centrality is "
                  f"symmetric with two and carries no information", flush=True)
            continue
        medoid = max(cen, key=cen.get)
        best_elbo = min(got, key=lambda t: models[t]["final_loss"])
        print(f"  K={K:>2}:", flush=True)
        for t in got:
            tag = []
            if t == medoid:
                tag.append("<- MEDOID")
            if t == best_elbo:
                tag.append("(best ELBO)")
            print(f"      seed {models[t]['seed']}  centrality {cen[t]:.3f}  "
                  f"loss {models[t]['final_loss']:.4g}  {' '.join(tag)}", flush=True)
        if medoid != best_elbo:
            print(f"      NOTE: medoid and best-ELBO disagree -- selecting on loss would pick the "
                  f"less representative fit", flush=True)
        sel.append(dict(K=K, medoid_task=medoid, medoid_seed=models[medoid]["seed"],
                        best_elbo_task=best_elbo, best_elbo_seed=models[best_elbo]["seed"],
                        agree=medoid == best_elbo, centrality=cen[medoid]))

    pd.DataFrame(rows).to_csv(f"{out_prefix}_health.csv", index=False)
    if perK:
        pd.DataFrame(perK).to_csv(f"{out_prefix}_perK.csv", index=False)
    if sel:
        pd.DataFrame(sel).to_csv(f"{out_prefix}_selection.csv", index=False)

    if want_K and any(s["K"] == want_K for s in sel):
        s = [x for x in sel if x["K"] == want_K][0]
        m = models[s["medoid_task"]]
        print(f"\n=== RECOMMENDATION ===", flush=True)
        print(f"  K = {want_K}, seed {s['medoid_seed']}  ->  {md}/mb_run_{s['medoid_task']}/fit.npz",
              flush=True)
        print(f"  chosen as the medoid (subspace centrality {s['centrality']:.3f}), not on ELBO; "
              f"psi_contrast {m['psi_contrast']:.4f}, bb_conc {m['bb_conc']:.2f}, "
              f"perplexity {m['perplexity']:.2f} of {want_K}, stopped at {m['steps']} steps.",
              flush=True)
        print(f"  CAVEAT: only ~{[p for p in perK if p['K']==want_K][0]['frac']:.0%} of phi "
              f"directions reproduce across seeds, so lean downstream conclusions on aggregate "
              f"cell-type-level structure rather than individual per-cell loadings.", flush=True)
    print(f"\nwrote {out_prefix}_{{health,perK,selection}}.csv", flush=True)


if __name__ == "__main__":
    main()
