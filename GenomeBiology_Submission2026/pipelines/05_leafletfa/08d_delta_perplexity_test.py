#!/usr/bin/env python
"""Step 08d: is the reported phi perplexity a FINDING, or a restatement of the prior?

THE PROBLEM
-----------
phi_c ~ Dirichlet(delta * pi). With pi ~ 1/K the per-component concentration is delta/K, and that
alone -- with NO data at all -- already implies a particular phi perplexity. At delta ~ 1 and K = 20,
delta/K = 0.05 implies a median perplexity of ~2.5. The manuscript reports a median SP perplexity
of 2.46.

That does not make the finding wrong, but it means perplexity ALONE cannot distinguish
"the model discovered that cells use ~2.5 programs" from "we chose a prior that says so".

THE TEST
--------
For each delta in the ablation (learned, 1.0, K), compare:

  observed  = perplexity of the FITTED phi
  prior     = perplexity of phi drawn from Dirichlet(delta * pi) with NO data (the null)

  observed ~= prior   -> the PRIOR is driving perplexity. Report it as a modelling choice, not a
                         result, and do not interpret the value biologically.
  observed <  prior   -> the DATA sharpened phi beyond what the prior implies -> a real finding.

Also reports whether observed perplexity TRACKS delta/K across the ablation. If it does, the number
is a knob, not a measurement.

Usage:
    python 08d_delta_perplexity_test.py <model_dir> [--out delta_perplexity.csv] [--draws 20000]
"""
import sys, os, json, gzip, pickle, argparse, numpy as np, pandas as pd


def load_model(run_dir):
    p = os.path.join(run_dir, "leafletfa_model.pkl.gz")
    if not os.path.exists(p):
        return None
    d = {}
    with gzip.open(p, "rb") as f:
        while True:
            try:
                d.update(pickle.load(f))
            except EOFError:
                break
    return d


def perplexity(phi):
    """exp(entropy) per cell = effective number of programs used. 1 = one program, K = uniform."""
    phi = np.asarray(phi, dtype=float)
    phi = phi / (phi.sum(1, keepdims=True) + 1e-12)
    return np.exp(-(phi * np.log(phi + 1e-12)).sum(1))


def prior_perplexity(delta, pi, n_draws, rng):
    """Perplexity of phi drawn from the PRIOR alone -- Dirichlet(delta * pi), no data."""
    alpha = np.asarray(delta, dtype=float) * np.asarray(pi, dtype=float)
    alpha = np.clip(alpha, 1e-6, None)
    return perplexity(rng.dirichlet(alpha, size=n_draws))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--out", default="delta_perplexity.csv")
    ap.add_argument("--draws", type=int, default=20000)
    a = ap.parse_args()

    params = json.load(open(os.path.join(a.model_dir, "parameter_combinations.json")))
    rng = np.random.default_rng(0)
    rows = []
    for i, c in enumerate(params):
        d = load_model(os.path.join(a.model_dir, f"run_{i}"))
        if d is None or "assign_post" not in d:
            continue
        phi = np.asarray(d["assign_post"], dtype=float)
        pi  = np.asarray(d.get("pi", np.full(phi.shape[1], 1.0 / phi.shape[1])), dtype=float)
        pi  = pi / pi.sum()
        K   = phi.shape[1]

        # delta actually USED: the ablation pins it, otherwise the model learned it
        delta = c.get("delta_fixed")
        learned = delta is None
        if learned:
            dc = d.get("dir_conc")
            delta = float(np.asarray(dc)) if dc is not None else np.nan

        obs = float(np.median(perplexity(phi)))
        pri = float(np.median(prior_perplexity(delta, pi, a.draws, rng))) if np.isfinite(delta) else np.nan
        rows.append(dict(run=i, arm=c["arm"], K=K, seed=c.get("seed"),
                         delta=delta, delta_learned=learned, delta_over_K=delta / K,
                         obs_perplexity=obs, prior_perplexity=pri,
                         ratio_obs_over_prior=obs / pri if pri and np.isfinite(pri) else np.nan))

    if not rows:
        sys.exit(f"no models with assign_post found under {a.model_dir}")
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)

    pd.set_option("display.width", 200)
    print("PER-CONFIG (median across cells; prior = Dirichlet(delta*pi) with NO data)\n")
    g = (df.groupby(["arm", "K", "delta_learned"], dropna=False)
           .agg(delta=("delta", "mean"), delta_over_K=("delta_over_K", "mean"),
                observed=("obs_perplexity", "mean"), observed_sd=("obs_perplexity", "std"),
                prior=("prior_perplexity", "mean"), ratio=("ratio_obs_over_prior", "mean"),
                n=("run", "size")).reset_index())
    print(g.round(3).to_string(index=False))

    print("\nVERDICT")
    for _, r in g.iterrows():
        if not np.isfinite(r["ratio"]):
            continue
        if r["ratio"] > 0.9:
            v = "PRIOR-DRIVEN -- observed ~= prior; perplexity is a modelling choice, not a finding"
        elif r["ratio"] < 0.7:
            v = "DATA-DRIVEN -- phi is sharper than the prior implies"
        else:
            v = "partially data-driven"
        print(f"  {r['arm']:<22} K={int(r['K']):<3} delta={r['delta']:.2f}  "
              f"obs={r['observed']:.2f} vs prior={r['prior']:.2f}  ratio={r['ratio']:.2f}  {v}")

    # does observed perplexity simply TRACK delta/K?
    ok = df.dropna(subset=["delta_over_K", "obs_perplexity"])
    if ok.delta_over_K.nunique() > 2:
        from scipy.stats import spearmanr
        rho, p = spearmanr(ok.delta_over_K, ok.obs_perplexity)
        print(f"\n  observed perplexity vs delta/K:  rho={rho:+.2f} (p={p:.2g})")
        print("  " + ("*** perplexity TRACKS the prior knob -- it is not a measurement of the data."
                      if abs(rho) > 0.7 else
                      "perplexity does not simply track delta/K -- the data constrains it."))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
