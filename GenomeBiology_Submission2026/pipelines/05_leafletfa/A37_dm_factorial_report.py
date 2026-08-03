#!/usr/bin/env python
"""A37: collect the A36 factorial and score it against the three tracker properties.

Run: python A37_dm_factorial_report.py --dir <DM_TEST dir>

The decision rule, fixed before the numbers land so it cannot be fitted to them:

  A cell WINS on calibration if its realised FDP is at or below nominal at alpha = 0.05, and
  no worse than 1.5x nominal at alpha = 0.10 and 0.02. Beating the baseline is not enough --
  the point of the exercise is calibration, not improvement.

  A cell is DISQUALIFIED if within-ATSE psi validity drops below the current 86%, or if power
  at alpha = 0.05 falls below half the baseline's. A calibrated model with no power is not a
  usable differential-splicing method, and the user's stated goal names both.

  Between qualifying cells, prefer higher power; break ties on lower seed-to-seed spread of
  realised FDP.

Comparison points carried in from earlier work, all at nominal 0.05 on the same simulations:
  current model (betabinomial + beta)   FDP 0.110 @ eff 0.25, 0.162 @ eff 0.50
  A35 (m-1)/m weighting                 FDP 0.053 @ eff 0.25, 0.144 @ eff 0.50
  LeafCutter                            FDP 0.109 @ eff 0.25, 0.047 @ eff 0.50; power 0.92/0.95
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

BASELINE = "betabinomial_beta"
PRIOR = {("sim_heart_K5_binom_eff0.25_frac0.3_seed0", "current"): 0.110,
         ("sim_heart_K5_binom_eff0.25_frac0.3_seed0", "mminus1"): 0.053,
         ("sim_heart_K5_binom_eff0.5_frac0.3_seed0", "current"): 0.162,
         ("sim_heart_K5_binom_eff0.5_frac0.3_seed0", "mminus1"): 0.144}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-steps", type=int, default=1000,
                    help="discard fits trained for fewer steps than this (dry runs)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "a36_*.tsv")))
    if not files:
        print(f"no a36_*.tsv under {args.dir}")
        return
    D = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)
    print(f"loaded {len(files)} result files, {len(D):,} rows")

    # A dry run writes to the same path as the real run and is NOT overwritten if the real job
    # is still going, so a 30-step smoke test can sit in the directory looking like a result.
    # It shows as zero calls at every alpha, which reads as "perfectly conservative" rather than
    # "untrained". Drop anything that did not train.
    if "steps" in D.columns:
        short = D[D.steps < args.min_steps]
        if len(short):
            for (sim, cell, seed, st), _ in short.groupby(["sim", "cell", "seed", "steps"]):
                print(f"  DROPPED {cell} seed {seed} on {sim}: only {st} steps "
                      f"(< --min-steps {args.min_steps})")
            D = D[D.steps >= args.min_steps]
    else:
        print("  WARNING: no `steps` column -- results predate the fit-settings guard, "
              "cannot verify these are full runs")
    have = D.groupby(["sim", "cell"]).seed.nunique()
    print(f"cells present: {D.cell.nunique()} | seeds per (sim, cell): "
          f"{have.min()}-{have.max()}\n")

    for sim, Dsim in D.groupby("sim"):
        print("=" * 84)
        print(sim)
        cur = PRIOR.get((sim, "current"))
        mm = PRIOR.get((sim, "mminus1"))
        if cur is not None:
            print(f"  reference at nominal 0.05: current model {cur:.3f}, "
                  f"A35 (m-1)/m {mm:.3f}")
        print("=" * 84)

        print("\n1. CALIBRATION  realised FDP by nominal alpha (mean over seeds and contrasts)")
        piv = Dsim.pivot_table(index="alpha", columns="cell", values="realised_fdp")
        print(piv.round(4).to_string())

        print("\n2. POWER  fraction of truly differential ATSEs called")
        pw = Dsim.pivot_table(index="alpha", columns="cell", values="power")
        print(pw.round(4).to_string())

        print(f"\n3. SUMMARY at nominal alpha = {args.alpha}")
        A = Dsim[np.isclose(Dsim.alpha, args.alpha)]
        rows = []
        base = A[A.cell == BASELINE]
        base_pw = float(base.power.mean()) if len(base) else np.nan
        base_valid = float(base.frac_valid.mean()) if len(base) else np.nan
        for cell, g in A.groupby("cell"):
            per_seed = g.groupby("seed").realised_fdp.mean()
            fdp = float(per_seed.mean())
            rows.append(dict(
                cell=cell, realised_fdp=fdp, fdp_seed_sd=float(per_seed.std()),
                ratio_to_nominal=fdp / args.alpha, power=float(g.power.mean()),
                frac_valid=float(g.frac_valid.mean()),
                n_saturated=float(g.n_saturated.mean())))
        S = pd.DataFrame(rows).sort_values("realised_fdp")
        print(S.round(4).to_string(index=False))

        print("\n4. VERDICT (rule fixed before the run -- see the module docstring)")
        for _, r in S.iterrows():
            a10 = Dsim[(Dsim.cell == r.cell) & np.isclose(Dsim.alpha, 0.10)].realised_fdp.mean()
            a02 = Dsim[(Dsim.cell == r.cell) & np.isclose(Dsim.alpha, 0.02)].realised_fdp.mean()
            calib = (r.realised_fdp <= args.alpha and a10 <= 1.5 * 0.10 and a02 <= 1.5 * 0.02)
            dq = []
            if r.frac_valid < 0.86:
                dq.append(f"psi validity {100*r.frac_valid:.0f}% < 86%")
            if np.isfinite(base_pw) and r.power < 0.5 * base_pw:
                dq.append(f"power {r.power:.3f} < half baseline {base_pw:.3f}")
            verdict = ("DISQUALIFIED: " + "; ".join(dq)) if dq else \
                      ("CALIBRATED" if calib else "not calibrated")
            print(f"  {r.cell:<32} {verdict}")
        if np.isfinite(base_valid):
            print(f"\n  baseline power {base_pw:.4f}, baseline psi validity "
                  f"{100*base_valid:.1f}%")

    print("\n" + "=" * 84)
    print("CONVERGENCE CAVEAT -- check before trusting any of the above")
    print("=" * 84)
    print("""  A fit that has not converged looks conservative: psi is shrunk toward the prior, effect
  sizes attenuate, and realised FDP falls for the wrong reason. The atlas work already
  measured this (1500 steps attenuates beta ~20% at scale), so a DM cell that appears
  calibrated ONLY because it trained more slowly would be a false win.

  Guard: the four cells share steps, lr, gamma and batch size, so a difference in convergence
  shows up as a difference in POWER at fixed alpha. A cell whose FDP improves while its power
  collapses is undertrained, not calibrated -- which is why the disqualification rule above
  cuts anything below half the baseline's power.""")

    out = os.path.join(args.dir, "a36_factorial_summary.tsv")
    D.to_csv(out, sep="\t", index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
