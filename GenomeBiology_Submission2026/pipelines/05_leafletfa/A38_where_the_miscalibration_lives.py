#!/usr/bin/env python
"""A38: decompose the FDR miscalibration. It is NOT saturation, and it is NOT the FDP rule.

Run: python A38_where_the_miscalibration_lives.py --dir <DS_CALIBRATION>

Why this exists
---------------
The working diagnosis all through this effort had two causes: an over-narrow posterior in the
mid range, and saturation at prob_de = 1.0 where the Bayesian-FDP rule reports zero risk for a
top set that is not actually clean. The Dirichlet-multinomial addresses the first exactly. When
it closed only about half the gap, saturation was the obvious suspect for the rest.

It is not. This script measures it directly on the cached per-ATSE scores, which cost nothing
to re-analyse, and the suspect has an alibi.

What it finds
-------------
Splitting the called set at prob_de >= 0.999, baseline model, nominal alpha 0.05:

                    calls   realised FDP    share of all false discoveries
  effect 0.25
    saturated        2069        0.0174                 5%
    NOT saturated    2302        0.2798                95%
  effect 0.50
    saturated        6032        0.0056                 2%
    NOT saturated    4127        0.4875                98%

The saturated ATSEs are the ACCURATE ones. They are 47-59% of all calls and contribute 2-5% of
the false discoveries. The Jeffreys floor applied earlier is still correct, but it was fixing a
negligible contributor. Essentially all of the miscalibration comes from calls in the mid range,
where prob_de is high but not pinned, and those are false 28% (effect 0.25) to 49% (effect 0.50)
of the time.

The FDP rule is also exonerated
-------------------------------
Its predicted FDP tracks the nominal alpha almost exactly (0.0497 at alpha 0.05), so the rule is
being applied correctly -- it is being fed dishonest inputs. The long-standing worry that its
independence assumption is the problem does not survive contact with this: E[FDP] = mean(1 - prob)
holds in expectation whenever prob is calibrated, whatever the dependence between ATSEs.
Correlation inflates the VARIANCE of the realised FDP, not its mean. A 3-4x bias cannot come
from there.

The actual mechanism: TV is positively biased under the null
------------------------------------------------------------
prob_de = P(TV > delta) with delta = 0.10 fixed a priori, and TV = 0.5 * sum_j |Delta_j| is a
sum of ABSOLUTE values. Estimation noise in psi and phi cannot cancel inside that sum, so a
truly null event does not score TV = 0:

                       true delta    estimated TV (median / mean / p90)   % of nulls over delta
  effect 0.25             0.0000          0.0217 / 0.0305 / 0.0670                3.9%
  effect 0.50             0.0000          0.0247 / 0.0392 / 0.0879                8.4%

And the separation is worse than that makes it sound. At effect 0.25 the median TV of a TRUE
positive is 0.0959 -- BELOW the 0.10 threshold that a tail of null events clears. True effects
sit under the threshold while null noise crosses it, which is exactly the shape that produces
low power and high FDP at the same time, and it is what the low-rank ceiling
(|beta_j| <= |dphi|_1 / 2) does to effect sizes.

Why this reorders the work
--------------------------
A bias is not fixed by widening a posterior. That is why the DM -- which does correct the
effective sample size exactly, and does widen q(psi) by 1.29x -- only got about halfway. No
likelihood change will fix a statistic compared against a threshold it should not be compared
against.

The fix is to stop treating delta = 0.10 as an absolute threshold and calibrate TV against its
own null distribution: per-ATSE null TV -> empirical p -> BH. That was item E in the plan,
listed as independent of the model work. It is not independent; it is the main event.

The DM and the simplex remain worth keeping on their own merits -- 100% valid compositions and
about 9x lower seed-to-seed spread -- but they should be presented as fixing identifiability and
consistency, not as the calibration fix.
"""

import argparse
import glob
import os

import numpy as np


def fdr_select(prob, alpha):
    """Bayesian FDR: rank by prob_de, take the largest set whose mean lfdr stays under alpha."""
    order = np.argsort(-prob)
    lfdr = 1.0 - prob[order]
    cum = np.cumsum(lfdr) / np.arange(1, len(lfdr) + 1)
    k = int(np.searchsorted(cum, alpha, side="right"))
    sel = np.zeros(len(prob), dtype=bool)
    sel[order[:k]] = True
    return sel, (cum[k - 1] if k else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--sat", type=float, default=0.999,
                    help="prob_de at or above this counts as saturated")
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.10])
    args = ap.parse_args()

    for eff in ("0.25", "0.5"):
        fs = sorted(glob.glob(os.path.join(
            args.dir, f"simscores_sim_heart_K5_binom_eff{eff}_frac0.3_seed0_*.npz")))
        if not fs:
            continue
        print("=" * 88)
        print(f"BASELINE model, effect {eff}  ({len(fs)} contrasts)")
        print("=" * 88)

        print("\n1. IS IT SATURATION?  split the called set at prob_de >= "
              f"{args.sat}")
        print(f"{'alpha':>6} {'predicted':>10} {'realised':>9} {'ratio':>6} | "
              f"{'sat n':>7} {'sat FDP':>8} | {'unsat n':>8} {'unsat FDP':>10}")
        for a in args.alphas:
            P = R = S = SF = U = UF = 0.0
            for f in fs:
                z = np.load(f)
                p = z["prob_de"].astype(np.float64)
                nul = z["is_null"]
                sel, pred = fdr_select(p, a)
                if not sel.sum():
                    continue
                P += pred * sel.sum()
                R += (sel & nul).sum()
                sat = sel & (p >= args.sat)
                uns = sel & (p < args.sat)
                S += sat.sum(); SF += (sat & nul).sum()
                U += uns.sum(); UF += (uns & nul).sum()
            tot = S + U
            print(f"{a:>6.2f} {P/tot:>10.4f} {R/tot:>9.4f} {R/max(P,1e-9):>6.2f} | "
                  f"{S:>7.0f} {SF/max(S,1):>8.4f} | {U:>8.0f} {UF/max(U,1):>10.4f}")
        print("  The saturated set is the ACCURATE one. The rule's predicted FDP tracks alpha,")
        print("  so the rule is fine -- prob_de is not.")

        print("\n2. WHY:  TV is positively biased under the null")
        tv_n, tv_a, p_n, td_n = [], [], [], []
        for f in fs:
            z = np.load(f)
            nul = z["is_null"]
            tv_n.append(z["tv"][nul]); tv_a.append(z["tv"][~nul])
            p_n.append(z["prob_de"][nul]); td_n.append(z["true_delta"][nul])
        tv_n = np.concatenate(tv_n); tv_a = np.concatenate(tv_a)
        p_n = np.concatenate(p_n); td_n = np.concatenate(td_n)
        print(f"  null ATSEs: n={len(tv_n):,}, true delta median {np.median(td_n):.4f}")
        print(f"    estimated TV: median {np.median(tv_n):.4f}, mean {tv_n.mean():.4f}, "
              f"p90 {np.percentile(tv_n, 90):.4f}")
        print(f"    over the delta={args.delta} threshold: {100*(tv_n > args.delta).mean():.1f}% "
              f"of nulls")
        print(f"    their prob_de > 0.5: {100*(p_n > 0.5).mean():.1f}%, "
              f"> 0.9: {100*(p_n > 0.9).mean():.1f}%")
        print(f"  TRUE positives: TV median {np.median(tv_a):.4f}"
              f"{'  <-- BELOW the threshold' if np.median(tv_a) < args.delta else ''}")
        print("  A sum of absolute values cannot cancel noise, so a null event scores TV > 0.")
        print("  Widening a posterior does not remove a bias -- which is why the DM only")
        print("  got about halfway.\n")


if __name__ == "__main__":
    main()
