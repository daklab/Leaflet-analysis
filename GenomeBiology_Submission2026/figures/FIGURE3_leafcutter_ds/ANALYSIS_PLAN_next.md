# What to test next, to find what actually improves the model

Written 2026-08-03 while the current round runs. Purpose: replace ad-hoc single tests with one
design that isolates which change buys which property.

## Three properties, one metric each

Fixing a metric per property up front stops us optimising whichever number happens to move.

| property | metric | current value |
|---|---|---|
| **consistency** | cross-seed correlation of the ATSE-level TV statistic, plus top-2000 overlap | psi composition r = 0.95–0.96, but the TV posterior sd (0.0137) is no larger than the across-seed spread (0.0130) |
| **calibration** | realised FDP vs nominal on simulations with known truth, at nominal 0.05 | 0.156 at effect 0.25, 0.199 at effect 0.50 (target: ≤ 0.05) |
| **meaningful ATSEs** | fraction of program-ATSE pairs with psi summing to [0.9, 1.1]; and AUROC vs LeafCutter restricted to high-effect events | 86% valid; AUROC 0.62–0.66 |

Report all three for every candidate. A change that fixes one and breaks another is not a win.

## Five candidate changes

Three alter the model and are separable, so they can be crossed. Two are post-hoc and layer on
top of whatever model wins.

**Model changes**
- **A. psi simplex** — within-ATSE softmax so a program's junction usages sum to 1 by construction.
  Directly targets *meaningful ATSEs*; predicted to help *consistency* by removing the
  unidentified sum direction where seeds disagree (r = 0.51–0.66).
- **B. effective n** — down-weight each junction's likelihood, or move to Dirichlet-multinomial.
  Targets *calibration* by widening an over-narrow posterior. Should also reduce the 2,542 ATSEs
  stuck at confidence exactly 1.0.
- **C. phi uncertainty** — save q(phi) during training and propagate it into the group means.
  Targets *calibration*; also the only route to a downsampling panel that responds to cell count.

**Post-hoc, no retrain**
- **D. Jeffreys estimator** — (successes + 0.5)/(draws + 1) for prob_de. One line, removes the
  absurd zero-FDP. Apply regardless of what else wins.
- **E. empirical calibration** — convert TV to an empirical p-value against the donor-split null,
  then BH. Does not depend on fixing the model at all, and is what Reviewer 1 asked for.

## The design

Run the **2 × 2 × 2 factorial of A, B, C on simulations**, 3 seeds each = 24 fits. Simulations
because truth is known, K=5 and 6,986 cells so each fit is ~70 min, and all 24 run in parallel.

That answers, for each property, how much each change buys on its own and whether they interact —
which the current one-at-a-time tests cannot. Specifically it settles:

- does A alone help consistency, or only B+C
- is B enough, or does calibration need C too (the DM decision hangs on this)
- does A cost anything at high-effect events, where the current slack appears load-bearing

Then take **only the winning combination** to the atlas, 3 seeds at one K, and confirm the same
three metrics hold on real data before touching the released models.

## What decides the Dirichlet-multinomial

Cell B in the factorial is the down-weighting approximation. If it closes most of the calibration
gap, the DM is a refinement and can wait. If it closes some but stalls, the DM is justified
because it applies the same correction exactly. If it closes none, effective sample size is not
the cause and the DM would be wasted — look instead at the guide.

## The contingency nobody has tested

The guide is `AutoNormal`, i.e. mean-field: it assumes every latent is independent in the
posterior. Mean-field variational inference is known to understate posterior variance, and psi and
phi are strongly coupled here because only their product is identified.

The evidence is suggestive: individual psi widths look reasonable (median logit sd 0.82–1.12,
larger than the across-seed spread), yet the aggregated TV statistic is too tight, with its
posterior sd no larger than seed-to-seed variation. Aggregating many individually-uncertain terms
under an independence assumption is exactly how that happens.

If the factorial leaves a calibration gap, the next candidate is a structured guide
(`AutoLowRankMultivariateNormal`) or an importance-weighted objective — a bigger change, held in
reserve rather than started now.

## Order of work

1. Read the current round (EFFN, SIMPLEX, phi E-step) — may already narrow the factorial
2. Apply D, it is free
3. Build E, it is independent of everything else and unblocks the figure
4. Launch the factorial
5. Winning combination to the atlas, confirm on real data
6. Only then decide on the DM or a structured guide
